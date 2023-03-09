#!/usr/bin/env python
# coding: utf-8

"""Blackboard (AI) Pattern Spike"""
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.llms import BaseLLM
from langchain.schema import AIMessage, BaseMessage, ChatGeneration, ChatResult, Generation, HumanMessage, SystemMessage

from blackboard_pagi.prompts.chat_chain import BooleanClarification, BooleanDecision, ChatChain


@dataclass
class Contribution:
    """
    A class to represent a contribution to a blackboard board
    """

    name: str
    origin: str
    content: str
    feedback: str
    confidence: str
    dependencies: list["Contribution"]

    def to_prompt_context(self):
        """
        Returns a prompt context for the contribution
        """
        return (
            f"## {self.name}\n"
            "\n"
            f"Origin: {self.origin}\n"
            f"Confidence (0 not confident, 1 very confident): {self.confidence}\n"
            f"Dependencies: {[dep.name for dep in self.dependencies]}\n"
            "\n"
            f"{self.content}\n"
            "\n"
            f"### Feedback\n"
            f"{self.feedback}\n"
            "\n"
        )


@dataclass
class OracleChain:
    """A chain of messages that is used to ask an oracle a question"""

    chat_model: ChatOpenAI
    chat_chain: ChatChain

    @property
    def response(self):
        return self.chat_chain.response

    def query(self, question: str) -> Tuple[str, "OracleChain"]:
        """Asks a question and returns the result in a single block."""
        reply, chain = self.chat_chain.query(self.chat_model, question)
        return reply, OracleChain(self.chat_model, chain)

    def branch(self) -> "OracleChain":
        """Branches the chain"""
        return OracleChain(self.chat_model, self.chat_chain.branch())


@dataclass
class Oracle:
    chat_model: ChatOpenAI
    text_model: BaseLLM

    def start_oracle_chain(self, context: str) -> OracleChain:
        """Starts an oracle chain with the given context"""
        # Build messages:
        messages = [
            SystemMessage(
                content="You are an oracle. You try to be truthful and helpful. "
                "You state when you are unsure about something. "
                "You think step by step."
            ),
            AIMessage(
                content="First, what is the context of your request? "
                "Then, let me know your questions, and I will answer each question exactly once."
            ),
            HumanMessage(content="The context is as follows:\n\n" + context),
            # TODO: we might want to add a prompt here to make sure the oracle understands the context.
            AIMessage(content="Ok, I understand the context. What is your question?"),
        ]
        return OracleChain(self.chat_model, ChatChain(messages))


@dataclass
class Blackboard:
    """
    A class to represent a blackboard board
    """

    goal: str
    contributions: list[Contribution] = dataclasses.field(default_factory=list)

    def to_prompt_context(self):
        """
        Returns a prompt context for the blackboard
        """

        context = f"# Goal\n{self.goal}\n\n"
        if self.contributions:
            context += "# Contributions\n" "\n".join(
                [contribution.to_prompt_context() for contribution in self.contributions]
            )
        return context


class KnowledgeSource:
    def can_contribute(self, blackboard: Blackboard) -> bool:
        """
        Returns whether the knowledge source can contribute to the blackboard
        """
        raise NotImplementedError

    def contribute(self, blackboard: Blackboard) -> Contribution:
        """
        Returns a contribution to the blackboard
        """
        raise NotImplementedError


def optionally_include_enumeration(prefix, enumeration, suffix=""):
    if enumeration:
        subprompt = prefix
        if len(enumeration) == 1:
            subprompt += " (" + enumeration[0] + ")"
        else:
            subprompt += ":\n"
            for i, item in enumerate(enumeration):
                subprompt += " - " + item
                # add ';' to the end of every item but the last.
                # add a '.' to the end of the last item.
                if i < len(enumeration) - 1:
                    subprompt += ";\n"

            subprompt += ".\n"
        subprompt += suffix
        return subprompt

    return ""


@dataclass
class Controller:
    blackboard: Blackboard
    oracle: Oracle
    knowledge_sources: list[KnowledgeSource]
    last_reported_success: BooleanDecision | None = None

    def update(self):
        """
        Updates the blackboard
        """
        for knowledge_source in self.knowledge_sources:
            if knowledge_source.can_contribute(self.blackboard):
                self.blackboard.contributions.append(knowledge_source.contribute(self.blackboard))

        reported_success, solution_attempt = self.try_solve()

        self.blackboard.contributions.append(solution_attempt)
        self.last_reported_success = reported_success

        return reported_success

    def try_solve(self) -> Tuple[BooleanDecision, Contribution]:
        """
        Returns a contribution to the blackboard
        """
        chain = self.oracle.start_oracle_chain(self.blackboard.to_prompt_context())
        how_response, chain = chain.query("How could we solve the goal using all the available information?")
        does_response, chain = chain.query("Does this solve the goal or provide a definite answer?")

        boolean_clarification = BooleanClarification()
        does_response_clarification, _ = chain.query(boolean_clarification.query)
        can_convert = boolean_clarification.can_convert(does_response_clarification)
        if not can_convert:
            raise NotImplementedError("We don't know how to handle this yet.")
        does_response = boolean_clarification.convert(does_response_clarification)

        if does_response:
            what_response, chain = chain.query("Hence, what is the solution?")
            how_response += "\n\n" + what_response

        feedback_response, chain = chain.query(
            "How could your response be improved? Is anything missing? Does anything look wrong in hindsight?"
        )
        confidence_response, chain = chain.query(
            "Given all this, how confident are you about your original reponse? Is it likely correct/consistent?"
            "Please self-report a confidence level: 0.0 = no confidence, 1.0 = full confidence."
        )

        name_query = "Please come up with a name for your original response."
        if self.blackboard.contributions:
            name_query += optionally_include_enumeration(
                "Make it unique. It should be different than existing subsections under 'Contributions'",
                [contribution.name for contribution in self.blackboard.contributions],
            )
        name, _ = chain.query(name_query)

        if self.blackboard.contributions:
            dependencies_query = "Please provide the subsection titles this solution directly depends on. "
            dependencies_query += optionally_include_enumeration(
                "(The following subsections exist under 'Contributions'",
                [contribution.name for contribution in self.blackboard.contributions],
                ")",
            )
            dependencies, _ = chain.query(dependencies_query)
        else:
            dependencies = []

        return does_response, Contribution(
            name=name,
            origin="Oracle",
            content=how_response,
            feedback=feedback_response,
            confidence=confidence_response,
            dependencies=[dependencies],
        )


#%%
import langchain
from langchain import OpenAI
from langchain.cache import SQLiteCache

langchain.llm_cache = SQLiteCache(".chat.langchain.db")


class CachedChatOpenAI(ChatOpenAI):
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        messages_prompt = repr(messages)
        if langchain.llm_cache:
            results = langchain.llm_cache.lookup(messages_prompt, self.model_name)
            if results:
                assert len(results) == 1
                result: Generation = results[0]
                chat_result = ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=result.text))],
                    llm_output=result.generation_info,
                )
                return chat_result
        chat_result = super()._generate(messages, stop)
        if langchain.llm_cache:
            assert len(chat_result.generations) == 1
            result = Generation(text=chat_result.generations[0].message.content, generation_info=chat_result.llm_output)
            langchain.llm_cache.update(messages_prompt, self.model_name, [result])
        return chat_result


chat_model = CachedChatOpenAI(max_tokens=512, model_kwargs=dict(temperature=0.0))

text_model = OpenAI(model_name="text-davinci-001", max_tokens=256, model_kwargs=dict(temperature=0.0))

blackboard = Blackboard("How can we solve the problem of global warming?")
oracle = Oracle(chat_model, text_model)
controller = Controller(blackboard, oracle, [])

#%%
controller.update()
