from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI
from langchain.llms import BaseLLM
from langchain.schema import HumanMessage, SystemMessage

from blackboard_pagi.prompts.chat_chain import ChatChain


@dataclass
class Oracle:
    chat_model: ChatOpenAI
    text_model: BaseLLM

    def start_oracle_chain(self, context: str) -> ChatChain:
        """Starts an oracle chain with the given context"""
        # Build messages:
        messages = [
            SystemMessage(
                content="You are an oracle. You try to be truthful and helpful. "
                "You state when you are unsure about something. "
                "You think step by step. You format your output as JSON following the given schema and instructions."
            ),
            # AIMessage(
            #     content="First, what is the context of your request? "
            #     "Then, let me know your questions, and I will answer each question exactly once."
            # ),
            HumanMessage(content="The context is as follows:\n\n" + context),
            # TODO: we might want to add a prompt here to make sure the oracle understands the context.
            # AIMessage(content="Ok, I understand the context. What is your question?"),
        ]
        return ChatChain(self.chat_model, messages)
