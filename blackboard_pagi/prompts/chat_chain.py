import dataclasses
from dataclasses import dataclass
from typing import ClassVar, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage


@dataclass
class ExtractedValue:
    """
    A class to represent a decision that is motivated by a reason.
    """

    value: object
    # TODO: add this as a weak ref map!
    reason: str


@dataclass
class BooleanDecision(ExtractedValue):
    value: bool

    def __bool__(self):
        return self.value


class ConversionDecision(BooleanDecision):
    pass


@dataclass
class StructuredClarification:
    """
    A class to represent a structured query that can be parsed with the help of another language model.
    """

    def can_convert(self, response: str) -> ConversionDecision:
        raise NotImplementedError

    def convert(self, response: str):
        """
        Converts the response to a python object.
        """
        raise NotImplementedError


class BooleanClarification(StructuredClarification):
    query: ClassVar[str] = """Does this mean yes or no? Please answer with either 'yes' or 'no' as your full answer."""
    no_synonyms: ClassVar[set[str]] = {"no", "false", "0"}
    yes_synonyms: ClassVar[set[str]] = {"yes", "true", "1"}

    def can_convert(self, response: str):
        reduced_response = response.lower().strip().rstrip('.')
        return ConversionDecision(
            reduced_response in self.no_synonyms | self.yes_synonyms,
            f"`{reduced_response}` in `{self.no_synonyms}` | `{self.yes_synonyms}`",
        )

    def convert(self, response: str) -> BooleanDecision:
        return BooleanDecision(any(synonym in response.lower() for synonym in self.yes_synonyms), response)


class StringClarification(StructuredClarification):
    query: ClassVar[str] = "Please repeat only the relevant answer as an string wrapped in \"\" as your full answer."

    def can_convert(self, response: str):
        # Check that the response is a wrapped in quotation marks after we strip whitespace
        return ConversionDecision(
            response.strip().startswith('"') and response.strip().endswith('"'),
            f"`{response}`.strip().startswith('\"') and .strip().endswith('\"')",
        )

    def convert(self, response: str) -> ExtractedValue:
        stripped_response = response.strip()
        assert stripped_response.startswith('"') and stripped_response.endswith('"')
        return ExtractedValue(stripped_response[1:-1], response)


@dataclass
class ChatChain:
    messages: list[BaseMessage]

    @property
    def response(self):
        assert len(self.messages) >= 1
        return self.messages[-1]

    def query(self, chat_model: ChatOpenAI, question: str) -> Tuple[str, "ChatChain"]:
        """Asks a question and returns the result in a single block."""
        # Build messages:
        messages = self.messages + [HumanMessage(content=question)]
        reply = chat_model(messages)
        messages.append(reply)
        return reply.content, ChatChain(messages)

    def branch(self) -> "ChatChain":
        return dataclasses.replace(self, messages=self.messages.copy())


x = StringClarification()
print(x.can_convert('"hello"'))
print(x.can_convert('hello'))
print(x.convert('"hello"'))
