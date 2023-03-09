from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from blackboard_pagi.prompts.chat_chain import ChatChain

T = TypeVar('T')


@dataclass
class LLMOptional:
    """
    A class to represent a potentially parsed value.
    """

    def is_missing(self):
        raise NotImplementedError()


@dataclass
class LLMValue(LLMOptional, Generic[T]):
    """
    A class to represent a potentially parsed value.
    """

    value: T
    source: str

    def is_missing(self):
        raise False


@dataclass
class LLMNone(LLMOptional):
    """
    A class to represent a missing value (failed conversion).
    """

    details: str

    def is_missing(self):
        return True


class ConversionFailure(LLMNone):
    pass


@dataclass
class LLMBool(LLMValue[bool]):
    def __bool__(self):
        return self.value


V = TypeVar("V", bound=LLMOptional)


class StructuredConverter(Generic[T]):
    query: ClassVar[str]
    """Follow-up query to use in prompt if the conversion fails."""

    def __call__(self, response: str) -> LLMValue[T] | LLMNone:
        raise NotImplementedError()

    def convert_from_chain_response(self, chat_chain: "ChatChain") -> LLMValue[T] | LLMNone:
        """
        Converts a chat chain to a value using the given converter.
        """
        result = self(chat_chain.response)
        if not result.is_missing():
            return result

        for _ in range(2):
            assert isinstance(result, LLMNone)

            retry_prompt = (
                f"""I'm sorry, I'm parsing your output and I failed. Error:\n{result.details}\n\n{self.query}"""
            )
            response, chat_chain = chat_chain.query(retry_prompt)
            result = self(response)
            if not result.is_missing():
                return result

        return result


class BooleanConverter(StructuredConverter):
    query: ClassVar[str] = """Does this mean yes or no? Please answer with either 'yes' or 'no' as your full answer."""
    no_synonyms: ClassVar[set[str]] = {"no", "false", "0"}
    yes_synonyms: ClassVar[set[str]] = {"yes", "true", "1"}

    @classmethod
    def __call__(cls, response: str) -> LLMBool | ConversionFailure:
        reduced_response = response.lower().strip().rstrip('.')
        if reduced_response in cls.no_synonyms | cls.yes_synonyms:
            return LLMBool(any(synonym in response.lower() for synonym in cls.yes_synonyms), response)
        else:
            return ConversionFailure(
                f"Expected response in `{cls.no_synonyms}`"
                " | `{self.yes_synonyms}`, "
                f"but got `{reduced_response}`!"
            )


class StringConverter(StructuredConverter):
    query: ClassVar[str] = "Please repeat only the relevant answer as an string wrapped in '\"' as your full answer."

    def __call__(self, response: str) -> LLMValue[str] | ConversionFailure:
        if response.strip().startswith('"') and response.strip().endswith('"'):
            return LLMValue(response.strip()[1:-1], response)
        else:
            return ConversionFailure(f"Expected response to be wrapped in \"\" but got `{response.strip()}`!")
