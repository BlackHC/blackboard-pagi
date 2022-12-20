import typing
from dataclasses import dataclass
from typing import ClassVar, TypeVar

import langchain.llms
import parse
import typing_extensions
from typing_extensions import dataclass_transform

T = TypeVar("T")


@dataclass
class PromptResult(typing.Generic[T]):
    prompt: T
    response: str


@dataclass
class PromptTemplateMixin:
    """
    A template for the prompt.
    """

    prompt_template: ClassVar[str]

    def render(self) -> str:
        """
        Returns a prompt string with the given arguments.
        """
        return self.__class__.prompt_template.format(**vars(self))

    @classmethod
    def parse(cls, text: str):
        result = parse.parse(cls.prompt_template, text)
        if result is None:
            raise ValueError(f"Could not parse prompt {text}")
        assert len(result.fixed) == 0
        assert set(result.named.keys()) == set(cls.__dataclass_fields__.keys()) - {"prompt_template"}  # type: ignore
        prompt_instance = cls(**result.named)  # type: ignore
        return prompt_instance

    def __call__(self, lmm: langchain.llms.LLM) -> PromptResult[typing_extensions.Self]:  # type: ignore
        response_text = lmm(self.render())
        return PromptResult(prompt=self, response=response_text)


def prompt_template(prompt_template: str, as_dataclass=True):
    """
    Wraps a class to make it a prompt template.
    """

    @dataclass_transform()
    def prompt_wrapper(class_definition: type) -> type:
        if as_dataclass:
            class_definition = dataclass(class_definition)

        new_type = type(
            class_definition.__name__,
            (class_definition, PromptTemplateMixin),
            {"prompt_template": prompt_template},
        )
        return new_type

    return prompt_wrapper
