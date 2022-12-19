from dataclasses import dataclass
from typing import ClassVar

import parse


@dataclass
class PromptTemplateMixin:
    """
    Provides
    """

    prompt_template: ClassVar[str]

    def to_prompt(self) -> str:
        """
        Returns a prompt string with the given arguments.
        """
        return self.__class__.prompt_template.format(**vars(self))

    @classmethod
    def from_prompt(cls, text: str):
        result = parse.parse(cls.prompt_template, text)
        if result is None:
            raise ValueError(f"Could not parse prompt {text}")
        assert len(result.fixed) == 0
        assert set(result.named.keys()) == set(cls.__dataclass_fields__.keys()) - {"prompt_template"}  # type: ignore
        prompt_instance = cls(**result.named)  # type: ignore
        return prompt_instance

    def __call__(self) -> str:
        return self.to_prompt()


def prompt_template(prompt_template: str, as_dataclass=True):
    """
    Wraps a class to make it a prompt template.
    """

    def prompt_wrapper(class_definition):
        if as_dataclass:
            class_definition = dataclass(class_definition)

        new_type = type(
            class_definition.__name__,
            (class_definition, PromptTemplateMixin),
            {"prompt_template": prompt_template},
        )
        return new_type

    return prompt_wrapper
