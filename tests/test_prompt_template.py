from dataclasses import dataclass

import pytest

from blackboard_pagi.prompts import prompt_template


@prompt_template.prompt_template("What is the meaning of life?")
class MeaningOfLife:
    pass


@prompt_template.prompt_template("What is the meaning of {term}?", as_dataclass=False)
@dataclass
class MeaningOf:
    term: str

    def __post_init__(self):
        self.term = self.term.lower()


def test_prompt_template():
    """Test prompt template."""
    assert MeaningOfLife().to_prompt() == "What is the meaning of life?"

    assert MeaningOfLife().from_prompt("What is the meaning of life?") == MeaningOfLife()


def test_prompt_template_with_dataclass():
    """Test prompt template with dataclass."""
    assert MeaningOf("Life").to_prompt() == "What is the meaning of life?"
    assert MeaningOf.from_prompt("What is the meaning of life?") == MeaningOf("Life")

    assert MeaningOf("Death").to_prompt() == "What is the meaning of death?"


def test_prompt_template_from_prompt_failure():
    """Test prompt template from prompt failure."""
    with pytest.raises(ValueError):
        MeaningOfLife.from_prompt("What is the meaning of death?")


@dataclass
class HelloName(prompt_template.PromptTemplateMixin):
    prompt_template = "Hello, {name}!"
    name: str


def test_prompt_template_mixin():
    """Test prompt template mixin with dataclass."""
    assert HelloName("Bob").render() == "Hello, Bob!"
    assert HelloName.parse("Hello, John!") == HelloName("John")
