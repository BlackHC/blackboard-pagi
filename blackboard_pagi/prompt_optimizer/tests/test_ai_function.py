import inspect

import pytest
from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from pydantic import Field, create_model

from blackboard_pagi.prompt_optimizer.ai_function import (
    AIFunctionSpec,
    ai_function,
    get_json_schema_hyperparameters,
    is_not_implemented,
    update_json_schema_hyperparameters,
)
from blackboard_pagi.prompt_optimizer.track_execution import ChatChain
from blackboard_pagi.testing.fake_llm import FakeLLM


def test_get_json_schema_hyperparameters():
    schema = {
        "title": "test",
        "description": "test",
        "properties": {
            "test": {"title": "test", "description": "test", "type": "string"},
            "test2": {"title": "test", "description": "test", "type": "string"},
        },
        "type": "object",
        "required": ["test", "test2"],
    }
    assert get_json_schema_hyperparameters(schema) == {
        "title": "test",
        "description": "test",
        "properties": {
            "test": {"title": "test", "description": "test"},
            "test2": {"title": "test", "description": "test"},
        },
    }


def test_update_json_schema_hyperparameters():
    schema = {
        "title": "test",
        "description": "test",
        "properties": {
            "test": {"title": "test", "description": "test", "type": "string"},
            "test2": {"title": "test", "description": "test", "type": "string"},
        },
        "type": "object",
        "required": ["test", "test2"],
    }
    hyperparameters = {
        "title": "test2",
        "description": "test2",
        "properties": {
            "test": {"title": "test2", "description": "test2"},
            "test2": {"title": "test2", "description": "test2"},
        },
    }
    update_json_schema_hyperparameters(schema, hyperparameters)
    assert schema == {
        "title": "test2",
        "description": "test2",
        "properties": {
            "test": {"title": "test2", "description": "test2", "type": "string"},
            "test2": {"title": "test2", "description": "test2", "type": "string"},
        },
        "type": "object",
        "required": ["test", "test2"],
    }


def not_implemented_function():
    raise NotImplementedError


def test_is_not_implemented_function():
    assert is_not_implemented(not_implemented_function)
    assert not is_not_implemented(lambda: 1)


def test_ai_function_spec_from_function():
    def f(llm: BaseLLM, a: str, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    ai_function_spec = AIFunctionSpec.from_function(f)
    assert ai_function_spec.docstring == "Test docstring."
    assert ai_function_spec.signature == inspect.signature(f)
    assert ai_function_spec.input_model.schema() == create_model("FInputs", a=(str, ...), b=(int, 1)).schema()
    assert ai_function_spec.output_model.schema() == create_model("FOutput", result=(str, ...)).schema()


def test_ai_function_first_param():
    def f(llm: BaseLLM, a: str, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    assert AIFunctionSpec.from_function(f).input_model.schema() == {
        'properties': {
            'a': {'title': 'A', 'type': 'string'},
            'b': {'title': 'B', 'type': 'integer', 'default': 1},
        },
        'required': ['a'],
        'title': 'FInputs',
        'type': 'object',
    }

    def g(chat_model: BaseChatModel, a: str, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    assert AIFunctionSpec.from_function(g).input_model.schema() == {
        'properties': {
            'a': {'title': 'A', 'type': 'string'},
            'b': {'title': 'B', 'type': 'integer', 'default': 1},
        },
        'required': ['a'],
        'title': 'GInputs',
        'type': 'object',
    }

    def h(chat_chain: ChatChain, a: str, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    assert AIFunctionSpec.from_function(h).input_model.schema() == {
        'properties': {
            'a': {'title': 'A', 'type': 'string'},
            'b': {'title': 'B', 'type': 'integer', 'default': 1},
        },
        'required': ['a'],
        'title': 'HInputs',
        'type': 'object',
    }

    # with a wrong type
    with pytest.raises(ValueError):

        def i(x: int, a: str, b: int = 1) -> str:
            """Test docstring."""
            raise NotImplementedError

        AIFunctionSpec.from_function(i)


def test_ai_function_spec_from_function_with_field():
    # Use Pydantic's Field to specify a default value.
    def f(llm: BaseLLM, a: str, b=Field(3)) -> str:
        """Test docstring."""
        raise NotImplementedError

    ai_function_spec = AIFunctionSpec.from_function(f)

    assert ai_function_spec.input_model.schema() == create_model("FInputs", a=(str, ...), b=(int, 3)).schema()


def test_ai_function_spec_from_function_with_field_description_no_default():
    # Use Pydantic's Field to specify a description.
    def f(llm: BaseLLM, a: str, b: int = Field(..., description="test")) -> str:
        """Test docstring."""
        raise NotImplementedError

    ai_function_spec = AIFunctionSpec.from_function(f)

    assert ai_function_spec.input_model.schema() == {
        'properties': {
            'a': {'title': 'A', 'type': 'string'},
            'b': {'description': 'test', 'title': 'B', 'type': 'integer'},
        },
        'required': ['a', 'b'],
        'title': 'FInputs',
        'type': 'object',
    }


def test_ai_function_spec_from_function_no_docstring():
    def f(llm: BaseLLM, a: str, b: int = 1) -> str:
        raise NotImplementedError

    with pytest.raises(ValueError):
        AIFunctionSpec.from_function(f)


def test_ai_function_spec_from_function_no_return_type():
    def f(llm: BaseLLM, a: str, b: int = 1):
        """Test docstring."""
        raise NotImplementedError

    with pytest.raises(ValueError):
        AIFunctionSpec.from_function(f)


def test_ai_function_spec_from_function_no_parameter_annotation():
    def f(llm: BaseLLM, a, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    with pytest.raises(ValueError):
        AIFunctionSpec.from_function(f)


def test_ai_function_spec_from_function_no_parameter_annotation_but_default():
    def f(llm: BaseLLM, a=1, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    ai_function_spec = AIFunctionSpec.from_function(f)
    assert ai_function_spec.input_model.schema() == create_model("FInputs", a=(int, 1), b=(int, 1)).schema()


def test_ai_function():
    @ai_function
    def add(llm: BaseLLM, a: int, b: int) -> int:
        """
        Add two numbers.
        """
        raise NotImplementedError

    fake_llm = FakeLLM(
        texts=[
            'Add two numbers.\n'
            'The inputs are formatted as JSON using the following schema:\n'
            '{"properties": {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}}, '
            '"required": ["a", "b"]}\n'
            '\n'
            'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n'
            '\n'
            'As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", '
            '"type": "array", "items": {"type": "string"}}}, "required": ["foo"]}}\n'
            'the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": '
            '{"foo": ["bar", "baz"]}} is not well-formatted.\n'
            '\n'
            'Here is the output schema:\n'
            '```\n'
            '{"properties": {"result": {"title": "Result", "type": "integer"}}, "required": ["result"]}\n'
            '```\n'
            'Now output the results for the following inputs:\n'
            '{"a": 1, "b": 2}'
            '{"result": 3}'
        ]
    )

    assert add.__doc__ == '\n        Add two numbers.\n        '
    print(inspect.signature(inspect.unwrap(add)))
    assert add(fake_llm, 1, 2) == 3
    assert add.__doc__ == inspect.unwrap(add).__doc__
