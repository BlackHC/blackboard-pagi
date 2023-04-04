import inspect
import re
import typing

import pytest
from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from pydantic import Field, create_model
from pydantic.generics import GenericModel

from blackboard_pagi.prompt_optimizer.llm_function import (
    LLMFunctionSpec,
    get_json_schema_hyperparameters,
    is_not_implemented,
    llm_function,
    update_json_schema_hyperparameters,
)
from blackboard_pagi.prompt_optimizer.track_execution import ChatChain
from blackboard_pagi.prompts.chat_chain import ChatChain as UntrackedChatChain
from blackboard_pagi.testing.fake_chat_model import FakeChatModel
from blackboard_pagi.testing.fake_llm import FakeLLM


def test_get_json_schema_hyperparameters():
    schema = {
        "title": "test",
        "description": "test",
        "properties": {
            "test": {"description": "test", "type": "string"},
            "test2": {"description": "test", "type": "string"},
        },
        "type": "object",
        "required": ["test", "test2"],
    }
    assert get_json_schema_hyperparameters(schema) == {
        "description": "test",
        "properties": {
            "test": {"description": "test"},
            "test2": {"description": "test"},
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


def test_llm_function_spec_from_function():
    def f(llm: BaseLLM, a: str, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    llm_function_spec = LLMFunctionSpec.from_function(f)
    assert llm_function_spec.docstring == "Test docstring."
    assert llm_function_spec.signature == inspect.signature(f)
    assert llm_function_spec.input_model.schema() == create_model("Inputs", a=(str, ...), b=(int, 1)).schema()
    assert llm_function_spec.output_model.schema() == create_model("Outputs", return_value=(str, ...)).schema()


def test_llm_function_first_param():
    def f(llm: BaseLLM, a: str, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    assert LLMFunctionSpec.from_function(f).input_model.schema() == {
        'properties': {
            'a': {'title': 'A', 'type': 'string'},
            'b': {'title': 'B', 'type': 'integer', 'default': 1},
        },
        'required': ['a'],
        'title': 'Inputs',
        'type': 'object',
    }

    def g(chat_model: BaseChatModel, a: str, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    assert LLMFunctionSpec.from_function(g).input_model.schema() == {
        'properties': {
            'a': {'title': 'A', 'type': 'string'},
            'b': {'title': 'B', 'type': 'integer', 'default': 1},
        },
        'required': ['a'],
        'title': 'Inputs',
        'type': 'object',
    }

    def h(chat_chain: ChatChain, a: str, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    assert LLMFunctionSpec.from_function(h).input_model.schema() == {
        'properties': {
            'a': {'title': 'A', 'type': 'string'},
            'b': {'title': 'B', 'type': 'integer', 'default': 1},
        },
        'required': ['a'],
        'title': 'Inputs',
        'type': 'object',
    }

    # with a wrong type
    with pytest.raises(ValueError):

        def i(x: int, a: str, b: int = 1) -> str:
            """Test docstring."""
            raise NotImplementedError

        LLMFunctionSpec.from_function(i)


def test_llm_function_spec_from_function_with_field():
    # Use Pydantic's Field to specify a default value.
    def f(llm: BaseLLM, a: str, b=Field(3)) -> str:
        """Test docstring."""
        raise NotImplementedError

    llm_function_spec = LLMFunctionSpec.from_function(f)

    assert llm_function_spec.input_model.schema() == create_model("Inputs", a=(str, ...), b=(int, 3)).schema()


def test_llm_function_spec_from_function_with_field_description_no_default():
    # Use Pydantic's Field to specify a description.
    def f(llm: BaseLLM, a: str, b: int = Field(..., description="test")) -> str:
        """Test docstring."""
        raise NotImplementedError

    llm_function_spec = LLMFunctionSpec.from_function(f)

    assert llm_function_spec.input_model.schema() == {
        'properties': {
            'a': {'title': 'A', 'type': 'string'},
            'b': {'description': 'test', 'title': 'B', 'type': 'integer'},
        },
        'required': ['a', 'b'],
        'title': 'Inputs',
        'type': 'object',
    }


def test_llm_function_spec_from_function_no_docstring():
    def f(llm: BaseLLM, a: str, b: int = 1) -> str:
        raise NotImplementedError

    with pytest.raises(ValueError):
        LLMFunctionSpec.from_function(f)


def test_llm_function_spec_from_function_no_return_type():
    def f(llm: BaseLLM, a: str, b: int = 1):
        """Test docstring."""
        raise NotImplementedError

    with pytest.raises(ValueError):
        LLMFunctionSpec.from_function(f)


def test_llm_function_spec_from_function_no_parameter_annotation():
    def f(llm: BaseLLM, a, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    with pytest.raises(ValueError):
        LLMFunctionSpec.from_function(f)


def test_llm_function_spec_from_function_no_parameter_annotation_but_default():
    def f(llm: BaseLLM, a=1, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    llm_function_spec = LLMFunctionSpec.from_function(f)
    assert llm_function_spec.input_model.schema() == create_model("Inputs", a=(int, 1), b=(int, 1)).schema()


def test_llm_function_spec_from_call():
    def f(llm: BaseLLM, a: str, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    llm_function_spec = LLMFunctionSpec.from_call(f, FakeLLM(), ("",), {})
    assert llm_function_spec.docstring == "Test docstring."
    assert llm_function_spec.signature == inspect.signature(f)
    assert llm_function_spec.input_model.schema() == create_model("FInputs", a=(str, ...), b=(int, 1)).schema()
    assert llm_function_spec.output_model.schema() == create_model("FOutputs", return_value=(str, ...)).schema()


def test_llm_function_from_call_first_param():
    def f(llm: BaseLLM, a: str, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    assert LLMFunctionSpec.from_call(f, FakeLLM(), ("",), dict(b=1)).input_model.schema() == {
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

    assert LLMFunctionSpec.from_call(g, FakeChatModel(), ("",), {}).input_model.schema() == {
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

    assert LLMFunctionSpec.from_call(h, UntrackedChatChain(FakeChatModel(), []), ("",), {}).input_model.schema() == {
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

        LLMFunctionSpec.from_call(i, 1, ("",), {})


def test_llm_function_spec_from_call_with_field():
    # Use Pydantic's Field to specify a default value.
    def f(llm: BaseLLM, a: str, b=Field(3)) -> str:
        """Test docstring."""
        raise NotImplementedError

    llm_function_spec = LLMFunctionSpec.from_call(f, FakeLLM(), ("",), {})

    assert llm_function_spec.input_model.schema() == create_model("FInputs", a=(str, ...), b=(int, 3)).schema()


def test_llm_function_spec_from_call_with_missing_default():
    # Use Pydantic's Field to specify a description.
    def f(llm: BaseLLM, a: str, b: int = Field(..., description="test")) -> str:
        """Test docstring."""
        raise NotImplementedError

    with pytest.raises(TypeError, match=re.escape("missing a required argument: 'b'")):
        LLMFunctionSpec.from_call(f, FakeLLM(), ("",), {})


def test_llm_function_spec_from_call_with_field_description_no_default():
    # Use Pydantic's Field to specify a description.
    def f(llm: BaseLLM, a: str, b: int = Field(..., description="test")) -> str:
        """Test docstring."""
        raise NotImplementedError

    llm_function_spec = LLMFunctionSpec.from_call(f, FakeLLM(), ("",), dict(b=1))

    assert llm_function_spec.input_model.schema() == {
        'properties': {
            'a': {'title': 'A', 'type': 'string'},
            'b': {'description': 'test', 'title': 'B', 'type': 'integer'},
        },
        'required': ['a', 'b'],
        'title': 'FInputs',
        'type': 'object',
    }


def test_llm_function_spec_from_call_no_docstring():
    def f(llm: BaseLLM, a: str, b: int = 1) -> str:
        raise NotImplementedError

    with pytest.raises(ValueError):
        LLMFunctionSpec.from_call(f, FakeLLM(), ("",), {})


def test_llm_function_spec_from_call_no_return_type():
    def f(llm: BaseLLM, a: str, b: int = 1):
        """Test docstring."""
        raise NotImplementedError

    with pytest.raises(ValueError):
        LLMFunctionSpec.from_call(f, FakeLLM(), ("",), {})


def test_llm_function_spec_from_call_no_parameter_annotation_but_default():
    def f(llm: BaseLLM, a=1, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    llm_function_spec = LLMFunctionSpec.from_call(f, FakeLLM(), (), {})
    assert llm_function_spec.input_model.schema() == create_model("FInputs", a=(int, 1), b=(int, 1)).schema()


def test_llm_function_spec_from_call_generic_input_outputs() -> None:
    T = typing.TypeVar("T")
    V = typing.TypeVar("V")

    class GenericType(GenericModel, typing.Generic[T]):
        value: T

    class GenericType2(GenericModel, typing.Generic[T, V]):
        value: T
        value2: V

    def f(llm: BaseLLM, a: GenericType[T], b: GenericType[V]) -> GenericType2[T, V]:
        """Test docstring."""
        raise NotImplementedError

    llm_function_spec = LLMFunctionSpec.from_call(
        f, FakeLLM(), (), dict(a=GenericType[int](value=0), b=GenericType[str](value=""))
    )
    assert (
        llm_function_spec.output_model.schema()
        == create_model("FOutputs", return_value=(GenericType2[int, str], ...)).schema()
    )


def test_llm_function_spec_from_call_generic_function() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    def f(llm: BaseLLM, a: T, b: S) -> dict[T, S]:
        """Test docstring."""
        raise NotImplementedError

    llm_function_spec = LLMFunctionSpec.from_call(f, FakeLLM(), (), dict(a=0, b=""))
    assert (
        llm_function_spec.output_model.schema() == create_model("FOutputs", return_value=(dict[int, str], ...)).schema()
    )


def test_llm_function_spec_from_call_generic_input_outputs_full_remap() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    U = typing.TypeVar("U")
    V = typing.TypeVar("V")

    class GenericType(GenericModel, typing.Generic[T]):
        value: T

    class GenericType2(GenericModel, typing.Generic[T, S]):
        value: T
        value2: S

    def f(llm: BaseLLM, a: GenericType[U], b: GenericType[V], c: GenericType[V]) -> GenericType2[U, V]:
        """Test docstring."""
        raise NotImplementedError

    llm_function_spec = LLMFunctionSpec.from_call(
        f, FakeLLM(), (), dict(a=GenericType[int](value=0), b=GenericType[str](value=""), c=GenericType[str](value=""))
    )
    assert (
        llm_function_spec.output_model.schema()
        == create_model("FOutputs", return_value=(GenericType2[int, str], ...)).schema()
    )


def test_llm_function_spec_from_call_generic_input_outputs_multiple_remap() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    U = typing.TypeVar("U")
    V = typing.TypeVar("V")

    class GenericType(GenericModel, typing.Generic[T, S]):
        value: T | None = None
        value2: S | None = None

    def f(
        llm: BaseLLM, a: GenericType[U, int], b: GenericType[int, V], c: GenericType[V, V], d: GenericType[U, U]
    ) -> GenericType[U, V]:
        """Test docstring."""
        raise NotImplementedError

    llm_function_spec = LLMFunctionSpec.from_call(
        f,
        FakeLLM(),
        (),
        dict(
            a=GenericType[int, int](), b=GenericType[int, str](), c=GenericType[str, str](), d=GenericType[int, int]()
        ),
    )
    assert (
        llm_function_spec.output_model.schema()
        == create_model("FOutputs", return_value=(GenericType[int, str], ...)).schema()
    )


def test_llm_function_spec_from_call_generic_input_outputs_full_remap_failed() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    U = typing.TypeVar("U")
    V = typing.TypeVar("V")

    class GenericType(GenericModel, typing.Generic[T]):
        value: T

    class GenericType2(GenericModel, typing.Generic[T, S]):
        value: T
        value2: S

    def f(llm: BaseLLM, a: GenericType[U], b: GenericType[V], c: GenericType[V]) -> GenericType2[U, V]:
        """Test docstring."""
        raise NotImplementedError

    with pytest.raises(
        ValueError,
        match=re.escape("Cannot resolve generic type ~V, conflicting resolution: <class 'str'> and <class 'float'>."),
    ):
        LLMFunctionSpec.from_call(
            f,
            FakeLLM(),
            (),
            dict(a=GenericType[int](value=0), b=GenericType[str](value=""), c=GenericType[float](value=0.0)),
        )


def test_llm_function_spec_get_generic_type_map() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    U = typing.TypeVar("U")
    V = typing.TypeVar("V")

    X = typing.TypeVar("X")

    class GenericType(GenericModel, typing.Generic[T, S]):
        value: T
        value2: S

    assert LLMFunctionSpec.get_generic_type_map(GenericType) == {T: T, S: S}
    assert LLMFunctionSpec.get_generic_type_map(GenericType[S, T]) == {T: S, S: T}
    assert LLMFunctionSpec.get_generic_type_map(GenericType[S, T][U, V]) == {T: U, S: V}  # type: ignore
    assert LLMFunctionSpec.get_generic_type_map(GenericType[S, T][U, V][X, X]) == {T: X, S: X}  # type: ignore
    assert LLMFunctionSpec.get_generic_type_map(GenericType[U, U][X]) == {T: X, S: X}  # type: ignore
    assert LLMFunctionSpec.get_generic_type_map(GenericType[int, U][str]) == {T: int, S: str}  # type: ignore


def test_llm_function_spec_resolve_generic_types() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    def f(a: T, b: S) -> dict[T, S]:
        """Test docstring."""
        raise NotImplementedError

    signature = inspect.signature(f)
    parameter_items = list(signature.parameters.items())
    assert LLMFunctionSpec.resolve_generic_types(parameter_items, dict(a=1, b="Hello")) == {
        T: int,
        S: str,
    }


def test_llm_function():
    @llm_function
    def add(llm: BaseLLM, a: int, b: int) -> int:
        """
        Add two numbers.
        """
        raise NotImplementedError

    fake_llm = FakeLLM(
        texts=[
            'Add two numbers.\n\nThe input and output are formatted as a JSON interface that conforms to the JSON '
            'schemas below.\n\nAs an example, for the schema {"properties": {"foo": {"description": "a list of '
            'strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}} the object {"foo": ['
            '"bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", '
            '"baz"]}} is not well-formatted.\n\nHere is the schema for additional date types:\n```\n{}\n```\n\nHere '
            'is the input schema:\n```\n{\n "properties": {\n  "a": {\n   "type": "integer"\n  },\n  "b": {\n   '
            '"type": "integer"\n  }\n },\n "required": [\n  "a",\n  "b"\n ]\n}\n```\n\nHere is the output '
            'schema:\n```\n{\n "properties": {\n  "return_value": {\n   "type": "integer"\n  }\n },\n "required": [\n '
            ' "return_value"\n ]\n}\n```\nNow output the results for the following inputs:\n```\n{\n "a": 1,'
            '\n "b": 2\n}\n```\n '
            '{"return_value": 3}'
        ]
    )

    assert add.__doc__ == '\n        Add two numbers.\n        '
    assert add(fake_llm, 1, 2) == 3
    assert add.__doc__ == inspect.unwrap(add).__doc__
