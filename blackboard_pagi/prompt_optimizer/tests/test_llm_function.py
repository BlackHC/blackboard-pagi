#  Blackboard-PAGI - LLM Proto-AGI using the Blackboard Pattern
#  Copyright (c) 2023. Andreas Kirsch
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

# type: ignore
import inspect
import re
import typing

import pytest
from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from pydantic import Field, create_model
from pydantic.generics import GenericModel

from blackboard_pagi.prompt_optimizer.llm_function import (
    LLMBoundSignature,
    LLMStructuredPrompt,
    Output,
    get_concise_type_repr,
    get_json_schema_hyperparameters,
    is_not_implemented,
    llm_function,
    update_json_schema_hyperparameters,
)
from blackboard_pagi.prompts.chat_chain import ChatChain
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


def test_get_concise_type_repr():
    assert get_concise_type_repr(int) == "int"
    assert get_concise_type_repr(typing.List[int]) == "list[int]"
    assert get_concise_type_repr(list[int]) == "list[int]"
    assert get_concise_type_repr(typing.List[typing.List[int]]) == "list[list[int]]"
    assert get_concise_type_repr(list[list[int]]) == "list[list[int]]"
    assert get_concise_type_repr(typing.Dict[str, int]) == "dict[str, int]"
    assert get_concise_type_repr(dict[str, int]) == "dict[str, int]"
    assert get_concise_type_repr(typing.Dict[str, typing.List[int]]) == "dict[str, list[int]]"
    assert get_concise_type_repr(dict[str, list[int]]) == "dict[str, list[int]]"
    assert get_concise_type_repr(typing.Dict[str, typing.Dict[str, int]]) == "dict[str, dict[str, int]]"
    assert get_concise_type_repr(typing.Set[int]) == "set[int]"
    assert get_concise_type_repr(set[int]) == "set[int]"
    assert get_concise_type_repr(typing.Set[typing.Set[int]]) == "set[set[int]]"
    assert get_concise_type_repr(set[set[int]]) == "set[set[int]]"
    assert get_concise_type_repr(typing.Set[typing.List[int]]) == "set[list[int]]"

    class A:
        pass

    assert get_concise_type_repr(A) == "A"
    assert get_concise_type_repr(typing.List[A]) == "list[A]"

    # generic class
    T = typing.TypeVar('T')

    class B(GenericModel, typing.Generic[T]):
        pass

    assert get_concise_type_repr(B) == "B[T]"
    assert get_concise_type_repr(typing.List[B]) == "list[B[T]]"
    assert get_concise_type_repr(typing.List[B[int]]) == "list[B[int]]"
    assert get_concise_type_repr(list[B[int]]) == "list[B[int]]"


def test_llm_function_first_param():
    def f(llm: BaseLLM, a: str = "", b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    assert LLMBoundSignature.from_call(f, (), {}).input_type.schema() == {
        'properties': {
            'a': {'default': '', 'title': 'A', 'type': 'string'},
            'b': {'default': 1, 'title': 'B', 'type': 'integer'},
        },
        'title': 'FInputs',
        'type': 'object',
    }

    def g(chat_model: BaseChatModel, a: str = "", b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    assert LLMBoundSignature.from_call(g, (), {}).input_type.schema() == {
        'properties': {
            'a': {'default': '', 'title': 'A', 'type': 'string'},
            'b': {'default': 1, 'title': 'B', 'type': 'integer'},
        },
        'title': 'GInputs',
        'type': 'object',
    }

    def h(chat_chain: ChatChain, a: str = "", b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    assert LLMBoundSignature.from_call(h, (), {}).input_type.schema() == {
        'properties': {
            'a': {'default': '', 'title': 'A', 'type': 'string'},
            'b': {'default': 1, 'title': 'B', 'type': 'integer'},
        },
        'title': 'HInputs',
        'type': 'object',
    }

    # with a wrong type
    with pytest.raises(ValueError):

        def i(x: int, a: str = "", b: int = 1) -> str:
            """Test docstring."""
            raise NotImplementedError

        LLMBoundSignature.from_call(i, (), {})


def test_llm_bound_signature_from_call():
    def f(llm: BaseLLM, a: str, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    llm_bound_signature = LLMBoundSignature.from_call(f, ("",), {})
    assert llm_bound_signature.docstring == "Test docstring."
    assert llm_bound_signature.signature == inspect.signature(f)
    assert llm_bound_signature.input_type.schema() == create_model("FInputs", a=(str, ...), b=(int, 1)).schema()
    assert llm_bound_signature.output_type.schema() == create_model("Output[str]", return_value=(str, ...)).schema()


def test_llm_function_from_call_first_param():
    def f(llm: BaseLLM, a: str, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    assert LLMBoundSignature.from_call(f, ("",), dict(b=1)).input_type.schema() == {
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

    assert LLMBoundSignature.from_call(g, ("",), {}).input_type.schema() == {
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

    assert LLMBoundSignature.from_call(h, ("",), {}).input_type.schema() == {
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

        LLMBoundSignature.from_call(i, ("",), {})


def test_llm_bound_signature_from_call_with_field():
    # Use Pydantic's Field to specify a default value.
    def f(llm: BaseLLM, a: str, b=Field(3)) -> str:
        """Test docstring."""
        raise NotImplementedError

    llm_bound_signature = LLMBoundSignature.from_call(f, ("",), {})

    assert llm_bound_signature.input_type.schema() == create_model("FInputs", a=(str, ...), b=(int, 3)).schema()


def test_llm_bound_signature_from_call_with_missing_default():
    # Use Pydantic's Field to specify a description.
    def f(llm: BaseLLM, a: str, b: int = Field(..., description="test")) -> str:
        """Test docstring."""
        raise NotImplementedError

    with pytest.raises(TypeError, match=re.escape("missing a required argument: 'b'")):
        LLMBoundSignature.from_call(f, ("",), {})


def test_llm_bound_signature_from_call_with_field_description_no_default():
    # Use Pydantic's Field to specify a description.
    def f(llm: BaseLLM, a: str, b: int = Field(..., description="test")) -> str:
        """Test docstring."""
        raise NotImplementedError

    llm_bound_signature = LLMBoundSignature.from_call(f, ("",), dict(b=1))

    assert llm_bound_signature.input_type.schema() == {
        'properties': {
            'a': {'title': 'A', 'type': 'string'},
            'b': {'description': 'test', 'title': 'B', 'type': 'integer'},
        },
        'required': ['a', 'b'],
        'title': 'FInputs',
        'type': 'object',
    }


def test_llm_bound_signature_from_call_no_docstring():
    def f(llm: BaseLLM, a: str, b: int = 1) -> str:
        raise NotImplementedError

    with pytest.raises(ValueError):
        LLMBoundSignature.from_call(f, ("",), {})


def test_llm_bound_signature_from_call_no_return_type():
    def f(llm: BaseLLM, a: str, b: int = 1):
        """Test docstring."""
        raise NotImplementedError

    with pytest.raises(ValueError):
        LLMBoundSignature.from_call(f, ("",), {})


def test_llm_bound_signature_from_generic_call_to_none():
    T = typing.TypeVar("T")

    def f(llm: BaseLLM, a: T) -> T:
        """Test docstring."""
        raise NotImplementedError

    with pytest.raises(ValueError):
        LLMBoundSignature.from_call(f, (None,), {})


def test_llm_bound_signature_from_call_no_parameter_annotation_but_default():
    def f(llm: BaseLLM, a=1, b: int = 1) -> str:
        """Test docstring."""
        raise NotImplementedError

    llm_bound_signature = LLMBoundSignature.from_call(f, (), {})
    assert llm_bound_signature.input_type.schema() == create_model("FInputs", a=(int, 1), b=(int, 1)).schema()


def test_llm_bound_signature_from_call_generic_input_outputs() -> None:
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

    llm_bound_signature = LLMBoundSignature.from_call(
        f, (), dict(a=GenericType[int](value=0), b=GenericType[str](value=""))
    )
    assert llm_bound_signature.output_type.schema() == Output[GenericType2[int, str]].schema()


def test_llm_bound_signature_from_call_generic_function() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    def f(llm: BaseLLM, a: T, b: S) -> dict[T, S]:
        """Test docstring."""
        raise NotImplementedError

    llm_bound_signature = LLMBoundSignature.from_call(f, (), dict(a=0, b=""))
    assert llm_bound_signature.output_type.schema() == Output[dict[int, str]].schema()


def test_llm_bound_signature_from_call_generic_input_outputs_full_remap() -> None:
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

    llm_bound_signature = LLMBoundSignature.from_call(
        f, (), dict(a=GenericType[int](value=0), b=GenericType[str](value=""), c=GenericType[str](value=""))
    )
    assert llm_bound_signature.output_type.schema() == Output[GenericType2[int, str]].schema()


def test_llm_bound_signature_from_call_generic_input_outputs_multiple_remap() -> None:
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

    llm_bound_signature = LLMBoundSignature.from_call(
        f,
        (),
        dict(
            a=GenericType[int, int](), b=GenericType[int, str](), c=GenericType[str, str](), d=GenericType[int, int]()
        ),
    )
    assert llm_bound_signature.output_type.schema() == Output[GenericType[int, str]].schema()


def test_llm_bound_signature_from_call_generic_input_outputs_full_remap_failed() -> None:
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
        LLMBoundSignature.from_call(
            f,
            (),
            dict(a=GenericType[int](value=0), b=GenericType[str](value=""), c=GenericType[float](value=0.0)),
        )


def test_get_generic_type_map() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    U = typing.TypeVar("U")
    V = typing.TypeVar("V")

    X = typing.TypeVar("X")

    class GenericType(GenericModel, typing.Generic[T, S]):
        value: T
        value2: S

    assert LLMStructuredPrompt.get_generic_type_map(GenericType) == {T: T, S: S}
    assert LLMStructuredPrompt.get_generic_type_map(GenericType[S, T]) == {T: S, S: T}
    assert LLMStructuredPrompt.get_generic_type_map(GenericType[S, T][U, V]) == {T: U, S: V}  # type: ignore
    assert LLMStructuredPrompt.get_generic_type_map(GenericType[S, T][U, V][X, X]) == {T: X, S: X}  # type: ignore
    assert LLMStructuredPrompt.get_generic_type_map(GenericType[U, U][X]) == {T: X, S: X}  # type: ignore
    assert LLMStructuredPrompt.get_generic_type_map(GenericType[int, U][str]) == {T: int, S: str}  # type: ignore


def test_resolve_generic_types() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    # Generic Pydantic model
    class GenericType(GenericModel, typing.Generic[T, S]):
        a: T
        b: S

    assert LLMStructuredPrompt.resolve_generic_types(GenericType, GenericType[int, str](a=1, b="Hello")) == {
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
            '"baz"]}} is not well-formatted.\n\nHere is the input schema:\n```\n{\n"properties": {\n"a": {\n"type": '
            '"integer"\n},\n"b": {\n"type": "integer"\n}\n},\n"required": [\n"a",\n"b"\n]\n}\n```\n\nHere is the '
            'output schema:\n```\n{\n"properties": {\n"return_value": {\n"type": "integer"\n}\n},\n"required": ['
            '\n"return_value"\n]\n}\n```\nNow output the results for the following inputs:\n```\n{\n"a": 1,'
            '\n"b": 2\n}\n```'
            '{"return_value": 3}'
        ]
    )

    assert add.__doc__ == '\n        Add two numbers.\n        '
    assert add(fake_llm, 1, 2) == 3
    assert add.__doc__ == inspect.unwrap(add).__doc__
