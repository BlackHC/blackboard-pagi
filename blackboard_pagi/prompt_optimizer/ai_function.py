import dis
import functools
import inspect
import json
import types
import typing
from copy import deepcopy
from dataclasses import dataclass

import pydantic
import typing_extensions
from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseLanguageModel, OutputParserException
from pydantic import BaseModel, create_model

from blackboard_pagi.prompt_optimizer.track_execution import ChatChain, prompt_hyperparameter, track_execution
from blackboard_pagi.prompts.chat_chain import ChatChain as UntrackedChatChain

T = typing.TypeVar("T")
P = typing_extensions.ParamSpec("P")


def get_json_schema_hyperparameters(schema: dict):
    """
    Get the hyperparameters from a JSON schema recursively.

    The hyperparameters are all fields for keys with "title" or "description".
    """
    hyperparameters = {}
    for key, value in schema.items():
        if key in ("title", "description"):
            hyperparameters[key] = value
        elif isinstance(value, dict):
            sub_hyperparameters = get_json_schema_hyperparameters(value)
            if sub_hyperparameters:
                hyperparameters[key] = sub_hyperparameters
    return hyperparameters


def update_json_schema_hyperparameters(schema: dict, hyperparameters: dict):
    """
    Nested merge of the schema dict with the hyperparameters dict.
    """
    for key, value in hyperparameters.items():
        if key in schema:
            if isinstance(value, dict):
                update_json_schema_hyperparameters(schema[key], value)
            else:
                schema[key] = value
        else:
            schema[key] = value


def unwrap_function(f: typing.Callable[P, T]) -> typing.Callable[P, T]:
    # is f a property?
    if isinstance(f, property):
        f = f.fget
    # is f a wrapped function?
    elif hasattr(f, "__wrapped__"):
        f = inspect.unwrap(f)
    elif inspect.ismethod(f):
        f = f.__func__
    else:
        return f

    return unwrap_function(f)


def is_not_implemented(f: typing.Callable) -> bool:
    """Check that a function only raises NotImplementedError."""
    unwrapped_f = unwrap_function(f)

    if not hasattr(unwrapped_f, "__code__"):
        raise ValueError(f"Cannot check whether {f} is implemented. Where is __code__?")

    # Inspect the opcodes
    code = unwrapped_f.__code__
    # Get the opcodes
    opcodes = list(dis.get_instructions(code))
    # Check that it only uses the following opcodes:
    # - RESUME
    # - LOAD_GLOBAL
    # - PRECALL
    # - CALL
    # - RAISE_VARARGS
    valid_opcodes = {
        "RESUME",
        "LOAD_GLOBAL",
        "PRECALL",
        "CALL",
        "RAISE_VARARGS",
    }
    # We allow at most a function of length len(valid_opcodes)
    if len(opcodes) > len(valid_opcodes):
        return False
    for opcode in opcodes:
        if opcode.opname not in valid_opcodes:
            return False
        # Check that the function only raises NotImplementedError
        if opcode.opname == "LOAD_GLOBAL" and opcode.argval != "NotImplementedError":
            return False
        if opcode.opname == "RAISE_VARARGS" and opcode.argval != 1:
            return False
        valid_opcodes.remove(opcode.opname)
    # Check that the function raises a NotImplementedError at the end.
    if opcodes[-1].opname != "RAISE_VARARGS":
        return False
    return True


class TyperWrapper(str):
    """
    A wrapper around a type that can be used to create a Pydantic model.

    This is used to support @classmethods.
    """

    @classmethod
    def __get_validators__(cls) -> typing.Iterator[typing.Callable]:
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(cls, v: type) -> str:
        if not isinstance(v, type):
            raise TypeError("type required")
        return v.__qualname__


@dataclass
class AIFunctionSpec:
    signature: inspect.Signature
    docstring: str
    input_model: typing.Type[BaseModel]
    output_model: typing.Type[BaseModel]

    @staticmethod
    def from_function(f: typing.Callable[P, T]) -> "AIFunctionSpec":
        """Create an AIFunctionSpec from a function."""

        if not is_not_implemented(f):
            raise ValueError("The function must not be implemented.")

        # get clean docstring of
        docstring = inspect.getdoc(f)
        if docstring is None:
            raise ValueError("The function must have a docstring.")
        # get the type of the first argument
        signature = inspect.signature(f, eval_str=True)
        # get all parameters
        parameters = signature.parameters
        # check that there is at least one parameter
        if not parameters:
            raise ValueError("The function must have at least one parameter.")
        # check that the first parameter has a type annotation that is an instance of BaseLanguageModel
        # or a TrackedChatChain
        first_parameter = next(iter(parameters.values()))
        if first_parameter.annotation is not inspect.Parameter.empty:
            if not issubclass(first_parameter.annotation, BaseLanguageModel | ChatChain):
                raise ValueError("The first parameter must be an instance of BaseLanguageModel or ChatChain.")

        # create a pydantic model from the parameters
        parameter_dict = {}
        for i, (parameter_name, parameter) in enumerate(parameters.items()):
            # skip the first parameter
            if i == 0:
                continue
            # every parameter must be annotated or have a default value
            annotation = parameter.annotation
            if annotation is type:
                annotation = TyperWrapper

            if annotation is inspect.Parameter.empty:
                # check if the parameter has a default value
                if parameter.default is inspect.Parameter.empty:
                    raise ValueError(f"The parameter {parameter_name} must be annotated or have a default value.")
                parameter_dict[parameter_name] = parameter.default
            elif parameter.default is inspect.Parameter.empty:
                parameter_dict[parameter_name] = (annotation, ...)
            else:
                parameter_dict[parameter_name] = (
                    annotation,
                    parameter.default,
                )

        # turn the function name into a class name
        class_name_prefix = f.__name__.capitalize()
        # create the model
        input_model = create_model(f"{class_name_prefix}Inputs", **parameter_dict)
        input_model.update_forward_refs()
        # get the return type
        return_type = signature.return_annotation
        if return_type is inspect.Signature.empty:
            raise ValueError("The function must have a return type.")
        # create the output model
        # the return type can be a type annotation or an Annotated type with annotation being a FieldInfo
        if typing.get_origin(return_type) is typing.Annotated:
            return_info = typing.get_args(return_type)
        else:
            return_info = (return_type, ...)
        output_model = create_model(f"{class_name_prefix}Output", return_value=return_info)
        output_model.update_forward_refs()

        return AIFunctionSpec(
            docstring=docstring,
            signature=signature,
            input_model=input_model,
            output_model=output_model,
        )


def get_call_schema(input_type: type[BaseModel], output_type: type[BaseModel]):
    schema = pydantic.schema.schema([input_type, output_type])
    definitions = deepcopy(schema["definitions"])
    # remove title and type from each sub dict in the definitions
    for value in definitions.values():
        value.pop("title")
        value.pop("type")

    return definitions


@dataclass
class AIFunction(typing.Generic[P, T], typing.Callable[P, T]):  # type: ignore
    """
    A callable that can be called with a chat model.
    """

    spec: AIFunctionSpec

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self.__wrapped__, name)

    def __call__(
        self,
        language_model_or_chat_chain: BaseLanguageModel | ChatChain | UntrackedChatChain,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        if isinstance(language_model_or_chat_chain, UntrackedChatChain):
            language_model_or_chat_chain = ChatChain(
                language_model_or_chat_chain.chat_model, language_model_or_chat_chain.messages
            )

        # bind the inputs to the signature
        bound_arguments = self.spec.signature.bind(language_model_or_chat_chain, *args, **kwargs)
        # get the arguments
        arguments = bound_arguments.arguments
        inputs = self.spec.input_model(**arguments)

        parser = PydanticOutputParser(pydantic_object=self.spec.output_model)

        # get the input adn output schema as JSON dict
        schema = get_call_schema(self.spec.input_model, self.spec.output_model)

        update_json_schema_hyperparameters(
            schema,
            prompt_hyperparameter("schema") @ get_json_schema_hyperparameters(schema),
        )

        # create the prompt
        prompt = (
            "{docstring}\n"
            "\n"
            "The input is formatted as a JSON interface of Inputs that conforms to the JSON schema below, "
            "and the output should be formatted as a JSON instance of Outputs that conforms to the JSON "
            "schema below.\n"
            "\n"
            'As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of '
            'strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}}} the object {{'
            '"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": '
            '["bar", "baz"]}}}} is not well-formatted.\n'
            "\n"
            "Here is the schema:\n"
            "```\n"
            "{schema}\n"
            "```\n"
            "\n"
            "Now output the results for the following inputs:\n"
            "```\n"
            "{inputs}\n"
            "```\n"
        ).format(docstring=self.spec.docstring, schema=json.dumps(schema), inputs=inputs.json())

        # get the response
        num_retries = prompt_hyperparameter("num_retries_on_parser_failure") @ 3

        if language_model_or_chat_chain is None:
            raise ValueError("The language model or chat chain must be provided.")
        if isinstance(language_model_or_chat_chain, ChatChain):
            chain = language_model_or_chat_chain
            for _ in range(num_retries):
                output, chain = chain.query(prompt)
                prompt_hyperparameter.track_chain(chain)

                try:
                    parsed_output = parser.parse(output)
                    break
                except OutputParserException as e:
                    prompt = (
                        prompt_hyperparameter("error_prompt") @ "Tried to parse your output but failed:\n\n"
                        + str(e)
                        + prompt_hyperparameter("retry_prompt") @ "\n\nPlease try again and avoid this issue."
                    )
            else:
                raise OutputParserException(f"Failed to parse the output after {num_retries} retries.")
        elif isinstance(language_model_or_chat_chain, BaseChatModel | BaseLLM):
            new_prompt = prompt
            output = ""
            for _ in range(num_retries):
                prompt = new_prompt
                if isinstance(language_model_or_chat_chain, BaseChatModel):
                    output = language_model_or_chat_chain.call_as_llm(prompt)
                elif isinstance(language_model_or_chat_chain, BaseLLM):
                    output = language_model_or_chat_chain(prompt)
                else:
                    raise ValueError("The language model or chat chain must be provided.")

                try:
                    parsed_output = parser.parse(output)
                    break
                except OutputParserException as e:
                    new_prompt = (
                        prompt
                        + prompt_hyperparameter("output_prompt") @ "\n\nReceived the output\n\n"
                        + output
                        + prompt_hyperparameter("error_prompt") @ "Tried to parse your output but failed:\n\n"
                        + str(e)
                        + prompt_hyperparameter("retry_prompt") @ "\n\nPlease try again and avoid this issue."
                    )
            else:
                prompt_hyperparameter.track_llm(prompt, output)
                raise ValueError(f"Failed to parse the output after {num_retries} retries.")
            prompt_hyperparameter.track_llm(prompt, output)
        else:
            raise ValueError("The language model or chat chain must be provided.")

        return parsed_output.return_value  # type: ignore


def ai_function(f: typing.Callable[P, T]) -> AIFunction[P, T]:
    """
    Decorator to wrap a function with a chat model.

    f is a function to a dataclass or Pydantic model.

    The docstring of the function provides instructions for the model.
    """
    if isinstance(f, classmethod):
        return classmethod(ai_function(f.__func__))
    elif isinstance(f, staticmethod):
        return staticmethod(ai_function(f.__func__))
    elif isinstance(f, property):
        return property(ai_function(f.fget), doc=f.__doc__)
    elif isinstance(f, types.MethodType):
        return types.MethodType(ai_function(f.__func__), f.__self__)
    elif hasattr(f, "__wrapped__"):
        return ai_function(f.__wrapped__)
    elif isinstance(f, AIFunction):
        return f
    elif not callable(f):
        raise ValueError(f"Cannot decorate {f} with llm_strategy.")

    specific_ai_function: AIFunction = track_execution(
        functools.wraps(f, updated=())(AIFunction(spec=AIFunctionSpec.from_function(f)))
    )  # type: ignore
    return specific_ai_function
