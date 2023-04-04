import dis
import functools
import inspect
import json
import re
import string
import types
import typing
from copy import deepcopy
from dataclasses import dataclass

import pydantic
import pydantic.schema
import typing_extensions
from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.schema import BaseLanguageModel, OutputParserException
from pydantic import BaseModel, ValidationError, create_model, generics
from pydantic.fields import FieldInfo, Undefined
from pydantic.generics import replace_types

from blackboard_pagi.prompt_optimizer.track_execution_old import ChatChain, prompt_hyperparameter, track_execution
from blackboard_pagi.prompts.chat_chain import ChatChain as UntrackedChatChain

T = typing.TypeVar("T")
P = typing_extensions.ParamSpec("P")
B = typing.TypeVar("B", bound=BaseModel)


def get_json_schema_hyperparameters(schema: dict):
    """
    Get the hyperparameters from a JSON schema recursively.

    The hyperparameters are all fields for keys with "title" or "description".
    """
    hyperparameters = {}
    for key, value in schema.items():
        if key == "description":
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
class LLMFunctionSpec:
    signature: inspect.Signature
    docstring: str
    input_model: typing.Type[BaseModel]
    output_model: typing.Type[BaseModel]
    return_type: type

    def get_call_schema(self) -> dict:
        schema = pydantic.schema.schema([self.input_model, self.output_model])
        definitions: dict = deepcopy(schema["definitions"])
        # remove title and type from each sub dict in the definitions
        for value in definitions.values():
            value.pop("title")
            value.pop("type")

            for property in value.get("properties", {}).values():
                if "title" in property:
                    property.pop("title")

        input_schema = definitions[self.input_model.__name__]
        output_schema = definitions[self.output_model.__name__]
        del definitions[self.input_model.__name__]
        del definitions[self.output_model.__name__]

        schema = dict(
            input_schema=input_schema,
            output_schema=output_schema,
            additional_definitions=definitions,
        )
        return schema

    def get_inputs(self, *args: P.args, **kwargs: P.kwargs) -> BaseModel:
        """Call the function and return the inputs."""
        # bind the inputs to the signature
        bound_arguments = self.signature.bind(None, *args, **kwargs)
        # get the arguments
        arguments = bound_arguments.arguments
        inputs = self.input_model(**arguments)
        return inputs

    @staticmethod
    def from_function(f: typing.Callable[P, T]) -> "LLMFunctionSpec":
        """Create an LLMFunctionSpec from a function."""

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
            if not issubclass(first_parameter.annotation, BaseLanguageModel | ChatChain | UntrackedChatChain):
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
        # create the model
        input_model = create_model("Inputs", **parameter_dict)
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
        output_model = create_model("Outputs", return_value=return_info)
        output_model.update_forward_refs()

        return LLMFunctionSpec(
            docstring=docstring,
            signature=signature,
            input_model=input_model,
            output_model=output_model,
            return_type=return_type,
        )

    @staticmethod
    def from_call(
        f: typing.Callable[P, T], language_model_or_chain, args: P.args, kwargs: P.kwargs
    ) -> "LLMFunctionSpec":
        """Create an LLMFunctionSpec from a function."""

        # get clean docstring of
        docstring = inspect.getdoc(f)
        if docstring is None:
            raise ValueError("The function must have a docstring.")
        # get the type of the first argument
        signature = inspect.signature(f, eval_str=True)
        # get all parameters
        parameters_items: list[tuple[str, inspect.Parameter]] = list(signature.parameters.items())  # type: ignore
        # check that there is at least one parameter
        if not parameters_items:
            raise ValueError("The function must have at least one parameter.")
        # check that the first parameter has a type annotation that is an instance of BaseLanguageModel
        # or a TrackedChatChain
        first_parameter: inspect.Parameter = parameters_items[0][1]
        if first_parameter.annotation is not inspect.Parameter.empty:
            if not issubclass(first_parameter.annotation, BaseLanguageModel | ChatChain):
                raise ValueError("The first parameter must be an instance of BaseLanguageModel or ChatChain.")

        # create a pydantic model from the parameters
        parameter_dict = LLMFunctionSpec.parameter_items_to_field_tuple(parameters_items[1:])

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

        return_type = return_info[0]

        # resolve generic types
        bound_arguments = LLMFunctionSpec.bind(args, kwargs, language_model_or_chain, signature)
        generic_type_map = LLMFunctionSpec.resolve_generic_types(parameters_items[1:], bound_arguments.arguments)
        return_type = LLMFunctionSpec.resolve_type(return_type, generic_type_map)
        return_info = (return_type, return_info[1])

        # turn function name into a class name
        class_name = string.capwords(f.__name__, sep="_").replace("_", "")

        # update parameter_dict types with bound_arguments
        # this ensures that serialize the actual types
        # might not be optimal because the language model won't be aware of original types
        for parameter_name in parameter_dict:
            if parameter_name in bound_arguments.arguments:
                parameter_dict[parameter_name] = (
                    type(bound_arguments.arguments[parameter_name]),
                    parameter_dict[parameter_name][1],
                )

        # create the input model
        input_model = create_model(f"{class_name}Inputs", __module__=f.__module__, **parameter_dict)
        input_model.update_forward_refs()

        output_model = create_model(f"{class_name}Outputs", __module__=f.__module__, return_value=return_info)
        output_model.update_forward_refs()

        return LLMFunctionSpec(
            docstring=docstring,
            signature=signature,
            input_model=input_model,
            output_model=output_model,
            return_type=return_type,
        )

    @staticmethod
    def parameter_items_to_field_tuple(parameters_items: list[tuple[str, inspect.Parameter]]):
        """
        Get the parameter definitions for a function call from the parameters and arguments.
        """
        parameter_dict: dict = {}
        for parameter_name, parameter in parameters_items:
            # every parameter must be annotated or have a default value
            annotation = parameter.annotation
            if annotation is type:
                annotation = TyperWrapper

            if parameter.default is inspect.Parameter.empty:
                parameter_dict[parameter_name] = (annotation, ...)
            else:
                parameter_dict[parameter_name] = (annotation, parameter.default)
        return parameter_dict

    @staticmethod
    def resolve_type(source_type: type, generic_type_map: dict[type, type]):
        """
        Resolve a type using the generic type map.

        Supports Pydantic.GenericModel and typing.Generic.
        """
        if source_type in generic_type_map:
            source_type = generic_type_map[source_type]

        if isinstance(source_type, type) and issubclass(source_type, generics.GenericModel):
            base_generic_type = LLMFunctionSpec.get_base_generic_type(source_type)
            generic_parameter_type_map = LLMFunctionSpec.get_generic_type_map(source_type, base_generic_type)
            # forward step using the generic type map
            resolved_generic_type_map = {
                generic_type: generic_type_map.get(target_type, target_type)
                for generic_type, target_type in generic_parameter_type_map.items()
            }
            resolved_tuple = tuple(
                resolved_generic_type_map[generic_type] for generic_type in base_generic_type.__parameters__
            )
            source_type = base_generic_type[resolved_tuple]  # type: ignore
        else:
            # we let Pydantic handle the rest
            source_type = replace_types(source_type, generic_type_map)

        return source_type

    @staticmethod
    def resolve_generic_types(parameters_items: list[tuple[str, inspect.Parameter]], arguments: dict):
        generic_type_map: dict = {}
        for parameter_name, parameter in parameters_items:
            annotation = parameter.annotation

            # if the annotation is an Annotated type, get the type annotation
            if typing.get_origin(annotation) is typing.Annotated:
                annotation = typing.get_args(annotation)[0]

            # if the annotation is a type var, resolve it into the generic type map
            if isinstance(annotation, typing.TypeVar):
                LLMFunctionSpec.add_resolved_type(generic_type_map, annotation, type(arguments[parameter_name]))
            # if the annotation is a type, check if it is a generic type
            elif issubclass(annotation, generics.GenericModel):
                # check if the type is in generics._assigned_parameters
                generic_parameters_type_map = LLMFunctionSpec.get_generic_type_map(annotation)

                argument_type = type(arguments[parameter_name])
                generic_argument_type_map = LLMFunctionSpec.get_generic_type_map(argument_type)

                assert list(generic_parameters_type_map.keys()) == list(generic_argument_type_map.keys())

                # update the generic type map
                # if the generic type is already in the map, check that it is the same
                for generic_parameter, generic_parameter_target in generic_parameters_type_map.items():
                    if generic_parameter_target not in annotation.__parameters__:
                        continue
                    resolved_type = generic_argument_type_map[generic_parameter]
                    LLMFunctionSpec.add_resolved_type(generic_type_map, generic_parameter_target, resolved_type)

        return generic_type_map

    @staticmethod
    def add_resolved_type(generic_type_map, source_type, resolved_type):
        """
        Add a resolved type to the generic type map.
        """
        if source_type in generic_type_map:
            # TODO: support finding the common base class?
            if (previous_resolution := generic_type_map[source_type]) is not resolved_type:
                raise ValueError(
                    f"Cannot resolve generic type {source_type}, conflicting "
                    f"resolution: {previous_resolution} and {resolved_type}."
                )
        else:
            generic_type_map[source_type] = resolved_type

    @staticmethod
    def get_or_create_pydantic_default(field: FieldInfo):
        if field.default is not Undefined:
            if field.default is Ellipsis:
                return inspect.Parameter.empty
            return field.default
        if field.default_factory is not None:
            return field.default_factory()
        return None

    @staticmethod
    def bind(args, kwargs, language_model_or_chain, signature):
        """
        Bind function taking into account Field definitions and defaults.
        """
        # resolve parameter defaults to FieldInfo.default if the parameter is a field
        signature_fixed_defaults = signature.replace(
            parameters=[
                parameter.replace(default=LLMFunctionSpec.get_or_create_pydantic_default(parameter.default))
                if isinstance(parameter.default, FieldInfo)
                else parameter
                for parameter in signature.parameters.values()
            ]
        )
        bound_arguments = signature_fixed_defaults.bind(language_model_or_chain, *args, **kwargs)
        bound_arguments.apply_defaults()
        return bound_arguments

    @staticmethod
    def get_generic_type_map(generic_type, base_generic_type=None):
        if base_generic_type is None:
            base_generic_type = LLMFunctionSpec.get_base_generic_type(generic_type)

        base_classes = inspect.getmro(generic_type)
        # we have to iterate through the base classes
        generic_parameter_type_map = {generic_type: generic_type for generic_type in generic_type.__parameters__}
        for base_class in base_classes:
            # skip baseclasses that are from pydantic.generic
            # this avoids a bug that is caused by generics.GenericModel.__parameterized_bases_
            if base_class.__module__ == "pydantic.generics":
                continue
            if issubclass(base_class, base_generic_type):
                if base_class in generics._assigned_parameters:
                    assignment = generics._assigned_parameters[base_class]
                    generic_parameter_type_map = {
                        old_generic_type: generic_parameter_type_map.get(new_generic_type, new_generic_type)
                        for old_generic_type, new_generic_type in assignment.items()
                    }

        return generic_parameter_type_map

    @staticmethod
    def get_argument_generic_type_map(base_generic_type: type[generics.GenericModel], generic_type):
        # find the first base class that is subclass of annotation
        # and in _assigned_parameters
        generic_argument_type_map: dict = {}
        for base_class in reversed(inspect.getmro(generic_type)):
            if issubclass(base_class, base_generic_type):
                if base_class in generics._assigned_parameters:
                    generic_argument_type_map.update(generics._assigned_parameters[base_class])
                else:
                    generic_argument_type_map.update(
                        {generic_type: generic_type for generic_type in base_class.__parameters__}
                    )
        return generic_argument_type_map

    @staticmethod
    def get_base_generic_type(argument_type) -> type[generics.GenericModel]:
        # get the base class name from annotation (which is without [])
        base_generic_name = argument_type.__name__
        if "[" in argument_type.__name__:
            base_generic_name = argument_type.__name__.split("[")[0]
        # get the base class from argument_type_base_classes with base_generic_name
        for base_class in reversed(inspect.getmro(argument_type)):
            if base_class.__name__ == base_generic_name and issubclass(argument_type, base_class):
                base_generic_type = base_class
                break
        else:
            raise ValueError(f"Could not find base generic type {base_generic_name} for {argument_type}.")
        return base_generic_type


@dataclass
class LLMFunction(typing.Generic[P, T], typing.Callable[P, T]):  # type: ignore
    """
    A callable that can be called with a chat model.
    """

    def get_spec_from_inputs(self, inputs: BaseModel) -> LLMFunctionSpec:
        return LLMFunctionSpec.from_call(self, None, args=(), kwargs=inputs.dict())

    def get_spec_from_args(self, *args, **kwargs) -> LLMFunctionSpec:
        return LLMFunctionSpec.from_call(self, None, args, kwargs)

    def get_inputs(self, *args, **kwargs) -> BaseModel:
        return self.get_spec_from_args(*args, **kwargs).get_inputs(*args, **kwargs)

    def __get__(self, instance: object, owner: type | None = None) -> typing.Callable:
        """Support instance methods."""
        if instance is None:
            return self

        # Bind self to instance as MethodType
        return types.MethodType(self, instance)

    def __getattr__(self, item):
        return getattr(self.__wrapped__, item)

    def explicit(
        self, language_model_or_chat_chain: BaseLanguageModel | ChatChain | UntrackedChatChain, inputs: BaseModel
    ):
        """Call the function with explicit inputs."""
        return self(language_model_or_chat_chain, **inputs.dict())

    def __call__(
        self,
        language_model_or_chat_chain: BaseLanguageModel | ChatChain | UntrackedChatChain,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Call the function."""
        if isinstance(language_model_or_chat_chain, UntrackedChatChain):
            language_model_or_chat_chain = ChatChain(
                language_model_or_chat_chain.chat_model, language_model_or_chat_chain.messages
            )

        # check that the first argument is an instance of BaseLanguageModel
        # or a TrackedChatChain or UntrackedChatChain
        if not isinstance(language_model_or_chat_chain, BaseLanguageModel | ChatChain):
            raise ValueError("The first parameter must be an instance of BaseLanguageModel or ChatChain.")

        spec = LLMFunctionSpec.from_call(self, language_model_or_chat_chain, args, kwargs)

        # bind the inputs to the signature
        bound_arguments = spec.signature.bind(language_model_or_chat_chain, *args, **kwargs)
        # get the arguments
        arguments = bound_arguments.arguments
        inputs = spec.input_model(**arguments)

        # get the input and output schema as JSON dict
        schema = spec.get_call_schema()
        # print(json.dumps(schema, indent=1))

        update_json_schema_hyperparameters(
            schema,
            prompt_hyperparameter("schema") @ get_json_schema_hyperparameters(schema),
        )

        # create the prompt
        prompt = (
            prompt_hyperparameter("llm_function_prompt")
            @ (
                "{docstring}\n"
                "\n"
                "The input and output are formatted as a JSON interface that conforms to the JSON schemas below.\n"
                "\n"
                'As an example, for the schema {{"properties": {{"foo": {{"description": "a list of '
                'strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}}} the object {{'
                '"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{'
                '"foo": '
                '["bar", "baz"]}}}} is not well-formatted.\n'
                "\n"
                "Here is the schema for additional date types:\n"
                "```\n"
                "{additional_definitions}\n"
                "```\n"
                "\n"
                "Here is the input schema:\n"
                "```\n"
                "{input_schema}\n"
                "```\n"
                "\n"
                "Here is the output schema:\n"
                "```\n"
                "{output_schema}\n"
                "```\n"
                "Now output the results for the following inputs:\n"
                "```\n"
                "{inputs}\n"
                "```\n"
            )
        ).format(
            docstring=spec.docstring,
            additional_definitions=json.dumps(schema['additional_definitions'], indent=1),
            input_schema=json.dumps(schema['input_schema'], indent=1),
            output_schema=json.dumps(schema['output_schema'], indent=1),
            inputs=inputs.json(indent=1),
        )

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
                    parsed_output = self.parse(output, spec.output_model)
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
                    parsed_output = self.parse(output, spec.output_model)
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

        print(f"Input: {inputs.json(indent=1)}")
        print(f"Output: {json.dumps(parsed_output.dict()['return_value'], indent=1)}")

        return parsed_output.return_value  # type: ignore

    @staticmethod
    def parse(text: str, output_model: type[B]) -> B:
        try:
            # Greedy search for 1st json candidate.
            match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
            json_str = ""
            if match:
                json_str = match.group()
            json_object = json.loads(json_str)
            return output_model.parse_obj(json_object)

        except (json.JSONDecodeError, ValidationError) as e:
            msg = f"Failed to parse the last reply. Expected: `{{\"return_value\": ...}}` Got: {e}"
            raise OutputParserException(msg)


def llm_function(f: typing.Callable[P, T]) -> LLMFunction[P, T]:
    """
    Decorator to wrap a function with a chat model.

    f is a function to a dataclass or Pydantic model.

    The docstring of the function provides instructions for the model.
    """
    if isinstance(f, classmethod):
        return classmethod(llm_function(f.__func__))
    elif isinstance(f, staticmethod):
        return staticmethod(llm_function(f.__func__))
    elif isinstance(f, property):
        return property(llm_function(f.fget), doc=f.__doc__)
    elif isinstance(f, types.MethodType):
        return types.MethodType(llm_function(f.__func__), f.__self__)
    elif hasattr(f, "__wrapped__"):
        return llm_function(f.__wrapped__)
    elif isinstance(f, LLMFunction):
        return f
    elif not callable(f):
        raise ValueError(f"Cannot decorate {f} with llm_strategy.")

    if not is_not_implemented(f):
        raise ValueError("The function must not be implemented.")

    specific_llm_function: LLMFunction = track_execution(functools.wraps(f)(LLMFunction()))
    return specific_llm_function
