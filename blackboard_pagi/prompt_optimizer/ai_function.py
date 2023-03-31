import dis
import functools
import inspect
import json
import typing
from dataclasses import dataclass

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from langchain.schema import BaseLanguageModel
from pydantic import BaseModel, create_model

from blackboard_pagi.prompt_optimizer.track_execution import prompt_hyperparameter, track_execution
from blackboard_pagi.prompts.chat_chain import ChatChain


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


def is_not_implemented(f: typing.Callable) -> bool:
    """Check that a function only raises NotImplementedError."""
    unwrapped_f = inspect.unwrap(f)

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


@dataclass
class AIFunctionSpec:
    signature: inspect.Signature
    docstring: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]

    @staticmethod
    def from_function(f):
        """Create an AIFunctionSpec from a function."""

        if not is_not_implemented(f):
            raise ValueError("The function must not be implemented.")

        # get clean docstring of
        docstring = inspect.getdoc(f)
        if docstring is None:
            raise ValueError("The function must have a docstring.")
        # get the type of the first argument
        signature = inspect.signature(f)
        # get all parameters
        parameters = signature.parameters
        # check that there is at least one parameter
        if not parameters:
            raise ValueError("The function must have at least one parameter.")
        # check that the first parameter has a type annotation that is an instance of BaseLanguageModel
        # or a TrackedChatChain
        first_parameter = next(iter(parameters.values()))
        if first_parameter.annotation is inspect.Parameter.empty:
            raise ValueError("The first parameter must have a type annotation.")
        if not issubclass(first_parameter.annotation, BaseLanguageModel | ChatChain):
            raise ValueError("The first parameter must be an instance of BaseLanguageModel or ChatChain.")

        # create a pydantic model from the parameters
        input_parameter_dict = {}
        for i, (parameter_name, parameter) in enumerate(parameters.items()):
            # skip the first parameter
            if i == 0:
                continue
            # every parameter must be annotated or have a default value
            if parameter.annotation is inspect.Parameter.empty:
                # check if the parameter has a default value
                if parameter.default is inspect.Parameter.empty:
                    raise ValueError(f"The parameter {parameter_name} must be annotated or have a default value.")
                input_parameter_dict[parameter_name] = parameter.default
            elif parameter.default is inspect.Parameter.empty:
                input_parameter_dict[parameter_name] = (parameter.annotation, ...)
            else:
                input_parameter_dict[parameter_name] = (
                    parameter.annotation,
                    parameter.default,
                )
        # turn the function name into a class name
        class_name_prefix = f.__name__.capitalize()
        # create the model
        input_model = create_model(f"{class_name_prefix}Inputs", **input_parameter_dict)  # type: ignore
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
        output_model = create_model(f"{class_name_prefix}Output", result=return_info)  # type: ignore

        return AIFunctionSpec(
            docstring=docstring,
            signature=signature,
            input_model=input_model,
            output_model=output_model,
        )


def get_clean_schema(object_type: type[BaseModel]):
    schema = object_type.schema()
    # remove title and type
    schema.pop("title")
    schema.pop("type")
    return schema


def ai_function(f) -> typing.Callable:
    """
    Decorator to wrap a function with a chat model.

    f is a function to a dataclass or Pydantic model.

    The docstring of the function provides instructions for the model.

    The first parameter of the function must be an instance of BaseLanguageModel or ChatChain.
    """

    ai_function_spec = AIFunctionSpec.from_function(f)

    # create the callable

    @functools.wraps(f)
    @track_execution
    def wrapped_f(language_model_or_chat_chain: BaseLanguageModel | ChatChain, *args, **kwargs):
        # bind the inputs to the signature
        bound_arguments = ai_function_spec.signature.bind(language_model_or_chat_chain, *args, **kwargs)
        # get the arguments
        arguments = bound_arguments.arguments
        inputs = ai_function_spec.input_model(**arguments)

        parser = PydanticOutputParser(pydantic_object=ai_function_spec.output_model)

        # get the input schema as JSON
        input_schema = get_clean_schema(ai_function_spec.input_model)
        output_schema = get_clean_schema(ai_function_spec.output_model)

        update_json_schema_hyperparameters(
            input_schema,
            prompt_hyperparameter("input_schema") @ get_json_schema_hyperparameters(input_schema),
        )

        update_json_schema_hyperparameters(
            output_schema,
            prompt_hyperparameter("output_schema") @ get_json_schema_hyperparameters(output_schema),
        )

        # create the prompt
        prompt = (
            f"{prompt_hyperparameter('docstring') @ ai_function_spec.docstring}\n"
            "The inputs are formatted as JSON using the following schema:\n"
            f"{json.dumps(input_schema)}\n"
            "\n"
            f"{PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=json.dumps(output_schema))}"
            "\n"
            "Now output the results for the following inputs:\n"
            f"{inputs.json()}"
        )

        # get the response
        if language_model_or_chat_chain is None:
            raise ValueError("The language model or chat chain must be provided.")
        if isinstance(language_model_or_chat_chain, ChatChain):
            output, _ = language_model_or_chat_chain.query(prompt)
        elif isinstance(language_model_or_chat_chain, BaseChatModel):
            output = language_model_or_chat_chain.call_as_llm(prompt)
        elif isinstance(language_model_or_chat_chain, BaseLLM):
            output = language_model_or_chat_chain(prompt)
        else:
            raise ValueError("The language model or chat chain must be provided.")

        # parse the output
        parsed_output = parser.parse(output)

        return parsed_output.result  # type: ignore

    return wrapped_f
