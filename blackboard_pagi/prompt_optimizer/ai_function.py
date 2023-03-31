import functools
import inspect
import json
import typing
from dataclasses import dataclass

from langchain.llms import BaseLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS
from pydantic import BaseModel, create_model

from blackboard_pagi.prompt_optimizer.chat_chain_optimizer import enable_prompt_optimizer, prompt_hyperparameter


@dataclass
class CallableWithLanguageModelProperty:
    """
    A callable that can be called with a chat model.
    """

    language_model: BaseLLM | None = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


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


@dataclass
class AIFunctionSpec:
    signature: inspect.Signature
    docstring: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]

    @staticmethod
    def from_function(f):
        # get clean docstring of
        docstring = inspect.getdoc(f)
        if docstring is None:
            raise ValueError("The function must have a docstring.")
        # get the type of the first argument
        signature = inspect.signature(f)
        # get all parameters
        parameters = signature.parameters
        # create a pydantic model from the parameters
        parameter_dict = {}
        for parameter_name, parameter in parameters.items():
            # every parameter must be annotated or have a default value
            if parameter.annotation is inspect.Parameter.empty:
                # check if the parameter has a default value
                if parameter.default is inspect.Parameter.empty:
                    raise ValueError(f"The parameter {parameter_name} must be annotated or have a default value.")
                parameter_dict[parameter_name] = parameter.default
            elif parameter.default is inspect.Parameter.empty:
                parameter_dict[parameter_name] = (parameter.annotation, ...)
            else:
                parameter_dict[parameter_name] = (
                    parameter.annotation,
                    parameter.default,
                )
        # turn the function name into a class name
        class_name_prefix = f.__name__.capitalize()
        # create the model
        input_model = create_model(f"{class_name_prefix}Inputs", **parameter_dict)  # type: ignore
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


def ai_function(f) -> CallableWithLanguageModelProperty:
    """
    Decorator to wrap a function with a chat model.

    f is a function to a dataclass or Pydantic model.

    The docstring of the function provides instructions for the model.
    """

    ai_function_spec = AIFunctionSpec.from_function(f)

    # create the callable
    class AIFunction(CallableWithLanguageModelProperty):
        @functools.wraps(f)
        @enable_prompt_optimizer
        def __call__(self, *args, **kwargs):
            # bind the inputs to the signature
            bound_arguments = ai_function_spec.signature.bind(*args, **kwargs)
            # get the arguments
            arguments = bound_arguments.arguments
            inputs = ai_function_spec.input_model(**arguments)

            parser = PydanticOutputParser(pydantic_object=ai_function_spec.output_model)

            # get the input schema as JSON
            input_schema = ai_function_spec.input_model.schema()
            # remove title and type
            input_schema.pop("title")
            input_schema.pop("type")

            output_schema = ai_function_spec.output_model.schema()
            # remove title and type
            output_schema.pop("title")
            output_schema.pop("type")

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

            # get the chat model
            language_model = self.language_model
            if language_model is None:
                raise ValueError("The chat model must be set.")

            # get the output
            output = language_model(prompt)

            # parse the output
            parsed_output: ai_function_spec.output_model = parser.parse(output)

            return parsed_output.result

    return AIFunction()
