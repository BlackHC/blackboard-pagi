"""
Defines a protocol for invoking a function as a black-box interface with a dataclass as input and output.
"""
import functools
import inspect
import typing
from dataclasses import dataclass

import langchain.llms
import typing_extensions
import yaml

from blackboard_pagi.prompts.dataclasses_schema import DataclassesSchema, deserialize_yaml, pretty_type_str
from blackboard_pagi.prompts.prompt_doc_string import extract_prompt_template
from blackboard_pagi.prompts.prompt_template import PromptTemplateMixin

# A type variable that must be a user-defined class?
T = typing.TypeVar("T")
P = typing_extensions.ParamSpec("P")


@dataclass
class TaskPrompt(PromptTemplateMixin):
    """
    A prompt that asks the user to execute a function with inputs and return the output.

    The function is described via a doc string. The prompt is generated from the doc string and the input
    and output types and a schema of all dataclasses that are used.

    """

    description: str
    inputs: str
    dataclasses_schema: str
    input_types: str
    output_type: str

    prompt_template = """Execute the following function that is described via a doc string:

{description}

# Task

Execute the function with the inputs that follow in the next section and finally return the output as YAML following the
Output Type.
{data_schema}
# Input Types

{input_types}

# Inputs

{inputs}

# Output Type

{output_type}

# Execution Scratch-Pad (Think Step by Step)

"""


@dataclass
class ExtractYamlPrompt(PromptTemplateMixin):
    full_answer: str

    prompt_template = """Extract the YAML from the end of the following text. Only print the YAML and ignore the rest.

{full_answer}

# YAML

"""

    def get_full_response(self, llm: langchain.llms.LLM) -> dict[str, typing.Any]:
        """Parse the outputs of a function call.

        Args:
            full_answer: The full answer containing a YAML object at the end.
            llm: The LLM to use to parse the outputs.
        """

        yaml_block = llm(self.render())
        yaml_object = yaml.safe_load(yaml_block)

        return yaml_object


def llm_implement(
    f: typing.Callable[P, T], llm: langchain.llms.LLM, parent_dataclasses_schema: DataclassesSchema | None = None
) -> typing.Callable[P, T]:
    # Get doc string from f
    docstring = f.__doc__ or ""
    # Get the prompt
    prompt_template: str | None = extract_prompt_template(docstring)
    # Get the signature
    signature = inspect.Signature.from_callable(f, eval_str=True)

    dataclasses_schema = DataclassesSchema.extend_parent(parent_dataclasses_schema)
    dataclasses_schema.add_return_annotation(signature)

    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Dump the inputs as YAML
        bound_inputs = signature.bind(*args, **kwargs)
        bound_inputs.apply_defaults()
        inputs = {name: value for name, value in bound_inputs.arguments.items()}
        inputs_yaml_block = yaml.safe_dump(inputs)

        input_types = {name: pretty_type_str(type) for name, type in signature.parameters.items()}
        input_types_yaml_block = yaml.safe_dump(input_types)

        runtime_schema = DataclassesSchema.extend_parent(dataclasses_schema)
        runtime_schema.add_bound_arguments(bound_inputs)

        if runtime_schema:
            runtime_schema_yaml = yaml.safe_dump(dict(types=runtime_schema.definitions))
            runtime_schema_text = f"\n# Dataclasses Schema\n\n{runtime_schema_yaml}\n"
        else:
            runtime_schema_text = ""

        assert signature.return_annotation
        output_type_text = signature.return_annotation.__name__

        task_prompt = TaskPrompt(
            description=prompt_template or docstring,
            inputs=inputs_yaml_block,
            input_types=input_types_yaml_block,
            dataclasses_schema=runtime_schema_text,
            output_type=output_type_text,
        )

        # Get the output
        full_answer = llm(task_prompt.render())
        # Convert the output to the return type
        yaml_extraction_prompt = ExtractYamlPrompt(full_answer)
        yaml_part = llm(yaml_extraction_prompt.render())
        yaml_dict = yaml.safe_load(yaml_part)

        result = deserialize_yaml(yaml_dict, signature.return_annotation)
        return result

    return wrapper
