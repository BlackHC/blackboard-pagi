"""
Defines a protocol for invoking a function as a black-box interface with a dataclass as input and output.
"""
import dataclasses
import dis
import inspect
import types
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

Execute the function with the inputs that follow in the next section and finally return the output using the output type
as YAML document in an # Output section. (If the value is a literal, then just write the value. We parse the text in the
# Output section using `yaml.safe_load` in Python.)
{dataclasses_schema}
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
    output_type: str

    prompt_template = """Extract the output into a YAML block from the end of the following text. The YAML comes after # YAML line:

{full_answer}

# Output Type

{output_type}

# YAML

"""

    def get_raw_response(self, llm: langchain.llms.LLM) -> str:
        """Parse the outputs of a function call.

        Args:
            full_answer: The full answer containing a YAML object at the end.
            llm: The LLM to use to parse the outputs.
        """

        yaml_block = llm(self.render())
        return yaml_block

    def get_yaml_response(self, llm: langchain.llms.LLM) -> dict[str, typing.Any]:
        """Parse the outputs of a function call.

        Args:
            full_answer: The full answer containing a YAML object at the end.
            llm: The LLM to use to parse the outputs.
        """

        yaml_block = self.get_raw_response(llm)
        yaml_object = yaml.safe_load(yaml_block)

        return yaml_object

    def get_structured_response(self, llm: langchain.llms.LLM, type_: type) -> typing.Any:
        """Parse the outputs of a function call.

        Args:
            full_answer: The full answer containing a YAML object at the end.
            llm: The LLM to use to parse the outputs.
            dataclasses_schema: The schema of all dataclasses that are used.
        """

        yaml_object = self.get_yaml_response(llm)

        return deserialize_yaml(yaml_object, type_)


# def llm_implement(
#     f: typing.Callable[P, T], llm: langchain.llms.LLM, parent_dataclasses_schema: DataclassesSchema | None = None
# ) -> typing.Callable[P, T]:
#     # Get doc string from f
#     docstring = f.__doc__
#     if docstring is None:
#         raise ValueError(f"Function {f} does not have a docstring.")
#
#     # Get the prompt
#     prompt_template: str | None = extract_prompt_template(docstring)
#     # Get the signature
#     signature = inspect.Signature.from_callable(f, eval_str=True)
#
#     dataclasses_schema = DataclassesSchema.extend_parent(parent_dataclasses_schema)
#     dataclasses_schema.add_return_annotation(signature)
#
#     @functools.wraps(f)
#     def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
#         # Dump the inputs as YAML
#         bound_inputs = signature.bind(*args, **kwargs)
#         bound_inputs.apply_defaults()
#         inputs = {name: value for name, value in bound_inputs.arguments.items()}
#         inputs_yaml_block = yaml.safe_dump(inputs)
#
#         input_types = {name: pretty_type_str(type) for name, type in signature.parameters.items()}
#         input_types_yaml_block = yaml.safe_dump(input_types)
#
#         runtime_schema = DataclassesSchema.extend_parent(dataclasses_schema)
#         runtime_schema.add_bound_arguments(bound_inputs)
#
#         if runtime_schema:
#             runtime_schema_yaml = yaml.safe_dump(dict(types=runtime_schema.definitions))
#             runtime_schema_text = f"\n# Dataclasses Schema\n\n{runtime_schema_yaml}\n"
#         else:
#             runtime_schema_text = ""
#
#         assert signature.return_annotation
#         output_type_text = signature.return_annotation.__name__
#
#         task_prompt = TaskPrompt(
#             description=prompt_template or docstring,
#             inputs=inputs_yaml_block,
#             input_types=input_types_yaml_block,
#             dataclasses_schema=runtime_schema_text,
#             output_type=output_type_text,
#         )
#
#         # Get the output
#         partial_answer = llm(task_prompt.render(), stop=["# Output"])
#         # Get the YAML
#         yaml_part = llm(task_prompt.render() + partial_answer)
#         # Convert the output to the return type
#         #yaml_extraction_prompt = ExtractYamlPrompt(full_answer)
#         #yaml_part = llm(yaml_extraction_prompt.render())
#         yaml_dict = yaml.safe_load(yaml_part)
#
#         result = deserialize_yaml(yaml_dict, signature.return_annotation)
#         return result
#
#     return wrapper


# def implement_ABC_dataclass(class_def: type):
#     """Implement all abstract methods of an ABC dataclass."""
#     @dataclass
#     class Implementer(class_def):
#         llm: langchain.llms.LLM
#
#         def __post_init__(self):
#             for name, value in self.__dict__.items():
#                 # Check whether the method is an abstract method
#                 if isinstance(value, abc.abstractmethod):


@dataclass
class NotImplementedDataclass:
    @staticmethod
    def static_method(a: int, b: int) -> int:
        """Add two numbers."""
        raise NotImplementedError()

    @classmethod
    def class_method(cls, a: int, b: int) -> int:
        """Add two numbers."""
        raise NotImplementedError()

    @property
    def property_getter(self) -> int:
        """Add two numbers."""
        raise NotImplementedError()

    def bound_method(self, a: int, b: int) -> int:
        """Add two numbers."""
        raise NotImplementedError()

    def bound_method_raises_class(self, a: int, b: int = 1) -> int:
        """Add two numbers."""
        raise NotImplementedError


# Check that a method (regular method, class method or static method) only raises NotImplementedError.
def check_not_implemented(f: typing.Callable) -> bool:
    # Inspect the opcodes
    code = f.__code__
    # Get the opcodes
    opcodes = list(dis.get_instructions(code))
    # Check that the we only use the following opcodes:
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
    # Check that the function raises a NotImplementedError
    if opcodes[-1].opname != "RAISE_VARARGS":
        return False
    return True


assert check_not_implemented(NotImplementedDataclass.static_method)
assert check_not_implemented(NotImplementedDataclass.bound_method)
assert check_not_implemented(NotImplementedDataclass.class_method)
assert check_not_implemented(NotImplementedDataclass.bound_method_raises_class)

# How do we determine if a field is a property?
if isinstance(NotImplementedDataclass.property_getter, property):
    print("It's a property!")

    assert check_not_implemented(NotImplementedDataclass.__dict__["property_getter"].fget)


@dataclass
class LLMCall:
    """
    A call to a LLM. This wraps a function with an LLM execution context.
    """

    dataclasses_schema: DataclassesSchema
    prompt_template: str
    signature: inspect.Signature
    __wrapped__: typing.Callable

    @staticmethod
    def wrap_callable(f: typing.Callable, parent_dataclasses_schema: DataclassesSchema) -> "LLMCall":
        """Create an LLMCall from a function."""
        include_parent_dataclasses_schema = False
        # is f a staticmethod?
        if isinstance(f, staticmethod):
            f = f.__func__
        # is f a classmethod?
        elif isinstance(f, classmethod):
            f = f.__func__
        # is f a property?
        elif isinstance(f, property):
            f = f.fget
        # is f a bound method?
        elif isinstance(f, types.MethodType):
            include_parent_dataclasses_schema = True

        docstring = f.__doc__
        if docstring is None:
            raise ValueError(f"Function {f} does not have a docstring.")

        prompt_template: str = extract_prompt_template(docstring) or docstring
        signature = inspect.Signature.from_callable(f, eval_str=True)

        # TODO: this is not exactly what we want but already better
        if include_parent_dataclasses_schema:
            dataclasses_schema = DataclassesSchema.extend_parent(parent_dataclasses_schema)
        else:
            dataclasses_schema = DataclassesSchema()
        dataclasses_schema.add_return_annotation(signature)

        llm_call = LLMCall(
            dataclasses_schema=dataclasses_schema,
            prompt_template=prompt_template,
            signature=signature,
            __wrapped__=f,
        )

        return llm_call

    def __call__(self, llm):
        def inner(*args, **kwargs):
            bound_inputs = self.signature.bind(*args, **kwargs)
            bound_inputs.apply_defaults()
            inputs = {name: dataclasses.asdict(value) for name, value in bound_inputs.arguments.items()}
            inputs_yaml_block = yaml.safe_dump(inputs)

            input_types = {
                name: pretty_type_str(parameter.annotation) for name, parameter in self.signature.parameters.items()
            }
            input_types_yaml_block = yaml.safe_dump(input_types)

            runtime_schema = DataclassesSchema.extend_parent(self.dataclasses_schema)
            runtime_schema.add_bound_arguments(bound_inputs)

            if runtime_schema.definitions:
                runtime_schema_yaml = yaml.safe_dump(dict(types=dict(runtime_schema.definitions)))
                runtime_schema_text = f"\n# Dataclasses Schema\n\n{runtime_schema_yaml}\n"
            else:
                runtime_schema_text = ""

            output_type_text = pretty_type_str(self.signature.return_annotation)

            task_prompt = TaskPrompt(
                description=self.prompt_template,
                inputs=inputs_yaml_block,
                input_types=input_types_yaml_block,
                dataclasses_schema=runtime_schema_text,
                output_type=output_type_text,
            )

            rendered_prompt = task_prompt.render()

            # Get the output
            partial_answer = llm(rendered_prompt, stop=["# Output"])
            # Get the YAML
            final_prompt = task_prompt.render() + partial_answer + "# Output\n"
            yaml_part = llm(final_prompt)
            # Convert the output to the return type
            # yaml_extraction_prompt = ExtractYamlPrompt(full_answer)
            # yaml_part = llm(yaml_extraction_prompt.render())

            yaml_dict = yaml.safe_load(yaml_part)

            result = deserialize_yaml(yaml_dict, self.signature.return_annotation)
            return result

        return inner


@typing_extensions.dataclass_transform()
@typing.no_type_check
def implement_with_LLM(dataclass_type: type[T], llm: langchain.llms.LLM) -> type[T]:
    global dataclasses_schema
    global current_llm
    current_llm = llm
    dataclasses_schema = DataclassesSchema()
    dataclasses_schema.add_dataclass_type(dataclass_type)

    key, value = None, None

    @dataclass
    class LLMImplementation(dataclass_type):
        global key, value
        for key, value in dataclass_type.__dict__.items():
            if isinstance(value, property):
                value = value.fget
            if callable(value):
                if not hasattr(value, "__code__"):
                    if hasattr(value, "__func__"):
                        value = value.__func__
                    else:
                        raise ValueError(f"What is {key}?")
            if callable(value) and check_not_implemented(value):
                exec(
                    f"""
global key, value, dataclass_schema
{key} = functools.wraps(value)(LLMCall.wrap_callable(value, dataclasses_schema)(current_llm))
    """
                )

    LLMImplementation.__name__ = f"{dataclass_type.__name__}[{llm.__class__.__name__}]"
    LLMImplementation.__qualname__ = f"{dataclass_type.__qualname__}[{llm.__class__.__name__}]"

    del key, value, current_llm, dataclasses_schema

    return LLMImplementation


#
# from blackboard_pagi.testing.fake_llm import FakeLLM
#
# fake_llm = FakeLLM(queries={})
#
# wrapped_dataclass = implement_with_LLM(NotImplementedDataclass, langchain.llms.OpenAI())
#
# #print(wrapped_dataclass.static_method(1,2))
# print(wrapped_dataclass().bound_method(1,2))
