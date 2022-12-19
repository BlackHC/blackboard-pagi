"""
Defines a protocol for invoking a function as a black-box interface with a dataclass as input and output.
"""
import dataclasses
import functools
import inspect
import typing
from dataclasses import dataclass

import docstring_parser.google
import docstring_parser.numpydoc
import langchain.llms
import typing_extensions
import yaml
from docstring_parser import ParseError

from blackboard_pagi.prompts.prompt_template import PromptTemplateMixin

GOOGLE_DEFAULT_SECTIONS = docstring_parser.google.DEFAULT_SECTIONS + [
    docstring_parser.google.Section("Prompt Template", "prompt_template", docstring_parser.google.SectionType.SINGULAR)
]

NUMPY_DEFAULT_SECTIONS = docstring_parser.numpydoc.DEFAULT_SECTIONS + [
    docstring_parser.numpydoc.Section("Prompt Template", "prompt_template")
]


# A type variable that must be a user-defined class?
T = typing.TypeVar("T")
P = typing_extensions.ParamSpec("P")


def try_parse_docstring(
    text: str, parsers: typing.List[typing.Callable[[str], docstring_parser.Docstring]]
) -> docstring_parser.Docstring:
    exc: Exception | None = None
    rets = []
    for parser in parsers:
        try:
            ret = parser(text)
        except ParseError as ex:
            exc = ex
        else:
            rets.append(ret)

    if not rets:
        assert exc
        raise exc

    return sorted(rets, key=lambda d: len(d.meta), reverse=True)[0]


def extract_prompt_template(docstring: str) -> typing.Optional[str]:
    """Either return the full doc string, or if there is a `Prompt:` section, return only that section.

    Note that additional sections might appear before and after the `Prompt:` section.

    Examples:
        Examples using different doc strings styles.

        >>> example_google_docstring = '''
        ... This is a doc string.
        ...
        ... Prompt:
        ... This is the prompt.
        ...
        ... This is more doc string.
        ... '''
        >>> extract_prompt_template(example_google_docstring)
        'This is the prompt.'


        >>> example_numpy_docstring = '''
        ... This is a doc string.
        ...
        ... This is more doc string.
        ...
        ... Parameters
        ... ----------
        ... input : int
        ...     This is the input.
        ...
        ... Prompt
        ... ------
        ... This is the prompt.
        ...
        ... Returns
        ... -------
        ... output : int
        ...     This is the output.
        ...'''
        >>> extract_prompt_template(example_numpy_docstring)
        'This is the prompt.'
    """
    # TODO support ReST style doc strings
    # TODO support other doc string styles
    # Example:
    #
    #     >>> example_rest_docstring = '''
    #     ... This is a doc string.
    #     ...
    #     ... This is more doc string.
    #     ...
    #     ... :param input: This is the input.
    #     ... :type input: int
    #     ... :return: This is the output.
    #     ... :rtype: int
    #     ... :prompt: This is the prompt.
    #     ... '''
    #     >>> extract_prompt_template(example_rest_docstring)
    #     'This is the prompt.'
    #
    # This code is adapted from docstring_parser (MIT License):
    parsed_docstring = try_parse_docstring(
        docstring,
        [
            docstring_parser.google.GoogleParser(GOOGLE_DEFAULT_SECTIONS).parse,
            docstring_parser.numpydoc.NumpydocParser(NUMPY_DEFAULT_SECTIONS).parse,  # type: ignore
        ],
    )
    prompt_template = [section for section in parsed_docstring.meta if section.args == ["prompt_template"]]
    assert len(prompt_template) <= 1
    if prompt_template:
        return prompt_template[0].description
    return None


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


def extract_output_yaml(full_answer: str, llm: langchain.llms.LLM) -> dict[str, typing.Any]:
    """Parse the outputs of a function call.

    Args:
        full_answer: The full answer containing a YAML object at the end.
        llm: The LLM to use to parse the outputs.
    """

    extraction_task = ExtractYamlPrompt(full_answer=full_answer)

    yaml_block = llm(extraction_task.render())
    yaml_object = yaml.safe_load(yaml_block)

    return yaml_object


def add_dataclass_type_to_full_schema(
    dataclass_type: typing.Type[T], full_schema: dict[str, dict[str, dict[str, str]]]
):
    """Convert a dataclass to a schema, which is a list of type definitions.

    Recursively convert nested dataclasses.

    Args:
        dataclass_type: The dataclass type to convert.
        full_schema: The full schema to add the type to.

    Example:

        >>> @dataclass
        ... class Foo:
        ...     a: int
        ...     b: str
        >>> convert_to_full_schema(Foo)
        {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}}

        >>> @dataclass
        ... class Bar:
        ...     c: Foo
        >>> convert_to_full_schema(Bar)
        {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}, 'Bar': {'c': {'type': 'Foo'}}}

        >>> @dataclass
        ... class Baz:
        ...     d: Annotated[int, "Additional annotations."]
        >>> convert_to_full_schema(Baz)
        {'Baz': {'d': {'type': 'int', 'metadata': ['Additional annotations.']}}}
    """
    schema = full_schema[dataclass_type.__name__] = {}

    field: dataclasses.Field
    for field in dataclasses.fields(dataclass_type):
        if dataclasses.is_dataclass(field.type):
            if field.type.__name__ not in schema:
                add_dataclass_type_to_full_schema(field.type, full_schema)
        entry = get_type_and_metadata_entry(field)
        schema[field.name] = entry


def get_type_and_metadata_entry(field) -> dict[str, typing.Any]:
    # Support annotations
    if typing.get_origin(field.type) == typing_extensions.Annotated:
        type_ = typing.get_args(field.type)[0]
        entry = {'type': type_.__name__, 'metadata': typing.get_args(field.type)[1:]}
    else:
        type_ = field.type
        entry = {'type': type_.__name__}
    return entry


def add_dataclass_to_full_schema(dataclass_instance: T, full_schema: dict[str, dict[str, dict[str, typing.Any]]]):
    """Convert a dataclass to a schema, which is a list of type definitions and values.

    Recursively convert nested dataclasses.

    Args:
        dataclass_instance: The dataclass type to convert.
        full_schema: The full schema to add the type to.

    Example:

        >>> @dataclass
        ... class Foo:
        ...     a: int
        ...     b: str
        >>> convert_to_full_schema(Foo(a=1, b="2"))
        {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}}


        >>> @dataclass
        ... class Bar:
        ...     c: Foo
        >>> convert_to_full_schema(Bar)
        {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}, 'Bar': {'c': {'type': 'Foo'}}}

        >>> @dataclass
        ... class Baz:
        ...     d: Annotated[int, "Additional annotations."]
        >>> convert_to_full_schema(Baz)
        {'Baz': {'d': {'type': 'int', 'metadata': ['Additional annotations.']}}}

    """
    schema = full_schema[dataclass_instance.__class__.__name__] = {}

    field: dataclasses.Field
    for field in dataclasses.fields(dataclass_instance):
        field_value = getattr(dataclass_instance, field.name)
        if dataclasses.is_dataclass(field_value):
            if field.type.__name__ not in schema:
                add_dataclass_to_full_schema(field_value, full_schema)
        schema[field.name] = get_type_and_metadata_entry(field)


VALID_TYPES = (int, float, str, bool, type(None), dict, list, tuple)


def get_return_schema_from_signature(signature: inspect.Signature) -> dict[str, dict[str, dict[str, typing.Any]]]:
    """
    Get the return schema from a function signature.

    Args:
        signature: The signature to get the return schema from.

    Returns:
        The return schema.

    Example:

        >>> def foo(a: int, b: str) -> int:
        ...     pass
        >>> get_return_schema_from_signature(inspect.signature(foo))
        {}
    """
    full_schema: dict[str, dict[str, dict[str, typing.Any]]] = {}

    # TODO: add support for union types!

    # Get the schema for all inputs and the return type.
    if signature.return_annotation != inspect.Signature.empty:
        # assert that the return type is a dataclass or is a primitive type
        assert (
            dataclasses.is_dataclass(signature.return_annotation)
            or signature.return_annotation in VALID_TYPES
            or typing.get_origin(signature.return_annotation) in VALID_TYPES
        )
        if dataclasses.is_dataclass(signature.return_annotation):
            add_dataclass_type_to_full_schema(signature.return_annotation, full_schema)

    return full_schema


def add_bound_arguments_to_full_schema(
    bound_arguments: inspect.BoundArguments, full_schema: dict[str, dict[str, dict[str, typing.Any]]]
):
    """Add any dataclasses from BoundArguments to the full schema.

    Args:
        bound_arguments: The bound arguments to add.
        full_schema: The schema to add to.

    Example:

        >>> def foo(a: int, b: str) -> None:
        ...     pass
        >>> signature = inspect.signature(foo)
        >>> bound_arguments = signature.bind(1, "2")
        >>> schema = get_schema_from_signature(signature)
        >>> add_bound_arguments_to_schema(bound_arguments, schema)
        {}

        >>> @dataclass
        ... class Foo:
        ...     a: int
        ...     b: str
        >>> def bar(c) -> None:
        ...     pass
        >>> signature = inspect.signature(bar)
        >>> bound_arguments = signature.bind(Foo(1, "2"))
        >>> schema = get_schema_from_signature(signature)
        >>> add_bound_arguments_to_schema(bound_arguments, schema)
        {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}}
    """
    for name, value in bound_arguments.arguments.items():
        if dataclasses.is_dataclass(value):
            add_dataclass_to_full_schema(value, full_schema)


def deserialize_yaml(decoded_yaml, type_):
    """
    Deserialize `yaml` into an instance of `type_`. If `yaml` is a dict, then
    `type_` can be a dataclass. If `yaml` is a list, then `type_` can be a
    list of dataclasses.

    Args:
        yaml: The dict to deserialize.
        type_: The type of the dataclass.
        full_schema: The schema of the dataclasses (for nested dataclasses).

    Returns:
        The deserialized object.

    Example:

            >>> @dataclass
            ... class Foo:
            ...     a: int
            ...     b: str
            >>> deserialize_yaml({'a': 1, 'b': "2"}, Foo, {})
            Foo(a=1, b='2')

            >>> @dataclass
            ... class Bar:
            ...     c: Foo
            >>> deserialize_yaml({'c': {'a': 1, 'b': "2"}}, Bar, {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}})
            Bar(c=Foo(a=1, b='2'))
    """
    # If the type is a list, then deserialize each element of the list.
    if typing.get_origin(type_) == list:
        # Require that we have a type annotation for the list.
        assert typing.get_args(type_)
        # Require that the type annotation is a dataclass.
        item_type = typing.get_args(type_)[0]
        return [deserialize_yaml(item, item_type) for item in decoded_yaml]
    elif typing.get_origin(type_) == dict:
        # Require that we have a type annotation for the dict.
        assert typing.get_args(type_)
        # Require that the type annotation is a dataclass.
        item_type = typing.get_args(type_)[1]
        return {key: deserialize_yaml(item, item_type) for key, item in decoded_yaml.items()}
    elif dataclasses.is_dataclass(typing.get_origin(type_)):
        # If the type is a dataclass, then deserialize the dict into an instance of the dataclass.
        kwargs = {}
        fields = dataclasses.fields(type_)
        for field in fields:
            if field.name in yaml:
                kwargs[field.name] = deserialize_yaml(decoded_yaml[field.name], field.type)
        return type_(**kwargs)
    else:
        # If the type is not a dataclass, then just return the value.
        return yaml


def llm_implement(f: typing.Callable[P, T], llm: langchain.llms.LLM) -> typing.Callable[P, T]:
    # Get doc string from f
    docstring = f.__doc__ or ""
    # Get the prompt
    prompt_template: str | None = extract_prompt_template(docstring)
    # Get the signature
    signature = inspect.Signature.from_callable(f, eval_str=True)

    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Dump the inputs as YAML
        bound_inputs = signature.bind(*args, **kwargs)
        bound_inputs.apply_defaults()
        inputs = {name: value for name, value in bound_inputs.arguments.items()}
        inputs_yaml_block = yaml.safe_dump(inputs)

        input_types = {name: type(value).__name__ for name, value in bound_inputs.arguments.items()}
        input_types_yaml_block = yaml.safe_dump(input_types)

        dataclass_schema = get_return_schema_from_signature(signature)
        add_bound_arguments_to_full_schema(bound_inputs, dataclass_schema)

        if dataclass_schema:
            dataclass_schema = yaml.safe_dump(dict(types=dataclass_schema))
            dataclass_schema_text = f"\n# Dataclasses Schema\n\n{dataclass_schema}\n"
        else:
            dataclass_schema_text = ""

        assert signature.return_annotation
        output_type_text = signature.return_annotation.__name__

        task_prompt = TaskPrompt(
            description=prompt_template or docstring,
            inputs=inputs_yaml_block,
            input_types=input_types_yaml_block,
            dataclasses_schema=dataclass_schema_text,
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
