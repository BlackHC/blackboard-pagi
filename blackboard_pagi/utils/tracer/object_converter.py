import dataclasses
import inspect
import typing
from dataclasses import dataclass, field
from functools import partial

import pydantic

T = typing.TypeVar("T")


@dataclass
class ObjectConverter:
    """
    A class that converts objects to JSON dicts or strings.
    By default, only literals, tuples, lists, sets, dicts and dataclasses are converted.
    Other objects are {}.

    Additional classes can be added with the `add_converter` decorator.
    """

    converters: dict[type, typing.Callable[['ObjectConverter', typing.Any], dict]] = field(default_factory=dict)

    def __call__(self, obj: object):
        if type(obj) in self.converters:
            return self.converters[type(obj)](self, obj)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, tuple):
            return tuple(self(v) for v in obj)
        elif isinstance(obj, list):
            return [self(v) for v in obj]
        elif isinstance(obj, set):
            return {self(v) for v in obj}
        elif isinstance(obj, dict):
            return {self(k): self(v) for k, v in obj.items()}
        elif not isinstance(obj, type) and dataclasses.is_dataclass(obj):
            return {self(f.name): self(getattr(obj, f.name)) for f in dataclasses.fields(obj)}

        return repr(obj)

    def register_converter(
        self, func: typing.Callable[['ObjectConverter', T], dict] | None = None, type_: type[T] | None = None
    ):
        """
        Registers a converter for a type.

        This can be called as a decorator or as a function.
        """
        if func is None:
            # return a decorator that adds a converter for type_
            return partial(self.register_converter, type_=type_)

        if type_ is None:
            # get the type from the function annotation (the first argument to func)
            signature = inspect.signature(func)
            type_ = list(signature.parameters.values())[0].annotation

            # fail is there no annotation
            if type_ is inspect.Parameter.empty:
                raise ValueError(
                    "No type annotation for the first argument to the converter function. Please pass the "
                    "type as the type_ argument to register_converter or add a type annotation."
                )

            # check if type_ is Annoated
            if hasattr(type_, '__metadata__'):
                assert hasattr(type_, '__origin__')
                # get the actual type
                type_ = type_.__origin__  # type: ignore

            assert type_ is not None, "type_ is None"

        self.converters[type_] = func

        return func

    def add_converter(self, func: typing.Callable[['ObjectConverter', T], dict] | None = None):
        """
        Decorator that adds a converter to the ObjectConverter class that is wrapped.
        """

        def wrapper(type_: type):
            if func is None:
                # Check if type_ is a Pydanitc model
                if isinstance(type_, type) and issubclass(type_, pydantic.BaseModel):
                    # Add a converter for the model
                    self.register_converter(convert_pydantic_model, type_)
                else:
                    raise ValueError("add_converter can only be used with Pydantic models.")

            # Add a converter for the type
            self.register_converter(func, type_)

            return type_

        return wrapper


def convert_pydantic_model(converter: ObjectConverter, obj: pydantic.BaseModel) -> dict:
    """
    Converts a pydantic model to a dict.
    """
    # Iterate over the fields of the model
    return {f.name: converter(getattr(obj, f.name)) for f in obj.__fields__.values()}
