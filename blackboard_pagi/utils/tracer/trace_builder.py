"""
Simple logger/execution tracker that uses tracks the stack frames and 'data'.
"""
import inspect
import time
import traceback
import typing
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import ClassVar

from blackboard_pagi.utils.callable_wrapper import CallableWrapper
from blackboard_pagi.utils.tracer import module_filtering
from blackboard_pagi.utils.tracer.frame_info import FrameInfo, get_frame_infos
from blackboard_pagi.utils.tracer.object_converter import ObjectConverter
from blackboard_pagi.utils.tracer.trace_schema import Trace, TraceNode, TraceNodeKind
from blackboard_pagi.utils.weakrefs import WeakKeyIdMap

T = typing.TypeVar("T")
P = typing.ParamSpec("P")


def default_timer() -> int:
    """
    Default timer for the tracer.

    Returns:
        The current time in milliseconds.
    """
    return int(time.time() * 1000)


@dataclass
class TraceNodeBuilder:
    """
    A node builder in the trace tree.
    """

    kind: TraceNodeKind
    name: str | None
    event_id: int
    start_time_ms: int
    delta_frame_infos: list[FrameInfo]
    stack_height: int

    end_time_ms: int | None = None
    parent: 'TraceNodeBuilder | None' = None
    children: list['TraceNodeBuilder'] = field(default_factory=list)
    properties: dict[str, object] = field(default_factory=dict)

    @classmethod
    def create_root(cls):
        return cls(
            kind=TraceNodeKind.SCOPE,
            name=None,
            event_id=0,
            start_time_ms=0,
            delta_frame_infos=[],
            stack_height=0,
        )

    def get_delta_frame_infos(
        self, num_frames_to_skip: int = 0, module_filters: module_filtering.ModuleFilters | None = None, context=3
    ):
        frame_infos, full_stack_height = get_frame_infos(
            num_top_frames_to_skip=num_frames_to_skip + 1,
            num_bottom_frames_to_skip=self.stack_height,
            module_filters=module_filters,
            context=context,
        )

        return frame_infos, full_stack_height

    def build(self):
        return TraceNode(
            kind=self.kind,
            name=self.name,
            event_id=self.event_id,
            start_time_ms=self.start_time_ms,
            end_time_ms=self.end_time_ms,
            delta_frame_infos=self.delta_frame_infos,
            properties=self.properties,
            children=[sub_event.build() for sub_event in self.children],
        )


trace_object_converter = ObjectConverter()
trace_module_filters = None


@dataclass
class TraceBuilder:
    _current: ClassVar[ContextVar['TraceBuilder | None']] = ContextVar("current_trace_builder", default=None)

    module_filters: module_filtering.ModuleFilters
    stack_frame_context: int

    event_root: TraceNodeBuilder = field(default_factory=TraceNodeBuilder.create_root)
    object_map: WeakKeyIdMap[object, str] = field(default_factory=WeakKeyIdMap)
    unique_objects: dict[str, dict] = field(default_factory=dict)

    id_counter: int = 0
    current_event_node: TraceNodeBuilder | None = None

    def build(self):
        return Trace(
            name=self.event_root.name,
            properties=self.event_root.properties,
            traces=[child.build() for child in self.event_root.children],
            unique_objects=self.unique_objects,
        )

    def next_id(self):
        self.id_counter += 1
        return self.id_counter

    @contextmanager
    def scope(self, name: str | None = None):
        """
        Context manager that allows to trace our program execution.
        """
        assert self.current_event_node is None
        self.current_event_node = self.event_root

        token = self._current.set(self)
        try:
            with self.event_scope(name=name, kind=TraceNodeKind.SCOPE, skip_frames=2):
                yield self
        finally:
            self._current.reset(token)
            self.current_event_node = None

    @contextmanager
    def event_scope(
        self,
        name: str | None,
        properties: dict[str, object] | None = None,
        kind: TraceNodeKind = TraceNodeKind.SCOPE,
        skip_frames: int = 0,
    ):
        """
        Context manager that allows to trace our program execution.
        """
        assert self._current.get() is self
        assert self.current_event_node is not None

        if properties is None:
            properties = {}

        start_time = default_timer()
        delta_frame_infos, stack_height = self.current_event_node.get_delta_frame_infos(
            num_frames_to_skip=2 + skip_frames, module_filters=self.module_filters, context=self.stack_frame_context
        )
        event_node = TraceNodeBuilder(
            kind=kind,
            name=name,
            event_id=self.next_id(),
            start_time_ms=start_time,
            delta_frame_infos=delta_frame_infos,
            stack_height=stack_height - 1,
            parent=self.current_event_node,
            properties=dict(properties),
        )
        self.current_event_node.children.append(event_node)

        old_event_node = self.current_event_node
        self.current_event_node = event_node

        try:
            yield
        except BaseException as e:
            self.update_event_properties(exception='\n'.join(traceback.TracebackException.from_exception(e).format()))
            raise
        finally:
            event_node.end_time_ms = default_timer()
            self.current_event_node = old_event_node

    def register_object(self, obj: object, name: str, properties: dict[str, object]):
        # Make name unique if needed
        if name in self.unique_objects:
            # if we are in a scope, we can use the scope name as a prefix
            if self.current_event_node is not None:
                name = f"{self.current_event_node.name}_{name}"

            if name in self.unique_objects:
                i = 1
                while f"{name}[{i}]" in self.unique_objects:
                    i += 1
                name = f"{name}[{i}]"
        self.object_map[obj] = name
        self.unique_objects[name] = properties

    def convert_object(self, obj: object):
        # anything that is not JSON compatible is converted to a unique object with a name
        # and a set of properties that can be serialized using the object converter
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return [self.convert_object(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_object(item) for item in obj)
        elif isinstance(obj, set):
            return {self.convert_object(item) for item in obj}
        elif isinstance(obj, dict):
            return {key: self.convert_object(value) for key, value in obj.items()}
        else:
            # if the object is in the map, we return its name as a reference
            if obj in self.object_map:
                return dict(unique_object=self.object_map[obj])
            else:
                return trace_object_converter(obj)

    @classmethod
    def get_current(cls) -> 'TraceBuilder | None':
        return cls._current.get()

    @classmethod
    def get_current_node(cls) -> 'TraceNodeBuilder | None':
        current = cls.get_current()
        if current is None:
            return None
        else:
            return current.current_event_node

    def add_event(
        self,
        name: str,
        properties: dict[str, object] | None = None,
        kind: TraceNodeKind = TraceNodeKind.EVENT,
    ):
        """
        Add an event to the current scope.
        """
        if properties is None:
            properties = {}
        with self.event_scope(name, properties=properties, kind=kind, skip_frames=2):
            pass

    def update_event_properties(self, properties: dict[str, object] | None = None, /, **kwargs):
        """
        Update the properties of the current event.
        """
        assert self.current_event_node is not None
        if properties is None:
            properties = {}
        self.current_event_node.properties.update(self.convert_object(properties | kwargs))

    def update_name(self, name: str):
        """
        Update the name of the current event.
        """
        assert self.current_event_node is not None
        self.current_event_node.name = name


def trace_builder(module_filters: module_filtering.ModuleFiltersSpecifier | None = None, stack_frame_context: int = 3):
    """
    Context manager that allows to trace our program execution.
    """
    if not module_filters:
        module_filters = trace_module_filters

    return TraceBuilder(
        module_filters=module_filtering.module_filters(module_filters), stack_frame_context=stack_frame_context
    )


def add_event(name: str, properties: dict[str, object] | None = None, kind: TraceNodeKind = TraceNodeKind.EVENT):
    """
    Add an event to the current scope.
    """
    current = TraceBuilder.get_current()
    if current is not None:
        current.add_event(name, properties, kind)


def register_object(obj: object, name: str, properties: dict[str, object]):
    """
    Register an object as unique, so that it will be serialized only once.
    """
    current = TraceBuilder.get_current()
    if current is not None:
        current.register_object(obj, name, properties)


def update_event_properties(properties: dict[str, object] | None = None, /, **kwargs):
    """
    Update the properties of the current event.
    """
    current = TraceBuilder.get_current()
    if current is not None:
        current.update_event_properties(properties, **kwargs)


def update_name(name: str):
    """
    Update the name of the current event.
    """
    current = TraceBuilder.get_current()
    if current is not None:
        current.update_name(name)


@contextmanager
def event_scope(name: str, properties: dict[str, object] | None = None, kind: TraceNodeKind = TraceNodeKind.SCOPE):
    """
    Context manager that allows to trace our program execution.
    """
    current = TraceBuilder.get_current()
    if current is None:
        yield
    else:
        with current.event_scope(name, properties=properties, kind=kind, skip_frames=2):
            yield


@dataclass
class CallTracer(typing.Callable[P, T], typing.Generic[P, T], CallableWrapper):  # type: ignore
    __signature__: inspect.Signature
    __wrapped__: typing.Callable[P, T]
    __wrapped_name__: str
    __kind__: TraceNodeKind = TraceNodeKind.CALL
    __capture_return__: bool = False
    __capture_args__: bool | list[str] = False

    def __call__(self, *args, **kwargs):
        # check if we are in a trace
        builder = TraceBuilder.get_current()
        if builder is None:
            return self.__wrapped__(*args, **kwargs)

        # build properties
        properties = {}
        if self.__capture_args__:
            # bind the arguments to the signature
            bound_args = self.__signature__.bind(*args, **kwargs)
            # add the arguments to the properties
            if self.__capture_args__ is True:
                arguments = bound_args.arguments
            else:
                arguments = {arg: bound_args.arguments[arg] for arg in self.__capture_args__}

            # anything that can be stored in a json is okay
            converted_arguments = {}
            for arg, value in arguments.items():
                converted_arguments[arg] = builder.convert_object(value)
            properties["arguments"] = converted_arguments

        # create event scope
        with builder.event_scope(self.__wrapped_name__, properties, kind=self.__kind__, skip_frames=1):
            # call the function
            result = self.__wrapped__(*args, **kwargs)

            if self.__capture_return__:
                update_event_properties({"result": builder.convert_object(result)})
        return result


def trace_calls(
    func=None,
    *,
    name: str | None = None,
    kind: TraceNodeKind = TraceNodeKind.CALL,
    capture_return: bool = False,
    capture_args: bool | list[str] = False,
):
    """
    Decorator that allows to trace our program execution.
    """
    if func is None:
        return partial(
            trace_calls,
            name=name,
            kind=kind,
            capture_return=capture_return,
            capture_args=capture_args,
        )

    # get the signature of the function
    signature = inspect.signature(func)

    # if capture_args is an iterable, convert it to a set
    if isinstance(capture_args, typing.Iterable):
        assert not isinstance(capture_args, bool)
        arg_names = set(capture_args)

        # check that all the arguments are valid
        for arg in arg_names:
            if arg not in signature.parameters:
                raise ValueError(f"Argument '{arg}' is not a valid argument of function '{func.__name__}'!")

    # get the name of the function
    if name is None:
        name = func.__name__

    wrapped_function = wraps(func)(
        CallTracer(
            __signature__=signature,
            __wrapped__=func,
            __wrapped_name__=name,
            __kind__=kind,
            __capture_return__=capture_return,
            __capture_args__=capture_args,
        )
    )

    return wrapped_function
