"""
Simple logger/execution tracker that uses tracks the stack frames and 'data'.
"""
import dis
import inspect
import typing
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import ClassVar

from blackboard_pagi.utils import module_filter
from blackboard_pagi.utils.object_converter import ObjectConverter
from blackboard_pagi.utils.stopwatch_context import StopwatchContext
from blackboard_pagi.utils.weakrefs import WeakKeyIdMap


@dataclass
class FrameInfo:
    """
    A frame info object that is serializable.
    """

    module: str
    lineno: int
    function: str
    code_context: list[str] | None
    index: int | None
    positions: dis.Positions | None = None


def get_frame_infos(
    num_top_frames_to_skip: int = 0,
    num_bottom_frames_to_skip: int = 0,
    module_filters: module_filter.ModuleFilters | None = None,
    context: int = 3,
) -> tuple[list[FrameInfo], int]:
    # Get the current stack frame infos
    frame_infos: list[inspect.FrameInfo] = inspect.stack(context=context)

    # Get the stack frame infos for the delta stack summary
    caller_frame_infos = frame_infos[num_top_frames_to_skip + 1 :]

    stack_height = len(caller_frame_infos)

    # remove bottom frames
    relevant_inspect_frame_infos: list[inspect.FrameInfo] = caller_frame_infos[
        : stack_height - num_bottom_frames_to_skip
    ]

    relevant_frame_infos: list[FrameInfo] = [
        FrameInfo(
            module=module.__name__ if (module := inspect.getmodule(f.frame)) else "<unknown>",
            lineno=f.lineno,
            function=f.function,
            code_context=f.code_context,
            index=f.index,
            positions=f.positions,
        )
        for f in relevant_inspect_frame_infos
    ]

    # Filter the stack frame infos
    if module_filters is not None:
        relevant_frame_infos = [f for f in relevant_frame_infos if module_filters(f.module)]

    return relevant_frame_infos, stack_height


def test_get_frame_infos():
    def f():
        frame_infos = get_frame_infos(module_filters=module_filter.only_module(__name__))
        assert len(frame_infos) == 3
        assert frame_infos[0].function == "f"
        assert frame_infos[1].function == "g"
        assert frame_infos[2].function == "test_get_frame_infos"

    def g():
        f()

    g()


@dataclass
class EventNode:
    """
    A node in the trace tree.
    """

    name: str | None
    event_id: int
    start_time: float
    delta_frame_infos: list[FrameInfo]
    stack_height: int

    end_time: float | None = None
    parent: 'EventNode | None' = None
    sub_events: list['EventNode'] = field(default_factory=list)
    properties: dict[str, object] = field(default_factory=dict)

    @classmethod
    def create_root(cls):
        return cls('root', 0, 0, [], 0)

    def get_delta_frame_infos(
        self, num_frames_to_skip: int = 0, module_filters: module_filter.ModuleFilters | None = None, context=3
    ):
        frame_infos, full_stack_height = get_frame_infos(
            num_top_frames_to_skip=num_frames_to_skip + 1,
            num_bottom_frames_to_skip=self.stack_height,
            module_filters=module_filters,
            context=context,
        )

        return frame_infos, full_stack_height


trace_object_converter = ObjectConverter()
trace_module_filters = None


@dataclass
class TraceBuilder:
    _current: ClassVar[ContextVar['TraceBuilder | None']] = ContextVar("current_trace_builder", default=None)

    module_filters: module_filter.ModuleFilters
    stack_frame_context: int

    event_root: EventNode = field(default_factory=EventNode.create_root)
    object_map: WeakKeyIdMap[object, str] = field(default_factory=WeakKeyIdMap)
    unique_objects: dict[str, dict] = field(default_factory=dict)

    id_counter: int = 0
    current_event_node: EventNode | None = None
    stopwatch: StopwatchContext = field(default_factory=StopwatchContext)

    def build(self, include_timing: bool = True):
        # convert everything to a JSON-compatible dict
        # unique_objects is already compatible
        # we only need to convert all tree nodes
        return {
            'unique_objects': self.unique_objects,
            'event_tree': [
                self.convert_event_node(traces, include_timing=include_timing) for traces in self.event_root.sub_events
            ],
        }

    def convert_event_node(self, node: EventNode, include_timing: bool = True):
        converted: dict = {
            'name': node.name,
            'event_id': node.event_id,
            'properties': node.properties,
            'delta_stack':
            # convert the stack frames into a dict
            [
                {
                    'module': f.module,
                    'lineno': f.lineno,
                    'function': f.function,
                    'code_context': f.code_context,
                    'index': f.index,
                }
                for f in node.delta_frame_infos
            ],
            'sub_events': [self.convert_event_node(n, include_timing=include_timing) for n in node.sub_events],
        }
        if include_timing:
            converted['start_time'] = node.start_time
            converted['end_time'] = node.end_time
        return converted

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

        start_time = self.stopwatch.elapsed_time
        delta_frame_infos, stack_height = self.event_root.get_delta_frame_infos(
            num_frames_to_skip=2, module_filters=self.module_filters, context=self.stack_frame_context
        )
        scope_root = EventNode(
            name,
            self.next_id(),
            start_time,
            delta_frame_infos=delta_frame_infos,
            stack_height=stack_height - 1,
            parent=self.event_root,
        )
        self.event_root.sub_events.append(scope_root)
        self.current_event_node = scope_root

        token = self._current.set(self)
        try:
            with self.stopwatch:
                yield self
        finally:
            scope_root.end_time = self.stopwatch.elapsed_time
            self.current_event_node = None
            self._current.reset(token)

    @contextmanager
    def event_scope(self, name: str, properties: dict[str, object] | None = None, skip_frames: int = 0):
        """
        Context manager that allows to trace our program execution.
        """
        assert self._current.get() is self
        assert self.current_event_node is not None

        if properties is None:
            properties = {}

        start_time = self.stopwatch.elapsed_time
        delta_frame_infos, stack_height = self.current_event_node.get_delta_frame_infos(
            num_frames_to_skip=2 + skip_frames, module_filters=self.module_filters, context=self.stack_frame_context
        )
        event_node = EventNode(
            name,
            self.next_id(),
            start_time,
            delta_frame_infos=delta_frame_infos,
            stack_height=stack_height - 1,
            parent=self.current_event_node,
            properties=dict(properties),
        )
        self.current_event_node.sub_events.append(event_node)

        old_event_node = self.current_event_node
        self.current_event_node = event_node

        try:
            yield
        finally:
            event_node.end_time = self.stopwatch.elapsed_time
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
        elif isinstance(obj, (list, tuple, set)):
            return (self.convert_object(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: self.convert_object(value) for key, value in obj.items()}
        else:
            # if the object is already in the map, we just return the name
            if obj in self.object_map:
                return dict(ref=self.object_map[obj])
            else:
                # otherwise we register it and return its name
                name = f"{type(obj).__name__}_{self.next_id()}"
                self.register_object(obj, name, trace_object_converter(obj))
                return dict(ref=name)

    @classmethod
    def get_current(cls) -> 'TraceBuilder | None':
        return cls._current.get()

    @classmethod
    def get_current_node(cls) -> 'EventNode | None':
        current = cls.get_current()
        if current is None:
            return None
        else:
            return current.current_event_node

    def add_event(self, name: str, properties: dict[str, object] | None = None):
        """
        Add an event to the current scope.
        """
        if properties is None:
            properties = {}
        with self.event_scope(name, properties, skip_frames=2):
            pass


def trace_builder(module_filters: module_filter.ModuleFiltersSpecifier | None = None, stack_frame_context: int = 3):
    """
    Context manager that allows to trace our program execution.
    """
    if not module_filters:
        module_filters = trace_module_filters

    return TraceBuilder(
        module_filters=module_filter.module_filters(module_filters), stack_frame_context=stack_frame_context
    )


def register_object(obj: object, name: str, properties: dict[str, object]):
    """
    Register an object as unique, so that it will be serialized only once.
    """
    current = TraceBuilder.get_current()
    if current is not None:
        current.register_object(obj, name, properties)


def update_event_properties(properties: dict[str, object]):
    """
    Update the properties of the current event.
    """
    current = TraceBuilder.get_current_node()
    if current is not None:
        current.properties.update(properties)


@contextmanager
def event_scope(name: str, properties: dict[str, object] | None = None):
    """
    Context manager that allows to trace our program execution.
    """
    current = TraceBuilder.get_current()
    if current is None:
        yield
    else:
        with current.event_scope(name, properties, skip_frames=2):
            yield


def trace_calls(
    func=None,
    *,
    name: str | None = None,
    capture_return: bool = False,
    capture_exception: bool = True,
    capture_args: bool | list[str] = False,
):
    """
    Decorator that allows to trace our program execution.
    """
    if func is None:
        return partial(
            trace_calls,
            name=name,
            capture_return=capture_return,
            capture_exception=capture_exception,
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

    @wraps(func)
    def wrapper(*args, **kwargs):
        # check if we are in a trace
        builder = TraceBuilder.get_current()
        if builder is None:
            return func(*args, **kwargs)

        # build properties
        properties = {}
        if capture_args:
            # bind the arguments to the signature
            bound_args = signature.bind(*args, **kwargs)
            # add the arguments to the properties
            if capture_args:
                if capture_args is True:
                    arguments = bound_args.arguments
                else:
                    arguments = {arg: bound_args.arguments[arg] for arg in capture_args}

                # anything that can be stored in a json is okay
                for arg, value in arguments.items():
                    properties[arg] = builder.convert_object(value)

        # create event scope
        with builder.event_scope(name, properties, skip_frames=1):
            # call the function
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                if capture_exception:
                    update_event_properties({"exception": e})
                raise
            else:
                if capture_return:
                    update_event_properties({"result": result})
                return result

    return wrapper
