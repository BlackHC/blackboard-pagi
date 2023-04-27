import dis
import inspect

from pydantic.dataclasses import dataclass

from blackboard_pagi.utils.tracer import module_filtering


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
    module_filters: module_filtering.ModuleFilters | None = None,
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
        frame_infos = get_frame_infos(module_filters=module_filtering.only_module(__name__))
        assert len(frame_infos) == 3
        assert frame_infos[0].function == "f"
        assert frame_infos[1].function == "g"
        assert frame_infos[2].function == "test_get_frame_infos"

    def g():
        f()

    g()
