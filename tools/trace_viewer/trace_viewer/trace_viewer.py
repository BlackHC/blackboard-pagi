"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import pprint  # noqa: F401
import typing
from enum import Enum

import pynecone as pc
import pynecone.pc as cli
from pcconfig import config
from trace_viewer.flame_graph import FlameGraphNode, flame_graph
from trace_viewer.json_view import json_view

from blackboard_pagi.utils.tracer import Trace, TraceNode, TraceNodeKind

docs_url = "https://pynecone.io/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"


# solarized colors as HTML hex
# https://ethanschoonover.com/solarized/
class SolarizedColors(str, Enum):
    base03 = "#002b36"
    base02 = "#073642"
    base01 = "#586e75"
    base00 = "#657b83"
    base0 = "#839496"
    base1 = "#93a1a1"
    base2 = "#eee8d5"
    base3 = "#fdf6e3"
    yellow = "#b58900"
    orange = "#cb4b16"
    red = "#dc322f"
    magenta = "#d33682"
    violet = "#6c71c4"
    blue = "#268bd2"
    cyan = "#2aa198"
    green = "#859900"


class NodeInfo(pc.Base):
    node_name: str
    kind: TraceNodeKind
    event_id: int
    start_time_ms: int
    end_time_ms: int
    # delta_frame_infos: list[dict[str, object]]
    properties: object | None
    exception: object | None
    self_object: object | None
    result: object | None
    arguments: object | None


class Wrapper(pc.Base):
    inner: object = None

    def __init__(self, inner):
        super().__init__(inner=inner)


def load_example_trace():
    return Trace.parse_file('optimization_unit_trace_example.json')


# streaming_trace: Trace | None = None
#
# async def update_trace(trace: Trace):
#     streaming_trace = trace
#     # trigger update in state


def convert_trace_node_kind_to_color(kind: TraceNodeKind):
    if kind == TraceNodeKind.SCOPE:
        return SolarizedColors.base1
    elif kind == TraceNodeKind.AGENT:
        return SolarizedColors.green
    elif kind == TraceNodeKind.LLM:
        return SolarizedColors.blue
    elif kind == TraceNodeKind.CHAIN:
        return SolarizedColors.cyan
    elif kind == TraceNodeKind.CALL:
        return SolarizedColors.yellow
    elif kind == TraceNodeKind.EVENT:
        return SolarizedColors.orange
    elif kind == TraceNodeKind.TOOL:
        return SolarizedColors.magenta
    else:
        return SolarizedColors.base2


def convert_node_to_color(node: TraceNode):
    if "exception" in node.properties:
        return SolarizedColors.red
    else:
        return "black"


def convert_trace_to_flame_graph_data(trace: Trace) -> dict:
    def convert_node(node: TraceNode, discount=1.0) -> FlameGraphNode:
        children = []
        last_ms = node.start_time_ms
        for child in node.children:
            gap_s = child.start_time_ms - last_ms
            if gap_s > 0:
                children.append(
                    FlameGraphNode(
                        name="",
                        background_color="#00000000",
                        value=gap_s,
                        children=[],
                    )
                )
            children.append(convert_node(child, discount=discount * 0.95))
            last_ms = child.end_time_ms

        duration_s = node.end_time_ms - node.start_time_ms
        return FlameGraphNode(
            id=str(node.event_id),
            name=node.name,
            value=duration_s * discount,
            children=children,
            background_color=convert_trace_node_kind_to_color(node.kind),
            color=convert_node_to_color(node),
        )

    converted_node = convert_node(trace.traces[-1]).dict(exclude_unset=True)
    return converted_node


class State(pc.State):
    """The app state."""

    flame_graph_data: dict = FlameGraphNode(name="", value=1, background_color="#00000000", children=[]).dict()
    current_node: list[NodeInfo] = []
    _trace: Trace | None = None
    _event_id_map: dict[int, TraceNode] = dict()

    def reset_graph(self):
        self._trace = None
        self.update_flame_graph(self)  # type: ignore

    def load_default_flame_graph(self):
        self._trace = load_example_trace()
        self.update_flame_graph(self)  # type: ignore

    async def handle_trace_upload(self, file: pc.UploadFile):
        """Handle the upload of a file.

        Args:
            file: The uploaded file.
        """
        upload_data = await file.read()
        self._trace = Trace.parse_raw(upload_data)
        self.update_flame_graph(self)  # type: ignore

    def update_flame_graph(self):
        if self._trace is None:
            self.flame_graph_data = FlameGraphNode(name="", value=1, background_color="#00000000", children=[]).dict()
            self._event_id_map = {}
        else:
            self._event_id_map = self._trace.build_event_id_map()
            self.flame_graph_data = convert_trace_to_flame_graph_data(self._trace)
        self.current_node = []

    def update_current_node(self, chart_data: dict):
        node_id = chart_data["source"].get("id", None)
        if node_id is None:
            return

        event_id = int(node_id)

        trace_node = self._event_id_map[event_id]

        properties = trace_node.properties.copy()
        exception = properties.pop("exception", None)
        arguments = properties.pop("arguments", None)
        result = properties.pop("result", None)
        if arguments:
            assert isinstance(arguments, dict)
            arguments = arguments.copy()
            self_object = arguments.pop("self", None)
        else:
            self_object = None

        optional_properties = properties if properties else None
        self.current_node = [
            NodeInfo(
                node_name=trace_node.name,
                kind=trace_node.kind,
                event_id=trace_node.event_id,
                start_time_ms=trace_node.start_time_ms,
                end_time_ms=trace_node.end_time_ms,
                properties=optional_properties,
                exception=exception,
                self_object=self_object,
                result=result,
                arguments=arguments,
            )
        ]


def render_node_info(node_info: NodeInfo):
    """Renders node_info as a simple table."""
    header_style = dict(width="20%", text_align="right", vertical_align="top")

    return pc.table_container(
        pc.table(
            pc.tbody(
                pc.tr(
                    pc.th("Name", style=header_style),
                    pc.td(node_info.node_name, colspan=0),
                ),
                pc.tr(
                    pc.th("", style=header_style),
                    pc.td(
                        pc.stat_group(
                            pc.stat(
                                pc.stat_number(node_info.kind),
                                pc.stat_help_text("KIND"),
                            ),
                            pc.stat(
                                pc.stat_number((node_info.end_time_ms - node_info.start_time_ms) / 1000),
                                pc.stat_help_text("DURATION (S)"),
                            ),
                            width="100%",
                        )
                    ),
                ),
                pc.cond(
                    node_info.exception,
                    pc.tr(
                        pc.th("Exception", style=header_style),
                        pc.td(json_view(data=node_info.exception), colspan=0),
                    ),
                ),
                pc.cond(
                    node_info.arguments,
                    pc.tr(
                        pc.th("Arguments", style=header_style),
                        pc.td(json_view(data=node_info.arguments), colspan=0),
                    ),
                ),
                pc.cond(
                    node_info.self_object,
                    pc.tr(
                        pc.th("Self", style=header_style),
                        pc.td(json_view(data=node_info.self_object), colspan=0),
                    ),
                ),
                pc.cond(
                    node_info.result,
                    pc.tr(
                        pc.th("Result", style=header_style),
                        pc.td(json_view(data=node_info.result), colspan=0),
                    ),
                ),
                pc.cond(
                    node_info.properties,
                    pc.tr(
                        pc.th("Properties", style=header_style),
                        pc.td(json_view(data=node_info.properties), colspan=0),
                    ),
                ),
            ),
            variant="simple",
        ),
        style=dict(width=1024),
    )


@typing.no_type_check
def index() -> pc.Component:
    return pc.center(
        pc.vstack(
            pc.span(
                pc.heading("Trace Viewer", level=1, style=dict(display="inline-block", margin_right="16px")),
                pc.popover(
                    pc.popover_trigger(pc.button(pc.icon(tag="hamburger"), style=dict(vertical_align="top"))),
                    pc.popover_content(
                        pc.popover_header("Choose Trace Source"),
                        pc.popover_body(
                            pc.button("Load Example", on_click=State.load_default_flame_graph),
                            pc.divider(margin="0.5em"),
                            pc.upload(
                                pc.text("Drag and drop files here or click to select trace json file."),
                                border="1px dotted",
                                padding="2em",
                            ),
                            pc.button("Load", on_click=lambda: State.handle_trace_upload(pc.upload_files())),
                            pc.divider(margin="0.5em"),
                            pc.button("Reset", on_click=State.reset_graph),
                        ),
                        # pc.popover_footer(pc.text("Footer text.")),
                        pc.popover_close_button(),
                    ),
                ),
            ),
            flame_graph(
                width=1024,
                height=100,
                data=State.flame_graph_data,
                on_change=lambda data: State.update_current_node(data),  # type: ignore
            ),
            pc.foreach(
                State.current_node,
                render_node_info,
            ),
        ),
        padding_top="64px",
        padding_bottom="64px",
    )


# Add state and page to the app.
app = pc.App(
    state=State,
    stylesheets=[
        'react-json-view-lite.css',
    ],
)
app.add_page(index)
# app.api.add_api_route("/update_trace/", update_trace)
app.compile()

if __name__ == "__main__":
    cli.main()
