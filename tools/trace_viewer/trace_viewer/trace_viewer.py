"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import pynecone as pc
from pcconfig import config
from trace_viewer.flame_graph import FlameGraphNode, flame_graph

from blackboard_pagi.utils.tracer import Trace, TraceNode

docs_url = "https://pynecone.io/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"


class State(pc.State):
    """The app state."""

    pass


def load_trace():
    return Trace.parse_file('optimization_unit_trace_example.json')


def convert_trace_to_flame_graph_data(trace: Trace) -> dict:
    def convert_node(node: TraceNode) -> FlameGraphNode:
        return FlameGraphNode(
            name=node.name,
            value=(node.end_time_ms - node.start_time_ms) / 1000,
            children=[convert_node(child) for child in node.children],
        )

    return convert_node(trace.traces[0]).dict()


def index() -> pc.Component:
    return pc.center(
        pc.vstack(
            pc.heading("Welcome to Pynecone!", font_size="2em"),
            pc.box("Get started by editing ", pc.code(filename, font_size="1em")),
            pc.link(
                "Check out our docs!",
                href=docs_url,
                border="0.1em solid",
                padding="0.5em",
                border_radius="0.5em",
                _hover={
                    "color": "rgb(107,99,246)",
                },
            ),
            flame_graph(
                width=800,
                height=400,
                data=convert_trace_to_flame_graph_data(load_trace()),
                # {
                #     "name": "root",
                #     "value": 5,
                #     "children": [
                #         {
                #             "name": "custom tooltip",
                #             "value": 1,
                #             "tooltip": "Custom tooltip shown on hover"
                #         },
                #         {
                #             "name": "custom colors",
                #             "backgroundColor": "#35f",
                #             "color": "#fff",
                #             "value": 3,
                #             "children": [
                #                 {
                #                     "name": "leaf",
                #                     "value": 2
                #                 }
                #             ]
                #         }
                #     ]
                # }
            ),
            spacing="1.5em",
            font_size="2em",
        ),
        padding_top="10%",
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index)
app.compile()
