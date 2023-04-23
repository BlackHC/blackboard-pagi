from wandb.sdk.data_types import trace_tree

from blackboard_pagi.utils.tracer import EventNode, TraceBuilder

NAME_PREFIX_TO_SPAN_KIND = {
    "llm:": trace_tree.SpanKind.LLM,
    "chain:": trace_tree.SpanKind.CHAIN,
    "agent:": trace_tree.SpanKind.AGENT,
    "tool:": trace_tree.SpanKind.TOOL,
}


def convert_name(name: str | None) -> tuple[str, trace_tree.SpanKind | str]:
    if name is None:
        name = ""

    for prefix, span_kind in NAME_PREFIX_TO_SPAN_KIND.items():
        if name.startswith(prefix):
            return name[len(prefix) :], span_kind

    return name, "SCOPE"


def build_span(node: EventNode):
    span = trace_tree.Span()
    span_name, span_kind = convert_name(node.name)
    span.name = span_name
    span.span_kind = span_kind

    span.start_time_ms = node.start_time
    span.end_time_ms = node.end_time

    if "exception" not in node.properties:
        span.status_code = trace_tree.StatusCode.SUCCESS
    else:
        span.status_code = trace_tree.StatusCode.ERROR
        span.status_message = repr(node.properties["exception"])

    span.add_named_result(node.properties.get('arguments', {}), node.properties.get('result', {}))

    attributes = dict(node.properties)
    if "arguments" in attributes:
        del attributes["arguments"]
    if "result" in attributes:
        del attributes["result"]

    span.attributes = attributes
    span.child_spans = [build_span(child) for child in node.sub_events]
    return span


def build_trace_tree(trace_builder: TraceBuilder):
    root_span = build_span(trace_builder.event_root)
    media = trace_tree.WBTraceTree(root_span=root_span, model_dict=trace_builder.unique_objects)
    return media
