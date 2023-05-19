from blackboard_pagi.tools.trace_viewer.endpoint_integration import trace_viewer_send_trace_builder
from blackboard_pagi.utils.tracer import TraceBuilder, TraceBuilderEventHandler


class TraceViewerIntegration(TraceBuilderEventHandler):
    def on_scope_final(self, builder: 'TraceBuilder'):
        trace_viewer_send_trace_builder(builder, force=True)

    def on_event_scope_final(self, builder: 'TraceBuilder'):
        trace_viewer_send_trace_builder(builder, force=False)
