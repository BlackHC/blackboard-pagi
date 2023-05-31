import json
import os
from dataclasses import dataclass

from blackboard_pagi.utils.tracer.trace_builder import TraceBuilder, TraceBuilderEventHandler


@dataclass
class JsonFileWriter(TraceBuilderEventHandler):
    filename: str

    def on_event_scope_final(self, builder: 'TraceBuilder'):
        trace = builder.build()
        json_trace = trace.dict()

        tempfile = self.filename + ".new_tmp"

        with open(tempfile, "w") as f:
            json.dump(json_trace, f, indent=1)

        os.replace(tempfile, self.filename)
