import wandb
from blackboard_pagi.utils.tracer import event_scope, trace_builder
from blackboard_pagi.utils.tracer.wandb_integration import build_trace_tree

run = wandb.init(project="blackboard-pagi", name="wandb_integration_test", reinit=True)  # type: ignore

with trace_builder(module_filters=__name__, stack_frame_context=0).scope("first test") as builder:
    with event_scope("foo"):
        with event_scope("bar"):
            with event_scope("baz"):
                pass

with trace_builder(module_filters=__name__, stack_frame_context=0).scope("first test") as builder:
    with event_scope("foo"):
        with event_scope("bar"):
            with event_scope("baz"):
                pass


assert builder is not None

run.log({"trace": build_trace_tree(builder)})
run.finish()
