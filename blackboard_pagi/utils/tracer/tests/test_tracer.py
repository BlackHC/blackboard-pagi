from blackboard_pagi.utils.tracer import event_scope, trace_builder, trace_calls


def test_trace():
    with trace_builder(module_filters=__name__, stack_frame_context=0).scope() as builder:
        with event_scope("foo"):
            with event_scope("bar"):
                with event_scope("baz"):
                    pass

    assert builder is not None
    assert builder.build(include_timing=False, include_lineno=False)['event_tree'] == [
        {
            'delta_stack': [
                {
                    'code_context': None,
                    'function': 'test_trace',
                    'index': None,
                    'module': 'blackboard_pagi.utils.tracer.tests.test_tracer',
                }
            ],
            'event_id': 1,
            'name': None,
            'properties': {},
            'sub_events': [
                {
                    'delta_stack': [
                        {
                            'code_context': None,
                            'function': 'test_trace',
                            'index': None,
                            'module': 'blackboard_pagi.utils.tracer.tests.test_tracer',
                        }
                    ],
                    'event_id': 2,
                    'name': 'foo',
                    'properties': {},
                    'sub_events': [
                        {
                            'delta_stack': [
                                {
                                    'code_context': None,
                                    'function': 'test_trace',
                                    'index': None,
                                    'module': 'blackboard_pagi.utils.tracer.tests.test_tracer',
                                }
                            ],
                            'event_id': 3,
                            'name': 'bar',
                            'properties': {},
                            'sub_events': [
                                {
                                    'delta_stack': [
                                        {
                                            'code_context': None,
                                            'function': 'test_trace',
                                            'index': None,
                                            'module': 'blackboard_pagi.utils.tracer.tests.test_tracer',
                                        }
                                    ],
                                    'event_id': 4,
                                    'name': 'baz',
                                    'properties': {},
                                    'sub_events': [],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ]


def test_trace_calls():
    @trace_calls(capture_args=True, capture_return=True)
    def f(value: int):
        return value * 3

    with trace_builder(
        module_filters=__name__,
        stack_frame_context=0,
    ).scope() as builder:
        f(3)
        f(5)

    assert builder is not None
    event_tree = builder.build(include_timing=False, include_lineno=False)['event_tree']
    assert event_tree == [
        {
            'delta_stack': [
                {
                    'code_context': None,
                    'function': 'test_trace_calls',
                    'index': None,
                    'module': 'blackboard_pagi.utils.tracer.tests.test_tracer',
                }
            ],
            'event_id': 1,
            'name': None,
            'properties': {},
            'sub_events': [
                {
                    'delta_stack': [
                        {
                            'code_context': None,
                            'function': 'test_trace_calls',
                            'index': None,
                            'module': 'blackboard_pagi.utils.tracer.tests.test_tracer',
                        }
                    ],
                    'event_id': 2,
                    'name': 'f',
                    'properties': {'arguments': {'value': 3}, 'result': 9},
                    'sub_events': [],
                },
                {
                    'delta_stack': [
                        {
                            'code_context': None,
                            'function': 'test_trace_calls',
                            'index': None,
                            'module': 'blackboard_pagi.utils.tracer.tests.test_tracer',
                        }
                    ],
                    'event_id': 3,
                    'name': 'f',
                    'properties': {'arguments': {'value': 5}, 'result': 15},
                    'sub_events': [],
                },
            ],
        }
    ]
