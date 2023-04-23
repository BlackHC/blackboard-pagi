from blackboard_pagi.utils.tracer import event_scope, trace_builder, trace_calls


def test_trace():
    with trace_builder(module_filters=__name__).scope() as builder:
        with event_scope("foo"):
            with event_scope("bar"):
                with event_scope("baz"):
                    pass

    assert builder is not None
    assert builder.build(include_timing=False)['event_tree'] == [
        {
            'delta_stack': [
                {
                    'code_context': [
                        'def test_trace():\n',
                        '    with ' 'trace_builder(module_filters=__name__).scope() ' 'as builder:\n',
                        '        with event_scope("foo"):\n',
                    ],
                    'filename': 'blackboard_pagi.utils.tests.test_logger',
                    'function': 'test_trace',
                    'index': 1,
                    'lineno': 5,
                }
            ],
            'event_id': 1,
            'name': None,
            'properties': {},
            'sub_events': [
                {
                    'delta_stack': [
                        {
                            'code_context': [
                                '    with ' 'trace_builder(module_filters=__name__).scope() ' 'as builder:\n',
                                '        with ' 'event_scope("foo"):\n',
                                '            with ' 'event_scope("bar"):\n',
                            ],
                            'filename': 'blackboard_pagi.utils.tests.test_logger',
                            'function': 'test_trace',
                            'index': 1,
                            'lineno': 6,
                        }
                    ],
                    'event_id': 2,
                    'name': 'foo',
                    'properties': {},
                    'sub_events': [
                        {
                            'delta_stack': [
                                {
                                    'code_context': [
                                        '        ' 'with ' 'event_scope("foo"):\n',
                                        '            ' 'with ' 'event_scope("bar"):\n',
                                        '                ' 'with ' 'event_scope("baz"):\n',
                                    ],
                                    'filename': 'blackboard_pagi.utils.tests.test_logger',
                                    'function': 'test_trace',
                                    'index': 1,
                                    'lineno': 7,
                                }
                            ],
                            'event_id': 3,
                            'name': 'bar',
                            'properties': {},
                            'sub_events': [
                                {
                                    'delta_stack': [
                                        {
                                            'code_context': [
                                                '            ' 'with ' 'event_scope("bar"):\n',
                                                '                ' 'with ' 'event_scope("baz"):\n',
                                                '                    ' 'pass\n',
                                            ],
                                            'filename': 'blackboard_pagi.utils.tests.test_logger',
                                            'function': 'test_trace',
                                            'index': 1,
                                            'lineno': 8,
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
    ).scope() as builder:
        f(3)
        f(5)

    assert builder is not None
    event_tree = builder.build(include_timing=False)['event_tree']
    assert event_tree == [
        {
            'delta_stack': [
                {
                    'code_context': [
                        '\n',
                        '    with ' 'trace_builder(module_filters=__name__, ' ').scope() as builder:\n',
                        '        f(3)\n',
                    ],
                    'filename': 'blackboard_pagi.utils.tests.test_logger',
                    'function': 'test_trace_calls',
                    'index': 1,
                    'lineno': 102,
                }
            ],
            'event_id': 1,
            'name': None,
            'properties': {},
            'sub_events': [
                {
                    'delta_stack': [
                        {
                            'code_context': [
                                '    with ' 'trace_builder(module_filters=__name__, ' ').scope() as builder:\n',
                                '        f(3)\n',
                                '        f(5)\n',
                            ],
                            'filename': 'blackboard_pagi.utils.tests.test_logger',
                            'function': 'test_trace_calls',
                            'index': 1,
                            'lineno': 103,
                        }
                    ],
                    'event_id': 2,
                    'name': 'f',
                    'properties': {'result': 9, 'value': 3},
                    'sub_events': [],
                },
                {
                    'delta_stack': [
                        {
                            'code_context': ['        f(3)\n', '        f(5)\n', '\n'],
                            'filename': 'blackboard_pagi.utils.tests.test_logger',
                            'function': 'test_trace_calls',
                            'index': 1,
                            'lineno': 104,
                        }
                    ],
                    'event_id': 3,
                    'name': 'f',
                    'properties': {'result': 15, 'value': 5},
                    'sub_events': [],
                },
            ],
        }
    ]
