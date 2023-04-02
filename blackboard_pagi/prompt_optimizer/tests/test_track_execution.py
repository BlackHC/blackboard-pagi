import re

import pytest
from langchain.schema import AIMessage, HumanMessage, OutputParserException

from blackboard_pagi.prompt_optimizer.track_execution import ChatChain, prompt_hyperparameter, track_execution
from blackboard_pagi.testing.fake_chat_model import FakeChatModel


def test_no_description():
    @track_execution
    def f():
        return prompt_hyperparameter @ 1

    assert f() == 1

    f.all_hyperparameters[f.__qualname__][0] = 2
    assert f() == 2

    @track_execution
    def g():
        return prompt_hyperparameter @ "Hello" + prompt_hyperparameter @ "Hello"

    assert g() == "HelloHello"

    g.all_hyperparameters[g.__qualname__][1] = "World"
    assert g() == "HelloWorld"


def test_with_description():
    @track_execution
    def f():
        return prompt_hyperparameter("Hello") @ 1

    assert f() == 1

    f.all_hyperparameters[f.__qualname__]["Hello"] = 2
    assert f() == 2

    @track_execution
    def g():
        return prompt_hyperparameter("Hello") @ "Hello" + prompt_hyperparameter("Hello") @ "Hello"

    assert g() == "HelloHello"

    g.all_hyperparameters[g.__qualname__]["Hello"] = "World"
    assert g() == "WorldWorld"


def test_nested():
    @track_execution
    def f():
        return prompt_hyperparameter("Hello") @ 1 + prompt_hyperparameter("World") @ 2

    assert f() == 3

    @track_execution
    def g():
        return prompt_hyperparameter("Hello") @ 3 + f()

    assert g() == 6

    g.all_hyperparameters[g.__qualname__]["Hello"] = 4

    assert g() == 7

    g.all_hyperparameters[f.__qualname__]["Hello"] = 5

    assert g() == 11


def test_chat_chain():
    # Only test ChatChain
    chat_model = FakeChatModel.from_messages(
        [
            [
                HumanMessage(content="Hello"),
                AIMessage(content="World"),
                HumanMessage(content="How are you?"),
                AIMessage(content="Good. How are you?"),
            ],
            [
                HumanMessage(content="Hello"),
                AIMessage(content="World"),
                HumanMessage(content="What's up?"),
                AIMessage(content="Nothing. You?"),
            ],
        ]
    )

    @track_execution
    def f():
        chat_chain = ChatChain(chat_model, [HumanMessage(content="Hello")])

        assert chat_chain.response == "Hello"

        assert chat_chain.get_full_message_chain() == [
            HumanMessage(content="Hello"),
        ]

        chat_chain_2 = ChatChain(chat_model, [])

        with pytest.raises(AssertionError):
            chat_chain_2.response

        assert chat_chain_2.get_full_message_chain() == []

        response, chat_chain_3 = chat_chain_2.query("Hello", track=False)

        assert response == "World"

        assert chat_chain_3.get_full_message_chain() == [
            HumanMessage(content="Hello"),
            AIMessage(content="World"),
        ]

        assert chat_chain_3.response == "World"
        assert chat_chain_3.get_compact_subtree_dict(include_all=True) == {
            'branches': [],
            'messages': [{'content': 'Hello', 'role': 'user'}, {'content': 'World', 'role': 'assistant'}],
        }

        chat_chain_4 = chat_chain_3.branch()
        with pytest.raises(AssertionError):
            chat_chain_4.response

        assert chat_chain_4.get_full_message_chain() == [
            HumanMessage(content="Hello"),
            AIMessage(content="World"),
        ]

        assert chat_chain_2.get_compact_subtree_dict(include_all=True) == {
            'branches': [],
            'messages': [
                {'content': 'Hello', 'role': 'user'},
                {'content': 'World', 'role': 'assistant'},
            ],
        }

        assert chat_chain_2.get_compact_subtree_dict(include_all=False) == {}

        response, chat_chain_5 = chat_chain_4.query("How are you?", track=False)
        assert response == "Good. How are you?"
        assert chat_chain_5.get_full_message_chain() == [
            HumanMessage(content="Hello"),
            AIMessage(content="World"),
            HumanMessage(content="How are you?"),
            AIMessage(content="Good. How are you?"),
        ]

        assert chat_chain_2.get_compact_subtree_dict(include_all=True) == {
            'branches': [],
            'messages': [
                {'content': 'Hello', 'role': 'user'},
                {'content': 'World', 'role': 'assistant'},
                {'content': 'How are you?', 'role': 'user'},
                {'content': 'Good. How are you?', 'role': 'assistant'},
            ],
        }

        response, chat_chain_6 = chat_chain_4.query("What's up?", track=False)
        assert response == "Nothing. You?"
        assert chat_chain_6.get_full_message_chain() == [
            HumanMessage(content="Hello"),
            AIMessage(content="World"),
            HumanMessage(content="What's up?"),
            AIMessage(content="Nothing. You?"),
        ]

        assert chat_chain_2.get_compact_subtree_dict(include_all=True) == {
            'messages': [{'content': 'Hello', 'role': 'user'}, {'content': 'World', 'role': 'assistant'}],
            'branches': [
                {
                    'branches': [],
                    'messages': [
                        {'content': 'How are you?', 'role': 'user'},
                        {'content': 'Good. How are you?', 'role': 'assistant'},
                    ],
                },
                {
                    'branches': [],
                    'messages': [
                        {'content': "What's up?", 'role': 'user'},
                        {'content': 'Nothing. You?', 'role': 'assistant'},
                    ],
                },
            ],
        }

        chat_chain_6.track()

    f()

    assert f.hyperparameters == {}
    assert f.all_chat_chains == [
        {'branches': [], 'messages': [{'content': 'Hello', 'role': 'user'}]},
        {
            'branches': [
                {
                    'branches': [],
                    'messages': [
                        {'content': 'How are you?', 'role': 'user'},
                        {'content': 'Good. How are you?', 'role': 'assistant'},
                    ],
                },
                {
                    'branches': [],
                    'messages': [
                        {'content': "What's up?", 'role': 'user'},
                        {'content': 'Nothing. You?', 'role': 'assistant'},
                    ],
                },
            ],
            'messages': [{'content': 'Hello', 'role': 'user'}, {'content': 'World', 'role': 'assistant'}],
        },
    ]
    assert f.tracked_chat_chains == [
        {
            'branches': [
                {
                    'branches': [],
                    'messages': [
                        {'content': "What's up?", 'role': 'user'},
                        {'content': 'Nothing. You?', 'role': 'assistant'},
                    ],
                }
            ],
            'messages': [{'content': 'Hello', 'role': 'user'}, {'content': 'World', 'role': 'assistant'}],
        }
    ]


def test_chat_chain_structured_query():
    chat_model = FakeChatModel.from_messages(
        [
            [
                HumanMessage(
                    content='Return 1 as string\n\nThe output should be formatted as a JSON instance that conforms to '
                    'the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": '
                    '"Foo", "description": "a list of strings", "type": "array", "items": {"type": '
                    '"string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a '
                    'well-formatted instance of the schema. The object {"properties": {"foo": ["bar", '
                    '"baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {'
                    '"result": {"title": "Result", "type": "string"}}, "required": ["result"]}\n```',
                    additional_kwargs={},
                ),
                AIMessage(content='{"result": "1"}', additional_kwargs={}),
            ]
        ]
    )

    @track_execution
    def f():
        chain = ChatChain(chat_model, [])
        result, new_chain = chain.structured_query("Return 1 as string", str)
        assert result == "1"
        assert len(new_chain.get_full_message_chain()) == 2

    f()


def test_chat_chain_structured_query_retry():
    chat_model = FakeChatModel.from_messages(
        [
            [
                HumanMessage(
                    content='Return 1 as string\n\nThe output should be formatted as a JSON instance that conforms to '
                    'the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": '
                    '"Foo", "description": "a list of strings", "type": "array", "items": {"type": '
                    '"string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a '
                    'well-formatted instance of the schema. The object {"properties": {"foo": ["bar", '
                    '"baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {'
                    '"result": {"title": "Result", "type": "string"}}, "required": ["result"]}\n```',
                    additional_kwargs={},
                ),
                AIMessage(content='The result is: "1".', additional_kwargs={}),
                HumanMessage(
                    content='Tried to parse your last output but failed:\n\nFailed to parse StructuredOutput from '
                    'completion The result is: "1".. Got: Expecting value: line 1 column 1 (char 0)\n\nPlease '
                    'try again and avoid this issue.',
                    additional_kwargs={},
                ),
                AIMessage(content='My apologies. The result should be: {"result": "1"}', additional_kwargs={}),
            ]
        ]
    )

    @track_execution
    def f():
        chain = ChatChain(chat_model, [])
        result, new_chain = chain.structured_query("Return 1 as string", str)
        assert result == "1"
        assert len(new_chain.get_full_message_chain()) == 4

    f()


def test_chat_chain_structured_query_retry_fail():
    chat_model = FakeChatModel.from_messages(
        [
            [
                HumanMessage(
                    content='Return 1 as string\n\nThe output should be formatted as a JSON instance that conforms to '
                    'the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": '
                    '"Foo", "description": "a list of strings", "type": "array", "items": {"type": '
                    '"string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a '
                    'well-formatted instance of the schema. The object {"properties": {"foo": ["bar", '
                    '"baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {'
                    '"result": {"title": "Result", "type": "string"}}, "required": ["result"]}\n```',
                    additional_kwargs={},
                ),
                AIMessage(content='The result is: "1".', additional_kwargs={}),
                HumanMessage(
                    content='Tried to parse your last output but failed:\n\nFailed to parse StructuredOutput from '
                    'completion The result is: "1".. Got: Expecting value: line 1 column 1 (char 0)\n\nPlease '
                    'try again and avoid this issue.',
                    additional_kwargs={},
                ),
                AIMessage(content='My apologies. The result should be: {"result": "1"}', additional_kwargs={}),
            ]
        ]
    )

    @track_execution
    def f():
        chain = ChatChain(chat_model, [])
        result, new_chain = chain.structured_query("Return 1 as string", str)
        assert result == "1"
        assert len(new_chain.get_full_message_chain()) == 4

    ChatChain.structured_query.hyperparameters['num_retries_on_parser_failure'] = 0
    with pytest.raises(OutputParserException, match=re.escape("Failed to parse output")):
        f()
