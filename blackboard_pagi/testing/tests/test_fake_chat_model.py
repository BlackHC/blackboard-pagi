import pytest
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from blackboard_pagi.testing import fake_chat_model


def test_fake_chat_model_query():
    """Test that the fake LLM returns the correct query."""
    chat_model = fake_chat_model.FakeChatModel.from_messages(
        [[SystemMessage(content="foo"), HumanMessage(content="bar"), AIMessage(content="doo")]]
    )
    assert chat_model([SystemMessage(content="foo"), HumanMessage(content="bar")]) == AIMessage(content="doo")


def test_fake_chat_model_query_with_stop_raise():
    """Test that the fake LLM returns the correct query."""
    chat_model = fake_chat_model.FakeChatModel.from_messages(
        [[SystemMessage(content="foo"), HumanMessage(content="bar"), AIMessage(content="doo")]]
    )

    with pytest.raises(AssertionError):
        chat_model([SystemMessage(content="foo"), HumanMessage(content="bar")], stop=["a"])


def test_chat_model_llm_missing_query():
    """Test that the fake LLM raises an error if the query is missing."""
    chat_model = fake_chat_model.FakeChatModel(messages_tuples_bag=set())
    with pytest.raises(NotImplementedError):
        chat_model([SystemMessage(content="foo"), HumanMessage(content="bar")])
