from langchain.schema import AIMessage, HumanMessage

from blackboard_pagi.prompt_optimizer.adapters import ChatModelAsLLM, LLMAsChatModel
from blackboard_pagi.testing.fake_chat_model import FakeChatModel
from blackboard_pagi.testing.fake_llm import FakeLLM


def test_chat_model_as_llm():
    fake_chat_model = FakeChatModel.from_messages(
        [[HumanMessage(content='Hello', additional_kwargs={}), AIMessage(content='World', additional_kwargs={})]]
    )

    chat_model_as_llm = ChatModelAsLLM(chat_model=fake_chat_model)

    assert chat_model_as_llm("Hello") == "World"


def test_llm_as_chat_model():
    fake_llm = FakeLLM(texts={('<|im_start|>user\nHello<|im_end|><|im_start|>assistant\n' 'World<|im_end|>')})

    chat_model_as_llm = LLMAsChatModel(llm=fake_llm)

    assert chat_model_as_llm([HumanMessage(content="Hello")]) == AIMessage(content="World")
