#  Blackboard-PAGI - LLM Proto-AGI using the Blackboard Pattern
#  Copyright (c) 2023. Andreas Kirsch
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
