import dataclasses
from dataclasses import dataclass
from typing import Tuple

from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage


@dataclass
class ChatChain:
    messages: list[BaseMessage]
    chat_model: ChatOpenAI

    @property
    def response(self):
        assert len(self.messages) >= 1
        return self.messages[-1].content

    def query(self, question: str) -> Tuple[str, "ChatChain"]:
        """Asks a question and returns the result in a single block."""
        # Build messages:
        messages = self.messages + [HumanMessage(content=question)]
        reply = self.chat_model(messages)
        messages.append(reply)
        return reply.content, dataclasses.replace(self, messages=messages)

    def branch(self) -> "ChatChain":
        return dataclasses.replace(self, messages=self.messages.copy())
