import typing
from typing import List, Optional

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.schema import BaseMessage, ChatMessage, ChatResult, LLMResult
from pydantic import BaseModel, Field

T = typing.TypeVar("T")
P = typing.ParamSpec("P")


class ChatTree(BaseModel):
    message: BaseMessage | None
    children: list['ChatTree']

    @staticmethod
    def create_root():
        return ChatTree(message=None, children=[])

    def insert(self, messages: list[BaseMessage]) -> 'ChatTree':
        node = self
        for message in messages:
            for child in node.children:
                if child.message == message:
                    node = child
                    break
            else:
                new_node = ChatTree(message=message, children=[])
                node.children.append(new_node)
                node = new_node
        return node

    def build_compact_dict(self) -> dict:
        """
        Returns a compact JSON representation of the chat tree.
        If we only have one child, we concatenate the messages until we hit a child with more than one child.

        We include the role of the message in the JSON.
        """
        node = self
        messages_json = []
        while True:
            message = node.message
            if message is not None:
                if message.type == "human":
                    role = "user"
                elif message.type == "ai":
                    role = "assistant"
                elif message.type == "system":
                    role = "system"
                elif message.type == "chat":
                    assert isinstance(message, ChatMessage)
                    role = message.role
                else:
                    raise ValueError(f"Unknown message type {message.type}")
                messages_json.append({"role": role, "content": message.content})
            if len(node.children) == 1:
                node = node.children[0]
            else:
                break

        return {
            "messages": messages_json,
            "children": [child.build_compact_dict() for child in node.children],
        }


class PromptTree(BaseModel):
    fragment: str = ""
    children: list['PromptTree']

    @staticmethod
    def create_root():
        return PromptTree(message=None, children=[])

    def insert(self, text: str) -> 'PromptTree':
        node = self
        while len(text):
            for child in node.children:
                if text.startswith(child.fragment):
                    node = child
                    text = text.removeprefix(child.fragment)
                    break
            else:
                new_node = PromptTree(fragment=text, children=[])
                node.children.append(new_node)
                return new_node
        return node

    def build_compact_dict(self) -> dict:
        """
        Returns a compact JSON representation of the chat chain.
        If we only have one child, we concatenate the messages until we hit a child with more than one child.

        We include the role of the message in the JSON.
        """
        node = self
        fragments_json = []
        while True:
            if len(node.fragment):
                fragments_json.append(node.fragment)
            if len(node.children) == 1:
                node = node.children[0]
            else:
                break

        return {
            "fragments": fragments_json,
            "children": [child.build_compact_dict() for child in node.children],
        }


class TrackedLLM(BaseLLM):
    llm: BaseLLM
    tracked_prompts: PromptTree = Field(default_factory=PromptTree.create_root)

    def __call__(self, prompt: str, stop: Optional[List[str]] = None, track: bool = False) -> str:
        node = self.tracked_prompts.insert(prompt)
        response = self.llm(prompt, stop)
        node.insert(response)
        return response

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        raise NotImplementedError()

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        raise NotImplementedError()

    @property
    def _llm_type(self) -> str:
        return self._llm_type


class TrackedChatModel(BaseChatModel):
    chat_model: BaseChatModel
    tracked_chats: ChatTree = Field(default_factory=ChatTree.create_root)

    def __call__(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> BaseMessage:
        response_message = self.chat_model(messages, stop)
        self.tracked_chats.insert(messages + [response_message])
        return response_message

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        chat_result = self.chat_model._generate(messages, stop)
        node = self.tracked_chats.insert(messages)
        for generation in chat_result.generations:
            node.insert([generation.message])
        return chat_result

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        chat_result = await self.chat_model._agenerate(messages, stop)
        node = self.tracked_chats.insert(messages)
        for generation in chat_result.generations:
            node.insert([generation.message])
        return chat_result
