import dataclasses
import functools
import typing
from dataclasses import dataclass
from typing import List, Optional

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.schema import BaseMessage, ChatMessage, HumanMessage, LLMResult

from blackboard_pagi.prompts import chat_chain


@dataclass
class TrackedState:
    hyperparameters: dict
    all_hyperparameters: dict
    unique_id: int = 0

    tracked_chat_chains: dict = dataclasses.field(default_factory=dict)
    all_chat_chains: dict = dataclasses.field(default_factory=dict)

    tracked_prompts: dict = dataclasses.field(default_factory=dict)
    all_prompts: dict = dataclasses.field(default_factory=dict)

    def set_default(self, *, default, key=None):
        if key is None:
            key = self.unique_id
            self.unique_id += 1
        result = self.hyperparameters.setdefault(key, default)
        return result


current_tracked_state: TrackedState | None = None


@dataclass
class _HyperparameterWDescription:
    description: str

    def __matmul__(self, default):
        return current_tracked_state.set_default(key=self.description, default=default)

    def set_default(self, default):
        return current_tracked_state.set_default(key=self.description, default=default)


class _Hyperparameter:
    @staticmethod
    def set_default(default):
        return current_tracked_state.set_default(default=default)

    def __matmul__(self, default):
        return current_tracked_state.set_default(default=default)

    def __call__(self, description):
        return _HyperparameterWDescription(description)

    def track_chain(self, chain: 'ChatChain'):
        chain.track()

    def train_llm(self, prompt, response):
        current_tracked_state.tracked_prompts[prompt] = response


prompt_hyperparameter = _Hyperparameter()


class WrappedLLM(BaseLLM):
    inner_llm: BaseLLM

    def __call__(self, prompt: str, stop: Optional[List[str]] = None, track: bool = False) -> str:
        assert current_tracked_state is not None
        result = self.inner_llm.__call__(prompt, stop)
        current_tracked_state.all_prompts[prompt] = result
        if track:
            current_tracked_state.tracked_prompts[prompt] = result
        return result

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        raise NotImplementedError()

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        raise NotImplementedError()

    @property
    def _llm_type(self) -> str:
        return self.inner_llm._llm_type


class WrappedChatModelAsLLM(BaseLLM):
    inner_chat_model: BaseChatModel

    def __call__(self, prompt: str, stop: Optional[List[str]] = None, track: bool = False) -> str:
        assert current_tracked_state is not None
        result = self.inner_chat_model.call_as_llm(prompt, stop)
        current_tracked_state.all_prompts[prompt] = result
        if track:
            current_tracked_state.tracked_prompts[prompt] = result
        return result

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        raise NotImplementedError()

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        raise NotImplementedError()

    @property
    def _llm_type(self) -> str:
        return self.inner_chat_model.__repr_name__()


@dataclass
class ChatChain(chat_chain.ChatChain):
    """
    A chat chain is a chain of messages that can be used to query a chat model.

    The `messages` attribute is a list of messages that are sent to the chat model.

    Compared to the regular ChatChain, this class also keeps track of the parent and children of the chat chain.
    """

    chat_model: BaseChatModel
    messages: list[BaseMessage]
    parent: 'ChatChain | None' = None
    children: list['ChatChain'] = dataclasses.field(default_factory=list)
    include_in_record: bool = False
    """This is private. Use the `track` method instead."""

    def __post_init__(self):
        if self.parent is None:
            current_tracked_state.all_chat_chains[id(self)] = self

    @property
    def response(self):
        assert len(self.messages) >= 1
        return self.messages[-1].content

    def append(self, messages: list[BaseMessage]) -> "ChatChain":
        new_chain = dataclasses.replace(self, messages=messages, parent=self, children=[])
        self.children.append(new_chain)
        return new_chain

    # overload operator +
    def __add__(self, other: list[BaseMessage]) -> "ChatChain":
        return self.append(other)

    def get_full_message_chain(self):
        if self.parent is None:
            return self.messages
        else:
            return self.parent.get_full_message_chain() + self.messages

    def query(self, question: str) -> typing.Tuple[str, "ChatChain"]:
        """Asks a question and returns the result in a single block."""
        # Build messages:
        question_message = HumanMessage(content=question)
        messages = self.get_full_message_chain() + [question_message]
        reply = self.chat_model(messages)
        new_chain = dataclasses.replace(self, messages=[question_message, reply], parent=self, children=[])
        self.children.append(new_chain)
        return reply.content, new_chain

    def branch(self) -> "ChatChain":
        new_chain = dataclasses.replace(self, messages=[], parent=self, children=[])
        self.children.append(new_chain)
        return new_chain

    def track(self):
        """
        Recursively mark all parents as being included in the record.
        """
        node = self
        while True:
            node.include_in_record = True
            if node.parent is None:
                current_tracked_state.tracked_chat_chains[id(node)] = node
                break
            node = node.parent

    def get_compact_subtree_dict(self, include_all: bool = False):
        """
        Returns a compact JSON representation of the chat chain.
        If we only have one child, we concatenate the messages until we hit a child with more than one child.

        We include the role of the message in the JSON.
        """
        node = self
        messages_json = []
        while node.include_in_record or include_all:
            for message in node.messages:
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
                return {
                    "messages": messages_json,
                    "branches": [
                        child.get_compact_subtree_dict(include_all=include_all)
                        for child in node.children
                        if child.include_in_record or include_all
                    ],
                }
        return {}


class ProtocolCallableWPromptHyperparams(typing.Protocol):
    hyperparameters: dict[str, dict[str | int, object]]
    tracked_chat_chains: list[dict]
    all_chat_chains: list[dict]
    tracked_prompts: dict[str, str]
    all_prompts: dict[str, str]

    def __call__(self, *args, **kwargs):
        pass


def track_execution(f) -> ProtocolCallableWPromptHyperparams:
    @functools.wraps(f)
    def decorator(*args, **kwargs):
        global current_tracked_state
        old_tracker_state = current_tracked_state

        try:
            current_tracked_state = TrackedState(
                hyperparameters=decorator.hyperparameters.setdefault(decorator.__qualname__, {}),
                all_hyperparameters=decorator.hyperparameters,
            )
            result = f(*args, **kwargs)
            if decorator.hyperparameters[decorator.__qualname__] == {}:
                del decorator.hyperparameters[decorator.__qualname__]
            if old_tracker_state is not None:
                old_tracker_state.all_hyperparameters.update(current_tracked_state.all_hyperparameters)
            else:
                decorator.all_chat_chains = [
                    chain.get_compact_subtree_dict(include_all=True)
                    for chain in current_tracked_state.all_chat_chains.values()
                ]
                decorator.tracked_chat_chains = [
                    chain.get_compact_subtree_dict(include_all=False)
                    for chain in current_tracked_state.tracked_chat_chains.values()
                    if chain.include_in_record
                ]
                decorator.tracked_prompts = current_tracked_state.tracked_prompts
                decorator.all_prompts = current_tracked_state.all_prompts

            decorator.hyperparameters = current_tracked_state.all_hyperparameters
        finally:
            current_tracked_state = old_tracker_state

        return result

    # ignore attr-defined from mypy
    decorator.hyperparameters = {}  # type: ignore
    decorator.all_chat_chains = []  # type: ignore
    decorator.tracked_chat_chains = []  # type: ignore
    decorator.tracked_prompts = {}  # type: ignore
    decorator.all_prompts = {}  # type: ignore

    return decorator  # type: ignore


__all__ = ["prompt_hyperparameter", "track_execution", "ProtocolCallableWPromptHyperparams", "ChatChain"]
