import dataclasses
import functools
import typing
from dataclasses import dataclass

from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatMessage, HumanMessage

from blackboard_pagi.prompts import chat_chain


@dataclass
class _HyperparameterWDescription:
    description: str

    def __matmul__(self, default):
        return _Hyperparameter.set_default(self.description, default)

    def set_default(self, default):
        return _Hyperparameter.set_default(self.description, default)


class _Hyperparameter:
    hyperparameters: dict | None = None
    tracked_chat_chains: dict | None = None
    all_chat_chains: dict | None = None

    unique_id: int = 0

    @staticmethod
    def set_default(key, default):
        return _Hyperparameter.hyperparameters.setdefault(key, default)

    def __matmul__(self, default):
        result = _Hyperparameter.set_default(self.unique_id, default)
        _Hyperparameter.unique_id += 1
        return result

    def __call__(self, description):
        return _HyperparameterWDescription(description)


prompt_hyperparameter = _Hyperparameter()


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
            _Hyperparameter.all_chat_chains[id(self)] = self

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
                _Hyperparameter.tracked_chat_chains[id(node)] = node
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
    hyperparameters: dict[object, object]
    tracked_chat_chains: list[dict]
    all_chat_chains: list[dict]

    def __call__(self, *args, **kwargs):
        pass


def enable_prompt_optimizer(f) -> ProtocolCallableWPromptHyperparams:
    @functools.wraps(f)
    def decorator(*args, **kwargs):
        old_hyperparameters = _Hyperparameter.hyperparameters
        if old_hyperparameters is None:
            _Hyperparameter.hyperparameters = decorator.hyperparameters
            _Hyperparameter.tracked_chat_chains = {}
            _Hyperparameter.all_chat_chains = {}
        else:
            _Hyperparameter.hyperparameters = old_hyperparameters.get(decorator.__qualname__, decorator.hyperparameters)

        old_unique_id = _Hyperparameter.unique_id
        _Hyperparameter.unique_id = 0

        result = f(*args, **kwargs)
        if old_hyperparameters is not None:
            old_hyperparameters[decorator.__qualname__] = _Hyperparameter.hyperparameters
        else:
            decorator.all_chat_chains = [
                chain.get_compact_subtree_dict(include_all=True) for chain in _Hyperparameter.all_chat_chains.values()
            ]
            decorator.tracked_chat_chains = [
                chain.get_compact_subtree_dict(include_all=False)
                for chain in _Hyperparameter.tracked_chat_chains.values()
                if chain.include_in_record
            ]
            _Hyperparameter.tracked_chat_chains = dict()
            _Hyperparameter.all_chat_chains = dict()

        _Hyperparameter.hyperparameters = old_hyperparameters
        _Hyperparameter.unique_id = old_unique_id

        return result

    # ignore attr-defined from mypy
    decorator.hyperparameters = {}  # type: ignore
    decorator.all_chat_chains = []  # type: ignore
    decorator.tracked_chat_chains = []  # type: ignore
    return decorator  # type: ignore


__all__ = ["prompt_hyperparameter", "enable_prompt_optimizer", "ProtocolCallableWPromptHyperparams", "ChatChain"]
