import dataclasses
import functools
import types
import typing
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseMessage, ChatMessage, HumanMessage, LLMResult, OutputParserException
from pydantic import create_model

from blackboard_pagi.prompts import chat_chain

T = typing.TypeVar("T")
P = typing.ParamSpec("P")


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

    @staticmethod
    def track_chain(chain: 'ChatChain'):
        chain.track()

    @staticmethod
    def track_llm(prompt, response):
        current_tracked_state.tracked_prompts[prompt] = response
        current_tracked_state.all_prompts[prompt] = response

    @staticmethod
    def merge(root, updated_hyperparameters: dict):
        """
        Merge the hyperparameters into the root's all_hyperparameters.

        root.all_hyperparameters points to all relevant hyperparameter dicts.
        """
        all_hyperparameters = root.all_hyperparameters
        for qualname, hyperparameters in updated_hyperparameters.items():
            if qualname not in all_hyperparameters:
                raise ValueError(f"{qualname} not in {root.__qualname__}.all_hyperparameters!")
            all_hyperparameters[qualname].update(hyperparameters)


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
class TrackedFunction(typing.Callable[P, T], typing.Generic[P, T]):  # type: ignore
    """
    A callable that can be called with a chat model.
    """

    __wrapped__: typing.Callable[P, T]
    hyperparameters: dict[str | int, object]
    all_hyperparameters: dict[str, dict[str | int, object]]
    tracked_chat_chains: list[dict]
    all_chat_chains: list[dict]
    tracked_prompts: dict[str, str]
    all_prompts: dict[str, str]

    @staticmethod
    def from_function(f: typing.Callable[P, T]):
        tracked_function: TrackedFunction = functools.wraps(f)(
            TrackedFunction(
                f,
                hyperparameters={},
                all_hyperparameters=defaultdict(dict),
                tracked_chat_chains=[],
                all_chat_chains=[],
                tracked_prompts={},
                all_prompts={},
            )
        )

        return tracked_function

    def __get__(self, instance: object, owner: type | None = None) -> typing.Callable:
        """Support instance methods."""
        if instance is None:
            return self

        # Bind self to instance as MethodType
        return types.MethodType(self, instance)

    def __getattr__(self, item):
        return getattr(self.__wrapped__, item)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        global current_tracked_state
        old_tracker_state = current_tracked_state

        try:
            self.all_hyperparameters[self.__qualname__] = self.hyperparameters
            current_tracked_state = TrackedState(
                hyperparameters=self.hyperparameters,
                all_hyperparameters=self.all_hyperparameters,
            )
            result = self.__wrapped__(*args, **kwargs)
            if self.all_hyperparameters[self.__qualname__] == {}:
                del self.all_hyperparameters[self.__qualname__]
            if old_tracker_state is not None:
                old_tracker_state.all_hyperparameters.update(current_tracked_state.all_hyperparameters)
            else:
                self.all_chat_chains = [
                    chain.get_compact_subtree_dict(include_all=True)
                    for chain in current_tracked_state.all_chat_chains.values()
                ]
                self.tracked_chat_chains = [
                    chain.get_compact_subtree_dict(include_all=False)
                    for chain in current_tracked_state.tracked_chat_chains.values()
                    if chain.include_in_record
                ]
                self.tracked_prompts = current_tracked_state.tracked_prompts
                self.all_prompts = current_tracked_state.all_prompts

            self.all_hyperparameters = current_tracked_state.all_hyperparameters
        finally:
            current_tracked_state = old_tracker_state

        return result


def track_execution(f: typing.Callable[P, T]) -> TrackedFunction[P, T]:
    return TrackedFunction.from_function(f)


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

    def query(self, question: str, track: bool = True) -> typing.Tuple[str, "ChatChain"]:
        """Asks a question and returns the result in a single block."""
        # Build messages:
        question_message = HumanMessage(content=question)
        messages = self.get_full_message_chain() + [question_message]
        reply = self.chat_model(messages)
        new_chain = dataclasses.replace(self, messages=[question_message, reply], parent=self, children=[])
        self.children.append(new_chain)
        if track:
            new_chain.track()
        return reply.content, new_chain

    @track_execution
    def structured_query(self, question: str, return_type: type[T], track: bool = True) -> Tuple[T, "ChatChain"]:
        """Asks a question and returns the result in a single block."""
        # TOOD: deduplicate
        if typing.get_origin(return_type) is typing.Annotated:
            return_info = typing.get_args(return_type)
        else:
            return_info = (return_type, ...)

        output_model = create_model("StructuredOutput", result=return_info)  # type: ignore
        parser = PydanticOutputParser(pydantic_object=output_model)
        question_and_formatting = question + "\n\n" + parser.get_format_instructions()

        num_retries = prompt_hyperparameter("num_retries_on_parser_failure") @ 3
        prompt = question_and_formatting
        chain = self
        for _ in range(num_retries):
            try:
                reply_content, chain = chain.query(prompt, track=track)
                parsed_reply = parser.parse(reply_content)
                break
            except OutputParserException as e:
                prompt = (
                    prompt_hyperparameter("error_prompt") @ "Tried to parse your last output but failed:\n\n"
                    + str(e)
                    + prompt_hyperparameter("retry_prompt") @ "\n\nPlease try again and avoid this issue."
                )
        else:
            raise OutputParserException(f"Failed to parse output after {num_retries} retries.")

        result = parsed_reply.result  # type: ignore
        return result, chain

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


__all__ = ["prompt_hyperparameter", "track_execution", "TrackedFunction", "ChatChain"]
