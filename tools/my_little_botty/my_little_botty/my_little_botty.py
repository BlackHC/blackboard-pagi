"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import pprint  # noqa: F401
import re
import time
import types
import typing
from datetime import datetime
from enum import Enum
from typing import Dict

import pydantic
import pynecone as pc
import pynecone.pc as cli
from pydantic.fields import ModelField
from pynecone import Var
from pynecone.components.media.icon import ChakraIconComponent
from pynecone.utils.imports import ImportDict, merge_imports
from pynecone.var import BaseVar, ComputedVar


class TypedVar(Var):
    def __try_type_get(self, name, outer_type_: type):
        # check if type_ is a Union
        if isinstance(outer_type_, types.UnionType):
            for sub_type_ in outer_type_.__args__:
                var = self.__try_type_get(name, sub_type_)
                if var is not None:
                    return var
        if hasattr(outer_type_, "__fields__") and name in outer_type_.__fields__:
            type_ = outer_type_.__fields__[name].outer_type_
            if isinstance(type_, ModelField):
                type_ = type_.type_
            return BaseVar(
                name=f"{self.name}.{name}",
                type_=type_,
                state=self.state,
            )
        return None

    def __getattribute__(self, name: str) -> pc.Var:
        """Get a var attribute.

        Args:
            name: The name of the attribute.

        Returns:
            The var attribute.

        Raises:
            Exception: If the attribute is not found.
        """
        try:
            return super().__getattribute__(name)
        except Exception as e:
            # Check if the attribute is one of the class fields.
            if not name.startswith("_"):
                var = self.__try_type_get(name, self.type_)
                if var is not None:
                    return var

            raise e


Var.register(TypedVar)


class TypedBaseVar(TypedVar, pc.Base):
    """A base (non-computed) var of the app state."""

    # The name of the var.
    name: str

    # The type of the var.
    type_: typing.Any

    # The name of the enclosing state.
    state: str = ""

    # Whether this is a local javascript variable.
    is_local: bool = False

    # Whether this var is a raw string.
    is_string: bool = False

    def __hash__(self) -> int:
        """Define a hash function for a var.

        Returns:
            The hash of the var.
        """
        return hash((self.name, str(self.type_)))

    @typing.no_type_check
    def get_default_value(self) -> typing.Any:
        """Get the default value of the var.

        Returns:
            The default value of the var.
        """
        type_ = self.type_.__origin__ if types.is_generic_alias(self.type_) else self.type_
        if issubclass(type_, str):
            return ""
        if issubclass(type_, types.get_args(typing.Union[int, float])):
            return 0
        if issubclass(type_, bool):
            return False
        if issubclass(type_, list):
            return []
        if issubclass(type_, dict):
            return {}
        if issubclass(type_, tuple):
            return ()
        return set() if issubclass(type_, set) else None

    def get_setter_name(self, include_state: bool = True) -> str:
        """Get the name of the var's generated setter function.

        Args:
            include_state: Whether to include the state name in the setter name.

        Returns:
            The name of the setter function.
        """
        setter = pc.constants.SETTER_PREFIX + self.name
        if not include_state or self.state == "":
            return setter
        return ".".join((self.state, setter))

    def get_setter(self) -> typing.Callable[[pc.State, typing.Any], None]:
        """Get the var's setter function.

        Returns:
            A function that that creates a setter for the var.
        """

        def setter(state: State, value: typing.Any):
            """Get the setter for the var.

            Args:
                state: The state within which we add the setter function.
                value: The value to set.
            """
            setattr(state, self.name, value)

        setter.__qualname__ = self.get_setter_name()

        return setter


BaseVar.register(TypedBaseVar)


# Redefine ComputedVar to revert the baseclasses
class TypedComputedVar(TypedVar, property):
    """A field with computed getters."""

    @property
    def name(self) -> str:
        """Get the name of the var.

        Returns:
            The name of the var.
        """
        assert self.fget is not None, "Var must have a getter."
        return self.fget.__name__

    @property
    def type_(self):
        """Get the type of the var.

        Returns:
            The type of the var.
        """
        if "return" in self.fget.__annotations__:
            return self.fget.__annotations__["return"]
        return typing.Any


pc.var = TypedComputedVar
ComputedVar.register(pc.var)


class ReactIcon(ChakraIconComponent):
    """An image icon."""

    tag: str = "Icon"
    as_: Var[pc.EventChain]

    @classmethod
    def create(cls, *children, as_: str, **props):
        """Initialize the Icon component.

        Run some additional checks on Icon component.

        Args:
            children: The positional arguments
            props: The keyword arguments

        Raises:
            AttributeError: The errors tied to bad usage of the Icon component.
            ValueError: If the icon tag is invalid.

        Returns:
            The created component.
        """
        as_var = TypedBaseVar(name=as_, type_=pc.EventChain)
        return super().create(*children, as_=as_var, **props)

    def _get_imports(self) -> ImportDict:
        icon_name = self.as_.name
        # the icon name uses camel case, e.g. BsThreeDotsVertical. The first part (Bs) is the icon sub package
        # convert camel case to snake case
        camel_case = re.sub(r"(?<!^)(?=[A-Z])", "_", icon_name).lower()
        icon_sub_package = camel_case.split("_")[0]
        return merge_imports(super()._get_imports(), {f"react-icons/{icon_sub_package}": {self.as_.name}})


react_icon = ReactIcon.create


# solarized colors as HTML hex
# https://ethanschoonover.com/solarized/
class SolarizedColors(str, Enum):
    base03 = "#002b36"
    base02 = "#073642"
    base01 = "#586e75"
    base00 = "#657b83"
    base0 = "#839496"
    base1 = "#93a1a1"
    base2 = "#eee8d5"
    base3 = "#fdf6e3"
    yellow = "#b58900"
    orange = "#cb4b16"
    red = "#dc322f"
    magenta = "#d33682"
    violet = "#6c71c4"
    blue = "#268bd2"
    cyan = "#2aa198"
    green = "#859900"


class ColorStyle:
    HUMAN = (SolarizedColors.base3, SolarizedColors.base03)
    SYSTEM = (SolarizedColors.base2, SolarizedColors.base03)
    BOTS = [
        SolarizedColors.yellow,
        SolarizedColors.orange,
        SolarizedColors.magenta,
        SolarizedColors.violet,
        SolarizedColors.blue,
        SolarizedColors.cyan,
        SolarizedColors.green,
    ]


class MessageRole(str, Enum):
    SYSTEM = "System"
    HUMAN = "Human"
    BOT = "Bot"


class MessageSource(pc.Base):
    model_type: str | None = None
    model_dict: dict = {}
    edited: bool = False


class Message(pc.Base):
    uid: str

    role: MessageRole
    source: MessageSource
    creation_time: int
    content: str | None
    error: str | None

    def compute_editable_style(self) -> 'StyledMessage':
        return StyledMessage.create(self, [], [])


class LinkedMessage(pc.Base):
    thread_uid: str
    uid: str


class StyledMessage(Message):
    background_color: str
    foreground_color: str
    align: str
    fmt_creation_datetime: str
    fmt_header: str
    prev_linked_messages: list[LinkedMessage]
    next_linked_messages: list[LinkedMessage]

    @classmethod
    def create(
        cls, message: Message, prev_linked_messages: list[LinkedMessage], next_lined_messages: list[LinkedMessage]
    ) -> 'StyledMessage':
        fmt_creation_datetime = cls.format_datetime(message.creation_time)

        if message.role == MessageRole.SYSTEM:
            background_color, foreground_color = ColorStyle.SYSTEM
            align = "left"
        elif message.role == MessageRole.HUMAN:
            background_color, foreground_color = ColorStyle.HUMAN
            align = "right"
        elif message.role == MessageRole.BOT:
            align = "left"
            if message.source.model_type is not None:
                background_color = ColorStyle.BOTS[sum(map(ord, message.source.model_type)) % len(ColorStyle.BOTS)]
                foreground_color = ColorStyle.HUMAN[1]
            else:
                background_color, foreground_color = ColorStyle.HUMAN

        fmt_header = cls.format_message_header(message.role, message.source)

        return cls(
            align=align,
            background_color=background_color,
            foreground_color=foreground_color,
            fmt_creation_datetime=fmt_creation_datetime,
            fmt_header=fmt_header,
            prev_linked_messages=prev_linked_messages,
            next_linked_messages=next_lined_messages,
            **dict(Message(**dict(message))),
        )

    @staticmethod
    def format_datetime(creation_time: int):
        # format datetime as "DD.MM.YYYY HH:MM"
        # except if it's yesterday, then just "Yesterday HH:MM"
        # except if it's today, then just "HH:MM"
        # except if it's less than 1 hour ago, then just "MM minutes ago"
        # except if it's less than 1 minute ago, then just "now"
        delta = datetime.now() - datetime.fromtimestamp(creation_time)
        if delta.days > 1:
            return datetime.fromtimestamp(creation_time).strftime("%d.%m.%Y %H:%M")
        elif delta.days == 1:
            return datetime.fromtimestamp(creation_time).strftime("Yesterday %H:%M")
        elif delta.seconds > 3600:
            return datetime.fromtimestamp(creation_time).strftime("%H:%M")
        elif delta.seconds > 60:
            return f"{int(delta.seconds / 60)} minutes ago"
        else:
            return "now"

    @staticmethod
    def format_message_header(role: MessageRole, source: MessageSource):
        if role == MessageRole.SYSTEM:
            fmt_header = 'System'
            if source.model_type is not None:
                fmt_header += f' (by {source.model_type})'
        elif role == MessageRole.HUMAN:
            fmt_header = 'Human'
            if source.model_type is not None:
                fmt_header += f' (by {source.model_type})'
        elif role == MessageRole.BOT:
            if source.model_type is not None:
                fmt_header = source.model_type
            else:
                fmt_header = 'Bot (by Human)'
        else:
            raise ValueError(f'Unknown role: {role}')

        if source.model_type is not None and source.edited:
            fmt_header += " [edited]"

        return fmt_header


class MessageThread(pc.Base):
    uid: str
    messages: list[Message] = pydantic.Field(default_factory=list)

    def get_message_by_uid(self, uid: str) -> Message | None:
        filtered_message = [message for message in self.messages if message.uid == uid]
        if not filtered_message:
            return None
        assert len(filtered_message) == 1
        return filtered_message[0]


class StyledMessageThread(MessageThread):
    messages: list[StyledMessage]


class MessageGrid(pc.Base):
    thread_uids: list[str]
    message_grid: list[list[Message | None]]


class StyledGridCell(pc.Base):
    message: StyledMessage | None = None
    row_idx: int
    col_idx: int
    thread_uid: str
    col_span: int = 1
    skip: bool = False


class ModelOptions(pc.Base):
    model_type: str
    max_tokens: int
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float


class MessageExploration(pc.Base):
    uid: str
    title: str = "Untitled"
    note: str = ""
    tags: list[str] = pydantic.Field(default_factory=list)

    current_message_thread_uid: str
    message_threads: list[MessageThread]
    hidden_message_thread_uids: set[str] = pydantic.Field(default_factory=set)

    _message_uid_map: dict[str, Message] = pydantic.PrivateAttr(default_factory=dict)

    @property
    def message_uid_map(self) -> dict[str, Message]:
        return self._message_uid_map

    def __init__(self, **data):
        super().__init__(**data)
        self._message_uid_map = self._compute_message_uid_map()

    @property
    def current_message_thread(self) -> MessageThread:
        return self.get_message_thread_by_uid(self.current_message_thread_uid)

    def get_message_thread_by_uid(self, uid: str):
        matches = [mt for mt in self.message_threads if mt.uid == uid]
        assert len(matches) == 1
        return matches[0]

    def update_message_thread(self, message_thread: MessageThread):
        # insert the message thread after the current one
        idx = self.message_threads.index(self.current_message_thread)
        self.message_threads.insert(idx + 1, message_thread)
        self.current_message_thread_uid = message_thread.uid

        # update the message uid map
        self._message_uid_map = self._compute_message_uid_map()

    def delete_message_thread(self, message_thread_uid: str):
        """
        Remove the message thread with the given uid from the list of message threads.
        Update the current message thread to the previous one if possible.
        Remove the message thread from the list of hidden message threads if it was hidden.
        """
        idx = next(i for i, mt in enumerate(self.message_threads) if mt.uid == message_thread_uid)
        self.message_threads.pop(idx)

        if idx > 0:
            self.current_message_thread_uid = self.message_threads[idx - 1].uid
        elif len(self.message_threads) > 0:
            self.current_message_thread_uid = self.message_threads[0].uid
        else:
            self.message_threads = [MessageThread(uid=UID.create())]
            self.current_message_thread_uid = self.message_threads[0].uid

        self.hidden_message_thread_uids.discard(message_thread_uid)

        # update the message uid map
        self._message_uid_map = self._compute_message_uid_map()

    def set_message_thread_visible(self, message_thread_uid: str, visible: bool):
        if visible:
            self.hidden_message_thread_uids.discard(message_thread_uid)
        else:
            self.hidden_message_thread_uids.add(message_thread_uid)

    def _deduplicate_message_threads(self, message_threads, row):
        cleaned_message_threads = []
        message_uids = set()
        for message_thread in message_threads:
            if len(message_thread.messages) <= row:
                continue
            message_uid = message_thread.messages[row].uid
            if message_uid in message_uids:
                continue
            message_uids.add(message_uid)
            cleaned_message_threads.append(message_thread)
        return cleaned_message_threads

    def get_message_grid(self) -> MessageGrid:
        # for each message thread create a tuple of all message uids in the thread
        message_threads_map = {
            tuple(message.uid for message in message_thread.messages): message_thread
            for message_thread in self.message_threads
        }
        # sort lexigraphically by message uid tuples
        sorted_message_threads_items = sorted(list(message_threads_map.items()), key=lambda kv: kv[0])

        # create grid
        row_count = max(len(message_thread.messages) for message_thread in self.message_threads)
        message_grid = [[None for _ in message_threads_map] for _ in range(row_count)]
        message_thread_uids = []
        for col, (_, message_thread) in enumerate(sorted_message_threads_items):
            message_thread_uids.append(message_thread.uid)
            for row, message in enumerate(message_thread.messages):
                message_grid[row][col] = message  # type: ignore

        return MessageGrid(thread_uids=message_thread_uids, message_grid=message_grid)

    def get_styled_current_message_thread(self) -> StyledMessageThread:
        """
        For the current message thread, we compute a StyledMessageThread.
        """
        current_message_thread = self.current_message_thread

        message_thread_beam = self.message_threads

        styled_messages = []
        for row, current_message in enumerate(current_message_thread.messages):
            # find all message threads in the beam that have a message with the same uid in the same row (if available)
            new_message_thread_beam = []
            for message_thread in message_thread_beam:
                if len(message_thread.messages) > row and message_thread.messages[row].uid == current_message.uid:
                    new_message_thread_beam.append(message_thread)
            # split message thread beam into three parts:
            # 1. message threads thare are also in the new message thread beam, and
            #    message threads that are not in the new message thread beam:
            #    2. message threads that come before the current message thread
            #    3. message threads that come after the current message thread

            message_threads = [mt for mt in message_thread_beam]
            message_threads = self._deduplicate_message_threads(message_threads, row)

            # obtain the index of the current message thread in the new message thread beam
            current_message_idx = next(
                i for i, mt in enumerate(message_threads) if mt.messages[row].uid == current_message.uid
            )

            linked_messages_before: list[LinkedMessage] = [
                LinkedMessage(thread_uid=message_thread.uid, uid=message_thread.messages[row].uid)
                for message_thread in message_threads[:current_message_idx]
            ]
            linked_messages_after: list[LinkedMessage] = [
                LinkedMessage(thread_uid=message_thread.uid, uid=message_thread.messages[row].uid)
                for message_thread in message_threads[current_message_idx + 1 :]
            ]
            styled_message = StyledMessage.create(current_message, linked_messages_before, linked_messages_after)
            styled_messages.append(styled_message)

            message_thread_beam = new_message_thread_beam

        # pprint.pprint(styled_messages)

        return StyledMessageThread(
            uid=current_message_thread.uid,
            messages=styled_messages,
        )

    def _compute_message_uid_map(self):
        message_uid_map = {}
        for message_thread in self.message_threads:
            for message in message_thread.messages:
                stored_message = message_uid_map.setdefault(message.uid, message)
                assert stored_message == message
        return message_uid_map


def render_static_message_content(message: StyledMessage):
    is_not_bot = message.role != "Bot"
    return pc.cond(is_not_bot, pc.markdown(message.content), pc.cond(message.content, pc.markdown(message.content)))


def render_edit_thread_button(message: StyledMessage):
    return pc.button(
        pc.icon(tag="edit"),
        size='xs',
        variant='ghost',
        on_click=lambda: EditableMessageState.enter_editing(message.uid),  # type: ignore
    )


def render_fork_thread_button(message: StyledMessage):
    return pc.button(
        pc.icon(tag="repeat_clock"),
        size='xs',
        variant='ghost',
        on_click=lambda: EditableMessageState.enter_forking(message.uid),  # type: ignore
    )


@typing.no_type_check
def render_message_toolbar(message: StyledMessage):
    num_threads = 1 + message.next_linked_messages.length() + message.prev_linked_messages.length()
    thread_index = 1 + message.prev_linked_messages.length()
    is_not_bot = message.role != "Bot"
    return pc.fragment(
        pc.cond(
            message.uid == EditableMessageState.editable_message_uid,
            pc.hstack(
                pc.button(
                    pc.icon(tag="check"), size='xs', variant='ghost', on_click=EditableMessageState.submit_editing
                ),
                pc.divider(orientation="vertical", height="1em"),
                pc.button(
                    pc.icon(tag="close"), size='xs', variant='ghost', on_click=EditableMessageState.cancel_editing
                ),
                pc.divider(orientation="vertical", height="1em"),
            ),
            pc.fragment(
                render_message_menu(message),
                pc.cond(is_not_bot, render_fork_thread_button(message), render_edit_thread_button(message)),
            ),
        ),
        pc.fragment(
            pc.button(
                pc.icon(tag="chevron_left"),
                size='xs',
                variant='ghost',
                is_disabled=EditableMessageState.is_editing | (message.prev_linked_messages.length() == 0),
                on_click=lambda: NavigationState.go_to(
                    message.prev_linked_messages[-1].thread_uid,
                    message.prev_linked_messages[-1].uid,
                ),
            ),
            pc.button(
                react_icon(as_="ImTree"),
                size='xs',
                variant='ghost',
                is_disabled=EditableMessageState.is_editing
                | ((message.next_linked_messages.length() + message.prev_linked_messages.length()) == 0),
                on_click=MessageOverviewState.show_drawer(message.uid),
            ),
            pc.button(
                pc.icon(tag="chevron_right"),
                size='xs',
                variant='ghost',
                is_disabled=EditableMessageState.is_editing | (message.next_linked_messages.length() == 0),
                on_click=lambda: NavigationState.go_to(
                    message.next_linked_messages[0].thread_uid,
                    message.next_linked_messages[0].uid,
                ),
            ),
        ),
        pc.cond(
            num_threads > 1,
            pc.fragment(
                pc.text(thread_index),
                pc.text("/", color=SolarizedColors.base0),
                pc.text(num_threads, color=SolarizedColors.base0),
            ),
        ),
    )


def render_carousel_message(message: StyledMessage):
    return pc.grid_item(
        pc.button(
            pc.vstack(
                pc.box(
                    pc.flex(
                        pc.text(message.fmt_header),
                        pc.spacer(),
                        pc.text(message.fmt_creation_datetime),
                        width="100%",
                    ),
                    background_color=message.background_color,
                    color=message.foreground_color,
                    border_radius="15px 15px 0 0",
                    width="100%",
                    padding="0.25em 0.25em 0 0.5em",
                    margin_bottom="0",
                ),
                pc.cond(
                    message.error,
                    pc.fragment(
                        pc.cond(
                            message.content,
                            pc.box(
                                render_markdown(message.content),
                                border_radius="0 0 0 0",
                                color=message.foreground_color,
                                border_width="medium",
                                border_color=message.foreground_color,
                                width="100%",
                                padding="0 0.25em 0 0.25em",
                            ),
                        ),
                        pc.box(
                            pc.markdown(message.error),
                            border_radius="0 0 15px 15px",
                            color=message.foreground_color,
                            background_color=SolarizedColors.red,
                            width="100%",
                            padding="0 0.25em 0 0.25em",
                        ),
                    ),
                    pc.cond(
                        message.content,
                        pc.box(
                            render_markdown(message.content),
                            border_radius="0 0 15px 15px",
                            color=message.foreground_color,
                            border_width="medium",
                            border_color=message.foreground_color,
                            width="100%",
                            padding="0 0.25em 0 0.25em",
                        ),
                    ),
                ),
                width="30ch",
                max_width="30ch",
                padding_top="0.5em",
                padding_bottom="0.5em",
                spacing="0em",
            ),
            width="30ch",
            max_width="32ch",
            height="100%",
            on_click=lambda: MessageOverviewState.close_drawer(message.uid),  # type: ignore
            variant="solid",
        ),
        row_start=1,
    )


def render_markdown(content: str | None):
    return pc.box(
        pc.cond(content, pc.markdown(content)),
        max_height="50vh",
        style={"word-wrap": "break-word", "white-space": "pre-wrap", "text-align": "left", "overflow": "scroll"},
    )


def render_message_overview_carousel():
    return pc.flex(
        pc.box(
            pc.grid(
                pc.foreach(MessageOverviewState.styled_previous_messages, render_carousel_message),
                render_carousel_message(MessageOverviewState.styled_current_message),
                pc.foreach(MessageOverviewState.styled_next_messages, render_carousel_message),
                gap="1ch",
                width="fit-content",
                zoom="0.75",
            ),
            max_width="100%",
            overflow="scroll",
        ),
        justify_content="center",
        width="100vw",
        max_width="100vw",
        margin_left="50%",
        transform="translateX(-50%)",
        padding_bottom="0.5em",
    )


@typing.no_type_check
def render_grid_cell(styled_grid_cell: StyledGridCell):
    message = styled_grid_cell.message

    return pc.cond(
        ~styled_grid_cell.skip,
        pc.grid_item(
            pc.cond(
                message,
                pc.vstack(
                    pc.box(
                        pc.flex(
                            pc.text(message.fmt_header),
                            pc.spacer(),
                            pc.text(message.fmt_creation_datetime),
                            width="100%",
                        ),
                        background_color=message.background_color,
                        color=message.foreground_color,
                        border_radius="15px 15px 0 0",
                        width="100%",
                        padding="0.25em 0.25em 0 0.5em",
                        margin_bottom="0",
                    ),
                    pc.cond(
                        message.error,
                        pc.fragment(
                            pc.cond(
                                message.content,
                                pc.box(
                                    render_markdown(message.content),
                                    border_radius="0 0 0 0",
                                    color=message.foreground_color,
                                    border_width="medium",
                                    border_color=message.foreground_color,
                                    width="100%",
                                    padding="0 0.25em 0 0.25em",
                                ),
                            ),
                            pc.box(
                                pc.markdown(message.error),
                                border_radius="0 0 15px 15px",
                                color=message.foreground_color,
                                background_color=SolarizedColors.red,
                                width="100%",
                                padding="0 0.25em 0 0.25em",
                            ),
                        ),
                        pc.cond(
                            message.content,
                            pc.box(
                                render_markdown(message.content),
                                border_radius="0 0 15px 15px",
                                color=message.foreground_color,
                                border_width="medium",
                                border_color=message.foreground_color,
                                width="100%",
                                padding="0 0.25em 0 0.25em",
                            ),
                        ),
                    ),
                    width="30ch",
                    max_width="30ch",
                    padding="0.5em",
                    spacing="0em",
                ),
            ),
            row_start=styled_grid_cell.row_idx + 1,
            col_start=styled_grid_cell.col_idx,
            col_span=styled_grid_cell.col_span,
            justify_self="center",
            style={
                "border-width": "0 thin 0 thin",
                "border_color": SolarizedColors.base1,
                "width": "100%",
                "justify-content": "center",
                "display": "flex",
            },
        ),
    )


def render_grid_row(message_row: list[StyledGridCell]):
    return pc.foreach(message_row, render_grid_cell)


def render_grid_column_button(thread_uid: str, index):
    return pc.fragment(
        pc.grid_item(
            pc.button(pc.icon(tag="delete"), variant="outline", margin="1em"),
            row_start=1,
            row_span=1,
            col_start=index.to(int) + 1,
            col_span=1,
            on_click=lambda: MessageGridState.remove_thread(thread_uid),  # type: ignore
        ),
        pc.grid_item(
            pc.button(
                width="100%",
                height="100%",
                on_click=lambda: MessageGridState.go_to_thread(thread_uid),  # type: ignore
                variant="ghost",
                opacity=0.5,
                minWidth="30ch",
                min_height="30ch",
            ),
            row_start=2,
            row_span=MessageGridState.styled_grid_cells.length(),
            col_start=index.to(int) + 1,
            col_span=1,
        ),
    )


def render_message_grid():
    return pc.flex(
        pc.box(
            pc.grid(
                pc.foreach(MessageGridState.styled_grid_cells, render_grid_row),
                pc.foreach(MessageGridState.column_thread_uids, render_grid_column_button),
                width="fit-content",
                zoom="0.75",
            ),
            max_width="100%",
            overflow="scroll",
        ),
        justify_content="center",
        width="100vw",
        max_width="100vw",
        margin_left="50%",
        transform="translateX(-50%)",
        padding_bottom="0.5em",
    )


def render_static_message(message: StyledMessage):
    return pc.hstack(
        pc.vstack(
            pc.box(
                pc.flex(
                    render_message_toolbar(message),
                    pc.spacer(),
                    pc.text(message.fmt_header),
                    pc.cond(
                        message.role == "System",
                        pc.button(pc.icon(tag="info"), size="xs", variant="outline"),
                    ),
                    pc.spacer(),
                    pc.text(message.fmt_creation_datetime),
                    width="100%",
                ),
                background_color=message.background_color,
                color=message.foreground_color,
                border_radius="15px 0 0 0",
                width="100%",
                padding="0.25em 0.25em 0.25em 0.5em",
                margin_bottom="0",
            ),
            pc.cond(
                message.error,
                pc.fragment(
                    pc.cond(
                        message.content,
                        pc.box(
                            render_static_message_content(message),
                            border_radius="0 0 0 0",
                            color=message.foreground_color,
                            border_width="medium",
                            border_color=message.foreground_color,
                            width="100%",
                            padding="0 0.25em 0 0.25em",
                        ),
                    ),
                    pc.box(
                        pc.markdown(message.error),
                        border_radius="0 0 15px 0",
                        color=message.foreground_color,
                        background_color=SolarizedColors.red,
                        width="100%",
                        padding="0 0.25em 0 0.25em",
                    ),
                ),
                pc.cond(
                    message.content,
                    pc.box(
                        render_static_message_content(message),
                        border_radius="0 0 15px 0",
                        color=message.foreground_color,
                        border_width="medium",
                        border_color=message.foreground_color,
                        width="100%",
                        padding="0 0.25em 0 0.25em",
                    ),
                ),
            ),
            width="60ch",
            padding_bottom="0.5em",
            spacing="0em",
        ),
        justify_content=message.align,
        width="100%",
    )


def render_editable_message(message: StyledMessage):
    return pc.hstack(
        pc.vstack(
            pc.center(
                pc.flex(
                    render_message_toolbar(message),
                    pc.spacer(),
                    pc.box(
                        pc.select(
                            [role.value for role in MessageRole],
                            default_value=message.role,
                            # on_change=SelectState.set_option,
                            variant="unstyled",
                            on_change=EditableMessageState.set_editable_message_role,
                        ),
                        width='30%',
                    ),
                    pc.spacer(),
                    pc.text(message.fmt_creation_datetime),
                    width="100%",
                ),
                background_color=message.background_color,
                color=message.foreground_color,
                border_radius="15px 0 0 0",
                width="100%",
                padding="0.25em 0.25em 0 0.5em",
                margin_bottom="0",
            ),
            pc.cond(
                message.error,
                pc.fragment(
                    render_editable_message_content(message, with_error=False), render_editable_message_error(message)
                ),
                render_editable_message_content(message, with_error=False),
            ),
            width="75%",
            padding_bottom="0.5em",
            spacing="0em",
        ),
        justify_content=message.align,
        width="100%",
    )


def render_editable_message_error(message):
    return pc.box(
        pc.flex(
            pc.box(
                pc.markdown(message.error),
            ),
            pc.spacer(),
            pc.button(
                pc.icon(tag="delete"),
                size='xs',
                variant='ghost',
                on_click=EditableMessageState.clear_editable_message_error,
            ),
            height="100%",
        ),
        border_radius="0 0 15px 0",
        color=message.foreground_color,
        background_color=SolarizedColors.red,
        width="100%",
        padding="0 0.25em 0 0.25em",
    )


class AutoFocusTextArea(pc.TextArea):
    auto_focus: Var[typing.Union[str, int, bool]]


auto_focus_text_area = AutoFocusTextArea.create


def render_editable_message_content(message, with_error: bool):
    return auto_focus_text_area(
        auto_focus=True,
        default_value=message.content,
        border_radius="0 0 0 0" if with_error else "0 0 15px 0",
        color=message.foreground_color,
        border_width="medium",
        width="100%",
        padding="0 0.25em 0 0.25em",
        on_blur=EditableMessageState.set_editable_message_content,
    )


def render_message_menu(message: StyledMessage):
    return pc.popover(
        pc.popover_trigger(
            pc.button(pc.icon(tag="hamburger"), size='xs', variant='ghost', is_disabled=EditableMessageState.is_editing)
        ),
        pc.popover_content(
            pc.popover_arrow(),
            pc.popover_body(
                pc.hstack(
                    # pc.button(pc.icon(tag="add"), size='xs', variant='ghost'),
                    # pc.button(pc.icon(tag="delete"), size='xs', variant='ghost'),
                    # pc.divider(orientation="vertical", height="1em"),
                    render_edit_thread_button(message),
                    render_fork_thread_button(message),  # type: ignore
                )
            ),
            pc.popover_close_button(),
            width='7em',  # '12em',
        ),
        trigger="hover",
    )


def render_message(message: StyledMessage):
    return pc.cond(
        message.uid == EditableMessageState.editable_message_uid,
        render_editable_message(EditableMessageState.styled_editable_message),
        pc.cond(
            message.uid == MessageOverviewState.current_message_uid,
            render_message_overview_carousel(),
            render_static_message(message),
        ),
    )


def render_message_thread_menu(message_thread: MessageThread):
    return pc.popover(
        pc.popover_trigger(pc.button(pc.icon(tag="hamburger"), size='xs', variant='ghost')),
        pc.popover_content(
            pc.popover_arrow(),
            pc.popover_body(
                pc.hstack(
                    pc.button(pc.icon(tag="close"), size='xs', variant='ghost'),
                    pc.button(pc.icon(tag="info_outline"), size='xs', variant='ghost'),
                    pc.button(pc.icon(tag="copy"), size='xs', variant='ghost'),
                    pc.button(pc.icon(tag="delete"), size='xs', variant='ghost'),
                    pc.divider(orientation="vertical", height="1em"),
                    pc.button(pc.icon(tag="lock"), size='xs', variant='ghost'),
                    pc.button(pc.icon(tag="edit"), size='xs', variant='ghost'),
                )
            ),
            pc.popover_close_button(),
            width='17em',
        ),
        trigger="hover",
    )


def render_note_editor():
    return pc.cond(
        ~MessageExplorationState.note_editor,
        pc.button(
            pc.flex(
                pc.cond(
                    MessageExplorationState.note,
                    pc.markdown(MessageExplorationState.note),
                    pc.markdown("Add a note here...", color=SolarizedColors.base1),
                ),
                pc.spacer(),
                pc.icon(tag="edit", size="xs"),
                width="100%",
                style={"font-weight": "normal"},
            ),
            on_click=MessageExplorationState.start_note_editing(None),
            variant="ghost",
            width="100%",
            padding="0.25em",
        ),
        pc.editable(
            pc.editable_preview(),
            pc.editable_textarea(),
            start_with_edit_view=True,
            default_value=MessageExplorationState.note,
            on_submit=MessageExplorationState.update_note,
            on_cancel=MessageExplorationState.cancel_note_editing,
            margin="0.25em",
        ),
    )


# BUG(blackhc): https://github.com/pynecone-io/pynecone/issues/925
class FixedSlider(pc.Slider):
    @classmethod
    def get_controlled_triggers(cls) -> Dict[str, Var]:
        return {}

    def get_triggers(cls) -> set[str]:
        return {"on_change_end"}


fixed_slider = FixedSlider.create


def float_slider(value: Var[float], setter, min_: float, max_: float, step: float = 0.01):
    def on_change_end(final_value: Var[int]):
        event_spec: pc.event.EventSpec = setter(final_value * step + min_)
        return event_spec.copy(update=dict(local_args={final_value.name}))

    return pc.fragment(
        pc.text(BaseVar(name=f"{value.full_name}.toFixed(2)", type_=str)),
        fixed_slider(
            min_=0,
            max_=int((max_ - min_ + step / 2) // step),
            default_value=((value - min_ + step / 2) // step).to(int),
            on_change_end=on_change_end(pc.EVENT_ARG),
        ),
    )


def render_model_options():
    return pc.table_container(
        pc.table(
            # pc.thead(
            #     pc.tr(
            #         pc.th("Name"),
            #         pc.th("Age"),
            #     )
            # ),
            pc.tbody(
                pc.tr(
                    pc.th("Model"),
                    pc.td(
                        pc.select(
                            [
                                "GPT-3.5",
                                "GPT-4",
                            ],
                            default_value="GPT-3.5",
                        ),
                        style={"min-width": "20em"},
                    ),
                ),
                pc.tr(
                    pc.th("Temperature"),
                    pc.td(
                        float_slider(
                            ModelOptionsState.temperature, ModelOptionsState.set_temperature, min_=0, max_=1, step=0.01
                        ),
                    ),
                ),
                pc.tr(
                    pc.th("Max Tokens"),
                    pc.td(
                        pc.text(ModelOptionsState.max_tokens),
                        pc.slider(
                            min_=0,
                            max_=2048,
                            default_value=ModelOptionsState.max_tokens,
                            on_change_end=ModelOptionsState.set_max_tokens,
                        ),
                    ),
                ),
                pc.tr(
                    pc.th("Top P"),
                    pc.td(
                        float_slider(ModelOptionsState.top_p, ModelOptionsState.set_top_p, min_=0, max_=1, step=0.01),
                    ),
                ),
                pc.tr(
                    pc.th("Frequency Penalty"),
                    pc.td(
                        float_slider(
                            ModelOptionsState.frequency_penalty,
                            ModelOptionsState.set_frequency_penalty,
                            min_=0,
                            max_=2,
                            step=0.01,
                        ),
                    ),
                ),
                pc.tr(
                    pc.th("Presence Penalty"),
                    pc.td(
                        float_slider(
                            ModelOptionsState.presence_penalty,
                            ModelOptionsState.set_presence_penalty,
                            min_=0,
                            max_=1,
                            step=0.01,
                        ),
                    ),
                ),
            ),
            width="50%",
        ),
    )


def render_message_exploration(message_thread: MessageThread):
    return pc.box(
        pc.hstack(
            pc.fragment(
                render_message_thread_menu(message_thread),
                pc.button(
                    react_icon(as_="BsFillGrid3X3GapFill"),
                    size='xs',
                    variant='ghost',
                    on_click=MessageGridState.toggle_grid,
                ),
            ),
            pc.heading(
                pc.editable(
                    pc.editable_preview(),
                    pc.editable_input(),
                    default_value=MessageExplorationState.title,
                    placeholder="Untitled",
                ),
                size="md",
            ),
        ),
        pc.divider(margin="0.5em"),
        pc.box(
            render_note_editor(),
            padding="0.5em",
        ),
        pc.divider(margin="0.5em"),
        pc.cond(
            MessageGridState.is_grid_visible,
            render_message_grid(),
            pc.box(
                pc.foreach(message_thread.messages, render_message),
                pc.divider(margin="0.5em"),
                pc.button(
                    "Request Continuation",
                ),
                pc.cond(
                    State.chat_mode,
                    pc.fragment(
                        pc.divider(margin="0.5em"),
                        pc.hstack(
                            pc.text_area(
                                placeholder="Write a message...",
                                default_value="",
                                width="100%",
                                min_height="5em",
                            ),
                            pc.button(react_icon(as_="TbSend"), size="lg", variant="outline", float="right"),
                        ),
                    ),
                ),
                pc.box(
                    pc.accordion(
                        pc.accordion_item(
                            pc.accordion_button(
                                pc.hstack(
                                    pc.heading("Model Options", size="sm"),
                                    pc.accordion_icon(),
                                    min_width="100%",
                                ),
                            ),
                            pc.accordion_panel(
                                render_model_options(),
                            ),
                        ),
                    ),
                    margin_top="0.5em",
                ),
            ),
        ),
        width="100%",
    )


class UID:
    _last_time_idx: tuple[str, int] | None = None

    @staticmethod
    def create() -> str:
        now_dt = datetime.now()
        # format is YYYYMMDDHHMMSS
        now_str = now_dt.strftime("%Y%m%d%H%M%S")
        # if we have the same time as last time, increment the counter
        if UID._last_time_idx is not None and UID._last_time_idx[0] == now_str:
            UID._last_time_idx = (now_str, UID._last_time_idx[1] + 1)
        else:
            UID._last_time_idx = (now_str, 0)

        return f"{now_str}#{UID._last_time_idx[1]:04d}"


example_message_thread = MessageThread(
    uid=UID.create(),
    messages=[
        Message(
            uid=UID.create(),
            role=MessageRole.SYSTEM,
            creation_time=0,
            source=MessageSource(),
            content="# Be whatever you want to be.",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.HUMAN,
            creation_time=0,
            source=MessageSource(),
            content="What is 2+2?",
            error="I don't know.\n\nCan you help me?",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.BOT,
            creation_time=0,
            source=MessageSource(model_type="GPT-4"),
            content=None,
            error="I don't know.\n\nCan you help me?",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.BOT,
            creation_time=0,
            source=MessageSource(model_type="GPT-4"),
            content="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
            "What is 2+2? Lorem ipsum possum dolor sit amet. 4. ",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.HUMAN,
            creation_time=0,
            source=MessageSource(),
            content="What is 2+2?",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.BOT,
            creation_time=0,
            source=MessageSource(model_type="GPT-4"),
            content="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
            "What is 2+2? Lorem ipsum possum dolor sit amet. 4. ",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.HUMAN,
            creation_time=0,
            source=MessageSource(),
            content="What is 2+2?",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.BOT,
            creation_time=0,
            source=MessageSource(model_type="GPT-4"),
            content="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
            "What is 2+2? Lorem ipsum possum dolor sit amet. 4. ",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.HUMAN,
            creation_time=0,
            source=MessageSource(),
            content="What is 2+2?",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.BOT,
            creation_time=0,
            source=MessageSource(model_type="GPT-4"),
            content="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
            "What is 2+2? Lorem ipsum possum dolor sit amet. 4. ",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.HUMAN,
            creation_time=0,
            source=MessageSource(),
            content="What is 2+2?",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.BOT,
            creation_time=0,
            source=MessageSource(model_type="GPT-4"),
            content="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
            "What is 2+2? Lorem ipsum possum dolor sit amet. 4. ",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.HUMAN,
            creation_time=0,
            source=MessageSource(),
            content="What is 2+2?",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.BOT,
            creation_time=0,
            source=MessageSource(model_type="GPT-4"),
            content="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
            "What is 2+2? Lorem ipsum possum dolor sit amet. 4. ",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.HUMAN,
            creation_time=0,
            source=MessageSource(),
            content="What is 2+2?",
        ),
        Message(
            uid=UID.create(),
            role=MessageRole.BOT,
            creation_time=0,
            source=MessageSource(model_type="GPT-4"),
            content="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
            "What is 2+2? Lorem ipsum possum dolor sit amet. 4. ",
        ),
    ],
    forked_threads=[],
)

example_message_exploration = MessageExploration(
    uid=UID.create(),
    title="Untitled",
    note="Blabla",
    tags=[],
    current_message_thread_uid=example_message_thread.uid,
    message_threads=[example_message_thread],
    hidden_message_thread_uids=set(),
)


class State(pc.State):
    """The app state."""

    _message_exploration: MessageExploration = example_message_exploration.copy(deep=True)
    auto_focus_uid: str | None = None

    @pc.var
    def chat_mode(self) -> bool:
        messages = self._message_exploration.current_message_thread.messages
        if len(messages) == 0 or messages[-1].role != MessageRole.HUMAN:
            return True
        return False

    @property
    def message_map(self):
        return self._message_exploration.message_uid_map

    @pc.var
    def styled_message_thread(self) -> StyledMessageThread:
        return self._message_exploration.get_styled_current_message_thread()


class MessageExplorationState(State):
    note_editor: bool = False

    def cancel_note_editing(self, _: typing.Any):
        self.note_editor = False
        self.mark_dirty()

    def start_note_editing(self, _: typing.Any):
        self.note_editor = True
        self.mark_dirty()

    @pc.var
    def note(self) -> str:
        return self._message_exploration.note

    @pc.var
    def title(self) -> str:
        return self._message_exploration.title

    def update_note(self, content):
        self._message_exploration.note = content
        self.note_editor = False
        self.mark_dirty()

    def update_title(self, content):
        self._message_exploration.title = content
        self.mark_dirty()


class NavigationState(State):
    def go_to(self, thread_id, message_uid):
        self.auto_focus_uid = message_uid
        self._message_exploration.current_message_thread_uid = thread_id
        self.mark_dirty()


class EditableMessageState(State):
    _editable_message: Message | None = None
    fork_mode: bool = False

    @pc.var
    def editable_message_uid(self) -> str | None:
        if self._editable_message is None:
            return None
        else:
            return self._editable_message.uid

    @pc.var
    def is_editing(self) -> bool:
        return self._editable_message is not None

    @typing.no_type_check
    @pc.var
    def styled_editable_message(self) -> StyledMessage | None:
        if self._editable_message is None:
            return None
        else:
            return self._editable_message.compute_editable_style()

    def cancel_editing(self):
        self._editable_message = None
        self.mark_dirty()

    def submit_editing(self):
        if self._editable_message is None:
            return

        if self.fork_mode:
            EditableMessageState._fork_thread(self)
        else:
            EditableMessageState._edit_thread(self)

        self._editable_message = None
        self.mark_dirty()

    def enter_editing(self, uid: str):
        self._editable_message = self.message_map[uid].copy()
        self.fork_mode = False
        self.mark_dirty()

    def enter_forking(self, uid: str):
        self._editable_message = self.message_map[uid].copy()
        self.fork_mode = True
        self.mark_dirty()

    def set_editable_message_content(self, content: str):
        assert self._editable_message is not None
        self._editable_message.content = content

    def set_editable_message_role(self, role: str):
        assert self._editable_message is not None
        self._editable_message.role = MessageRole(role)

    def clear_editable_message_error(self):
        assert self._editable_message is not None
        self._editable_message.error = None
        self.mark_dirty()

    def _edit_thread(self):
        original_message = self.message_map[self._editable_message.uid]

        print(original_message, self._editable_message)
        if self._editable_message == original_message:
            return

        # copy the current message thread
        message_thread = self._message_exploration.current_message_thread.copy()
        message_thread.uid = UID.create()

        # update the message in the message thread
        messages = list(message_thread.messages)

        # create a message with a new uid
        new_message = self._editable_message.copy()
        new_message.uid = UID.create()
        new_message.source = new_message.source.copy()
        new_message.source.edited = True

        messages[messages.index(original_message)] = new_message
        message_thread.messages = messages

        # update the message exploration
        self._message_exploration.update_message_thread(message_thread)

    def _fork_thread(self):
        original_message = self.message_map[self._editable_message.uid]
        if original_message == self._editable_message:
            return

        # copy the current message thread
        message_thread = self._message_exploration.current_message_thread.copy()
        message_thread.uid = UID.create()

        # update the message in the message thread
        message_index = message_thread.messages.index(original_message)
        messages = list(message_thread.messages[: message_index + 1])

        # create a message with a new uid
        new_message = self._editable_message.copy()
        new_message.uid = UID.create()
        if new_message.source.model_type is not None:
            new_message.source = new_message.source.copy()
            new_message.source.edited = True
        else:
            new_message.creation_time = time.time()
        new_message.error = None

        messages[-1] = new_message
        message_thread.messages = messages

        # update the message exploration
        self._message_exploration.update_message_thread(message_thread)


class MessageOverviewState(State):
    current_message_uid: str | None = None

    def show_drawer(self, message_uid: str):
        self.current_message_uid = message_uid

    def close_drawer(self, target_message_uid: str):
        styled_current_message: StyledMessage | None = MessageOverviewState._get_current_message(self)
        if styled_current_message is None:
            return
        # find target_message_uid in prev_linked_messages and next_linked_messages
        linked_message = [
            message
            for message in styled_current_message.prev_linked_messages + styled_current_message.next_linked_messages
            if message.uid == target_message_uid
        ]
        if len(linked_message) != 0:
            linked_message = linked_message[0]
            # set the current message thread to the thread of the linked message
            self._message_exploration.current_message_thread_uid = linked_message.thread_uid

        self.current_message_uid = None
        self.mark_dirty()

    def _get_current_message(self) -> StyledMessage | None:
        return self.styled_message_thread.get_message_by_uid(self.current_message_uid)

    @pc.var
    def styled_current_message(self) -> StyledMessage | None:
        styled_current_message: StyledMessage | None = MessageOverviewState._get_current_message(self)
        if styled_current_message is None:
            return None
        return styled_current_message.compute_editable_style()

    @pc.var
    def styled_previous_messages(self) -> list[StyledMessage]:
        styled_current_message: StyledMessage | None = MessageOverviewState._get_current_message(self)
        if styled_current_message is None:
            return []
        return [
            self.message_map[prev_linked_message.uid].compute_editable_style()
            for prev_linked_message in styled_current_message.prev_linked_messages
        ]

    @pc.var
    def styled_next_messages(self) -> list[StyledMessage]:
        styled_current_message: StyledMessage | None = MessageOverviewState._get_current_message(self)
        if styled_current_message is None:
            return []
        return [
            self.message_map[next_linked_message.uid].compute_editable_style()
            for next_linked_message in styled_current_message.next_linked_messages
        ]


class MessageGridState(State):
    _grid: MessageGrid | None = None

    def remove_thread(self, thread_uid: str):
        self._message_exploration.delete_message_thread(thread_uid)
        self._grid = self._message_exploration.get_message_grid()
        self.mark_dirty()

    def toggle_grid(self):
        if self._grid is None:
            self._grid = self._message_exploration.get_message_grid()
        else:
            self._grid = None
        self.mark_dirty()

    def go_to_thread(self, thread_uid: str):
        self._message_exploration.current_message_thread_uid = thread_uid
        self._grid = None
        self.mark_dirty()

    @pc.var
    def is_grid_visible(self) -> bool:
        return self._grid is not None

    @pc.var
    def column_thread_uids(self) -> list[str]:
        if self._grid is None:
            return []
        return self._grid.thread_uids

    @pc.var
    def styled_grid_cells(self) -> list[list[StyledGridCell]]:
        if self._grid is None:
            return [[]]

        styled_grid_cells = []
        for row_idx, messages in enumerate(self._grid.message_grid):
            styled_row = []
            for col_idx, message in enumerate(messages):
                styled_grid_cell = StyledGridCell(
                    message=message.compute_editable_style() if message is not None else None,
                    row_idx=row_idx + 1,
                    col_idx=col_idx + 1,
                    thread_uid=self._grid.thread_uids[col_idx],
                )
                styled_row.append(styled_grid_cell)
            styled_grid_cells.append(styled_row)

        # compress the grid row-by-row using col_span for duplicated cells
        for row_idx, row in enumerate(styled_grid_cells):
            col_idx = 0
            while col_idx < len(row):
                cell = row[col_idx]
                if cell.message is None:
                    col_idx += 1
                    continue
                col_span = 1
                while col_idx + col_span < len(row) and row[col_idx + col_span].message == cell.message:
                    col_span += 1
                if col_span > 1:
                    row[col_idx].col_span = col_span
                    for i in range(col_idx + 1, col_idx + col_span):
                        row[i].skip = True
                col_idx += col_span
        return styled_grid_cells


class ModelOptionsState(State):
    """The model options state."""

    model_type: str = "GPT-3.5"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


def index() -> pc.Component:
    return pc.container(
        pc.el.script(
            """
            import { extendTheme, withDefaultColorScheme } from '@chakra-ui/react'

            const customTheme = extendTheme(withDefaultColorScheme({ colorScheme: 'red' }))
            """
        ),
        pc.vstack(
            render_message_exploration(State.styled_message_thread),
            width="100",
        ),
        padding_top="64px",
        padding_bottom="64px",
        max_width="80ch",
    )


# Add state and page to the app.
app = pc.App(
    state=State,
    stylesheets=[
        'react-json-view-lite.css',
    ],
)
app.add_page(index)
app.compile()

if __name__ == "__main__":
    cli.main()
