"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import pprint  # noqa: F401
import typing
from datetime import datetime
from enum import Enum

import pynecone as pc
import pynecone.pc as cli
from pynecone import Var
from pynecone.var import ComputedVar


# Redefine ComputedVar to revert the baseclasses
class _ComputedVar(Var, property):
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


pc.var = _ComputedVar

ComputedVar.register(pc.var)


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
    HUMAN = (SolarizedColors.base1, SolarizedColors.base03)
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

    def compute_style(self):
        return StyledMessage.create(self)


class StyledMessage(Message):
    background_color: str
    foreground_color: str
    align: str
    fmt_creation_datetime: str
    fmt_header: str

    @classmethod
    def create(cls, message: Message) -> str:
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
            **dict(message),
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
    title: str
    note: str
    tags: list[str]
    messages: list[Message]
    forked_threads: list['MessageThread']

    def compute_style(self) -> 'StyledMessageThread':
        return StyledMessageThread(
            uid=self.uid,
            title=self.title,
            note=self.note,
            tags=self.tags,
            messages=[StyledMessage.create(message) for message in self.messages],
            forked_threads=[thread.compute_style() for thread in self.forked_threads],
        )


class StyledMessageThread(MessageThread):
    messages: list[StyledMessage]


def render_message_content(message_content: str | None):
    return pc.cond(message_content, pc.markdown(message_content))  # type: ignore


def render_message_toolbar(message: StyledMessage):
    is_editing = message.uid == EditableMessageState.editable_message_uid
    return pc.fragment(
        pc.cond(
            is_editing,
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
            render_message_menu(message),
        ),
        pc.button(pc.icon(tag="chevron_left"), size='xs', variant='ghost', is_disabled=is_editing),
        pc.button(pc.icon(tag="search"), size='xs', variant='ghost', is_disabled=is_editing),
        pc.button(pc.icon(tag="chevron_right"), size='xs', variant='ghost', is_disabled=is_editing),
    )


def render_static_message(message: StyledMessage):
    return pc.hstack(
        pc.vstack(
            pc.box(
                pc.flex(
                    render_message_toolbar(message),
                    pc.spacer(),
                    pc.text(message.fmt_header),
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
                    pc.cond(
                        message.content,
                        pc.box(
                            render_message_content(message.content),
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
                        render_message_content(message.content),
                        border_radius="0 0 15px 0",
                        color=message.foreground_color,
                        border_width="medium",
                        border_color=message.foreground_color,
                        width="100%",
                        padding="0 0.25em 0 0.25em",
                    ),
                ),
            ),
            width="75%",
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


def render_editable_message_content(message, with_error: bool):
    return pc.text_area(
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
        pc.popover_trigger(pc.button(pc.icon(tag="hamburger"), size='xs', variant='ghost')),
        pc.popover_content(
            pc.popover_arrow(),
            pc.popover_body(
                pc.hstack(
                    pc.button(pc.icon(tag="add"), size='xs', variant='ghost'),
                    pc.button(pc.icon(tag="delete"), size='xs', variant='ghost'),
                    pc.divider(orientation="vertical", height="1em"),
                    pc.button(
                        pc.icon(tag="edit"),
                        size='xs',
                        variant='ghost',
                        on_click=lambda: EditableMessageState.set_editing(message.uid),  # type: ignore
                    ),
                    pc.button(pc.icon(tag="repeat_clock"), size='xs', variant='ghost'),
                )
            ),
            pc.popover_close_button(),
            width='12em',
        ),
        trigger="hover",
    )


def render_message(message: StyledMessage):
    return pc.cond(
        message.uid == EditableMessageState.editable_message_uid,
        render_editable_message(EditableMessageState.styled_editable_message),
        render_static_message(message),
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
                    pc.divider(orientation="vertical", height="1em"),
                    pc.button(pc.icon(tag="drag_handle"), size='xs', variant='ghost'),
                )
            ),
            pc.popover_close_button(),
            width='20em',
        ),
        trigger="hover",
    )


def render_message_thread(message_thread: MessageThread):
    return pc.box(
        pc.hstack(
            render_message_thread_menu(message_thread),
            pc.heading(message_thread.title, size="md"),
        ),
        pc.divider(margin="0.5em"),
        pc.markdown(message_thread.note),
        pc.divider(margin="0.5em"),
        pc.box(
            pc.foreach(message_thread.messages, render_message),
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
    title="Untitled",
    note="",
    tags=[],
    messages=[
        Message(
            uid=UID.create(),
            role=MessageRole.SYSTEM,
            creation_time=0,
            source=MessageSource(),
            content="Be whatever you want to be.",
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


class State(pc.State):
    """The app state."""

    _message_thread: MessageThread = example_message_thread
    _message_map: dict[str, Message] = {}

    @property
    def message_map(self):
        self._message_map = {
            m.uid: m for thread in [self._message_thread] + self._message_thread.forked_threads for m in thread.messages
        }
        return self._message_map

    @pc.var
    def styled_message_thread(self) -> StyledMessageThread:
        return self._message_thread.compute_style()


class EditableMessageState(State):
    _editable_message: Message | None = None

    @pc.var
    def editable_message_uid(self) -> str | None:
        if self._editable_message is None:
            return None
        else:
            return self._editable_message.uid

    @typing.no_type_check
    @pc.var
    def styled_editable_message(self) -> StyledMessage:
        if self._editable_message is None:
            return None
        else:
            return self._editable_message.compute_style()

    def cancel_editing(self):
        self._editable_message = None

    def submit_editing(self):
        self._editable_message = None

    def set_editing(self, uid: str):
        self._editable_message = self.message_map[uid].copy()

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


def index() -> pc.Component:
    return pc.container(
        pc.vstack(
            render_message_thread(State.styled_message_thread),
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
