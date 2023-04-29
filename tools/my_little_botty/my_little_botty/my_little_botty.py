"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
import pprint  # noqa: F401
from datetime import datetime
from enum import Enum

import pynecone as pc
import pynecone.pc as cli
from pcconfig import config
from pydantic import BaseModel

docs_url = "https://pynecone.io/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"


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


class MessageContent(pc.Base):
    markdown: str


def render_message_content(message_content: MessageContent | None):
    return pc.cond(message_content, pc.markdown(message_content.markdown))  # type: ignore


class Message(pc.Base):
    uid: str

    role: MessageRole
    source: MessageSource
    creation_time: int
    content: MessageContent | None
    error: str | None


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


class StyledMessage(Message):
    background_color: str
    foreground_color: str
    align: str
    fmt_creation_datetime: str

    @classmethod
    def create(cls, message: Message) -> str:
        fmt_creation_datetime = format_datetime(message.creation_time)

        if message.role == MessageRole.SYSTEM:
            background_color, foreground_color = ColorStyle.SYSTEM
            align = "left"
        elif message.role == MessageRole.HUMAN:
            background_color, foreground_color = ColorStyle.HUMAN
            align = "right"
        elif message.role == MessageRole.BOT:
            align = "left"
            assert message.source.model_type is not None
            background_color = ColorStyle.BOTS[sum(map(ord, message.source.model_type)) % len(ColorStyle.BOTS)]
            foreground_color = ColorStyle.HUMAN[1]

        return cls(
            align=align,
            background_color=background_color,
            foreground_color=foreground_color,
            fmt_creation_datetime=fmt_creation_datetime,
            **dict(message),
        )


class IconButton(pc.IconButton):
    size: str = "md"


icon_button = IconButton.create


def render_message(message: Message):
    return pc.hstack(
        pc.vstack(
            pc.center(
                pc.flex(
                    pc.popover(
                        pc.popover_trigger(pc.button(pc.icon(tag="hamburger"), size='xs', variant='ghost')),
                        pc.popover_content(
                            pc.popover_arrow(),
                            pc.popover_body(
                                pc.hstack(
                                    pc.button(pc.icon(tag="add"), size='xs', variant='ghost'),
                                    pc.button(pc.icon(tag="delete"), size='xs', variant='ghost'),
                                    pc.divider(orientation="vertical", height="1em"),
                                    pc.button(pc.icon(tag="edit"), size='xs', variant='ghost'),
                                    pc.button(pc.icon(tag="repeat_clock"), size='xs', variant='ghost'),
                                )
                            ),
                            pc.popover_close_button(),
                            width='12em',
                        ),
                        trigger="hover",
                    ),
                    pc.button(pc.icon(tag="chevron_left"), size='xs', variant='ghost'),
                    pc.button(pc.icon(tag="search"), size='xs', variant='ghost'),
                    pc.button(pc.icon(tag="chevron_right"), size='xs', variant='ghost'),
                    pc.spacer(),
                    pc.text(message.source.model_type | message.role),  # type: ignore
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
            pc.center(
                pc.vstack(
                    pc.cond(message.content, render_message_content(message.content)),
                    pc.cond(message.error, pc.text(message.error)),
                ),
                border_radius="0 0 15px 0",
                color=message.foreground_color,
                border_width="medium",
                border_color=message.foreground_color,
                width="100%",
                padding="0 0.25em 0 0.25em",
            ),
            width="75%",
            padding_bottom="0.5em",
            spacing="0em",
        ),
        justify_content=message.align,
        width="100%",
    )


class MessageThread(BaseModel):
    uid: str
    title: str
    note: str
    tags: list[str]
    messages: list[Message]
    forked_threads: list['MessageThread']

    def apply_style(self):
        return StyledMessageThread(
            uid=self.uid,
            title=self.title,
            note=self.note,
            tags=self.tags,
            messages=[StyledMessage.create(message) for message in self.messages],
            forked_threads=[thread.apply_style() for thread in self.forked_threads],
        )


class StyledMessageThread(MessageThread, pc.Base):
    messages: list[StyledMessage]


def render_message_thread(message_thread: MessageThread):
    return pc.box(
        pc.hstack(
            pc.popover(
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
            ),
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


class State(pc.State):
    """The app state."""

    message_thread: StyledMessageThread = MessageThread(
        uid="",
        title="Untitled",
        note="",
        tags=[],
        messages=[
            Message(
                uid="",
                role=MessageRole.SYSTEM,
                creation_time=0,
                source=MessageSource(),
                content=MessageContent(markdown="Be whatever you want to be."),
            ),
            Message(
                uid="",
                role=MessageRole.HUMAN,
                creation_time=0,
                source=MessageSource(),
                content=MessageContent(markdown="What is 2+2?"),
            ),
            Message(
                uid="",
                role=MessageRole.BOT,
                creation_time=0,
                source=MessageSource(model_type="GPT-4"),
                content=MessageContent(
                    markdown="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                    "What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                ),
            ),
            Message(
                uid="",
                role=MessageRole.HUMAN,
                creation_time=0,
                source=MessageSource(),
                content=MessageContent(markdown="What is 2+2?"),
            ),
            Message(
                uid="",
                role=MessageRole.BOT,
                creation_time=0,
                source=MessageSource(model_type="GPT-4"),
                content=MessageContent(
                    markdown="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                    "What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                ),
            ),
            Message(
                uid="",
                role=MessageRole.HUMAN,
                creation_time=0,
                source=MessageSource(),
                content=MessageContent(markdown="What is 2+2?"),
            ),
            Message(
                uid="",
                role=MessageRole.BOT,
                creation_time=0,
                source=MessageSource(model_type="GPT-4"),
                content=MessageContent(
                    markdown="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                    "What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                ),
            ),
            Message(
                uid="",
                role=MessageRole.HUMAN,
                creation_time=0,
                source=MessageSource(),
                content=MessageContent(markdown="What is 2+2?"),
            ),
            Message(
                uid="",
                role=MessageRole.BOT,
                creation_time=0,
                source=MessageSource(model_type="GPT-4"),
                content=MessageContent(
                    markdown="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                    "What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                ),
            ),
            Message(
                uid="",
                role=MessageRole.HUMAN,
                creation_time=0,
                source=MessageSource(),
                content=MessageContent(markdown="What is 2+2?"),
            ),
            Message(
                uid="",
                role=MessageRole.BOT,
                creation_time=0,
                source=MessageSource(model_type="GPT-4"),
                content=MessageContent(
                    markdown="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                    "What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                ),
            ),
            Message(
                uid="",
                role=MessageRole.HUMAN,
                creation_time=0,
                source=MessageSource(),
                content=MessageContent(markdown="What is 2+2?"),
            ),
            Message(
                uid="",
                role=MessageRole.BOT,
                creation_time=0,
                source=MessageSource(model_type="GPT-4"),
                content=MessageContent(
                    markdown="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                    "What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                ),
            ),
            Message(
                uid="",
                role=MessageRole.HUMAN,
                creation_time=0,
                source=MessageSource(),
                content=MessageContent(markdown="What is 2+2?"),
            ),
            Message(
                uid="",
                role=MessageRole.BOT,
                creation_time=0,
                source=MessageSource(model_type="GPT-4"),
                content=MessageContent(
                    markdown="What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                    "What is 2+2? Lorem ipsum possum dolor sit amet. 4. "
                ),
            ),
        ],
        forked_threads=[],
    ).apply_style()


def index() -> pc.Component:
    return pc.container(
        pc.vstack(
            render_message_thread(State.message_thread),
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
