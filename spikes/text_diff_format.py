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
"""A simple datatype for representing the differences between two texts."""
import re
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class _TextEdit(BaseModel):
    edit_type: str = Field(description="The type of edit.")

    def apply(self, text) -> str:
        """Apply the edit to the text."""
        raise NotImplementedError()


class InsertParagraph(_TextEdit):
    """An insertion of a new paragraph after an existing paragraph."""

    edit_type: Literal["INSERT"] = "INSERT"
    name: str | None = Field(
        description="The name of the paragraph after which the text is inserted or None if we "
        "insert at the beginning."
    )
    new_name: str = Field(description="The name of the new paragraph.")
    text: str = Field(description="The text that is inserted.")

    def apply(self, text) -> str:
        """Apply the edit to the text."""
        if self.name is None:
            return f"§{self.new_name}§\n{self.text}\n§/{self.new_name}§\n{text}"
        else:
            return text.replace(
                f"§/{self.name}§", f"§/{self.name}§\n§{self.new_name}§\n{self.text}\n§/{self.new_name}§"
            )


class DeleteParagraph(_TextEdit):
    """A deletion of a paragraph."""

    edit_type: Literal["DELETE"] = "DELETE"
    name: str = Field(description="The name of the paragraph that is deleted.")

    def apply(self, text) -> str:
        """Apply the edit to the text."""
        # Find the whole paragraph and remove it
        return re.sub(f"§{self.name}§\\n.*?\\n§/{self.name}§", "", text, flags=re.DOTALL | re.MULTILINE)


class ReplaceText(_TextEdit):
    """A text replacement within a paragraph."""

    edit_type: Literal["REPLACE"] = "REPLACE"
    name: str = Field(description="The name of the paragraph that is replaced.")
    prefix: str | None = Field(
        description="Where to insert the new text after if everything from the beginning is replaced."
    )
    suffix: str | None = Field(
        description="Where to insert the new text before or None if everything up to end is replaced."
    )
    new_text: str = Field(description="The new text that replaces the old text.")

    def apply(self, text) -> str:
        """Apply the edit to the text."""
        # Find the whole paragraph first and then replace the text within it
        regex_result = re.search(f"§{self.name}§\n(.*?)\n§/{self.name}§", text, flags=re.DOTALL | re.MULTILINE)
        full_paragraph = regex_result.group(0)
        paragraph = regex_result.group(1)

        if self.prefix is None:
            prefix = ""
            rest_paragraph = paragraph
        else:
            splits = paragraph.split(self.prefix)
            prefix = splits[0] + self.prefix
            rest_paragraph = splits[1]

        if self.suffix is None:
            suffix = ""
        else:
            suffix = self.suffix + rest_paragraph.split(self.suffix)[1]

        new_paragraph = f"§{self.name}§\n" + prefix + self.new_text + suffix + f"\n§/{self.name}§"
        return text.replace(full_paragraph, new_paragraph)


TextEdit = Annotated[Union[InsertParagraph, DeleteParagraph, ReplaceText], Field(discriminator='edit_type')]


class TextEdits(BaseModel):
    """
    The text that we edit has markers before and after each paragraph of the form "§<name>§...§/name§".

    Text edits are a list of edits, each of which is either a InsertParagraph, DeleteParagraph, or ReplaceText.
    """

    edits: list[TextEdit] = Field(description="The list of edits.")

    @staticmethod
    def prepare_text(text) -> str:
        """Split the text into paragraphs and insert markers.

        A paragraph is a sequence of lines that are separated by one or multiple empty lines.
        """
        # replace a line that only contains whitespace with an empty line
        text = f'\n{text}\n'
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n+\n", "\n\n", text, flags=re.MULTILINE)
        paragraphs = text.split("\n\n")
        paragraphs = [p for p in paragraphs if p]
        paragraphs = [f"§Org{i}§\n{p}\n§/Org{i}§" for i, p in enumerate(paragraphs)]
        return "\n".join(paragraphs)

    def apply(self, text) -> str:
        """Apply the edits to the text."""
        for edit in self.edits:
            text = edit.apply(text)
        return text


def test_prepare_text():
    """Test the text diff format."""
    text = """
    This is a test text.
    It has two paragraphs.

    This is the second paragraph.
    It has two lines.
    """
    text = TextEdits.prepare_text(text)
    assert text == (
        '§Org0§\n'
        '    This is a test text.\n'
        '    It has two paragraphs.\n'
        '§/Org0§\n'
        '§Org1§\n'
        '    This is the second paragraph.\n'
        '    It has two lines.\n'
        '§/Org1§'
    )


def test_text_diff_format():
    """Test the text diff format."""
    text = """
    This is a test text.
    It has two paragraphs.

    This is the second paragraph.
    It has two lines.
    """
    text = TextEdits.prepare_text(text)

    text_edits = TextEdits(
        edits=[
            InsertParagraph(name=None, new_name="NewParagraph", text="This is a new paragraph."),
            DeleteParagraph(name="Org0"),
            ReplaceText(
                name="Org1",
                prefix="This is the second paragraph.",
                suffix="It has two lines.",
                new_text="This is the replaced paragraph.",
            ),
        ]
    )
    text = text_edits.apply(text)
    assert text == (
        '§NewParagraph§\n'
        'This is a new paragraph.\n'
        '§/NewParagraph§\n'
        '\n'
        '§Org1§\n'
        '    This is the second paragraph.This is the replaced paragraph.It has two '
        'lines.\n'
        '§/Org1§'
    )
