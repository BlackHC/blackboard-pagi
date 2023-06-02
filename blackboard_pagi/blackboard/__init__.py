"""
Based on docs/concepts/blackboard.md TK overview actually chatgpt TK:

> PAGI is an artificial general intelligence that uses a data sandbox called the Blackboard to store information about
 its reference data (the current state of the world), knowledge (the current state of its own mind), tasks (the current
 state of its plans), skills (the current state of its abilities and experience), and self (the current state of its
 processes). The Blackboard is stored in a git repository and uses Obsidian markdown files to store metadata and allow
 for viewing in a web browser or desktop application. Git's version control system and branching/merging capabilities
 allow for tracking changes to the Blackboard and for exploration and experimentation.

This file implements the Blackboard class, which is a wrapper around a git repository of Obsidian markdown files. The
Blackboard class provides methods for reading and writing to the Blackboard, as well as methods for committing changes
to the Blackboard.
"""
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

import pathlib
import shutil
from dataclasses import dataclass

import frontmatter
import git


@dataclass
class MarkdownContent:
    """Represents the content of a markdown file.

    We have the following attributes:
        - 'frontmatter': The frontmatter of the markdown file (as dict).
        - 'body': The body of the markdown file.

    We also have the following methods:
        - 'dumps': Returns the combined content as a string.

    We also have the following static methods:
        - 'loads': Creates a MarkdownContent object from a string.

    Note: because of the frontmatter library, the frontmatter does not support properties with the
    names "content", "metadata", or "self"... :|
    """

    frontmatter: dict
    body: str

    def dumps(self) -> str:
        """Returns the combined content as a string."""
        return frontmatter.dumps(frontmatter.Post(body=self.body, **self.frontmatter))

    @staticmethod
    def loads(string):
        """Creates a MarkdownContent object from a string."""
        post = frontmatter.loads(string)
        return MarkdownContent(post.metadata, post.content)


class Blackboard:
    """
    A wrapper around a git repository of Obsidian markdown files.

    The Blackboard class is responsible for:
        - Opening a git working directory (or creating a repository if it does not exist)
        - Reading and writing to the Blackboard
        - Committing changes to the Blackboard

    If the Blackboard is not initialized, it will be initialized when the Blackboard class is instantiated:
        - A git repository will be created in the specified directory
        - A .gitignore file will be created in the specified directory
        - A README.md file will be created in the specified directory
    """

    def __init__(self, directory):
        """
        Initialize the Blackboard class.
        """
        self.working_dir = pathlib.Path(directory)

        if not self.working_dir.is_dir():
            # TODO: move this into its own method.
            # Copy files in the template directory (under this sub-package) to Blackboard directory
            template_dir = pathlib.Path(__file__).parent / "template"
            shutil.copytree(template_dir, directory)
            # Create a git repository if it doesn't exist
            repo = git.Repo.init(directory)
            # Add all files to the index
            repo.index.add(["."])
            # Commit all initial changes
            repo.index.commit("Initialize Blackboard")
        else:
            # Check if the directory is a git repository
            if not (self.working_dir / ".git").is_dir():
                raise Exception("The specified directory is not a git repository.")

        # Open the git repository
        self.repo = git.Repo(directory)
        # Ensure the git repository is not bare
        if self.repo.bare:
            raise Exception("The specified directory is a bare git repository.")
        # Create a git index
        self.index = self.repo.index
        # Create a git tree
        self.tree = self.repo.tree()
        # Create a git head
        self.head = self.repo.head

    def get_local_path(self, path):
        """
        Get the local path of a file or directory.
        """
        return self.working_dir / path

    def create_persisted_directory(self, path):
        """
        Create an empty directory with a .gitkeep file.
        """
        (self.working_dir / path).mkdir()
        (self.working_dir / path / ".gitkeep").touch()

    def touch_file(self, path):
        """
        Create an empty file.
        """
        (self.working_dir / path).touch()

    def read_text_file(self, path):
        """
        Read the contents of a text file.
        """
        return (self.working_dir / path).read_text()

    def write_text_file(self, path, contents):
        """
        Write contents to a text file.
        """
        (self.working_dir / path).write_text(contents)

    def read_markdown_file(self, path):
        """
        Read the contents of a markdown file.
        """
        return MarkdownContent.loads(self.read_text_file(path))

    def write_markdown_file(self, path, contents):
        """
        Write contents to a markdown file.
        """
        self.write_text_file(path, contents.dumps())

    def commit(self, message):
        """
        Commit changes to the Blackboard.
        """
        self.index.commit(message)

    def store(self, data):
        """
        Store data to the Blackboard.
        """
        raise NotImplementedError()

    def lookup(self, query):
        """
        Lookup data in the Blackboard.
        """
        raise NotImplementedError()
