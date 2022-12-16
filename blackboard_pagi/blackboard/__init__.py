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
import pathlib

import git


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
            # Create a git repository if it doesn't exist
            repo = git.Repo.init(directory)
            # Create a .gitignore file if it doesn't exist
            (self.working_dir / ".gitignore").write_text(".DS_Store")
            # Create a README.md file if it doesn't exist
            (self.working_dir / "README.md").write_text("# Blackboard")
            # Create directories for: reference data, knowledge, tasks, skills, and self
            self.create_persisted_directory("reference data")
            self.create_persisted_directory("knowledge")
            self.create_persisted_directory("tasks")
            self.create_persisted_directory("skills")
            self.create_persisted_directory("self")
            # Create README.md files for: reference data, knowledge, tasks, skills, and self with
            # a short description of what each directory is for
            self.write_markdown(
                "reference data/README.md",
                "# Reference Data\n\nThis directory contains reference data, which is data that is used to "
                "represent the current state of the world. Reference data is stored in Obsidian markdown files.",
            )
            self.write_markdown(
                "knowledge/README.md",
                "# Knowledge\n\nThis directory contains knowledge, which is data that is used to represent the "
                "current state of the agent's mind. Knowledge is stored in Obsidian markdown files.",
            )
            self.write_markdown(
                "tasks/README.md",
                "# Tasks\n\nThis directory contains tasks, which are actions that the agent plans to perform. "
                "Tasks are stored in Obsidian markdown files.",
            )
            self.write_markdown(
                "skills/README.md",
                "# Skills\n\nThis directory contains skills, which are actions that the agent can perform. "
                "Skills are stored in Obsidian markdown files.",
            )
            self.write_markdown(
                "self/README.md",
                "# Self\n\nThis directory contains self, which is data that is used to represent the current "
                "state of the agent's processes. Self is stored in Obsidian markdown files.",
            )
            # Commit all initial changes
            repo.index.add(["."])
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

    def read_markdown(self, path):
        """
        Read the contents of a markdown file.
        """
        return (self.working_dir / path).read_text()

    def write_markdown(self, path, contents):
        """
        Write contents to a markdown file.
        """
        (self.working_dir / path).write_text(contents)

    def commit(self, message):
        """
        Commit changes to the Blackboard.
        """
        self.index.commit(message)
