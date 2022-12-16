"""
Based on docs/concepts/blackboard.md TK overview actually chatgpt TK:

> PAGI is an artificial general intelligence that uses a data sandbox called the Blackboard to store information about its reference data (the current state of the world), knowledge (the current state of its own mind), tasks (the current state of its plans), skills (the current state of its abilities and experience), and self (the current state of its processes). The Blackboard is stored in a git repository and uses Obsidian markdown files to store metadata and allow for viewing in a web browser or desktop application. Git's version control system and branching/merging capabilities allow for tracking changes to the Blackboard and for exploration and experimentation.

This file implements the Blackboard class, which is a wrapper around a git repository of Obsidian markdown files. The Blackboard class provides methods for reading and writing to the Blackboard, as well as methods for committing changes to the Blackboard.
"""
import git
import os

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
        if not os.path.isdir(directory):
            # Create a git repository if it doesn't exist
            repo = git.Repo.init(directory)
            # Create a .gitignore file if it doesn't exist
            with open(os.path.join(directory, ".gitignore"), "w") as file:
                file.write(".DS_Store")
            # Create a README.md file if it doesn't exist
            with open(os.path.join(directory, "README.md"), "w") as file:
                file.write("# Blackboard")
            # Commit the initial changes
            repo.index.add([".gitignore", "README.md"])
            repo.index.commit("Initialize Blackboard")
        else:
            # Check if the directory is a git repository
            if not os.path.isdir(os.path.join(directory, ".git")):
                raise Exception("The specified directory is not a git repository.")

        # Open the git repository
        self.repo = git.Repo(directory)
        # Ensure the git repository is not bare
        if self.repo.bare:
            raise Exception("The specified directory is a bare git repository.")
        # Create a git working directory
        self.working_dir = self.repo.working_dir
        # Create a git index
        self.index = self.repo.index
        # Create a git tree
        self.tree = self.repo.tree()
        # Create a git head
        self.head = self.repo.head

    def read_markdown(self, path):
        """
        Read the contents of a markdown file.
        """
        with open(os.path.join(self.working_dir, path), "r") as file:
            return file.read()

    def write_markdown(self, path, contents):
        """
        Write contents to a markdown file.
        """
        with open(os.path.join(self.working_dir, path), "w") as file:
            file.write(contents)

    def commit(self, message):
        """
        Commit changes to the Blackboard.
        """
        self.index.commit(message)

