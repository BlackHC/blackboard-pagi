"""
Tests for the Blackboard class using pytest.
"""
import tempfile
from blackboard_pagi.blackboard import Blackboard


def test_blackboard_class():
    """
    Test the Blackboard class.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as directory:
        # Initialize the Blackboard class
        blackboard = Blackboard(directory + "/blackboard")
        # Read the contents of the README.md file
        readme = blackboard.read_markdown("README.md")
        # Ensure the README.md file contains the expected contents
        assert readme == "# Blackboard"

        # Write the contents of the README.md file
        blackboard.write_markdown("README.md", "Hello, World!")
        # Read the contents of the README.md file
        readme = blackboard.read_markdown("README.md")
        # Ensure the README.md file contains the expected contents
        assert readme == "Hello, World!"

        # Commit changes to the Blackboard
        blackboard.commit("Update README.md")

        # Read the contents of the README.md file
        readme = blackboard.read_markdown("README.md")
        # Ensure the README.md file contains the expected contents
        assert readme == "Hello, World!"
