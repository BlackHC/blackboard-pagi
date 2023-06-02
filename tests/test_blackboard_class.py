"""
Tests for the Blackboard class using pytest.
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

        # Write the contents of the README.md file
        blackboard.write_text_file("README.md", "Hello, World!")
        # Read the contents of the README.md file
        readme = blackboard.read_text_file("README.md")
        # Ensure the README.md file contains the expected contents
        assert readme == "Hello, World!"

        # Commit changes to the Blackboard
        blackboard.commit("Update README.md")

        # Read the contents of the README.md file
        readme = blackboard.read_text_file("README.md")
        # Ensure the README.md file contains the expected contents
        assert readme == "Hello, World!"
