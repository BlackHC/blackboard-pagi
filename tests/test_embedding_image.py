"""
PyTest for embedding_image.py
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

import numpy as np

from blackboard_pagi.embedding_image import (
    base64_to_embedding,
    create_inline_markdown_embedding,
    decode_embedding,
    embedding_to_base64,
    encode_embedding,
    parse_inline_markdown_embedding,
    remove_inlined_embeddings,
)


def test_encode_decode():
    """Test encode and decode."""
    embedding = np.random.rand(1536)
    encoded = encode_embedding(embedding)
    decoded = decode_embedding(encoded)
    assert np.allclose(embedding, decoded)


def test_base64():
    """Test base64 conversion."""
    embedding = np.random.rand(1536)
    base64_png = embedding_to_base64(embedding)
    decoded = base64_to_embedding(base64_png)
    assert np.allclose(embedding, decoded)


def test_inline_markdown():
    """Test inline markdown conversion."""
    embedding = np.random.rand(1536)
    markdown = create_inline_markdown_embedding(embedding)
    decoded = parse_inline_markdown_embedding(markdown)
    assert np.allclose(embedding, decoded)


def test_inline_markdown_no_separator():
    """Test inline markdown conversion."""
    embedding = np.random.rand(1536)
    markdown = create_inline_markdown_embedding(embedding)
    markdown = markdown.replace("---", "")
    decoded = parse_inline_markdown_embedding(markdown)
    assert np.allclose(embedding, decoded)


def test_inline_markdown_return_first_out_of_two():
    """Test inline markdown conversion."""
    embedding = np.random.rand(1536)
    embedding2 = np.random.rand(1536)
    markdown = create_inline_markdown_embedding(embedding)
    markdown += create_inline_markdown_embedding(embedding2)
    decoded = parse_inline_markdown_embedding(markdown)
    assert np.allclose(embedding, decoded)


def test_remove_inlined_embeddings():
    """Test inline markdown conversion."""
    embedding = np.random.rand(1536)
    markdown = create_inline_markdown_embedding(embedding)
    markdown = markdown.replace("---", "")
    decoded = parse_inline_markdown_embedding(markdown)
    assert np.allclose(embedding, decoded)

    markdown = remove_inlined_embeddings(markdown)
    assert "inline_embedding" not in markdown
    assert "---" not in markdown
