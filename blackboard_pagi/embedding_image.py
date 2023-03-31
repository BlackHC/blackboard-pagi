"""
Encode an embedding in a way that looks pleasing to the eye as well.
"""
import base64
import io
import re

import numpy as np
from PIL import Image


def encode_embedding(embedding: np.ndarray):
    """Encode an embedding in a way that looks pleasing to the eye as well."""
    # Convert to float32
    embedding = embedding.astype(np.float32)
    # Convert embeddings to raw byte array
    embedding_bytes = embedding.tobytes()
    assert len(embedding_bytes) == embedding.size * embedding.itemsize
    embedding_np_bytes = np.frombuffer(embedding_bytes, dtype=np.uint8)
    square_size = (len(embedding_bytes) / 3) ** 0.5
    square_size = int(np.ceil(square_size))
    # Pad with zeros
    embedding_np_bytes = np.pad(
        embedding_np_bytes, (0, square_size**2 * 3 - len(embedding_bytes)), mode="constant", constant_values=0
    )
    # Reshape
    embedding_np_bytes = embedding_np_bytes.reshape((square_size, square_size, 3))
    # Convert to png
    image = Image.fromarray(embedding_np_bytes, mode="RGB")
    image_bytes_io = io.BytesIO()
    image.save(image_bytes_io, format="png")
    image_bytes = image_bytes_io.getvalue()
    return image_bytes


def decode_embedding(png_bytes, embedding_size=1536):
    """Decode an embedding from a png."""
    image = Image.open(io.BytesIO(png_bytes))
    # Turn into uint8 RGB image
    image = np.array(image, dtype=np.uint8).flatten()
    # Convert to bytes
    image = image.tobytes()
    # Turn the raw bytes into an embedding of float32 again
    # Drop mod float32
    image = image[: embedding_size * 4]
    embedding = np.frombuffer(image, dtype=np.float32)
    return embedding


def embedding_to_base64(embedding: np.ndarray):
    """Convert an embedding to a base64 string."""
    png_bytes = encode_embedding(embedding)
    return base64.b64encode(png_bytes).decode("utf-8")


def base64_to_embedding(base64_png):
    """Convert a base64 encoded png to an embedding."""
    png_bytes = base64.b64decode(base64_png)
    return decode_embedding(png_bytes)


def create_inline_markdown_embedding(embedding: np.ndarray):
    """Convert an embedding to an inline markdown image."""
    base64_png = embedding_to_base64(embedding)
    return f"\n\n---\n\n![inline_embedding](data:image/png;base64,{base64_png})"


def parse_inline_markdown_embedding(markdown):
    """Find the first base64 encoded png that uses the inlined_embedding description in the markdown and decode it."""
    matches = re.findall(r"!\[inline_embedding]\((data:image/png;base64,[a-zA-Z0-9+/=]+)\)", markdown)
    if not matches:
        raise ValueError("No inlined embedding found")
    embedding = base64_to_embedding(matches[0].split(",")[1])
    return embedding


def remove_inlined_embeddings(markdown):
    """Remove all inlined embedding from the markdown.

    Also remove a potential --- separator before it.

    If there is none, the markdown is returned unchanged.
    """
    return re.sub(r"(\n\n---\n\n)?!\[inline_embedding]\(data:image/png;base64,[a-zA-Z0-9+/=]+\)", "", markdown)


def show_embedding(embedding: np.ndarray):
    """Show an embedding in a notebook, for example."""
    import matplotlib.pyplot as plt

    png_bytes = encode_embedding(embedding)
    plt.imshow(Image.open(io.BytesIO(png_bytes)))
    plt.show()
