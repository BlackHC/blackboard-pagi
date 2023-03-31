# default / --min-python-version 3.9.0
from dataclasses import dataclass
from typing import Annotated


@dataclass
class LLMField:
    """
    A class to represent a field in a LLM.
    """

    description: str

    def __class_getitem__(cls, params):
        return Annotated[params[0], LLMField(params[1])]
