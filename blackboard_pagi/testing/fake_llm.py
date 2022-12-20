from typing import Any, Mapping

from langchain.llms import LLM
from pydantic import BaseModel


class FakeLLM(LLM, BaseModel):
    """Fake LLM wrapper for testing purposes."""

    queries: dict[str, str]

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(self, prompt: str, stop: list[str] | None = None) -> str:
        """Return the query if it exists, else print the code to update the query."""
        if self.queries is not None and prompt in self.queries:
            answer = self.queries[prompt]
            # Emulate stop behavior
            if stop is not None:
                for stop_word in stop:
                    if stop_word in answer:
                        # Only return the answer up to the stop word
                        return answer[: answer.index(stop_word)]
            return answer

        # If no queries are provided, print the code to update the query
        code_snippet = f"""# Add the following to the queries dict:
{prompt!r}: "foo", # TODO: Update this
"""
        raise NotImplementedError("No query provided. Add the following to the queries dict:\n\n" + code_snippet)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}
