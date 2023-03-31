"""
Spike for a meta-loop that optimizes the prompt templates we use.
"""

import langchain
from langchain import OpenAI
from langchain.cache import SQLiteCache
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

from blackboard_pagi.cached_chat_model import CachedChatOpenAI


class PydanticDataclassOutputParser(PydanticOutputParser):
    def parse(self, text: str):
        # Ignore type mismatch
        # noinspection PyTypeChecker
        return super().parse(text)


langchain.llm_cache = SQLiteCache(".execution_llm_spike.langchain.db")

# chat_model = CachedChatOpenAI(model_name="gpt-4", max_tokens=512)
chat_model = CachedChatOpenAI(max_tokens=512)

text_model = OpenAI(
    model_name="text-davinci-003",
    max_tokens=256,
    model_kwargs=dict(temperature=0.0),
)


# %%


class ExperimentRun(BaseModel):
    """
    The experiment run. This is the data we use to optimize the hyperparameters.
    """

    task_description: dict = Field(..., description="The task description as JSON")
    hyperparameters: dict = Field(
        ...,
        description="The hyperparameters used for the experiment. We optimize these.",
    )
    outputs: dict = Field(..., description="The outputs of the experiment.")


class ExperimentReflection(BaseModel):
    """
    The reflection on the experiment. These are the lessons we learn from each experiment run.
    """

    evaluation: str = Field(..., description="The evaluation of the outputs given the task.")
    feedback: str = Field(
        ...,
        description="What should be improved in the outputs given the evaluation for the given task.",
    )
    hyperparameter_suggestion: str = Field(
        ...,
        description="Reflection on how we might want to change the hyperparameters to improve them.",
    )


class Experiment(BaseModel):
    """
    The experiment run and the reflection on the experiment.
    """

    run: ExperimentRun = Field(..., description="The experiment run.")
    reflection: ExperimentReflection = Field(..., description="The reflection on the experiment.")


class OptimizationInfo(BaseModel):
    """
    The optimization step. This is the data we use to optimize the hyperparameters.
    """

    log_summary: str | None = Field(
        ...,
        description="A summary of previous experiments and the proposed changes with the goal of avoiding trying the same changes repeatedly.",
    )
    experiments: list[Experiment] = Field(..., description="The experiments we have run so far.")


class OptimizationStep(BaseModel):
    """
    The optimization step. New hyperparameters we want to try given the previous experiments.
    """

    optimization_info: OptimizationInfo = Field(..., description="The optimization info.")
    suggestion: str = Field(..., description="The suggestion for the next experiment.")


@dataclass
class Context:
    knowledge: dict[str, str]


# We want to define dataclasses for different actions the model can execute (e.g. "add a new contribution")
# and then use the model to decide which action to execute.
# We want to parse the actions from the model's output, and then execute them.

# Can we use pydantic discriminators to do this?
