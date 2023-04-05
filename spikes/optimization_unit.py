"""
Spike for a meta-loop that optimizes the prompt templates we use.
"""
import functools
import typing
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, TypeVar

import langchain
from langchain import OpenAI
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

from blackboard_pagi.cached_chat_model import CachedChatOpenAI
from blackboard_pagi.prompt_optimizer.llm_function import LLMFunction, llm_function
from blackboard_pagi.prompt_optimizer.track_execution import TrackedChatModel
from blackboard_pagi.prompt_optimizer.track_hyperparameters import (
    HyperparametersBuilder,
    hyperparameters_scope,
    track_hyperparameters,
)
from blackboard_pagi.prompts.chat_chain import ChatChain

langchain.llm_cache = SQLiteCache(".self_optimization.langchain.db")

chat_model = ChatOpenAI(model_name="gpt-4", max_tokens=1024, model_kwargs=dict(temperature=0.1))
# chat_model = CachedChatOpenAI(max_tokens=1024)

text_model = OpenAI(
    model_name="text-davinci-003",
    max_tokens=256,
    model_kwargs=dict(temperature=0.0),
)

# %%

TaskParameters = TypeVar('TaskParameters')
TaskResults = TypeVar('TaskResults')
Hyperparameters = TypeVar('Hyperparameters')


class TaskRun(GenericModel, Generic[TaskParameters, TaskResults, Hyperparameters]):
    """
    The task run. This is the 'data' we use to optimize the hyperparameters.
    """

    task_parameters: TaskParameters = Field(..., description="The task parameters.")
    hyperparameters: Hyperparameters = Field(
        ...,
        description="The hyperparameters used for the task. We optimize these.",
    )
    all_chat_chains: list[dict] = Field(..., description="The chat chains from the task execution.")
    # all_prompts: dict[str, str] = Field(..., description="The prompts and outputs from the task execution.")
    return_value: TaskResults = Field(..., description="The results of the task.")


class TaskReflection(BaseModel):
    """
    The reflections on the task.

    This contains the lessons we learn from each task run to come up with better hyperparameters to try.
    """

    feedback: str = Field(
        ...,
        description="Only look at the final results field. Does its content satisfy the task description and task "
        "parameters? "
        "Does it contain all the relevant information from the all_chains and all_prompts fields? What "
        "could be improved in the results?",
    )
    evaluation: str = Field(
        ...,
        description="The evaluation of the outputs given the task. Is the output satisfying? "
        "What is wrong? What is missing?",
    )
    hyperparameter_suggestion: str = Field(
        ...,
        description="How we want to change the hyperparameters to improve the results. What could we try to change?",
    )
    hyperparameter_missing: str = Field(
        ...,
        description="What hyperparameters are missing to improve the results? What could be changed that is not "
        "exposed via hyperparameters?",
    )


class TaskInfo(GenericModel, Generic[TaskParameters, TaskResults, Hyperparameters]):
    """
    The task run and the reflection on the experiment.
    """

    task_parameters: TaskParameters = Field(..., description="The task parameters.")
    hyperparameters: Hyperparameters = Field(
        ...,
        description="The hyperparameters used for the task. We optimize these.",
    )
    reflection: TaskReflection = Field(..., description="The reflection on the task.")


class OptimizationInfo(GenericModel, Generic[TaskParameters, TaskResults, Hyperparameters]):
    """
    The optimization information. This is the data we use to optimize the hyperparameters.
    """

    older_task_summary: str | None = Field(
        None,
        description="A summary of previous experiments and the proposed changes with the goal of avoiding trying the same changes repeatedly.",
    )
    task_infos: list[TaskInfo[TaskParameters, TaskResults, Hyperparameters]] = Field(
        ..., description="The most recent tasks we have run and our reflections on them."
    )
    best_hyperparameters: Hyperparameters = Field(..., description="The best hyperparameters we have found so far.")


class OptimizationStep(GenericModel, Generic[TaskParameters, TaskResults, Hyperparameters]):
    """
    The next optimization steps. New hyperparameters we want to try given the previous experiments and
    new task parameters we want to evaluate on given the previous experiments.
    """

    best_hyperparameters: Hyperparameters = Field(
        ..., description="The best hyperparameters we have found so far given " "task_infos and history."
    )
    suggestion: str = Field(
        ...,
        description="The suggestions for the next experiments. What could we try to change?"
        "We will try several tasks next and several sets of hyperparameters. Let's think step by "
        "step.",
    )
    task_parameters_suggestions: list[TaskParameters] = Field(
        ..., description="The task parameters we want to try next.", hint_min_items=1, hint_max_items=4
    )
    hyperparameter_suggestions: list[dict] = Field(
        ..., description="The hyperparameters we want to try next.", hint_min_items=1, hint_max_items=2
    )


@llm_function
def reflect_on_task_run(
    language_model, task_run: TaskRun[TaskParameters, TaskResults, Hyperparameters]
) -> TaskReflection:
    """
    Reflect on the task run.
    """
    raise NotImplementedError()


@llm_function
def summarize_optimization_info(
    language_model, optimization_info: OptimizationInfo[TaskParameters, TaskResults, Hyperparameters]
) -> str:
    """
    Summarize the optimization info. We want to preserve all relevant knowledge for
    improving the hyperparameters in the future. All information from previous experiments will be
    forgotten except for what this summary.
    """
    raise NotImplementedError()


@llm_function
def suggest_next_optimization_step(
    language_model, optimization_info: OptimizationInfo[TaskParameters, TaskResults, Hyperparameters]
) -> OptimizationStep[TaskParameters, TaskResults, Hyperparameters]:
    """
    Suggest the next optimization step.
    """
    raise NotImplementedError()


@llm_function
def probability_for_improvement(
    language_model, optimization_info: OptimizationInfo[TaskParameters, TaskResults, Hyperparameters]
) -> typing.Annotated[
    float,
    Field(
        ge=0.0,
        le=1.0,
        description="The self-reported probability that the next optimization steps will improve the hyperparameters.",
    ),
]:
    """
    Return the probability for improvement.
    """
    raise NotImplementedError()


@dataclass
class CapturedFuncArgsKWArgs:
    f: typing.Callable
    args: tuple
    kwargs: dict


P = typing.ParamSpec("P")
T = typing.TypeVar('T')


class CaptureFuncArgsKWArgs:
    def __matmul__(self, f: typing.Callable[P, T]) -> typing.Callable[P, CapturedFuncArgsKWArgs]:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return CapturedFuncArgsKWArgs(f=f, args=args, kwargs=kwargs)

        return wrapper


capture_func_args_kwargs = CaptureFuncArgsKWArgs()


def capture_task_run(
    chat_model,
    task_executor: LLMFunction,
    task_parameters: TaskParameters,
    hyperparameters: BaseModel,
) -> TaskRun[TaskParameters, TaskResults, Hyperparameters]:
    """
    Capture the task run.
    """
    tracked_chat_model = TrackedChatModel(chat_model=chat_model)

    llm_function_spec = task_executor.get_spec_from_inputs(task_parameters)
    with hyperparameters_scope(hyperparameters) as scope:
        return_value = task_executor(chat_model, **task_parameters.dict())

    return TaskRun[llm_function_spec.input_model, llm_function_spec.return_type, type(scope.hyperparameters)](
        task_parameters=task_parameters,
        hyperparameters=scope.hyperparameters,
        all_chat_chains=tracked_chat_model.tracked_chats.build_compact_dict(),
        return_value=return_value,
    )


@track_hyperparameters
def optimize_hyperparameters(
    chat_model: BaseChatModel,
    task_executor,
    seed_task_runs: list[TaskRun[TaskParameters, TaskResults, Hyperparameters]],
) -> OptimizationStep[TaskParameters, TaskResults, Hyperparameters]:
    """
    Optimize the hyperparameters.
    """

    root_chain = ChatChain(chat_model, [])

    llm_function_spec = task_executor.get_spec_from_inputs(seed_task_runs[0].task_parameters)

    task_infos = [
        TaskInfo[llm_function_spec.input_model, llm_function_spec.output_model](
            task_parameters=task_run.task_parameters,
            hyperparameters=task_run.hyperparameters,
            results=task_run.return_value,
            reflection=reflect_on_task_run(root_chain, task_run),
        )
        for task_run in seed_task_runs
    ]

    hyperparameters_builder = HyperparametersBuilder()
    for task_info in task_infos:
        hyperparameters_builder.update(task_info.hyperparameters)

    hyperparameter_model = hyperparameters_builder.build()
    hyperparameter_type = type(hyperparameter_model)

    optimization_info = OptimizationInfo[
        llm_function_spec.input_model, llm_function_spec.output_model, hyperparameter_type
    ](
        task_infos=task_infos,
        best_hyperparameters=hyperparameter_model,
    )

    optimization_step = None
    for _ in range(3):
        optimization_step = suggest_next_optimization_step(root_chain, optimization_info)

        optimization_info.best_hyperparameters = optimization_step.best_hyperparameters

        for task_parameters in optimization_step.task_parameters_suggestions:
            for hyperparameters in optimization_step.hyperparameter_suggestions:
                task_run = capture_task_run(root_chain, task_executor, task_parameters, hyperparameters)
                task_info = TaskInfo[
                    llm_function_spec.input_model, llm_function_spec.output_model, type(hyperparameters)
                ](
                    task_parameters=task_run.task_parameters,
                    hyperparameters=task_run.hyperparameters,
                    results=task_run.return_value,
                    reflection=reflect_on_task_run(root_chain, task_run),
                )
                optimization_info.task_infos.append(task_info)

        if len(optimization_info.task_infos) >= 20:
            optimization_info.older_task_summary = summarize_optimization_info(
                root_chain,
                OptimizationInfo[llm_function_spec.input_model, llm_function_spec.output_model, hyperparameter_type](
                    older_task_summary=optimization_info.older_task_summary,
                    task_infos=optimization_info.task_infos[:-10],
                    best_hyperparameters=optimization_step.best_hyperparameters,
                ),
            )
            optimization_info.task_infos = optimization_info.task_infos[-10:]

        if probability_for_improvement(root_chain, optimization_info) < 0.5:
            break

    return optimization_step


@llm_function
def write_essay(chat_model, essay_topic: str = Field(..., description="The essay topic.")) -> str:
    """
    Write an essay at the level of an Oxford undergraduate student. Please write about 500 words.
    Use markdown to format the essay.
    """
    raise NotImplementedError()


@llm_function
def create_essay_topics(
    chat_model,
    domain: str = Field(..., description="The domain or general area of the essay topic"),
    n: int = Field(..., description="Number of topics to generate"),
) -> list[str]:
    """
    Create a list of essay topics.
    """
    raise NotImplementedError()


# Several very varied essay topics in various domains (e.g., science, technology, art, philosophy, etc.).
essay_topics = [
    "The Interplay Between Artificial Intelligence and Human Ethics: A Philosophical Inquiry",
    "Examining the Effects of Climate Change on Global Food Security and Agricultural Practices",
    "A Comparative Analysis of Ancient Greek and Roman Political Structures: Democracy vs. Republic",
    "The Psychological Impact of Social Media: A Deep Dive into Mental Health and Connectivity",
    "The Fusion of Quantum Mechanics and General Relativity: The Road Towards a Unified Theory",
    "Exploring Gender Roles in Shakespeare's Plays: A Cross-Sectional Study of Female Characters",
    "The Evolution of Human Language: Examining Linguistic Diversity and its Implications",
    "The Intersection of Neuroscience and Music: How Rhythm and Melody Influence Cognitive Function",
    "The Socioeconomic Impacts of Renewable Energy Adoption: A Comparative Study of Developed and Developing Nations",
    "Deconstructing the Notion of Free Will: A Multidisciplinary Analysis of Human Agency",
]

#%%

seed_task_runs = [
    capture_task_run(
        ChatChain(chat_model, []),
        create_essay_topics,
        create_essay_topics.get_inputs(domain="comedy", n=2),
        BaseModel(),
    )
    for topic in essay_topics[:2]
]

all_hyperparameters = deepcopy(create_essay_topics.all_hyperparameters)

# Update seed task runs with the hyperparameters we found.
for task_run in seed_task_runs:
    task_run.hyperparameters = all_hyperparameters

#%%

optimization_step = optimize_hyperparameters(chat_model, create_essay_topics, seed_task_runs)
