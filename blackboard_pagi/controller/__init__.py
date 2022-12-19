"""
A simple controller for the Blackboard PAGI (to get started with the project).

The idea is that we use LLMs for (almost) everything and provide a very simple kernel that gracefully handles
failure cases.

The only obvious failure case for an LLM is:
    * the LLM hits its token limit (that is we cannot generate a further response).

We can handle this by:
    * ensuring we have enough takens to generate a response (e.g. 1024 tokens); and otherwise
    * inducing the LLM to summarize the context so far to reduce the prompt token count and ensure enough tokens are
        available to generate a response.
"""
from dataclasses import dataclass
from typing import Optional

import langchain
import yaml

from blackboard_pagi import prompt_template


@dataclass
class MainPromptTemplate(prompt_template.PromptTemplateMixin):
    """A main prompt for the Blackboard PAGI."""

    user_prompt: str

    prompt_template = """You are a large language model and act as a virtual assistant that has access to a scratchpad for documents,
which we call a blackboard.

# Instructions

You have access to several tools. You can invoke tools by ending your answer with a YAML block. For example, you
can invoke a Google search for "OpenAI" by ending your answer with:

```yaml

---
tool: search-google
query: OpenAI
```

Note that we do not include the `---` at the end of the YAML block in the answer.

You must include a tool property in the YAML block with the name of the tool you want to invoke.
You can also pass parameters using YAML syntax depending on the tool (see the definitions below).

If you don't want to invoke a tool, you can end your answer with a YAML block that contains a `tool` property with
the value `none` to indicate that you don't want to invoke a tool.

You have the following tools available with the parameters specified using $parameter syntax:
- "search-google": search Google for a $query and return the top $num_results results (default: 1).
- "search-wikipedia": search Wikipedia for a $query and return the top $num_results results (default: 1)..
- "memorize": memorize $note on the blackboard. You can then recall the note using the `recall` tool below.
- "recall": recall the $association from the blackboard. We match the $association using its semantic to
    all the notes on the blackboard and return the top $num_results results (default: 1).
- "ask-user": ask the user a $question and return the user's response.

After you stop, we will execute the tool you invoked and return the result in the YAML block in a 'result' property.
For example, if you invoke the `search-google` tool, we will return the top result in the YAML block as follows:

For example:
```yaml

---
tool: search-google
query: OpenAI
result:
    title: OpenAI
    url: https://openai.com/
    description: OpenAI is an artificial intelligence research company founded in 2015 by Elon Musk, Sam Altman, Ilya
        Sutskever, Greg Brockman, and Wojciech Zaremba.
---

```

Note the extra newlines and final --- at the end of the YAML block. This is required to indicate the end of the
YAML block.

Now, we will give your the main task. Think step by step and try to solve the problem. You can use the tools
described above to help you solve the problem. Try to write valid markdown in your answer except for the YAML
block at the end of your answer.

# Prompt

{user_prompt}

# Answer

"""


@dataclass
class SummarizePromptTemplate(prompt_template.PromptTemplateMixin):
    """A prompt for summarizing the context so far."""

    user_prompt: str
    answer: str

    prompt_template = """You are a large language assistant. You have hit your token limit for a request and we are going to summarize the
answer below to reduce the prompt token count and ensure enough tokens are available to generate a full response by you
later.

Summarize the given answer so far with regard to the main goal, such that you can continue to answer the
prompt with the summarized answer later. Stop as soon as you have summarized the answer so far. Do not attempt to
answer the prompt yourself.

The original answer includes YAML blocks. You can summarize the YAML blocks as well, but preserve the YAML syntax for
them.

# Prompt

{user_prompt}

# Answer

{answer}

# Shortened Partial Answer

"""


def approximate_token_count(text: str) -> int:
    """Approximate the token count of the text.

    Each word is counted as at least one token. Whitespace is counted as one token.
    A token can be at most 4 characters long (approximating the tokenization of the GPT-3 tokenizer).
    """
    return sum(1 + (len(word) - 1) // 4 for word in text.split())


def extract_last_yaml_block(prompt: str) -> Optional[str]:
    """Extract the last YAML block from the prompt. Fail gracefully if there is no YAML block."""
    yaml_blocks = prompt.split("\n---\n")
    if len(yaml_blocks) == 1:
        return None
    else:
        return yaml_blocks[-1]


def wrap_yaml_blocks_as_source(text: str) -> str:
    """Wrap YAML blocks in Markdown source blocks (e.g. ```yaml\n---\n[...]\n---\n```).

    We start and end outside of a YAML block. E.g.

    > Some text
    > ---
    > tool: search-google
    > query: OpenAI
    > ---
    > Some more text
    >
    > ---
    > tool: search-google
    > query: OpenAI GPT-3
    > ---
    >
    > Some more text

    becomes

    > Some text
    > ```yaml
    > ---
    > tool: search-google
    > query: OpenAI
    > ---
    > ```
    > Some more text
    >
    > ```yaml
    > ---
    > tool: search-google
    > query: OpenAI GPT-3
    > ---
    > ```
    >
    > Some more text
    """
    # Add a newline before and after the text
    text = f"\n{text}\n"
    blocks = text.split("\n---\n")

    # We have an odd number of blocks and end with a text block.
    assert len(blocks) % 2 == 1
    # We start with a text block.
    transformed_text = blocks[0]
    for i in range(1, len(blocks), 2):
        transformed_text += f"\n```yaml\n---\n{blocks[i]}\n---\n```\n{blocks[i + 1]}"
    return transformed_text.strip()


@dataclass
class SearchGooglePromptTemplate(prompt_template.PromptTemplateMixin):
    """A prompt for the search-google tool."""

    invocation: str

    prompt_template = """You are a large language assistant. You are emulating Google search now (for integration testing purposes).
You are given a YAML block with a query (`query` property) and a number of results to return (`num_results` property,
or default: 1).

You return the top results as a in the YAML block in a `results` property. Each result is a dictionary with the
following properties:
- `title`: the title of the result.
- `url`: the URL of the result.
- `description`: the description of the result.

For example, for:
```yaml
---
tool: search-google
query: OpenAI
num_results: 2
---
```

You return:
```yaml
---
tool: search-google
query: OpenAI
result:
    - title: OpenAI
      url: https://openai.com/
      description: OpenAI is an artificial intelligence research company founded in 2015 by Elon Musk, Sam Altman,
                    Ilya Sutskever, Greg Brockman, and Wojciech Zaremba.
    - title: OpenAI GPT-3
      url: https://openai.com/blog/openai-api/
      description: OpenAI API is a new service that gives developers access to OpenAI’s state-of-the-art language
                    models.
---
```

# Invocation

---
{invocation}
---

# Result YAML Block

"""


def generate_note(user_prompt: str, answer: str) -> str:
    """Generate a note for the blackboard."""
    document = f"""# Prompt
{user_prompt}

# Answer

{answer}
"""
    document = wrap_yaml_blocks_as_source(document)
    return document


class Kernel:
    """A simple kernel for the Blackboard PAGI.

    This is a very simple spike kernel that does not make use of all that langchain has to offer currently.

    We also use the simplest possible API: llm(prompt) -> response.
    """

    def __init__(self, llm: langchain.llms.LLM):
        self.llm = llm

    def summarize(self, user_prompt: str, answer: str) -> str:
        """Summarize the answer so far."""
        meta_prompt = SummarizePromptTemplate(user_prompt, answer)
        summarized_answer = self.llm(meta_prompt())

        return summarized_answer

    def _main_prompt(self, user_prompt, token_limit=2000):
        """Execute a prompt and return the answer.

        Extract YAMLs from the continunations and execute them as needed.
        If needed also summarize the answer so far.
        """
        original_answers = []
        answer_blocks = []
        main_prompt = MainPromptTemplate(user_prompt)
        while True:
            full_context = main_prompt + "".join(answer_blocks)
            answer = self.llm(full_context)
            answer_blocks.append(answer)

            # Check if we have hit the token limit.
            if approx_num_tokens := approximate_token_count(full_context) > token_limit:
                # We have hit the token limit. Summarize the answer so far.
                full_answer = "".join(answer_blocks)
                original_answers.append(full_answer)
                summarized_answer = self.summarize(user_prompt, full_answer)
                approx_summarized_num_tokens = approximate_token_count(main_prompt + summarized_answer)
                assert approx_summarized_num_tokens <= token_limit and approx_summarized_num_tokens < approx_num_tokens

                answer_blocks = [summarized_answer]

            # Check if we have a YAML block
            yaml_block = extract_last_yaml_block(answer)
            if yaml_block is None:
                print(answer)
                # No YAML block, ask our user if we want to continue (input y or n)
                print("> Continue? [y/n]")
                while (choice := input()) not in ["y", "n"]:
                    print("> Please enter y or n.")
                if choice != "n":
                    break
            else:
                print(answer)

                # We have a YAML block, extract the tool and execute it
                yaml_block = list(yaml.safe_load_all(yaml_block))[0]
                tool = yaml_block["tool"]
                if tool == "none":
                    answer_blocks.append("\n---\n")
                    break

                # Prompt the user whether to execute the tool
                print(f"> Execute tool {tool}? [y/n]")
                while (choice := input()) not in ["y", "n"]:
                    print("> Please enter y or n.")
                if choice == "n":
                    answer_blocks.append("\n---\n")
                    break

                if tool == "search-google":
                    # Search Google for the query and return the top result
                    query = yaml_block["query"]
                    num_results = yaml_block.get("num_results", 1)
                    result = self.search_google(query, num_results)
                    # result = "Time-out :("
                    answer_blocks.append(f"""\nresult: {result}\n---\n\n""")
                elif tool == "ask-user":
                    # Ask the user for input
                    question = yaml_block["question"]
                    print(f"> PAGI: {question}")
                    answer = input()
                    answer_blocks.append(f"""\nresult: {answer}\n---\n\n""")
                else:
                    raise NotImplementedError(f"Unknown tool {tool}")

        return generate_note(user_prompt, "".join(answer_blocks))

    def search_google(self, yaml_parameters: str):
        """Search Google for the query and return the top result."""
        # Use the emulator for now
        prompt = SearchGooglePromptTemplate(yaml_parameters)
        answer = self.llm(prompt())

        # Print the prompt and answer
        print(prompt)
        print(answer)

        return answer

    def __call__(self, prompt: str) -> str:
        """Generate a response to a prompt.

        Args:
            prompt: The prompt to generate a response to.

        Returns:
            The response.
        """
        return self._main_prompt(prompt)
