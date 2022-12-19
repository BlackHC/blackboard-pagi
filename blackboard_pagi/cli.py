"""Console script for blackboard_pagi."""

import click
import langchain
from langchain.cache import SQLiteCache
from langchain.llms.openai import OpenAI

import blackboard_pagi.controller

langchain.llm_cache = SQLiteCache()


@click.command()
def main():
    """Main entrypoint."""
    click.echo("blackboard-pagi")
    click.echo("=" * len("blackboard-pagi"))
    click.echo("Proto-AGI using a Blackboard System (for the LLM Hackathon by Ben's Bites)")

    click.echo("What is your prompt?")
    prompt = click.prompt("Prompt", default="How many colleges are there in Oxford and Cambridge?")
    # default="What is the meaning of life?")

    kernel = blackboard_pagi.controller.Kernel(OpenAI())
    note = kernel(prompt)

    click.echo("Here is your note:")
    click.echo(note)


if __name__ == "__main__":
    main()  # pragma: no cover
