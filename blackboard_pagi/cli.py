"""Console script for blackboard_pagi."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("blackboard-pagi")
    click.echo("=" * len("blackboard-pagi"))
    click.echo("Proto-AGI using a Blackboard System (for the LLM Hackathon by Ben's Bites)")


if __name__ == "__main__":
    main()  # pragma: no cover
