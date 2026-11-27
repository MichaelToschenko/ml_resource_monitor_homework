import typer

from resmon import __version__

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Монитор ресурсов для ML-задач."""


@app.command()
def version() -> None:
    """Показать версию resmon."""
    typer.echo(__version__)
