import logging
import signal
import subprocess
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from resmon import __version__
from resmon.core import Collector, Monitor, default_collectors
from resmon.report import compare_table, section_stats, stats_table
from resmon.sink import make_sink, read_records, setup_logging

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True)
console = Console()

LogOption = Annotated[Path, typer.Option("--log", help="Куда писать замеры: .jsonl или .db")]
IntervalOption = Annotated[float, typer.Option("--interval", help="Пауза между замерами, с")]
LogArgument = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, help="Файл с замерами: .jsonl или .db")
]


@app.callback()
def main() -> None:
    """Монитор ресурсов для ML-задач."""
    setup_logging()


@contextmanager
def watched(
    log: Path, interval: float, collectors: list[Collector] | None = None
) -> Iterator[Monitor]:
    """Монитор с записью замеров в файл. В конце печатает отчёт, даже если всё упало."""
    with (
        closing(make_sink(log)) as sink,
        Monitor(interval, sinks=[sink], collectors=collectors) as monitor,
    ):
        try:
            yield monitor
        finally:
            console.print(stats_table(section_stats(monitor.records)))


@app.command()
def demo(
    texts: Annotated[int, typer.Option(min=1, help="Сколько отзывов скачать")] = 500,
    log: LogOption = Path("logs/demo.jsonl"),
    interval: IntervalOption = 0.5,
) -> None:
    """Демо: скачать отзывы, посчитать эмбеддинги и найти похожие на запрос."""
    # Демо тянет torch, поэтому импортируем его только когда оно нужно
    from resmon.demo import run_demo

    with watched(log, interval) as monitor:
        found = run_demo(monitor, texts)
    console.print("Самые похожие отзывы:")
    for text in found:
        console.print(f"- {text[:120]}")


@app.command(context_settings={"allow_interspersed_args": False})
def run(
    command: Annotated[list[str], typer.Argument(help="Программа, например python train.py")],
    log: LogOption = Path("logs/run.jsonl"),
    interval: IntervalOption = 0.5,
) -> None:
    """Запустить чужую программу и следить за ней снаружи: resmon run -- python train.py."""
    try:
        process = subprocess.Popen(command)
    except OSError as error:
        logger.error("программа не запустилась: %s", error)
        raise typer.Exit(127) from None
    # Ctrl+C получит и сама программа. Пусть она спокойно завершится, а мы досмотрим до конца
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with watched(log, interval, default_collectors(process.pid)):
            code = process.wait()
    finally:
        signal.signal(signal.SIGINT, previous)
    # Если программу убили сигналом, wait вернёт минус номер сигнала. В shell принято 128 + номер
    raise typer.Exit(128 - code if code < 0 else code)


@app.command()
def report(log: LogArgument) -> None:
    """Показать отчёт по секциям из файла с замерами."""
    console.print(stats_table(section_stats(read_records(log))))


@app.command()
def compare(old: LogArgument, new: LogArgument) -> None:
    """Сравнить два запуска: было и стало по каждой секции."""
    console.print(compare_table(section_stats(read_records(old)), section_stats(read_records(new))))


@app.command()
def version() -> None:
    """Показать версию resmon."""
    typer.echo(__version__)
