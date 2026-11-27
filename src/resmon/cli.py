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
from resmon.forecast import OomForecast, lead_time
from resmon.report import compare_table, last_run, section_stats, stats_table
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
        Monitor(interval, sinks=[sink, OomForecast()], collectors=collectors) as monitor,
    ):
        try:
            yield monitor
        finally:
            console.print(stats_table(section_stats(monitor.records)))


@app.command()
def demo(
    texts: Annotated[int, typer.Option(min=1, help="Сколько отзывов скачать")] = 500,
    leak_mb: Annotated[
        int, typer.Option(min=0, help="Сколько МБ утечь в конце, чтобы показать прогноз")
    ] = 0,
    log: LogOption = Path("logs/demo.jsonl"),
    interval: IntervalOption = 0.5,
) -> None:
    """Демо: скачать отзывы, посчитать эмбеддинги и найти похожие на запрос."""
    # Демо тянет torch, поэтому импортируем его только когда оно нужно
    from resmon.demo import run_demo

    with watched(log, interval) as monitor:
        found = run_demo(monitor, texts, leak_mb)
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
        with watched(log, interval, default_collectors(process.pid)) as monitor:
            # Если программу убили сигналом, wait вернёт минус его номер. В shell это 128 + номер
            code = process.wait()
            code = 128 - code if code < 0 else code
            monitor.mark("exit", code=code)
    finally:
        signal.signal(signal.SIGINT, previous)
    raise typer.Exit(code)


@app.command()
def report(log: LogArgument) -> None:
    """Показать отчёт по секциям из файла с замерами."""
    console.print(stats_table(section_stats(read_records(log))))


@app.command()
def compare(old: LogArgument, new: LogArgument) -> None:
    """Сравнить два запуска: было и стало по каждой секции."""
    console.print(compare_table(section_stats(read_records(old)), section_stats(read_records(new))))


@app.command()
def postmortem(log: LogArgument) -> None:
    """Разобрать запуск по логу: чем он кончился и успел ли прогноз предупредить."""
    records = last_run(read_records(log))
    memory = [r for r in records if r["event"] == "sample" and "mem_used_mb" in r]
    if not memory:
        logger.error("в логе нет замеров памяти, разбирать нечего")
        raise typer.Exit(1)
    console.print(stats_table(section_stats(records)))
    # Без метки stop процесс умер вместе с монитором. В resmon run смотрим код выхода программы
    code = next((r["code"] for r in records if r["event"] == "exit"), 0)
    stopped = any(r["event"] == "stop" for r in records)
    ending = "завершился сам" if stopped and code < 128 else "оборвался"
    last, lead = memory[-1], lead_time(records)
    console.print(f"Запуск {ending}, код выхода {code}, последняя секция: {last['section'] or '-'}")
    console.print(f"Занято памяти: {last['mem_used_mb']:.0f} из {last['mem_limit_mb']:.0f} МБ")
    console.print("Прогноз молчал" if lead is None else f"Прогноз предупредил за {lead:.0f} с")


@app.command()
def version() -> None:
    """Показать версию resmon."""
    typer.echo(__version__)
