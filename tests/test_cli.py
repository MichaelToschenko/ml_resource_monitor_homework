import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from resmon import __version__, cli
from resmon.core import Monitor
from resmon.sink import make_sink, read_records


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> Iterator[CliRunner]:
    """Запуск команд как из терминала. Широкая консоль, чтобы таблица не обрезалась."""
    monkeypatch.setattr(cli, "console", Console(width=200))
    yield CliRunner()
    # Команда настроила логи на свой поток вывода, после теста он уже закрыт
    logging.getLogger().handlers.clear()


@pytest.fixture
def log_file(monitor: Monitor, tmp_path: Path) -> Path:
    """Файл с замерами небольшого запуска с секцией load."""
    path = tmp_path / "run.jsonl"
    sink = make_sink(path)
    monitor.sinks.append(sink)
    monitor.start()
    with monitor.section("load"):
        pass
    monitor.stop()
    sink.close()
    return path


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.usefixtures("fake_model", "hf_api")
def test_demo_command(runner: CliRunner, tmp_path: Path) -> None:
    log = tmp_path / "demo.db"
    result = runner.invoke(cli.app, ["demo", "--texts", "100", "--log", str(log)])
    assert result.exit_code == 0
    assert "batch" in result.output
    assert "review 7 horror" in result.output
    assert log.exists()


@pytest.mark.parametrize("code", [0, 3])
def test_run_watches_other_program(runner: CliRunner, tmp_path: Path, code: int) -> None:
    log = tmp_path / "logs" / "run.db"
    program = f"import sys, time; data = b'x' * 100_000_000; time.sleep(0.5); sys.exit({code})"
    args = ["run", "--log", str(log), "--interval", "0.1"]
    result = runner.invoke(cli.app, [*args, "--", sys.executable, "-c", program])
    # Код выхода программы передаётся дальше
    assert result.exit_code == code
    assert "весь запуск" in result.output
    # Монитор видел память чужой программы, а не свою
    assert max(r.get("rss_mb", 0) for r in read_records(log)) > 100


@pytest.mark.parametrize(
    ("command", "code"),
    [(["no-such-program"], 127), ([sys.executable, "-c", "import os; os.abort()"], 134)],
)
def test_run_bad_endings(runner: CliRunner, tmp_path: Path, command: list[str], code: int) -> None:
    result = runner.invoke(cli.app, ["run", "--log", str(tmp_path / "run.jsonl"), "--", *command])
    assert result.exit_code == code


def test_report_missing_file(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["report", "nope.jsonl"])
    assert result.exit_code == 2


def test_report_command(runner: CliRunner, log_file: Path) -> None:
    result = runner.invoke(cli.app, ["report", str(log_file)])
    assert result.exit_code == 0
    assert "load" in result.output


def test_compare_command(runner: CliRunner, log_file: Path) -> None:
    result = runner.invoke(cli.app, ["compare", str(log_file), str(log_file)])
    assert result.exit_code == 0
    assert "100.0 → 100.0 (+0%)" in result.output


def write_log(path: Path, records: list[dict[str, object]]) -> Path:
    """Записать готовые записи в файл с замерами."""
    sink = make_sink(path)
    for record in records:
        sink(record)
    sink.close()
    return path


def leak_records(stopped: bool, code: int | None = None) -> list[dict[str, object]]:
    """Утечка по 20 МБ/с до лимита 1000 МБ. С меткой stop или без, с кодом выхода или без."""
    records: list[dict[str, object]] = [
        {
            "event": "sample",
            "t": i * 0.5,
            "section": "leak",
            "mem_used_mb": 500 + 10 * i,
            "mem_limit_mb": 1000,
        }
        for i in range(50)
    ]
    if code is not None:
        records.append({"event": "exit", "t": 25, "section": "", "code": code})
    if stopped:
        records.append({"event": "stop", "t": 25, "section": ""})
    return records


@pytest.mark.parametrize(
    ("stopped", "code", "ending"),
    [
        (False, None, "оборвался, код выхода 0"),  # монитор умер вместе с процессом
        (True, 137, "оборвался, код выхода 137"),  # resmon run: программу убили по памяти
        (True, 0, "завершился сам, код выхода 0"),
    ],
)
def test_postmortem(
    runner: CliRunner, tmp_path: Path, stopped: bool, code: int | None, ending: str
) -> None:
    log = write_log(tmp_path / "run.jsonl", leak_records(stopped, code))
    result = runner.invoke(cli.app, ["postmortem", str(log)])
    assert f"Запуск {ending}, последняя секция: leak" in result.output
    # Окно набралось за 5 с, а до конца записи ещё около 20 с
    assert "Прогноз предупредил за 20 с" in result.output


def test_postmortem_without_memory(runner: CliRunner, tmp_path: Path) -> None:
    log = write_log(
        tmp_path / "old.jsonl", [{"event": "sample", "t": 0, "section": "", "rss_mb": 1}]
    )
    result = runner.invoke(cli.app, ["postmortem", str(log)])
    assert result.exit_code == 1


def test_killed_program_is_reported(runner: CliRunner, tmp_path: Path) -> None:
    log = str(tmp_path / "run.jsonl")
    # Программа убивает себя сигналом 9, так её убил бы Docker при нехватке памяти
    program = "import os, time; time.sleep(0.3); os.kill(os.getpid(), 9)"
    result = runner.invoke(cli.app, ["run", "--log", log, "--", sys.executable, "-c", program])
    assert result.exit_code == 137
    result = runner.invoke(cli.app, ["postmortem", log])
    assert "Запуск оборвался, код выхода 137" in result.output
