import json
import logging
import sys
from pathlib import Path

import pytest

from resmon.core import Monitor, Record
from resmon.sink import JsonFormatter, JsonlSink, make_sink, read_records


def test_json_formatter_keeps_extra_and_errors() -> None:
    try:
        raise ValueError("плохо")
    except ValueError:
        fields = {"msg": "упало", "levelname": "ERROR", "step": 3, "exc_info": sys.exc_info()}
        data = json.loads(JsonFormatter().format(logging.makeLogRecord(fields)))
    assert data["message"] == "упало"
    assert data["level"] == "ERROR"
    assert data["step"] == 3
    assert "ValueError: плохо" in data["error"]


@pytest.mark.parametrize("name", ["metrics.jsonl", "metrics.db"])
def test_records_round_trip(monitor: Monitor, tmp_path: Path, name: str) -> None:
    sink = make_sink(tmp_path / name)
    monitor.sinks.append(sink)
    with monitor.section("load"), monitor.section("read"):
        pass
    sink.close()
    assert read_records(tmp_path / name) == monitor.records


def test_rotation_keeps_all_records(monitor: Monitor, tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    # Файл на 1 КБ, чтобы ротация случилась уже на десятке записей
    sink = JsonlSink(path, max_mb=0.001, backups=50)
    monitor.sinks.append(sink)
    for _ in range(30):
        monitor.sample()
    sink.close()
    assert (tmp_path / "metrics.jsonl.1").exists()
    assert read_records(path) == monitor.records


def test_broken_tail_is_skipped(monitor: Monitor, tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = JsonlSink(path)
    monitor.sinks.append(sink)
    monitor.sample()
    sink.close()
    # Так выглядит файл, если процесс убили посреди записи
    with path.open("a") as f:
        f.write('{"event": "sam')
    assert read_records(path) == monitor.records


def test_new_run_after_kill(monitor: Monitor, tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    # Прошлый запуск убили посреди строки, потом в тот же файл пишет новый
    path.write_text('{"event": "sample", "t": 1}\n{"event": "sam')
    sink = JsonlSink(path)
    monitor.sinks.append(sink)
    monitor.mark("start")
    sink.close()
    assert [r["event"] for r in read_records(path)] == ["sample", "start"]


def test_failing_sink_does_not_break_program(monitor: Monitor) -> None:
    def full_disk(record: Record) -> None:
        raise OSError("нет места")

    monitor.sinks.append(full_disk)
    with monitor.section("work"):
        pass
    assert len(monitor.records) == 3
