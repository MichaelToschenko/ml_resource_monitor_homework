import pytest
from rich.console import Console
from rich.table import Table

from resmon.core import Monitor, Record
from resmon.report import compare_table, section_stats, stats_table


def sample(t: float, section: str = "", **metrics: float) -> Record:
    """Запись замера в том виде, как её пишет монитор."""
    return {"event": "sample", "t": t, "section": section, **metrics}


def section(path: str, start: float, end: float) -> Record:
    """Запись о закрытой секции."""
    return {"event": "section", "section": path, "start": start, "end": end}


@pytest.fixture
def records() -> list[Record]:
    """Запуск на 5 секунд: секция load, внутри неё read."""
    return [
        sample(0, rss_mb=100, cpu_percent=10, gc_ms=0),
        sample(1, rss_mb=120, cpu_percent=20, gc_ms=1),
        sample(2, "load", rss_mb=200, cpu_percent=30, gc_ms=3),
        sample(3, "load/read", rss_mb=300, cpu_percent=50, gc_ms=8),
        section("load/read", 2, 3),
        sample(4, "load", rss_mb=150, cpu_percent=40, gc_ms=9),
        section("load", 1, 4),
        sample(5, rss_mb=100, cpu_percent=0, gc_ms=9),
    ]


def text(table: Table) -> str:
    """Таблица в виде текста, как её увидит человек в терминале."""
    console = Console(record=True, width=200)
    console.print(table)
    return console.export_text()


def test_nested_sections(records: list[Record]) -> None:
    stats = section_stats(records)
    # Весь запуск первым, родитель перед ребёнком и включает его пик
    assert list(stats) == ["", "load", "load/read"]
    assert stats[""] == pytest.approx(
        {"time_s": 5, "share": 100, "rss_mb": 300, "cpu_percent": 28, "gc_ms": 9}
    )
    assert stats["load"] == pytest.approx(
        {"calls": 1, "time_s": 3, "share": 60, "rss_mb": 300, "cpu_percent": 40, "gc_ms": 8}
    )
    assert stats["load/read"] == pytest.approx(
        {"calls": 1, "time_s": 1, "share": 20, "rss_mb": 300, "cpu_percent": 50, "gc_ms": 5}
    )


def test_repeated_calls_add_up() -> None:
    records = [
        sample(0),
        sample(1, "embed", rss_mb=150),
        section("embed", 0, 1),
        sample(2),
        sample(3, "embed", rss_mb=250),
        section("embed", 2, 3),
    ]
    stats = section_stats(records)["embed"]
    assert stats["calls"] == 2
    assert stats["time_s"] == 2
    assert stats["rss_mb"] == 250


def test_killed_process_keeps_open_section() -> None:
    # Процесс убили внутри train: записи о закрытии train нет
    records = [
        sample(0),
        sample(1, "train/epoch", rss_mb=1000),
        section("train/epoch", 0, 1),
        sample(2, "train", rss_mb=5000),
    ]
    stats = section_stats(records)
    assert list(stats) == ["", "train", "train/epoch"]
    assert "calls" not in stats["train"]
    assert stats["train"]["rss_mb"] == 5000


@pytest.mark.parametrize(("header", "shown"), [("RAM пик", True), ("видеопамять", False)])
def test_table_hides_empty_columns(records: list[Record], header: str, shown: bool) -> None:
    out = text(stats_table(section_stats(records)))
    assert (header in out) == shown
    assert "весь запуск" in out
    assert "  read" in out


def test_compare_shows_change(records: list[Record]) -> None:
    old = section_stats(records)
    new = {path: {**row, "time_s": row["time_s"] * 1.5} for path, row in old.items()}
    out = text(compare_table(old, new))
    assert "3.0 → 4.5 (+50%)" in out


def test_report_from_real_monitor(monitor: Monitor) -> None:
    with monitor.section("a"), monitor.section("b"):
        pass
    stats = section_stats(monitor.records)
    assert list(stats) == ["", "a", "a/b"]
    assert stats["a/b"]["rss_mb"] == 100


def test_empty_log() -> None:
    assert section_stats([]) == {}


def test_cpu_mean_is_weighted_by_time() -> None:
    # 10 секунд на 100 %, потом короткая секция на 0 %
    records = [sample(0), sample(10, cpu_percent=100), sample(10.001, "short", cpu_percent=0)]
    assert section_stats(records)[""]["cpu_percent"] == pytest.approx(100, abs=0.1)


def test_only_last_run_is_reported() -> None:
    first_run = [{"event": "start", "t": 0, "section": ""}, sample(0), sample(1000, rss_mb=900)]
    last_run = [{"event": "start", "t": 2000, "section": ""}, sample(2000), sample(2001, rss_mb=50)]
    stats = section_stats(first_run + last_run)[""]
    assert stats["time_s"] == 1
    assert stats["rss_mb"] == 50


def test_compare_takes_columns_from_both_runs(records: list[Record]) -> None:
    old = section_stats(records)
    new = {path: {**row, "gpu_mem_mb": 512.0} for path, row in old.items()}
    assert "- → 512.0" in text(compare_table(old, new))
