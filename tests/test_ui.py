import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from resmon import ui
from resmon.core import Record
from resmon.forecast import OomForecast
from resmon.sink import make_sink

LEAK = [500.0] * 20 + [500 + 10 * i for i in range(1, 30)]  # утечка 20 МБ/с


def run(used: list[float], end: float) -> list[Record]:
    """Замеры раз в 0.5 с в секции work, последний в момент end. Лимит 1000 МБ."""
    start = end - (len(used) - 1) * 0.5
    return [
        {
            "event": "sample",
            "t": start + i * 0.5,
            "section": "work",
            "mem_used_mb": value,
            "mem_limit_mb": 1000,
            "cpu_percent": 50,
            "rss_mb": value - 100,
        }
        for i, value in enumerate(used)
    ]


def open_dashboard(monkeypatch: pytest.MonkeyPatch, log: Path) -> AppTest:
    """Открыть страницу дашборда без браузера и дождаться первой отрисовки."""
    monkeypatch.setenv("RESMON_LOG", str(log))
    page = AppTest.from_file(ui.__file__, default_timeout=30).run()
    assert not page.exception
    return page


@pytest.mark.parametrize(("used", "countdown"), [(LEAK, " с"), ([500.0] * 40, "не грозит")])
def test_dashboard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, used: list[float], countdown: str
) -> None:
    sink = make_sink(tmp_path / "run.jsonl")
    for record in [{"event": "start", "t": 0, "section": ""}, *run(used, time.time())]:
        sink(record)
    sink({"event": "section", "section": "work", "start": 0, "end": time.time()})
    sink.close()
    page = open_dashboard(monkeypatch, tmp_path / "run.jsonl")
    metrics = {m.label: m.value for m in page.metric}
    assert metrics["секция"] == "work"
    assert metrics["до нехватки памяти"].endswith(countdown)
    assert page.dataframe[0].value["секция"].tolist() == ["весь запуск", "work"]


@pytest.mark.parametrize(
    ("extra", "age", "expected"),
    [
        ([], 0, " с"),  # тревога идёт, отсчёт от прогноза
        ([], 30, "замеров нет уже 30 с"),  # процесс, похоже, умер
        ([{"event": "stop", "t": 0, "section": ""}], 0, "запуск окончен"),
    ],
)
def test_countdown(extra: list[Record], age: float, expected: str) -> None:
    records = run(LEAK, end=100)
    forecast = OomForecast()
    for record in records:
        forecast.update(record)
    assert ui.countdown(records + extra, forecast, now=100 + age).endswith(expected)


def test_dashboard_waits_for_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    page = open_dashboard(monkeypatch, tmp_path / "nope.jsonl")
    assert "Жду файл" in page.info[0].value


def test_dashboard_on_empty_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path / "run.jsonl"
    log.write_text('{"event": "start", "t": 0, "section": ""}\n')
    page = open_dashboard(monkeypatch, log)
    assert "нет замеров" in page.info[0].value
