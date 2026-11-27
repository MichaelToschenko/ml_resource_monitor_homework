import logging

import numpy as np
import pytest

from resmon.core import Record
from resmon.forecast import OomForecast, fit_trend, lead_time, time_to_limit

TIMES = list(range(10))


def run(used: list[float], limit: float = 1000.0, dt: float = 0.5) -> list[Record]:
    """Записи запуска с заданной занятой памятью в каждом замере."""
    return [
        {"event": "sample", "t": i * dt, "section": "", "mem_used_mb": v, "mem_limit_mb": limit}
        for i, v in enumerate(used)
    ]


def leak_run(flat: int = 40, leak: int = 50, rate: float = 10.0) -> list[Record]:
    """20 с память стоит на 500 МБ, потом утекает по rate МБ за замер до лимита 1000 МБ."""
    return run([500.0] * flat + [500 + rate * i for i in range(1, leak + 1)])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([100 + 10 * t for t in TIMES], 11.0),  # 190 МБ и растёт на 10 в секунду до 300
        ([250 - 10 * t for t in TIMES], None),  # память уходит
        ([100.0] * 10, None),  # не меняется
        ([100 + 30 * t for t in TIMES], 0.0),  # уже на лимите
    ],
)
def test_linear_forecast(values: list[float], expected: float | None) -> None:
    ttl = time_to_limit(fit_trend(TIMES, values), limit=300)
    assert ttl == (None if expected is None else pytest.approx(expected))


def test_parabola_sees_acceleration() -> None:
    values = [100 + t**2 for t in TIMES]
    exact = np.sqrt(200) - 9  # 100 + t^2 дойдёт до 300 в момент корень из 200
    parabola = time_to_limit(fit_trend(TIMES, values, degree=2), limit=300)
    line = time_to_limit(fit_trend(TIMES, values, degree=1), limit=300)
    assert parabola == pytest.approx(exact)
    assert line is not None and parabola is not None
    # Прямая не видит разгона и обещает больше времени, чем есть
    assert line > parabola


def test_sink_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    forecast = OomForecast()
    with caplog.at_level(logging.WARNING):
        for record in leak_run():
            forecast(record)
    # Тревога включилась один раз и не повторяется на каждом замере
    assert caplog.text.count("память кончится") == 1
    # В окне только последние 10 секунд: 21 замер через 0.5 с
    assert len(forecast.points) == 21


def test_sink_ignores_other_records() -> None:
    forecast = OomForecast()
    forecast({"event": "section", "section": "a", "start": 0, "end": 1})
    forecast({"event": "sample", "t": 0, "section": "", "cpu_percent": 10})
    assert not forecast.points


@pytest.mark.parametrize("degree", [1, 2])
def test_lead_time_beats_threshold(degree: int) -> None:
    lead = lead_time(leak_run(), degree)
    # Порог 90 % сработал бы на 900 МБ, то есть за 5 с до конца
    assert lead is not None and lead >= 2 * 5


@pytest.mark.parametrize("seed", range(20))
def test_no_alarm_on_flat_noise(seed: int) -> None:
    rng = np.random.default_rng(seed)
    assert lead_time(run(list(500 + rng.normal(0, 5, 100)))) is None


@pytest.mark.parametrize(
    "used",
    [
        [300.0] * 20 + [800.0] * 40,  # модель загрузилась одним скачком
        [300.0] * 20 + [600.0] * 6 + [800.0] * 40,  # два скачка подряд
        [300.0] * 20 + [300 + 50 * i for i in range(1, 7)] + [600.0] * 40,  # 300 МБ за 3 с
    ],
)
def test_loading_is_not_a_leak(used: list[float]) -> None:
    forecast = OomForecast()
    assert not any(forecast.update(r) for r in run(used, limit=2000))


def test_alarm_does_not_blink_near_limit(caplog: pytest.LogCaptureFixture) -> None:
    rng = np.random.default_rng(0)
    # Сначала утечка до 950 из 1000 МБ, потом долго стоим у самого лимита
    used = [500 + 10 * i for i in range(46)] + list(950 + rng.normal(0, 5, 400))
    forecast = OomForecast()
    with caplog.at_level(logging.WARNING):
        for record in run(used):
            forecast(record)
    assert caplog.text.count("память кончится") <= 2


def test_last_jump_keeps_alarm() -> None:
    # Утечка, а перед смертью ещё один большой кусок памяти на 150 МБ
    leak = [500 + 10 * i for i in range(1, 41)]
    lead = lead_time(run([500.0] * 40 + leak + [leak[-1] + 150]))
    assert lead is not None and lead > 10
