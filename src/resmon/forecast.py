import logging
from collections import deque
from collections.abc import Sequence

import numpy as np

from resmon.core import Record

log = logging.getLogger(__name__)

WINDOW_S = 10.0  # по скольким последним секундам строим тренд
HORIZON_S = 3 * WINDOW_S  # дальше трёх окон тренд вперёд не тянем, предупреждаем внутри
JUMP_MB = 100.0  # настолько точка может уйти от тренда, дальше это уже скачок


def fit_trend(times: Sequence[float], values: Sequence[float], degree: int = 1) -> np.ndarray:
    """Прямая или парабола по точкам. Время считаем от последней точки, так проще дальше."""
    return np.polyfit(np.asarray(times) - times[-1], values, degree)


def time_to_limit(trend: np.ndarray, limit: float) -> float | None:
    """Через сколько секунд тренд дойдёт до лимита. None, если память сейчас не растёт."""
    # Последний коэффициент - значение сейчас, предпоследний - скорость роста сейчас.
    # Рост меньше 1e-9 МБ/с - это погрешность вычислений, на x86 у ровной линии она бывает > 0
    if trend[-2] <= 1e-9:
        return None
    if trend[-1] >= limit:
        return 0.0
    shifted = np.array(trend, dtype=float)
    shifted[-1] -= limit
    future = [root.real for root in np.roots(shifted) if abs(root.imag) < 1e-9 and root.real > 0]
    return min(future, default=None)


class OomForecast:
    """Sink для монитора: следит за трендом занятой памяти и предупреждает о скорой нехватке."""

    def __init__(self, degree: int = 1, window_s: float = WINDOW_S, horizon_s: float = HORIZON_S):
        self.degree = degree
        self.window_s = window_s
        self.horizon_s = horizon_s
        self.points: deque[tuple[float, float]] = deque()
        self.trend: np.ndarray | None = None
        self.ttl: float | None = None
        self.alarm_since: float | None = None
        self.quiet_since: float | None = None

    def update(self, record: Record) -> bool:
        """Учесть замер, пересчитать прогноз. Вернуть True, если тревога только что включилась."""
        if record["event"] != "sample" or not {"mem_used_mb", "mem_limit_mb"} <= record.keys():
            return False
        t, used = record["t"], record["mem_used_mb"]
        # Удивление: точка ушла от тренда, значит был скачок, а не утечка. Окно начинаем заново.
        # Пока тренда нет, ждём точку рядом с прошлой
        if self.points:
            last_t, last_used = self.points[-1]
            expected = last_used if self.trend is None else np.polyval(self.trend, t - last_t)
            if abs(expected - used) > JUMP_MB:
                self.points.clear()
        self.points.append((t, used))
        while t - self.points[0][0] > self.window_s:
            self.points.popleft()

        self.trend, self.ttl = None, None
        # Пока окно не набралось хотя бы наполовину, тренд слишком случайный
        if t - self.points[0][0] >= self.window_s / 2 and len(self.points) > self.degree + 2:
            times, values = zip(*self.points, strict=True)
            self.trend = fit_trend(times, values, self.degree)
            self.ttl = time_to_limit(self.trend, record["mem_limit_mb"])

        if self.ttl is not None and self.ttl < self.horizon_s:
            self.quiet_since = None
            if self.alarm_since is None:
                self.alarm_since = t
                return True
            return False
        # Тревогу снимаем, только если прогноз молчит целое окно. Иначе у лимита она мигает
        if self.quiet_since is None:
            self.quiet_since = t
        if t - self.quiet_since >= self.window_s:
            self.alarm_since = None
        return False

    def __call__(self, record: Record) -> None:
        """Учесть замер и при новой тревоге написать предупреждение в лог."""
        if self.update(record) and self.ttl is not None:
            extra = {"ttl_s": round(self.ttl, 1), "mem_used_mb": round(record["mem_used_mb"])}
            log.warning("память кончится примерно через %.0f с", self.ttl, extra=extra)


def lead_time(records: list[Record], degree: int = 1, horizon_s: float = HORIZON_S) -> float | None:
    """За сколько секунд до конца записи прогноз начал предупреждать и уже не замолкал."""
    forecast = OomForecast(degree, horizon_s=horizon_s)
    for record in records:
        forecast.update(record)
    samples = [r for r in records if r["event"] == "sample"]
    return None if forecast.alarm_since is None else samples[-1]["t"] - forecast.alarm_since
