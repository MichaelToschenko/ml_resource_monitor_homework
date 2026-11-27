import gc
import time
from collections.abc import Callable
from functools import partial
from statistics import mean, median, stdev

import numpy as np
from rich.console import Console

from resmon.core import GcCollector, Monitor, Record, default_collectors
from resmon.forecast import OomForecast, lead_time

console = Console(highlight=False, soft_wrap=True)
LIMIT = 3000.0  # лимит памяти в сценариях прогноза, МБ
SEEDS = range(10)


def table(header: list[str], rows: list[list[str]]) -> None:
    """Напечатать таблицу в формате markdown, чтобы сразу вставить её в README."""
    lines = [header, ["---"] * len(header), *rows]
    console.print("\n".join("| " + " | ".join(line) + " |" for line in lines) + "\n", markup=False)


def workload() -> float:
    """Нагрузка для замеров: 10 раз по миллиону мелких словарей. Возвращает время, с."""
    start = time.perf_counter()
    for _ in range(10):
        rows = [{"id": i, "value": i * 0.5} for i in range(1_000_000)]
        sum(row["value"] for row in rows)
    return time.perf_counter() - start


def timed(make_monitor: Callable[[], Monitor] | None, repeats: int = 3) -> float:
    """Медиана времени нагрузки под монитором или без него."""
    times = []
    for _ in range(repeats):
        if make_monitor is None:
            times.append(workload())
            continue
        with make_monitor():
            times.append(workload())
    return median(times)


def light_monitor(interval: float) -> Monitor:
    """Монитор без tracemalloc, чтобы мерить цену только самих замеров."""
    return Monitor(interval, collectors=default_collectors(trace_python=False))


def overhead() -> None:
    """Вопрос 1: насколько монитор замедляет программу в зависимости от частоты замеров."""
    base = timed(None)
    modes: list[tuple[str, Callable[[], Monitor] | None]] = [
        ("без монитора", None),
        *[(f"замер раз в {s} с", partial(light_monitor, s)) for s in (1.0, 0.1, 0.01)],
        ("раз в 0.1 с и tracemalloc", partial(Monitor, 0.1)),
    ]
    rows = []
    for name, make in modes:
        seconds = base if make is None else timed(make)
        rows.append([name, f"{seconds:.2f}", f"{100 * (seconds / base - 1):+.0f}"])
    table(["режим", "время, с", "замедление, %"], rows)


def make_objects(count: int) -> list[list[int]]:
    """Много мелких списков. Списки - контейнеры, сборщик мусора за ними следит."""
    return [[i] for i in range(count)]


def gc_cost(make: Callable[[], object]) -> list[str]:
    """Время создания объектов, число сборок мусора и время на них по GcCollector из resmon."""
    collector = GcCollector()
    collector.start()
    before, start = collector.collect(), time.perf_counter()
    kept = make()
    seconds, after = time.perf_counter() - start, collector.collect()
    collector.stop()
    del kept
    runs = sum(after[f"gc_gen{i}"] - before[f"gc_gen{i}"] for i in range(3))
    return [f"{seconds:.2f}", f"{runs:.0f}", f"{after['gc_ms'] - before['gc_ms']:.0f}"]


def gc_growth() -> None:
    """Вопрос 2: как сборщик мусора ведёт себя, когда объектов становится миллионы."""
    rows = []
    for count in (1_000_000, 3_000_000, 5_000_000):
        rows.append([f"{count // 1_000_000} млн", *gc_cost(partial(make_objects, count))])
        gc.collect()
    table(["объектов", "время, с", "сборок мусора", "время сборок, мс"], rows)


def gc_tricks() -> None:
    """Вопрос 3: помогают ли gc.freeze() и выключение сборщика на горячем участке."""
    rows = []
    for name in ("как есть", "gc.disable() на горячем участке", "gc.freeze() после загрузки"):
        # Каждый приём начинаем с одинаковой чистой памяти, иначе сравнение нечестное
        gc.collect()
        # Долгоживущие данные, например загруженный датасет, а потом горячий участок
        base = make_objects(5_000_000)
        if name.startswith("gc.freeze"):
            gc.freeze()
        if name.startswith("gc.disable"):
            gc.disable()
        collector = GcCollector()
        collector.start()
        before, start = collector.collect(), time.perf_counter()
        hot = make_objects(2_000_000)
        hot_s = time.perf_counter() - start
        gc.enable()
        # Программа живёт дальше, и отложенная уборка случается уже здесь
        tail = make_objects(100_000)
        total_s, after = time.perf_counter() - start, collector.collect()
        collector.stop()
        full = after["gc_gen2"] - before["gc_gen2"]
        rows.append([name, f"{hot_s:.2f}", f"{total_s:.2f}", f"{full:.0f}"])
        gc.unfreeze()
        del base, hot, tail
        gc.collect()
    table(["приём", "горячий участок, с", "вместе с уборкой после, с", "полных уборок"], rows)


def scenario(kind: str, seed: int) -> list[Record]:
    """Занятая память раз в 0.5 с: фон 1500 МБ и шум 10 МБ. Запись обрывается на лимите."""
    t = np.arange(0, 120, 0.5)
    after_20s = np.clip(t - 20, 0, None)
    used = {
        "линейная утечка 20 МБ/с": 1500 + 20 * after_20s,
        "ускоряющаяся утечка": 1500 + 0.5 * after_20s**2,
        "пила: батчи и сборка мусора": 1500 + 400 * (t % 20) / 20,
        "скачок: загрузилась модель": 1500 + 800 * (t >= 30),
        "ровная память": np.full_like(t, 1500.0),
    }[kind] + np.random.default_rng(seed).normal(0, 10, t.size)
    if (used >= LIMIT).any():
        used = used[: int(np.argmax(used >= LIMIT)) + 1]
    return [
        {
            "event": "sample",
            "t": float(ti),
            "section": "",
            "mem_used_mb": float(u),
            "mem_limit_mb": LIMIT,
        }
        for ti, u in zip(t, used, strict=False)
    ]


def threshold_lead(records: list[Record]) -> float | None:
    """Порог 90 %: за сколько секунд до конца память впервые перешла 90 % лимита."""
    over = [r["t"] for r in records if r["mem_used_mb"] >= 0.9 * LIMIT]
    return records[-1]["t"] - over[0] if over else None


def forecast_quality() -> None:
    """Проверка изюминки: прогноз против порога 90 % на сценариях с падением и без."""
    methods: dict[str, Callable[[list[Record]], float | None]] = {
        "порог 90 %": threshold_lead,
        "прямая": lambda records: lead_time(records, degree=1),
        "парабола": lambda records: lead_time(records, degree=2),
    }
    rows = []
    for kind in ("линейная утечка 20 МБ/с", "ускоряющаяся утечка"):
        cells = []
        for method in methods.values():
            leads = [method(scenario(kind, seed)) or 0.0 for seed in SEEDS]
            cells.append(f"за {mean(leads):.1f} ± {stdev(leads):.1f} с")
        rows.append([kind, *cells])
    for kind in ("пила: батчи и сборка мусора", "скачок: загрузилась модель", "ровная память"):
        cells = [f"{false_alarms(kind, method)} из {len(SEEDS)}" for method in methods]
        rows.append([f"{kind} (ложные тревоги)", *cells])
    table(["сценарий", *methods], rows)


def false_alarms(kind: str, method: str) -> int:
    """Сколько прогонов без падения подняли хотя бы одну тревогу."""
    count = 0
    for seed in SEEDS:
        records = scenario(kind, seed)
        if method == "порог 90 %":
            count += threshold_lead(records) is not None
        else:
            forecast = OomForecast(degree=1 if method == "прямая" else 2)
            count += any(forecast.update(r) for r in records)
    return count


if __name__ == "__main__":
    for title, experiment in [
        ("Цена замеров", overhead),
        ("Сборщик мусора и миллионы объектов", gc_growth),
        ("gc.freeze() и выключение сборщика", gc_tricks),
        ("Прогноз нехватки памяти против порога", forecast_quality),
    ]:
        console.print(f"### {title}\n")
        experiment()
