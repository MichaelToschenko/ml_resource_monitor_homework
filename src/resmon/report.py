from collections import Counter, defaultdict
from itertools import pairwise
from statistics import fmean

from rich.markup import escape
from rich.table import Table

from resmon.core import Record

Stats = dict[str, dict[str, float]]

# Как сводить метрику по секции: пик, среднее или прирост счётчика
PEAKS = ("rss_mb", "py_peak_mb", "gpu_mem_mb")
MEANS = ("cpu_percent", "gpu_percent")
DELTAS = ("gc_ms", "disk_read_mb", "disk_write_mb", "net_recv_mb", "net_sent_mb")

COLUMNS = {
    "calls": "вызовы",
    "time_s": "время, с",
    "share": "доля, %",
    "cpu_percent": "CPU, %",
    "gpu_percent": "GPU, %",
    "rss_mb": "RAM пик, МБ",
    "py_peak_mb": "Python пик, МБ",
    "gpu_mem_mb": "видеопамять пик, МБ",
    "gc_ms": "GC, мс",
    "disk_read_mb": "диск чтение, МБ",
    "disk_write_mb": "диск запись, МБ",
    "net_recv_mb": "сеть приём, МБ",
    "net_sent_mb": "сеть отдача, МБ",
}


def last_run(records: list[Record]) -> list[Record]:
    """Записи последнего запуска: в одном файле их может быть несколько."""
    starts = [i for i, r in enumerate(records) if r["event"] == "start"]
    return records[starts[-1] if starts else 0 :]


def section_stats(records: list[Record]) -> Stats:
    """Итоги по секциям последнего запуска в порядке дерева. Секция "" - это весь запуск."""
    records = last_run(records)
    samples = [r for r in records if r["event"] == "sample"]
    calls = Counter(r["section"] for r in records if r["event"] == "section")
    # Для каждой метрики храним пары (значение, длина отрезка)
    values: dict[str, defaultdict[str, list[tuple[float, float]]]] = {}
    first_seen: dict[str, int] = {}
    # Замер подводит итог отрезку с прошлого замера, отрезок идёт его секции и всем её родителям
    for i, (prev, cur) in enumerate(pairwise(samples)):
        parts = cur["section"].split("/") if cur["section"] else []
        for depth in range(len(parts) + 1):
            path = "/".join(parts[:depth])
            first_seen.setdefault(path, i)
            acc = values.setdefault(path, defaultdict(list))
            dt = cur["t"] - prev["t"]
            acc["time_s"].append((dt, dt))
            for key in PEAKS + MEANS:
                if key in cur:
                    acc[key].append((cur[key], dt))
            for key in DELTAS:
                if key in cur and key in prev:
                    acc[key].append((cur[key] - prev[key], dt))

    run_s = sum(v for v, _ in values[""]["time_s"]) if values else 0.0
    stats: Stats = {}
    for path, acc in values.items():
        row = {"time_s": sum(v for v, _ in acc["time_s"])}
        row["share"] = 100 * row["time_s"] / run_s if run_s else 0.0
        # У незакрытой секции (процесс убили внутри неё) записи о вызове нет
        if calls[path]:
            row["calls"] = calls[path]
        row |= {k: max(v for v, _ in acc[k]) for k in PEAKS if acc[k]}
        row |= {k: weighted_mean(acc[k]) for k in MEANS if acc[k]}
        row |= {k: sum(v for v, _ in acc[k]) for k in DELTAS if acc[k]}
        stats[path] = row

    def tree_key(path: str) -> list[int]:
        """Секции идут по первому появлению, а дети сразу после своего родителя."""
        parts = path.split("/") if path else []
        return [first_seen["/".join(parts[: i + 1])] for i in range(len(parts))]

    return {path: stats[path] for path in sorted(stats, key=tree_key)}


def weighted_mean(pairs: list[tuple[float, float]]) -> float:
    """Среднее по времени: короткие отрезки на входе и выходе секций весят меньше длинных."""
    values, weights = zip(*pairs, strict=True)
    return fmean(values, weights) if sum(weights) else fmean(values)


def cell(value: float | None, key: str) -> str:
    """Число для таблицы: вызовы целым, остальное с одним знаком после запятой."""
    if value is None:
        return "-"
    return f"{value:.0f}" if key == "calls" else f"{value:.1f}"


def stats_table(stats: Stats, title: str = "Отчёт по секциям") -> Table:
    """Таблица для терминала. Колонки без данных (например, GPU на ноутбуке) не показываем."""
    columns = [k for k in COLUMNS if any(k in row for row in stats.values())]
    table = Table("секция", *(COLUMNS[k] for k in columns), title=title)
    for path, row in stats.items():
        name = "  " * path.count("/") + path.rsplit("/", 1)[-1] if path else "весь запуск"
        # escape: квадратные скобки в имени секции rich иначе принял бы за разметку
        table.add_row(escape(name), *(cell(row.get(k), k) for k in columns))
    return table


def compare_rows(old: Stats, new: Stats) -> list[dict[str, str]]:
    """Сравнение двух запусков по общим секциям: было, стало и изменение в процентах."""
    keys = ["time_s", "cpu_percent", "rss_mb", "py_peak_mb", "gpu_mem_mb"]
    keys = [k for k in keys if any(k in row for run in (old, new) for row in run.values())]
    rows = []
    for path in (p for p in new if p in old):
        row = {"секция": path or "весь запуск"}
        for key in keys:
            a, b = old[path].get(key), new[path].get(key)
            change = f" ({(b - a) / a:+.0%})" if a and b is not None else ""
            row[COLUMNS[key]] = f"{cell(a, key)} → {cell(b, key)}{change}"
        rows.append(row)
    return rows


def compare_table(old: Stats, new: Stats) -> Table:
    """Сравнение двух запусков в виде таблицы для терминала."""
    rows = compare_rows(old, new)
    table = Table(*(rows[0] if rows else ["секция"]), title="Сравнение запусков: было → стало")
    for row in rows:
        table.add_row(*(escape(value) for value in row.values()))
    return table
