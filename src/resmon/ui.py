import os
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from resmon.core import Record
from resmon.forecast import HORIZON_S, WINDOW_S, OomForecast
from resmon.report import COLUMNS, Stats, cell, last_run, section_stats
from resmon.sink import read_records


def line_chart(samples: list[Record], title: str, lines: dict[str, str]) -> go.Figure:
    """График метрик во времени: по линии на каждую метрику, которая есть в замерах."""
    # Больше 2000 точек на графике не нужно, а страница от них тормозит
    samples = samples[:: max(len(samples) // 2000, 1)]
    times = [s["t"] - samples[0]["t"] for s in samples]
    figure = go.Figure()
    for key, name in lines.items():
        if any(key in s for s in samples):
            figure.add_scatter(x=times, y=[s.get(key) for s in samples], name=name)
    figure.update_layout(title=title, xaxis_title="секунды от старта", height=320)
    return figure


def icicle(stats: Stats) -> go.Figure:
    """Пламенный граф секций: ширина прямоугольника - время, внутри него вложенные секции."""
    # Знак / в именах секций запрещён, поэтому им можно обозначить корень
    ids = [path or "/" for path in stats]
    parents = ["" if not path else path.rpartition("/")[0] or "/" for path in stats]
    labels = [path.rsplit("/", 1)[-1] if path else "весь запуск" for path in stats]
    times = [row["time_s"] for row in stats.values()]
    # total: время родителя уже включает время детей, складывать их не надо
    icicle = go.Icicle(
        ids=ids,
        parents=parents,
        labels=labels,
        values=times,
        branchvalues="total",
        root_color="lightgrey",
        tiling={"orientation": "v"},  # корень сверху, вложенные секции под ним
        texttemplate="%{label}<br>%{value:.1f} с",
    )
    figure = go.Figure(icicle)
    figure.update_layout(title="Где программа провела время", height=320)
    return figure


def countdown(records: list[Record], forecast: OomForecast, now: float) -> str:
    """Текст плитки с обратным отсчётом. Учитываем, сколько прошло с последнего замера."""
    if any(r["event"] == "stop" for r in records):
        return "запуск окончен"
    age = now - max(r["t"] for r in records if r["event"] == "sample")
    if age > 5:
        return f"замеров нет уже {age:.0f} с"
    if forecast.alarm_since is None:
        return "не грозит"
    return "тревога" if forecast.ttl is None else f"{max(forecast.ttl - age, 0):.0f} с"


@st.fragment(run_every=2)
def live(log: Path) -> None:
    """Перечитать лог и перерисовать страницу. Streamlit вызывает это раз в 2 секунды."""
    if not log.exists():
        st.info(f"Жду файл {log}. Его создаст, например, команда resmon demo")
        return
    try:
        records = last_run(read_records(log))
    except FileNotFoundError:
        # Лог как раз ротируется, перечитаем через 2 секунды
        return
    samples = [r for r in records if r["event"] == "sample"]
    if not samples:
        st.info("В логе пока нет замеров")
        return
    last = samples[-1]
    # Прогнозу хватает последних секунд, старую историю не пересчитываем
    forecast = OomForecast()
    for sample in samples:
        if sample["t"] >= last["t"] - WINDOW_S - HORIZON_S:
            forecast.update(sample)

    cells = st.columns(4)
    cells[0].metric("секция", last["section"] or "-")
    cells[1].metric("CPU, %", f"{last.get('cpu_percent', 0):.0f}")
    if "mem_used_mb" in last:
        used, limit = last["mem_used_mb"] / 1024, last["mem_limit_mb"] / 1024
        cells[2].metric("занято памяти, ГБ", f"{used:.1f} из {limit:.1f}")
    cells[3].metric("до нехватки памяти", countdown(records, forecast, time.time()))

    charts = st.columns(3)
    program = {"rss_mb": "вся программа", "py_mb": "объекты Python"}
    charts[0].plotly_chart(line_chart(samples, "Память программы, МБ", program))
    # Отдельный график, иначе память всей машины сплющивает линию программы
    system = {"mem_used_mb": "занято", "mem_limit_mb": "лимит"}
    charts[1].plotly_chart(line_chart(samples, "Память машины или контейнера, МБ", system))
    load = {"cpu_percent": "CPU", "gpu_percent": "GPU"}
    charts[2].plotly_chart(line_chart(samples, "Загрузка, %", load))

    stats = section_stats(records)
    st.plotly_chart(icicle(stats))
    columns = [k for k in COLUMNS if any(k in row for row in stats.values())]
    rows = [
        {"секция": path or "весь запуск", **{COLUMNS[k]: cell(row.get(k), k) for k in columns}}
        for path, row in stats.items()
    ]
    st.dataframe(rows, hide_index=True)


st.set_page_config(page_title="resmon", layout="wide")
st.title("resmon: живой монитор ресурсов")
live(
    Path(st.sidebar.text_input("Файл с замерами", os.environ.get("RESMON_LOG", "logs/demo.jsonl")))
)
