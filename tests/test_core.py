import gc
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from resmon.core import (
    MB,
    Collector,
    GcCollector,
    GpuCollector,
    Monitor,
    ProcessCollector,
    PythonMemoryCollector,
    Record,
    SystemCollector,
    default_collectors,
)


def sections(monitor: Monitor) -> list[str]:
    """Имена закрытых секций по порядку."""
    return [r["section"] for r in monitor.records if r["event"] == "section"]


def test_nested_sections_build_paths(monitor: Monitor) -> None:
    with monitor.section("load"), monitor.section("read"):
        pass
    # Вложенная секция закрывается первой
    assert sections(monitor) == ["load/read", "load"]
    samples = [r["section"] for r in monitor.records if r["event"] == "sample"]
    assert samples == ["", "load", "load/read", "load"]


def test_decorator_keeps_result(monitor: Monitor) -> None:
    @monitor("embed")
    def double(x: int) -> int:
        return x * 2

    assert double(21) == 42
    assert sections(monitor) == ["embed"]


def test_section_closes_on_error(monitor: Monitor) -> None:
    with pytest.raises(ValueError), monitor.section("broken"):
        raise ValueError
    assert sections(monitor) == ["broken"]
    assert monitor.sample()["section"] == ""


def test_sinks_get_every_record(monitor: Monitor) -> None:
    got: list[Record] = []
    monitor.sinks.append(got.append)
    with monitor.section("a"):
        pass
    assert got == monitor.records
    assert got[0]["rss_mb"] == 100.0


class BrokenCollector(Collector):
    """Сборщик, который всегда падает."""

    def collect(self) -> dict[str, float]:
        raise RuntimeError("сломался")


def test_broken_collector_does_not_stop_monitor(monitor: Monitor) -> None:
    monitor.collectors.insert(0, BrokenCollector())
    with monitor.section("a"):
        pass
    assert monitor.records[-2]["rss_mb"] == 100.0


def test_start_and_stop_twice(monitor: Monitor) -> None:
    monitor.stop()
    monitor.start()
    monitor.start()
    monitor.stop()
    monitor.stop()
    # Метка старта, замер при старте и замер при остановке
    assert [r["event"] for r in monitor.records] == ["start", "sample", "sample"]


def test_slash_in_section_name(monitor: Monitor) -> None:
    with pytest.raises(ValueError), monitor.section("a/b"):
        pass


def test_background_thread_samples(monitor: Monitor) -> None:
    monitor.interval = 0.01
    with monitor:
        time.sleep(0.2)
    assert len(monitor.records) >= 5


class FakeProcess:
    """Подделка psutil.Process с фиксированными CPU и памятью."""

    def __init__(self, pid: int, kids: int = 0) -> None:
        self.pid = pid
        self.kids = kids

    def children(self, recursive: bool) -> list["FakeProcess"]:
        return [FakeProcess(self.pid + i + 1) for i in range(self.kids)]

    def cpu_percent(self) -> float:
        return 10.0

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=100 * MB)


@pytest.mark.parametrize("kids", [0, 1, 3])
def test_process_counts_children(monkeypatch: pytest.MonkeyPatch, kids: int) -> None:
    monkeypatch.setattr(psutil, "Process", lambda pid: FakeProcess(1, kids))
    metrics = ProcessCollector().collect()
    assert metrics == {"cpu_percent": 10.0 * (kids + 1), "rss_mb": 100.0 * (kids + 1)}


def test_system_metrics() -> None:
    assert {"ram_percent", "disk_read_mb", "net_recv_mb"} <= SystemCollector().collect().keys()


def test_real_allocation_is_visible() -> None:
    monitor = Monitor(collectors=[ProcessCollector(), PythonMemoryCollector()])
    with monitor:
        with monitor.section("greedy"):
            data = b"x" * (200 * MB)
        del data
    samples = [r for r in monitor.records if r["event"] == "sample"]
    # Первый замер с меткой greedy - выход из секции, замер перед ним - вход
    i = next(i for i, r in enumerate(samples) if r["section"] == "greedy")
    entry, leave = samples[i - 1], samples[i]
    assert leave["py_peak_mb"] > 190
    assert leave["rss_mb"] - entry["rss_mb"] > 150
    # После del память Python снова маленькая
    assert monitor.records[-1]["py_mb"] < 50


def test_gc_collector_counts_collections() -> None:
    collector = GcCollector()
    collector.start()
    before = collector.collect()
    gc.collect()
    after = collector.collect()
    collector.stop()
    assert after["gc_gen2"] == before["gc_gen2"] + 1
    assert after["gc_ms"] > 0


class NVMLError(Exception):
    pass


def fake_pynvml(driver: bool) -> SimpleNamespace:
    """Подделка pynvml: с драйвером или без."""

    def init() -> None:
        if not driver:
            raise NVMLError

    return SimpleNamespace(
        NVMLError=NVMLError,
        nvmlInit=init,
        nvmlDeviceGetHandleByIndex=lambda i: "gpu0",
        nvmlDeviceGetUtilizationRates=lambda h: SimpleNamespace(gpu=70),
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(used=512 * MB),
    )


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        (None, None),
        (fake_pynvml(driver=False), None),
        (fake_pynvml(driver=True), {"gpu_percent": 70.0, "gpu_mem_mb": 512.0}),
    ],
)
def test_gpu_detect(
    monkeypatch: pytest.MonkeyPatch,
    module: SimpleNamespace | None,
    expected: dict[str, float] | None,
) -> None:
    # None в sys.modules заставляет import упасть с ImportError
    monkeypatch.setitem(sys.modules, "pynvml", module)
    collector = GpuCollector.detect()
    assert (collector.collect() if collector else None) == expected


@pytest.mark.parametrize(("pid", "has_python"), [(None, True), (1, False)])
def test_default_collectors(
    monkeypatch: pytest.MonkeyPatch, pid: int | None, has_python: bool
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(psutil, "Process", lambda pid: FakeProcess(1))
    kinds = {type(c) for c in default_collectors(pid)}
    assert (GcCollector in kinds) == has_python
    assert (PythonMemoryCollector in kinds) == has_python
