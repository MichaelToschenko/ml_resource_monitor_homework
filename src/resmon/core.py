import contextlib
import functools
import gc
import logging
import threading
import time
import tracemalloc
from collections.abc import Callable, Iterable, Iterator
from types import ModuleType
from typing import Any, ParamSpec, Self, TypeVar

import psutil

log = logging.getLogger(__name__)

MB = 1024 * 1024
Record = dict[str, Any]
Sink = Callable[[Record], None]
P = ParamSpec("P")
R = TypeVar("R")


class Collector:
    """Источник метрик (паттерн Strategy): монитор опрашивает все источники одинаково."""

    def start(self) -> None:
        """Подготовиться к замерам."""

    def stop(self) -> None:
        """Убрать за собой."""

    def collect(self) -> dict[str, float]:
        """Вернуть текущие значения метрик."""
        raise NotImplementedError


class ProcessCollector(Collector):
    """CPU и память процесса вместе с его дочерними процессами."""

    def __init__(self, pid: int | None = None) -> None:
        self.process = psutil.Process(pid)
        self._known: dict[int, psutil.Process] = {}

    def collect(self) -> dict[str, float]:
        try:
            procs = [self.process, *self.process.children(recursive=True)]
        except psutil.NoSuchProcess:
            return {}
        # psutil считает CPU от прошлого вызова, поэтому храним старые объекты живых процессов
        self._known = {p.pid: self._known.get(p.pid, p) for p in procs}
        cpu = rss = 0.0
        for proc in self._known.values():
            with contextlib.suppress(psutil.NoSuchProcess):
                proc_cpu, proc_rss = proc.cpu_percent(), proc.memory_info().rss
                cpu += proc_cpu
                rss += proc_rss
        return {"cpu_percent": cpu, "rss_mb": rss / MB}


class SystemCollector(Collector):
    """Память всей машины и счётчики диска и сети с момента загрузки системы."""

    def collect(self) -> dict[str, float]:
        metrics = {"ram_percent": psutil.virtual_memory().percent}
        if disk := psutil.disk_io_counters():
            metrics |= {
                "disk_read_mb": disk.read_bytes / MB,
                "disk_write_mb": disk.write_bytes / MB,
            }
        if net := psutil.net_io_counters():
            metrics |= {"net_sent_mb": net.bytes_sent / MB, "net_recv_mb": net.bytes_recv / MB}
        return metrics


class GpuCollector(Collector):
    """Загрузка и память первой видеокарты NVIDIA."""

    def __init__(self, nvml: ModuleType) -> None:
        self.nvml = nvml
        self.handle = nvml.nvmlDeviceGetHandleByIndex(0)

    @classmethod
    def detect(cls) -> Self | None:
        """Вернуть сборщик, если есть pynvml и драйвер NVIDIA, иначе None."""
        try:
            import pynvml
        except ImportError:
            return None
        try:
            pynvml.nvmlInit()
            return cls(pynvml)
        except pynvml.NVMLError:
            return None

    def collect(self) -> dict[str, float]:
        util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {"gpu_percent": float(util.gpu), "gpu_mem_mb": mem.used / MB}


class PythonMemoryCollector(Collector):
    """Память под объекты Python через tracemalloc. Заметно замедляет программу."""

    def start(self) -> None:
        tracemalloc.start()

    def stop(self) -> None:
        tracemalloc.stop()

    def collect(self) -> dict[str, float]:
        current, peak = tracemalloc.get_traced_memory()
        # Сбрасываем пик, чтобы в каждом замере был пик именно с прошлого замера
        tracemalloc.reset_peak()
        return {"py_mb": current / MB, "py_peak_mb": peak / MB}


class GcCollector(Collector):
    """Число сборок мусора по поколениям и общее время, которое они заняли."""

    def __init__(self) -> None:
        self.gc_ms = 0.0
        self._started = 0.0

    def _on_gc(self, phase: str, info: dict[str, int]) -> None:
        """Python зовёт это перед сборкой мусора и после неё, info нам не нужен."""
        if phase == "start":
            self._started = time.perf_counter()
        else:
            self.gc_ms += (time.perf_counter() - self._started) * 1000

    def start(self) -> None:
        gc.callbacks.append(self._on_gc)

    def stop(self) -> None:
        gc.callbacks.remove(self._on_gc)

    def collect(self) -> dict[str, float]:
        counts = {f"gc_gen{i}": float(s["collections"]) for i, s in enumerate(gc.get_stats())}
        return counts | {"gc_ms": self.gc_ms}


def default_collectors(pid: int | None = None, trace_python: bool = True) -> list[Collector]:
    """Набор сборщиков. Память Python и сборщик мусора видны только в своём процессе."""
    collectors: list[Collector] = [ProcessCollector(pid), SystemCollector()]
    if pid is None:
        collectors.append(GcCollector())
        if trace_python:
            collectors.append(PythonMemoryCollector())
    if gpu := GpuCollector.detect():
        collectors.append(gpu)
    return collectors


class Monitor:
    """Опрашивает сборщики в фоновом потоке и помечает замеры секциями кода."""

    def __init__(
        self,
        interval: float = 0.5,
        sinks: Iterable[Sink] = (),
        collectors: list[Collector] | None = None,
    ) -> None:
        self.interval = interval
        self.sinks = list(sinks)
        self.collectors = default_collectors() if collectors is None else collectors
        self.records: list[Record] = []
        self._path: tuple[str, ...] = ()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Запустить фоновый опрос."""
        if self._thread:
            return
        for collector in self.collectors:
            collector.start()
        self._stop.clear()
        # Метка начала, чтобы отчёт отличал этот запуск от прошлых в том же файле
        self._emit({"event": "start", "t": time.time(), "section": ""})
        self.sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("монитор запущен", extra={"interval": self.interval})

    def stop(self) -> None:
        """Остановить опрос, сделать последний замер и отпустить сборщики."""
        if not self._thread:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        self.sample()
        for collector in self.collectors:
            collector.stop()
        log.info("монитор остановлен", extra={"records": len(self.records)})

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _loop(self) -> None:
        """Делать замер раз в interval, пока не попросят остановиться."""
        while not self._stop.wait(self.interval):
            self.sample()

    def _emit(self, record: Record) -> None:
        """Сохранить запись и отдать её всем sinks."""
        with self._lock:
            self.records.append(record)
            for sink in self.sinks:
                # Например, кончился диск. Программу из-за этого не роняем
                try:
                    sink(record)
                except Exception:
                    log.exception("хранилище %s упало", type(sink).__name__)

    def sample(self) -> Record:
        """Снять показания со всех сборщиков и отправить их в sinks."""
        # Замер идёт и из фонового потока, и из секций, поэтому под замком
        with self._lock:
            record: Record = {"event": "sample", "t": time.time(), "section": "/".join(self._path)}
            for collector in self.collectors:
                # Монитор не должен ронять программу, за которой следит
                try:
                    record |= collector.collect()
                except Exception:
                    log.exception("сборщик %s упал", type(collector).__name__)
            self._emit(record)
        return record

    @contextlib.contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Пометить блок кода. Вложенные секции дают путь вида load/read."""
        if "/" in name:
            raise ValueError("Знак / в имени секции занят: им разделяются вложенные секции")
        # Замер на входе закрывает отрезок родителя, поэтому он идёт до смены пути
        self.sample()
        parent = self._path
        self._path = (*parent, name)
        path, start = "/".join(self._path), time.time()
        try:
            yield
        finally:
            # Замер на выходе, чтобы даже короткая секция попала в отчёт
            self.sample()
            self._path = parent
            self._emit({"event": "section", "section": path, "start": start, "end": time.time()})

    def __call__(self, name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Декоратор: @monitor("embed") помечает каждый вызов функции секцией."""

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with self.section(name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator
