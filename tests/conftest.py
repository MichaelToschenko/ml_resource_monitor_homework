import pytest

from resmon.core import Collector, Monitor


class FakeCollector(Collector):
    """Сборщик с заранее заданными значениями вместо настоящего psutil."""

    def __init__(self, **metrics: float) -> None:
        self.metrics = metrics

    def collect(self) -> dict[str, float]:
        return dict(self.metrics)


@pytest.fixture
def monitor() -> Monitor:
    """Монитор без фонового потока и с предсказуемыми метриками."""
    return Monitor(collectors=[FakeCollector(rss_mb=100.0, cpu_percent=50.0)])
