from typing import Any

import httpx
import numpy as np
import pytest
import respx

from resmon import demo
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


class FakeModel:
    """Модель-заглушка: тексты про ужасы получают вектор [1, 0], остальные [0, 1]."""

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        return np.array([[1.0, 0.0] if "horror" in t else [0.0, 1.0] for t in texts])


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменить настоящую модель эмбеддингов заглушкой."""
    monkeypatch.setattr(demo, "load_model", FakeModel)


def hf_rows(request: httpx.Request) -> httpx.Response:
    """Ответ сервиса Hugging Face: столько отзывов, сколько попросили. Седьмой про ужасы."""
    offset, length = int(request.url.params["offset"]), int(request.url.params["length"])
    texts = [f"review {i}" + (" horror" if i == 7 else "") for i in range(offset, offset + length)]
    return httpx.Response(200, json={"rows": [{"row": {"text": t}} for t in texts]})


@pytest.fixture
def hf_api(respx_mock: respx.MockRouter) -> respx.Route:
    """Подменить сервис Hugging Face, настоящих запросов в сеть не будет."""
    return respx_mock.get(demo.ROWS_URL).mock(side_effect=hf_rows)
