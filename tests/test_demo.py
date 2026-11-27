from unittest.mock import patch

import httpx
import pytest
import respx

from resmon.core import Monitor
from resmon.demo import FALLBACK, ROWS_URL, fetch_texts, load_model, run_demo
from resmon.report import section_stats


@pytest.mark.parametrize(("count", "requests"), [(50, 1), (100, 1), (250, 3)])
def test_fetch_texts_by_pages(hf_api: respx.Route, count: int, requests: int) -> None:
    texts = fetch_texts(count)
    assert len(texts) == count
    assert hf_api.call_count == requests


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("нет сети"),
        httpx.Response(500),
        httpx.Response(429),
        httpx.Response(200, text="<html>страница прокси</html>"),
        httpx.Response(200, json={"error": "нет такого датасета"}),
        httpx.Response(200, json={"rows": []}),
    ],
)
def test_fallback_without_network(
    respx_mock: respx.MockRouter, failure: Exception | httpx.Response
) -> None:
    respx_mock.get(ROWS_URL).mock(side_effect=[failure])
    texts = fetch_texts(6)
    assert texts == [*FALLBACK, *FALLBACK[:2]]


@pytest.mark.usefixtures("fake_model", "hf_api")
def test_demo_finds_horror_review(monitor: Monitor) -> None:
    found = run_demo(monitor, count=150)
    assert found[0] == "review 7 horror"
    stats = section_stats(monitor.records)
    assert {"fetch_texts", "load_model", "embed", "embed/batch", "search"} <= stats.keys()
    # 150 текстов по 64 в пачке - три пачки
    assert stats["embed/batch"]["calls"] == 3


@pytest.mark.parametrize(
    ("env", "expected"),
    [(None, "intfloat/multilingual-e5-small"), ("my/tiny-model", "my/tiny-model")],
)
def test_model_name_from_env(
    monkeypatch: pytest.MonkeyPatch, env: str | None, expected: str
) -> None:
    monkeypatch.delenv("RESMON_MODEL", raising=False)
    if env:
        monkeypatch.setenv("RESMON_MODEL", env)
    # Настоящую модель не грузим, подменяем класс из sentence-transformers
    with patch("sentence_transformers.SentenceTransformer") as model_class:
        load_model()
    model_class.assert_called_once_with(expected, device="cpu")


@pytest.mark.slow
def test_demo_with_real_model() -> None:
    with Monitor() as monitor:
        found = run_demo(monitor, count=20)
    assert len(found) == 3
