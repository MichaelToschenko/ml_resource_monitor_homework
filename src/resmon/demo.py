import logging
import os
import time
from typing import Any

import httpx
import numpy as np

from resmon.core import MB, Monitor

log = logging.getLogger(__name__)

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = {"dataset": "stanfordnlp/imdb", "config": "plain_text", "split": "train"}
PAGE = 100  # больше строк за один запрос сервис не отдаёт
BATCH = 64
QUERY = "a scary horror movie with a lot of blood"
LEAK_STEP_MB = 10  # за шаг утечки, шаг раз в полсекунды

# Запасные отзывы на случай, если интернета нет
FALLBACK = (
    "A scary horror film, I could not sleep after it.",
    "Lovely romantic comedy with a happy ending.",
    "The plot was boring and the actors were bad.",
    "Great action scenes and a strong soundtrack.",
)


def fetch_texts(count: int) -> list[str]:
    """Скачать отзывы IMDB с Hugging Face страницами по 100. Если не вышло, вернуть запасные."""
    texts: list[str] = []
    try:
        with httpx.Client(timeout=10) as client:
            while len(texts) < count:
                params = DATASET | {"offset": len(texts), "length": min(PAGE, count - len(texts))}
                response = client.get(ROWS_URL, params=params)
                response.raise_for_status()
                rows = response.json()["rows"]
                if not rows:
                    break
                texts += [row["row"]["text"] for row in rows]
    # ValueError и KeyError - если вместо данных пришло что-то другое, например страница прокси
    except (httpx.HTTPError, ValueError, KeyError):
        log.warning("тексты не скачались, берём запасные", exc_info=True)
        texts = []
    return texts or [FALLBACK[i % len(FALLBACK)] for i in range(count)]


def load_model() -> Any:
    """Загрузить модель эмбеддингов. Имя берём из RESMON_MODEL."""
    # Импорт здесь, потому что torch тяжёлый и нужен только для демо
    from sentence_transformers import SentenceTransformer

    name = os.environ.get("RESMON_MODEL", "intfloat/multilingual-e5-small")
    return SentenceTransformer(name, device="cpu")


def leak(monitor: Monitor, total_mb: int) -> None:
    """Имитация утечки: кэш, который никто не чистит. Растёт на 20 МБ в секунду."""
    with monitor.section("leak"):
        cache = []
        for _ in range(total_mb // LEAK_STEP_MB):
            # np.ones заполняет память, поэтому система правда её выделяет
            cache.append(np.ones(LEAK_STEP_MB * MB, dtype=np.uint8))
            time.sleep(0.5)


def run_demo(monitor: Monitor, count: int = 500, leak_mb: int = 0) -> list[str]:
    """Скачать тексты, посчитать эмбеддинги и найти три отзыва, похожих на запрос."""
    with monitor.section("fetch_texts"):
        texts = fetch_texts(count)
    with monitor.section("load_model"):
        model = load_model()
    with monitor.section("embed"):
        parts = []
        for i in range(0, len(texts), BATCH):
            with monitor.section("batch"):
                # Модель e5 обучали с приставками passage и query
                batch = [f"passage: {text}" for text in texts[i : i + BATCH]]
                parts.append(
                    model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                )
        vectors = np.vstack(parts)
    with monitor.section("search"):
        query = model.encode(
            [f"query: {QUERY}"], normalize_embeddings=True, show_progress_bar=False
        )[0]
        best = np.argsort(vectors @ query)[::-1][:3]
    if leak_mb:
        leak(monitor, leak_mb)
    return [texts[i] for i in best]
