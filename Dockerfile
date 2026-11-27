FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv
WORKDIR /app
# Кэш uv в образе не нужен, он только раздувает его на гигабайт
ENV UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1

# Сначала только зависимости: этот слой пересобирается, только когда меняется uv.lock
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --all-extras --no-install-project

ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/app/hf RESMON_MODEL=intfloat/multilingual-e5-small
# Модель скачиваем при сборке, чтобы демо работало без интернета. Слой стоит до кода,
# чтобы правка кода не заставляла качать модель заново
RUN python -c "import os; from sentence_transformers import SentenceTransformer as M; M(os.environ['RESMON_MODEL'])"
ENV HF_HUB_OFFLINE=1

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --all-extras

# Работаем не от root. Папку для логов готовим заранее, иначе в общий том не записать
RUN useradd --create-home app && mkdir logs && chown app logs
USER app

ENTRYPOINT ["resmon"]
CMD ["--help"]
