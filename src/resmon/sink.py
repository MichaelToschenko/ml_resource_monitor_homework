import json
import logging
import sqlite3
from contextlib import closing
from logging.handlers import RotatingFileHandler
from pathlib import Path

from resmon.core import MB, Record


class JsonFormatter(logging.Formatter):
    """Пишет сообщение лога одной строкой JSON вместе с полями из extra."""

    # Поля, которые есть у любой записи лога. Всё остальное пришло через extra
    BASE_FIELDS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        data |= {k: v for k, v in vars(record).items() if k not in self.BASE_FIELDS}
        if record.exc_info:
            data["error"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Включить JSON-логи программы в stderr."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)


class JsonlSink:
    """Пишет каждую запись монитора строкой JSON в файл с ротацией."""

    def __init__(self, path: str | Path, max_mb: float = 10, backups: int = 5) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Обработчик сразу отдаёт строку системе, поэтому лог переживает даже kill -9
        self.handler = RotatingFileHandler(
            path, maxBytes=int(max_mb * MB), backupCount=backups, encoding="utf-8"
        )

    def __call__(self, record: Record) -> None:
        line = json.dumps(record, ensure_ascii=False)
        self.handler.handle(logging.makeLogRecord({"msg": line}))

    def close(self) -> None:
        """Закрыть файл."""
        self.handler.close()


class SQLiteSink:
    """Пишет записи монитора в базу SQLite: тип, секция и вся запись в JSON."""

    def __init__(self, path: str | Path) -> None:
        # Пишем из фонового потока монитора, по очереди нас пускает замок монитора
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        # Режим WAL, чтобы дашборд читал базу и не мешал записи
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS records (event TEXT, section TEXT, data TEXT)")

    def __call__(self, record: Record) -> None:
        row = (record["event"], record["section"], json.dumps(record, ensure_ascii=False))
        self.db.execute("INSERT INTO records VALUES (?, ?, ?)", row)

    def close(self) -> None:
        """Закрыть базу."""
        self.db.close()


def make_sink(path: str | Path) -> JsonlSink | SQLiteSink:
    """Выбрать хранилище по расширению: .db - SQLite, иначе JSONL."""
    return SQLiteSink(path) if Path(path).suffix == ".db" else JsonlSink(path)


def read_records(path: str | Path) -> list[Record]:
    """Прочитать записи из JSONL (вместе со старыми кусками после ротации) или из SQLite."""
    path = Path(path)
    if path.suffix == ".db":
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT data FROM records ORDER BY rowid")
            return [json.loads(data) for (data,) in rows]
    # Ротация переименовывает старые куски в .1, .2 и дальше, самый старый с большим номером
    old = [p for p in path.parent.glob(path.name + ".*") if p.suffix[1:].isdigit()]
    parts = [*sorted(old, key=lambda p: -int(p.suffix[1:])), path]
    # Последняя строка может быть недописана, если процесс убили во время записи
    lines = [line for part in parts for line in part.read_text("utf-8").split("\n")[:-1]]
    return [json.loads(line) for line in lines]
