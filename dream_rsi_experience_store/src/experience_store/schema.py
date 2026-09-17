from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema.sql"


def schema_sql() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")

