import hashlib, json, mimetypes
from pathlib import Path


def stable_id(prefix: str, *parts: object) -> str:
    raw = "\x1f".join(str(p) for p in parts).encode()
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:24]}"


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk): h.update(data)
    return h.hexdigest()


def jdump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_type(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("rollout-") and path.suffix == ".jsonl": return "codex_rollout"
    if name == "events.jsonl": return "events_jsonl"
    if name in ("manifest.json", "experiment.json", "result.json", "status.json"): return name[:-5]
    if path.suffix == ".jsonl": return "jsonl"
    if path.suffix == ".json": return "json"
    if path.suffix in (".log", ".txt", ".md", ".tsv", ".csv", ".diff"): return path.suffix[1:]
    return mimetypes.guess_type(path.name)[0] or "binary"


def mime_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def walk_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and "/.git/" not in path.as_posix():
            yield path
