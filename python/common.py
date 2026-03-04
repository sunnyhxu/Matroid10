from __future__ import annotations

import json
import pathlib
import time
import tomllib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List


def load_toml(path: str | pathlib.Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def monotonic_seconds() -> float:
    return time.monotonic()


def ensure_parent(path: str | pathlib.Path) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | pathlib.Path) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | pathlib.Path) -> Iterator[Dict[str, Any]]:
    p = pathlib.Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")))
            f.write("\n")


def append_jsonl(path: str | pathlib.Path, row: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")))
        f.write("\n")


def bitmask_to_subset(mask: int, n: int) -> List[int]:
    return [i for i in range(n) if (mask >> i) & 1]


def subset_to_bitmask(subset: Iterable[int]) -> int:
    out = 0
    for i in subset:
        out |= 1 << int(i)
    return out


def dump_json(path: str | pathlib.Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
