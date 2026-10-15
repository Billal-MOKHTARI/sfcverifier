"""Reading and writing activation traces as CSV (columns: id, timestamp, 0, 1, ..., n-1)."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

from .verifier import bits

TRUE_VALUES = {"true", "1", "yes", "t"}


def write_csv(path: str | Path, masks: list[int], n: int, dt: float = 1.0,
              start: datetime = datetime(2026, 1, 1)) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "timestamp", *range(n)])
        for t, m in enumerate(masks):
            active = set(bits(m))
            ts = (start + timedelta(seconds=t * dt)).isoformat(sep=" ")
            w.writerow([t, ts, *(str(i in active) for i in range(n))])


def read_csv(path: str | Path) -> tuple[list[int], list[str]]:
    """Return (masks, timestamps). Step columns are the purely numeric column names."""
    with open(path, newline="", encoding="utf-8") as f:
        return parse_rows(list(csv.reader(f)))


def parse_rows(rows: list[list[str]]) -> tuple[list[int], list[str]]:
    header = rows[0]
    steps = [(c, int(h)) for c, h in enumerate(header) if h.strip().isdigit()]
    ts_col = header.index("timestamp") if "timestamp" in header else None
    masks, stamps = [], []
    for row in rows[1:]:
        if not row:
            continue
        m = 0
        for c, s in steps:
            if row[c].strip().lower() in TRUE_VALUES:
                m |= 1 << s
        masks.append(m)
        stamps.append(row[ts_col] if ts_col is not None else str(len(stamps)))
    return masks, stamps
