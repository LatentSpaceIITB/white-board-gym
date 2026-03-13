from __future__ import annotations

from pathlib import Path
from typing import Sequence


def load_concepts(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"concept file not found: {path}")
    concepts: list[str] = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        concepts.append(text)
    if not concepts:
        raise ValueError(f"no concepts found in {path}")
    return concepts


def cycle_select(items: Sequence[str], start: int, count: int) -> list[str]:
    if not items:
        return []
    selected: list[str] = []
    for offset in range(count):
        selected.append(items[(start + offset) % len(items)])
    return selected
