"""Parse .excalidraw JSON → sorted ExcalidrawElement dataclasses."""

import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExcalidrawElement:
    id: str
    type: str           # "text" | "freedraw"
    x: float
    y: float
    width: float
    height: float
    stroke_color: str
    stroke_width: float
    updated: int        # Unix ms timestamp
    index: str          # e.g. "a0", "a1", "aA", ...
    rel_time: float     # (updated - min_updated) / 1000.0
    is_deleted: bool = False  # True → element was erased; updated = deletion time
    # text-only
    text: Optional[str] = None
    font_size: Optional[float] = None
    # freedraw-only
    points: List[List[float]] = field(default_factory=list)


def get_sorted_elements(filepath: str) -> List[ExcalidrawElement]:
    """Parse excalidraw file and return ALL elements sorted by (rel_time, index).

    Includes deleted elements (is_deleted=True) — their `updated` timestamp is
    the deletion time, and build_actions emits ERASE for them.
    """
    with open(filepath) as f:
        data = json.load(f)

    raw = data.get("elements", [])
    if not raw:
        return []

    min_updated = min(e["updated"] for e in raw)

    elements: List[ExcalidrawElement] = []
    for e in raw:
        rel_time = (e["updated"] - min_updated) / 1000.0
        elem = ExcalidrawElement(
            id=e["id"],
            type=e["type"],
            x=e["x"],
            y=e["y"],
            width=e.get("width", 0.0),
            height=e.get("height", 0.0),
            stroke_color=e.get("strokeColor", "#000000"),
            stroke_width=e.get("strokeWidth", 1.0),
            updated=e["updated"],
            index=e.get("index", ""),
            rel_time=rel_time,
            is_deleted=e.get("isDeleted", False),
            text=e.get("text"),
            font_size=e.get("fontSize"),
            points=e.get("points", []),
        )
        elements.append(elem)

    elements.sort(key=lambda e: (e.rel_time, e.index))
    return elements
