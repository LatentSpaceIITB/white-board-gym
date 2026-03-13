from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from spatial_rl.schema import ActionValidationError, normalize_action
from spatial_rl.types import JSONDict


@dataclass(slots=True)
class ParseResult:
    action: JSONDict
    action_text: str
    valid: bool
    error: str | None = None


def _extract_first_json_object(text: str) -> str | None:
    depth = 0
    start = None
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : idx + 1]
    return None


def parse_action_text(text: str) -> ParseResult:
    stripped = text.strip()
    if not stripped:
        finish = {"type": "finish"}
        return ParseResult(
            action=finish,
            action_text=json.dumps(finish),
            valid=False,
            error="empty output",
        )

    if stripped.lower() == "finish":
        finish = {"type": "finish"}
        return ParseResult(action=finish, action_text=json.dumps(finish), valid=True)

    candidate = _extract_first_json_object(stripped)
    if candidate is None:
        finish = {"type": "finish"}
        return ParseResult(
            action=finish,
            action_text=json.dumps(finish),
            valid=False,
            error="no json object found",
        )

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        finish = {"type": "finish"}
        return ParseResult(
            action=finish,
            action_text=json.dumps(finish),
            valid=False,
            error=f"json decode error: {exc}",
        )

    if not isinstance(parsed, dict):
        finish = {"type": "finish"}
        return ParseResult(
            action=finish,
            action_text=json.dumps(finish),
            valid=False,
            error="action must be an object",
        )

    try:
        normalized = normalize_action(parsed)
    except ActionValidationError as exc:
        finish = {"type": "finish"}
        return ParseResult(
            action=finish, action_text=json.dumps(finish), valid=False, error=str(exc)
        )

    return ParseResult(
        action=normalized,
        action_text=json.dumps(normalized, separators=(",", ":")),
        valid=True,
    )


def action_to_text(action: JSONDict) -> str:
    return json.dumps(action, separators=(",", ":"), sort_keys=False)


def text_to_action(text: str) -> JSONDict:
    result = parse_action_text(text)
    return result.action
