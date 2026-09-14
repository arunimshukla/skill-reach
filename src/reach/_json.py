# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Private JSON utilities for parsing model completion payloads."""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "extract_json_payload",
    "parse_model_json",
    "sanitize_json_string",
]


def _extract_outer_bounds(text: str) -> str:
    """Slice text to outermost matching JSON object or array bounds."""
    text = text.strip()
    brace_start = text.find("{")
    bracket_start = text.find("[")

    if brace_start != -1 and (bracket_start == -1 or brace_start < bracket_start):
        brace_end = text.rfind("}")
        if brace_end > brace_start:
            return text[brace_start : brace_end + 1]
    elif bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
        bracket_end = text.rfind("]")
        if bracket_end > bracket_start:
            return text[bracket_start : bracket_end + 1]
    return text


def extract_json_payload(raw: str) -> str:
    """Extract candidate JSON payload prioritizing outer code fences."""
    candidate = raw.strip()

    # Priority 1: Explicit ```json fence
    json_fence_pos = candidate.find("```json")
    if json_fence_pos != -1:
        content_start = json_fence_pos + 7
        closing_fence = candidate.rfind("```")
        if closing_fence > content_start:
            fenced = candidate[content_start:closing_fence].strip()
            return _extract_outer_bounds(fenced)
        return _extract_outer_bounds(candidate[content_start:].strip())

    # Priority 2: Generic ``` fence
    if "```" in candidate:
        first_fence = candidate.find("```")
        closing_fence = candidate.rfind("```")
        if closing_fence > first_fence:
            fence_end = candidate.find("\n", first_fence)
            start_pos = (
                fence_end + 1 if fence_end != -1 and fence_end < closing_fence else first_fence + 3
            )
            fenced = candidate[start_pos:closing_fence].strip()
            bounds = _extract_outer_bounds(fenced)
            if bounds.startswith(("{", "[")):
                return bounds

    return _extract_outer_bounds(candidate)


def sanitize_json_string(text: str) -> str:
    """Strip trailing commas outside string literals before closing braces and brackets."""
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            out.append(char)
            i += 1
        elif char == '"':
            in_string = True
            out.append(char)
            i += 1
        elif char == ",":
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j] in ("}", "]"):
                i += 1  # Skip trailing comma
            else:
                out.append(char)
                i += 1
        else:
            out.append(char)
            i += 1
    return "".join(out)


def parse_model_json(raw: str) -> Any:  # noqa: ANN401 (matches json.loads return type)
    """Parse JSON completion payload with trailing comma sanitization fallback."""
    candidate = raw.strip()
    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        pass

    extracted = extract_json_payload(candidate)
    try:
        return json.loads(extracted, strict=False)
    except json.JSONDecodeError:
        pass

    cleaned = sanitize_json_string(extracted)
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError as exc:
        msg = f"generator reply was not JSON: {exc}"
        raise ValueError(msg) from exc
