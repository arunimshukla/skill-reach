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

    candidates: list[tuple[int, int]] = []
    if brace_start != -1:
        brace_end = text.rfind("}")
        if brace_end > brace_start:
            candidates.append((brace_start, brace_end + 1))
    if bracket_start != -1:
        bracket_end = text.rfind("]")
        if bracket_end > bracket_start:
            candidates.append((bracket_start, bracket_end + 1))

    if not candidates:
        return text

    if len(candidates) == 1:
        start, end = candidates[0]
        return text[start:end]

    (b_start, b_end), (k_start, k_end) = candidates[0], candidates[1]

    # Bracket encloses brace or brace is prefix tag before bracket:
    if (k_start < b_start and k_end > b_end) or (b_start < k_start and k_end >= b_end):
        return text[k_start:k_end]

    return text[b_start:b_end]


def _scan_fenced_json(candidate: str, content_start: int) -> str | None:
    """Scan code fence blocks starting at content_start for valid or balanced JSON payload."""
    search_pos = content_start
    best_candidate: str | None = None
    while True:
        closing_fence = candidate.find("```", search_pos)
        if closing_fence == -1:
            break
        fenced = candidate[content_start:closing_fence].strip()
        bounds = _extract_outer_bounds(fenced)
        if bounds.startswith(("{", "[")) and bounds.endswith(("}", "]")):
            try:
                json.loads(bounds, strict=False)
                return bounds
            except json.JSONDecodeError:
                try:
                    json.loads(sanitize_json_string(bounds), strict=False)
                    return bounds
                except json.JSONDecodeError:
                    if best_candidate is None:
                        best_candidate = bounds
        search_pos = closing_fence + 3
    return best_candidate


def extract_json_payload(raw: str) -> str:
    """Extract candidate JSON payload prioritizing outer code fences."""
    candidate = raw.strip()

    # Priority 1: Explicit ```json fence
    json_fence_pos = candidate.find("```json")
    if json_fence_pos != -1:
        content_start = json_fence_pos + 7
        matched = _scan_fenced_json(candidate, content_start)
        if matched is not None:
            return matched
        return _extract_outer_bounds(candidate[content_start:].strip())

    # Priority 2: Generic ``` fence
    if "```" in candidate:
        first_fence = candidate.find("```")
        fence_end = candidate.find("\n", first_fence)
        start_pos = fence_end + 1 if fence_end != -1 else first_fence + 3
        matched = _scan_fenced_json(candidate, start_pos)
        if matched is not None:
            return matched

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

    # Try sanitizing raw text directly before slicing code fences
    try:
        return json.loads(sanitize_json_string(candidate), strict=False)
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
