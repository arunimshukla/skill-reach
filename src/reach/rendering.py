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

"""Format structured rows into standard CSV documents."""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


def csv_document(header: Sequence[object], rows: Iterable[Sequence[object]]) -> str:
    """Render header and row sequences into a newline-terminated CSV string."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


type RenderMap[T] = Mapping[str, Callable[[T], str]]


def dispatch_render[T](
    renderers: RenderMap[T],
    fmt: str,
    target: T,
) -> str:
    """Dispatch rendering of target to named format handler or raise descriptive ValueError."""
    renderer = renderers.get(fmt)
    if renderer is None:
        msg = f"unknown format {fmt!r}; expected one of {', '.join(sorted(renderers))}"
        raise ValueError(msg)
    return renderer(target)


def _normalize_annotation_level(severity: str) -> str:
    """Normalize arbitrary severity strings to valid GitHub Actions workflow levels."""
    level = severity.lower()
    if level in {"error", "warning", "notice"}:
        return level
    if "error" in level:
        return "error"
    if "warn" in level:
        return "warning"
    return "notice"


def _build_annotation_params(
    *,
    file: str | None,
    line: int | None,
    col: int | None,
    end_line: int | None,
    end_col: int | None,
    title: str | None,
) -> str:
    """Build key-value parameter string for GitHub Actions command."""
    params: list[str] = []
    if file:
        params.append(f"file={file}")
    if line is not None:
        params.append(f"line={line}")
    if col is not None:
        params.append(f"col={col}")
    if end_line is not None:
        params.append(f"endLine={end_line}")
    if end_col is not None:
        params.append(f"endColumn={end_col}")
    if title:
        params.append(f"title={title}")
    return f" {','.join(params)}" if params else ""


def format_github_annotation(
    severity: str,
    message: str,
    *,
    title: str | None = None,
    file: str | None = None,
    line: int | None = None,
    col: int | None = None,
    end_line: int | None = None,
    end_col: int | None = None,
) -> str:
    r"""Format a diagnostic message as a GitHub Actions workflow command.

    Encodes characters per the GitHub Actions workflow commands specification:
    '%' -> '%25', '\r' -> '%0D', '\n' -> '%0A'.
    """
    level = _normalize_annotation_level(severity)
    param_str = _build_annotation_params(
        file=file,
        line=line,
        col=col,
        end_line=end_line,
        end_col=end_col,
        title=title,
    )
    escaped_message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return f"::{level}{param_str}::{escaped_message}"
