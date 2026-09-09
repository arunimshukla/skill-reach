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

"""Render recorded evaluation run artifacts to terminal text or HTML formats."""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter
from pydantic import ValidationError as PydanticValidationError

from reach.artifact import ARTIFACT_SUFFIX, Artifact, read_artifact
from reach.view import render_view
from reach.views import Console, build_console, print_query_records, print_scorecard, print_wrote

from .app import LOOP, app
from .flags import SWITCH, Verbose


def _handle_browser_view(
    console: Console,
    recorded: Artifact,
    out: Path | None,
) -> int:
    """Save HTML view and open in default web browser."""
    target_out = out
    if target_out is None:
        reach_dir = Path(".reach")
        reach_dir.mkdir(parents=True, exist_ok=True)
        target_out = reach_dir / "report.html"

    target_out.parent.mkdir(parents=True, exist_ok=True)
    html_content = render_view(recorded, "html")
    target_out.write_text(html_content, encoding="utf-8")
    console.print(f"Opening [bold]{target_out}[/bold] in your default browser...")
    webbrowser.open(target_out.resolve().as_uri())
    return 0


def _handle_file_view(
    console: Console,
    recorded: Artifact,
    out: Path,
    format_opt: str,
    *,
    verbose: bool,
    show_queries: bool,
) -> int:
    """Render and write artifact output to a file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    match format_opt:
        case "html":
            content = render_view(recorded, "html")
        case "json":
            content = recorded.model_dump_json(indent=2)
        case "jsonl":
            content = "".join(query.model_dump_json() + "\n" for query in recorded.queries)
        case _:
            file_console = build_console(record=True)
            print_scorecard(file_console, recorded, verbose=verbose)
            if show_queries:
                print_query_records(file_console, recorded)
            content = file_console.export_text()
    out.write_text(content, encoding="utf-8")
    print_wrote(console, out)
    return 0


def _handle_stdout_view(
    console: Console,
    recorded: Artifact,
    format_opt: str,
    *,
    verbose: bool,
    show_queries: bool,
) -> int:
    """Render artifact output directly to stdout."""
    match format_opt:
        case "html":
            print(render_view(recorded, "html"))
        case "json":
            print(recorded.model_dump_json(indent=2))
        case "jsonl":
            for query in recorded.queries:
                print(query.model_dump_json())
        case _:
            print_scorecard(console, recorded, verbose=verbose)
            if show_queries:
                print_query_records(console, recorded)
    return 0


@app.command(name="view", group=LOOP)
def _view(
    artifact: Annotated[
        Path | None,
        Parameter(
            help="Path to the evaluation artifact JSON file (default: .reach/eval.json)",
        ),
    ] = None,
    *,
    out: Annotated[
        Path | None,
        Parameter(
            name=["--out", "-o"],
            help="Where to write the rendered report; defaults to stdout",
        ),
    ] = None,
    show_queries: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--show-queries",
            help="Display individual scored query records below the scorecard",
        ),
    ] = False,
    format: Annotated[
        Literal["text", "html", "json", "jsonl"],
        Parameter(
            help="Output format: 'text' (terminal scorecard), "
            "'html' (standalone interactive HTML report), "
            "'json' (raw artifact JSON), or "
            "'jsonl' (scored query records as JSON lines)",
        ),
    ] = "text",
    open_browser: Annotated[
        bool,
        SWITCH,
        Parameter(
            name=["--open", "-O"],
            help="Open the rendered HTML report directly in the default web browser",
        ),
    ] = False,
    verbose: Verbose = False,
) -> int:
    """Read back a recorded run and render what it measured."""
    target_artifact = artifact or Path(".reach/eval.json")
    if not target_artifact.is_file():
        msg = (
            f"No evaluation artifact found at {target_artifact}.\n\n"
            "Run 'reach eval' first, or pass an artifact path: 'reach view <artifact.json>'"
        )
        raise ValueError(msg)

    console = build_console()
    try:
        recorded = read_artifact(target_artifact)
    except PydanticValidationError as error:
        msg = (
            f"Cannot read artifact at {target_artifact}: expected an evaluation artifact "
            f"JSON file (written by `reach eval` as <results>{ARTIFACT_SUFFIX}), "
            f"not raw result rows ({error.error_count()} validation errors)."
        )
        raise ValueError(msg) from error

    if open_browser:
        return _handle_browser_view(console, recorded, out)

    if out is not None:
        return _handle_file_view(
            console, recorded, out, format, verbose=verbose, show_queries=show_queries
        )

    return _handle_stdout_view(
        console, recorded, format, verbose=verbose, show_queries=show_queries
    )
