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

"""Format experimental comparison reports across text, JSON, JSONL, and CSV formats."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from reach.rendering import csv_document, dispatch_render

if TYPE_CHECKING:
    from reach.diff import Comparison, Survey


def _points(value: float | None) -> str:
    """Format a float percentage rate as points string or '--' if None."""
    return "--" if value is None else f"{value * 100:.1f}"


def _headline_lines(comparison: Comparison) -> list[str]:
    """Format headline top-1 delta, standard errors, and noise floor for text output."""
    headline = comparison.headline
    control, treatment = comparison.control, comparison.treatment
    return [
        (
            f"  {'control':<16} {_points(headline.control):>7}%  "
            f"{control.top1_hits}/{control.scored} probes, "
            f"se {_points(control.standard_error)}"
        ),
        (
            f"  {'treatment':<16} {_points(headline.treatment):>7}%  "
            f"{treatment.top1_hits}/{treatment.scored} probes, "
            f"se {_points(treatment.standard_error)}"
        ),
        f"  {'delta':<16} {headline.delta * 100:>+7.1f}   points",
        (
            f"  {'noise floor':<16} {_points(headline.floor):>7}   points "
            f"({headline.confidence * 100:.0f}%, "
            f"{headline.noise_inflation:.2f}x over-dispersion)"
        ),
        (f"  {'resolvable':<16} {_points(headline.resolvable):>7}   points at this depth"),
    ]


ROSTER_SHOWN = 6


def _roster(names: tuple[str, ...]) -> str:
    """Format a list of skill names, truncating with count when exceeding limit."""
    if len(names) <= ROSTER_SHOWN:
        return ", ".join(names)
    rest = len(names) - ROSTER_SHOWN
    return f"{', '.join(names[:ROSTER_SHOWN])}, and {rest} more"


def _corroboration_lines(comparison: Comparison) -> list[str]:
    """Format corroboration lines showing digest movements and additions/removals."""
    check = comparison.corroboration
    control, treatment = comparison.control, comparison.treatment
    moved = {True: "moved", False: "held "}
    lines = [
        (
            f"  {'arm':<16} {moved[check.arm_moved]}  "
            f"{control.arm or '(unrecorded)'} -> "
            f"{treatment.arm or '(unrecorded)'}"
        ),
        (
            f"  {'corpus':<16} {moved[check.corpus_moved]}  "
            f"{control.corpus_digest or '(unrecorded)'} -> "
            f"{treatment.corpus_digest or '(unrecorded)'}"
        ),
        (
            f"  {'catalog':<16} "
            f"{moved[control.catalog_size != treatment.catalog_size]}  "
            f"{control.catalog_size} -> {treatment.catalog_size} skills"
        ),
    ]
    for caption, names in (
        ("resident added", check.added),
        ("resident gone", check.removed),
    ):
        if names:
            lines.append(f"  {caption:<16} {_roster(names)}")
    lines.append(
        f"  {'corroborated':<16} {'yes' if check.corroborated else 'no'}: {check.reason}",
    )
    return lines


def _skill_lines(comparison: Comparison) -> list[str]:
    """Format tabular lines showing per-skill recall deltas."""
    if not comparison.skills:
        return ["  (no skill was named by a query in both arms)"]
    width = max((len(delta.skill) for delta in comparison.skills), default=5)
    header = f"  {'skill':<{width}} {'control':>12} {'treatment':>12} {'delta':>8}  real"
    rows = []
    for delta in comparison.skills:
        left = f"{delta.control_reached}/{delta.control_probes} {delta.control_recall * 100:.0f}%"
        right = (
            f"{delta.treatment_reached}/{delta.treatment_probes} "
            f"{delta.treatment_recall * 100:.0f}%"
        )
        rows.append(
            f"  {delta.skill:<{width}} {left:>12} {right:>12} "
            f"{delta.delta * 100:>+8.1f}  {'yes' if delta.real else 'no'}",
        )
    return [header, *rows]


QUERIES_SHOWN = 8


def _query_lines(comparison: Comparison) -> list[str]:
    """Format tabular lines showing per-query deltas, prioritizing separated queries."""
    if not comparison.queries:
        return ["  (no query was probed by both arms)"]
    held = tuple(delta for delta in comparison.queries if not delta.real)
    ordered = (*comparison.separated, *held)
    shown = ordered[:QUERIES_SHOWN]
    width = max((len(delta.query_id) for delta in shown), default=8)
    lines = [
        f"  {'query':<{width}} {'control':>12} {'treatment':>12} {'delta':>8}  real",
    ]
    for delta in shown:
        left = f"{delta.control_hits}/{delta.control_probes} {delta.control_rate * 100:.0f}%"
        right = f"{delta.treatment_hits}/{delta.treatment_probes} {delta.treatment_rate * 100:.0f}%"
        lines.append(
            f"  {delta.query_id:<{width}} {left:>12} {right:>12} "
            f"{delta.delta * 100:>+8.1f}  {'yes' if delta.real else 'no'}",
        )
    if hidden := len(ordered) - len(shown):
        hidden_separated = max(0, len(comparison.separated) - len(shown))
        hidden_held = hidden - hidden_separated
        if hidden_separated > 0:
            lines.append(f"  ({hidden} more: {hidden_separated} separated, {hidden_held} held)")
        else:
            lines.append(f"  ({hidden} more, none of them separated)")
    return lines


def render_diff_text(comparison: Comparison) -> str:
    """Format a Comparison model as plain text for console display."""
    blocks = [
        (
            f"diff --vary {comparison.factor}  "
            f"{comparison.control.label} -> {comparison.treatment.label}  "
            f"({comparison.shared_queries} shared queries)"
        ),
        "\n".join(["Top-1 accuracy", *_headline_lines(comparison)]),
        "\n".join(["What moved", *_corroboration_lines(comparison)]),
        "\n".join(
            [
                (
                    f"Per-skill recall ({len(comparison.skills)}), "
                    "real when the intervals do not overlap"
                ),
                *_skill_lines(comparison),
            ],
        ),
        "\n".join(
            [
                (
                    f"Per-query hit rate ({len(comparison.queries)}), "
                    "real when the intervals do not overlap"
                ),
                *_query_lines(comparison),
            ],
        ),
        f"Verdict  {comparison.verdict}",
    ]
    return "\n\n".join(blocks)


WHERE_COLUMN = len("  treatment  ")
SURVEY_WIDTH = 76


def _shown(path: Path) -> Path:
    """Format path relative to current working directory if possible."""
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path


def _predicate(path: Path | None, reason: str) -> str:
    """Remove redundant leading path from an error message string."""
    prefix = f"{path} "
    return reason[len(prefix) :] if path is not None and reason.startswith(prefix) else reason


def _paragraph(head: str, body: str) -> list[str]:
    """Wrap error paragraph text cleanly with hanging indentation."""
    return textwrap.wrap(
        body,
        width=SURVEY_WIDTH,
        initial_indent=head,
        subsequent_indent=" " * WHERE_COLUMN,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [head.rstrip()]


def render_survey(survey: Survey) -> str:
    """Format a Survey report into plain text detailing comparison barriers."""
    if survey.comparable:
        return (
            f"nothing stands between {survey.control_path.name} and "
            f"{survey.treatment_path.name}: they compare on --vary {survey.factor}"
        )
    count = len(survey.walls)
    lines = [
        (
            f"cannot compare {survey.control_path.name} with "
            f"{survey.treatment_path.name} on --vary {survey.factor}. "
            f"{count} wall{'s' if count > 1 else ''}:"
        ),
        "",
    ]
    for wall in survey.walls:
        head = f"  {wall.where:<{WHERE_COLUMN - 2}}"
        if wall.path is not None:
            lines.append(f"{head}{_shown(wall.path)}")
            head = " " * WHERE_COLUMN
        lines.extend(_paragraph(head, _predicate(wall.path, wall.reason)))
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def render_diff_json(comparison: Comparison) -> str:
    """Serialize a Comparison model into formatted JSON."""
    return comparison.model_dump_json(indent=2)


def render_diff_csv(comparison: Comparison) -> str:
    """Export Comparison metrics across headline, skills, and queries as CSV."""
    headline = comparison.headline
    rows = [
        [
            "top1_accuracy",
            f"{headline.control:.6f}",
            f"{headline.treatment:.6f}",
            f"{headline.delta:.6f}",
            "" if headline.floor is None else f"{headline.floor:.6f}",
            "yes" if headline.real else "no",
            "noise floor",
        ],
        *(
            [
                f"recall:{delta.skill}",
                f"{delta.control_recall:.6f}",
                f"{delta.treatment_recall:.6f}",
                f"{delta.delta:.6f}",
                "",
                "yes" if delta.real else "no",
                "interval overlap",
            ]
            for delta in comparison.skills
        ),
        *(
            [
                f"query:{delta.query_id}",
                f"{delta.control_rate:.6f}",
                f"{delta.treatment_rate:.6f}",
                f"{delta.delta:.6f}",
                "",
                "yes" if delta.real else "no",
                "interval overlap",
            ]
            for delta in comparison.queries
        ),
    ]
    return csv_document(
        ["figure", "control", "treatment", "delta", "floor", "real", "test"],
        rows,
    )


def render_diff_jsonl(comparison: Comparison) -> str:
    """Render comparison delta records as JSONL lines."""
    return "".join(delta.model_dump_json() + "\n" for delta in comparison.deltas)


#: Supported export format handlers for comparison reports.
DIFF_RENDERERS = {
    "text": render_diff_text,
    "json": render_diff_json,
    "jsonl": render_diff_jsonl,
    "csv": render_diff_csv,
}


def render_diff(comparison: Comparison, fmt: str = "text") -> str:
    """Render a Comparison into the requested format (text, json, jsonl, csv)."""
    return dispatch_render(DIFF_RENDERERS, fmt, comparison)
