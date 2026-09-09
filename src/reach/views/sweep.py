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

"""Render multi-scale catalog scaling sweeps and loss decomposition tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from reach.rendering import csv_document, dispatch_render

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import Console

    from reach.sweep import ScalingPoint, ScalingStudy

__all__ = [
    "SWEEP_RENDERERS",
    "print_sweep",
    "render_ascii_curve",
    "render_sweep",
    "render_sweep_csv",
    "render_sweep_json",
]


_PASS_RATE_HIGH: float = 0.8
_PASS_RATE_MID: float = 0.5
_MIN_CURVE_POINTS: int = 2
_LEVEL_TOLERANCE: float = 0.125


def _print_corpus_capacity_sweep(console: Console, study: ScalingStudy) -> None:
    """Render a Rich table and capacity summary for whole-corpus scaling sweep."""
    header: list[tuple[str, str]] = [
        ("Corpus Capacity Scaling: ", "bold"),
        (f"{study.total_corpus_skills} skills", "cyan bold"),
        (" across library sizes", "dim"),
    ]
    if study.anchor_skills:
        header.extend(
            [
                (" (", "dim"),
                (f"{len(study.anchor_skills)} anchor skills", "green bold"),
                (")", "dim"),
            ]
        )
    console.print(Text.assemble(*header))

    decision_lines: list[tuple[str, str]] = []
    if study.sla_90_scale is not None:
        decision_lines.append(
            (f"  • Safe Operating Capacity (SLA ≥ 90% F1): K ≤ {study.sla_90_scale}", "bold green")
        )
        if study.sla_90_interpolated is not None:
            decision_lines.append((f" (continuous: {study.sla_90_interpolated:.1f})", "dim"))
        decision_lines.append(("\n", ""))

    if study.sla_85_scale is not None:
        decision_lines.append(
            (f"  • Degraded Capacity Limit (SLA ≥ 85% F1): K ≤ {study.sla_85_scale}", "bold yellow")
        )
        if study.sla_85_interpolated is not None:
            decision_lines.append((f" (continuous: {study.sla_85_interpolated:.1f})", "dim"))
        decision_lines.append(("\n", ""))

    if study.knee_scale is not None:
        decision_lines.append(
            (f"  • Capacity Knee Inflection (Kneedle k*): K = {study.knee_scale}", "bold cyan")
        )
        decision_lines.append(("\n", ""))

    if decision_lines:
        panel = Panel(
            Text.assemble(*decision_lines[:-1]),  # strip trailing newline
            title="[bold]Optimal Catalog Capacity[/]",
            title_align="left",
            box=box.ROUNDED,
            border_style="cyan",
            expand=False,
        )
        console.print(panel)

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        title="Corpus Scaling: Multi-Class Retrieval & Capacity Degradation",
    )
    table.add_column("Scale (K)", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Abstention", justify="right")
    table.add_column("F1 Score", justify="right")
    table.add_column("95% CI (F1)", justify="center", style="dim")
    table.add_column("Probes (In/Neg)", justify="right")
    table.add_column("Duration", justify="right", style="dim")

    for pt in study.points:
        f1_pct = f"{pt.f1_score * 100:.1f}%"
        rate_style = (
            "bold green"
            if pt.f1_score >= _PASS_RATE_HIGH
            else ("yellow" if pt.f1_score >= _PASS_RATE_MID else "bold red")
        )
        rec_pct = f"{pt.recall * 100:.1f}%"
        prec_pct = f"{pt.precision * 100:.1f}%"
        abs_pct = f"{pt.abstention_rate * 100:.1f}%" if pt.abstention_rate is not None else "—"
        f1_ci_str = f"[{pt.f1_interval[0] * 100:.1f}% - {pt.f1_interval[1] * 100:.1f}%]"
        probes_str = f"{pt.in_scope_probes}/{pt.negative_probes}"
        dur_str = f"{pt.duration_ms_mean:.0f}ms" if pt.duration_ms_mean > 0 else "—"

        table.add_row(
            str(pt.scale),
            rec_pct,
            prec_pct,
            abs_pct,
            Text(f1_pct, style=rate_style),
            f1_ci_str,
            probes_str,
            dur_str,
        )

    console.print(table)
    sparkline_parts = [f"K={pt.scale}: {pt.f1_score * 100:.0f}%" for pt in study.points]
    console.print(Text("F1 Trajectory: " + " ──> ".join(sparkline_parts), style="dim"))

    if len(study.points) >= _MIN_CURVE_POINTS:
        console.print()
        for line in render_ascii_curve(study.points, metric="f1"):
            console.print(Text(line, style="dim"))


def _print_single_skill_sweep(console: Console, study: ScalingStudy) -> None:
    """Render a Rich table and loss decomposition for single-skill scaling sweep."""
    target_display = study.target_skill or "all"
    console.print(
        Text.assemble(
            ("Scaling Sweep: ", "bold"),
            (target_display, "cyan bold"),
            (" across library sizes", "dim"),
        )
    )

    if study.knee_scale is not None:
        console.print(
            Text.assemble(
                ("Capacity Knee: ", "bold yellow"),
                (f"k* = {study.knee_scale}", "bold yellow"),
                (" (inflection point where distractor shadowing accelerates)", "dim"),
            )
        )

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        title="Reachability Decay Across Catalog Scales",
    )
    table.add_column("Scale (N)", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("95% CI", justify="center", style="dim")
    table.add_column("Δ Total", justify="right")
    table.add_column("Δ Context", justify="right", style="cyan")
    table.add_column("Δ Shadowing", justify="right", style="red")
    table.add_column("Probes", justify="right")
    table.add_column("Duration", justify="right", style="dim")

    for pt in study.points:
        pct = f"{pt.pass_rate * 100:.1f}%"
        rate_style = (
            "bold green"
            if pt.pass_rate >= _PASS_RATE_HIGH
            else ("yellow" if pt.pass_rate >= _PASS_RATE_MID else "bold red")
        )
        ci = f"[{pt.pass_rate_interval[0] * 100:.1f}% - {pt.pass_rate_interval[1] * 100:.1f}%]"

        tot_str = f"{pt.delta_vs_baseline * 100:+.1f}%" if pt.delta_vs_baseline != 0 else "0.0%"
        ctx_str = f"{pt.delta_context * 100:+.1f}%" if pt.delta_context != 0 else "0.0%"
        shd_str = f"{pt.delta_shadowing * 100:+.1f}%" if pt.delta_shadowing != 0 else "0.0%"
        probes_str = f"{pt.probes_executed - pt.probes_failed}/{pt.probes_executed}"
        dur_str = f"{pt.duration_ms_mean:.0f}ms" if pt.duration_ms_mean > 0 else "—"

        table.add_row(
            str(pt.scale),
            Text(pct, style=rate_style),
            ci,
            tot_str,
            ctx_str,
            shd_str,
            probes_str,
            dur_str,
        )

    console.print(table)
    sparkline_parts = [f"N={pt.scale}: {pt.pass_rate * 100:.0f}%" for pt in study.points]
    console.print(Text("Trajectory: " + " ──> ".join(sparkline_parts), style="dim"))

    if len(study.points) >= _MIN_CURVE_POINTS:
        console.print()
        for line in render_ascii_curve(study.points, metric="pass_rate"):
            console.print(Text(line, style="dim"))


def print_sweep(console: Console, study: ScalingStudy) -> None:
    """Render a Rich table and summary of the scaling sweep study."""
    if study.is_corpus_sweep:
        _print_corpus_capacity_sweep(console, study)
    else:
        _print_single_skill_sweep(console, study)


def render_ascii_curve(
    points: Sequence[ScalingPoint],
    metric: str = "pass_rate",
) -> list[str]:
    """Render an ASCII scaling curve depicting metric progression across catalog scales."""
    if not points:
        return []

    prefix = "K" if metric == "f1" else "N"
    title = "Scaling Curve (F1):" if metric == "f1" else "Scaling Curve:"
    levels = [1.0, 0.75, 0.5, 0.25, 0.0]
    scale_cols = [f"{prefix}={p.scale}" for p in points]
    col_width = max(max(len(c) for c in scale_cols) + 2, 6)

    lines: list[str] = [title]
    for lvl in levels:
        row_cells: list[str] = [f"{int(lvl * 100):3d}% |"]
        for p in points:
            val = p.f1_score if metric == "f1" else p.pass_rate
            symbol = "●" if abs(val - lvl) <= _LEVEL_TOLERANCE else " "
            row_cells.append(symbol.center(col_width))
        lines.append("".join(row_cells))

    axis_sep = "     +" + "-" * (col_width * len(points))
    lines.append(axis_sep)
    axis_labels = "      " + "".join(c.center(col_width) for c in scale_cols)
    lines.append(axis_labels)
    return lines


def render_sweep_json(study: ScalingStudy) -> str:
    """Serialize ScalingStudy to JSON."""
    return study.model_dump_json(indent=2)


def render_sweep_csv(study: ScalingStudy) -> str:
    """Export ScalingStudy metrics to formatted CSV."""
    if study.is_corpus_sweep:
        headers = [
            "scale",
            "recall",
            "recall_ci_low",
            "recall_ci_high",
            "precision",
            "precision_ci_low",
            "precision_ci_high",
            "abstention_rate",
            "f1_score",
            "f1_ci_low",
            "f1_ci_high",
            "in_scope_probes",
            "negative_probes",
            "duration_ms",
        ]
        rows = [
            [
                pt.scale,
                f"{pt.recall:.4f}",
                f"{pt.recall_interval[0]:.4f}",
                f"{pt.recall_interval[1]:.4f}",
                f"{pt.precision:.4f}",
                f"{pt.precision_interval[0]:.4f}",
                f"{pt.precision_interval[1]:.4f}",
                f"{pt.abstention_rate:.4f}" if pt.abstention_rate is not None else "",
                f"{pt.f1_score:.4f}",
                f"{pt.f1_interval[0]:.4f}",
                f"{pt.f1_interval[1]:.4f}",
                pt.in_scope_probes,
                pt.negative_probes,
                f"{pt.duration_ms_mean:.2f}",
            ]
            for pt in study.points
        ]
        return csv_document(headers, rows)

    rows_single: list[list[object]] = [
        [
            study.target_skill or "all",
            pt.scale,
            f"{pt.pass_rate:.4f}",
            f"{pt.pass_rate_interval[0]:.4f}",
            f"{pt.pass_rate_interval[1]:.4f}",
            f"{pt.delta_vs_baseline:.4f}",
            f"{pt.delta_context:.4f}",
            f"{pt.delta_shadowing:.4f}",
            pt.probes_executed,
            pt.probes_failed,
            f"{pt.duration_ms_mean:.2f}",
        ]
        for pt in study.points
    ]

    headers_single = [
        "skill",
        "scale",
        "pass_rate",
        "wilson_low",
        "wilson_high",
        "delta_total",
        "delta_context",
        "delta_shadowing",
        "probes_executed",
        "probes_failed",
        "duration_ms",
    ]
    return csv_document(headers_single, rows_single)


SWEEP_RENDERERS = {
    "csv": render_sweep_csv,
    "json": render_sweep_json,
}


def render_sweep(study: ScalingStudy, fmt: str) -> str:
    """Render ScalingStudy into the requested format."""
    return dispatch_render(SWEEP_RENDERERS, fmt, study)
