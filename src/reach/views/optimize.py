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

"""Render closed-loop description optimization reports, candidate tables, and diffs."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

from rich import box
from rich.table import Table

if TYPE_CHECKING:
    from rich.console import Console

    from reach.optimize import OptimizationCandidate, OptimizationReport


def render_optimization_diff(report: OptimizationReport) -> str:
    """Generate a unified diff showing changes between baseline and best candidate description."""
    if not report.best_candidate:
        return ""
    baseline_lines = ["description: >-\n", f"  {report.baseline_description}\n"]
    candidate_lines = ["description: >-\n", f"  {report.best_candidate.description}\n"]
    diff = difflib.unified_diff(
        baseline_lines,
        candidate_lines,
        fromfile=f"a/{report.skill_name}/SKILL.md",
        tofile=f"b/{report.skill_name}/SKILL.md",
    )
    return "".join(diff)


def _print_optimization_baseline(console: Console, report: OptimizationReport) -> None:
    """Render baseline description, rivals, terms, and initial evaluation metrics."""
    console.print(f"[bold]Current Description:[/] {report.baseline_description}")
    if report.rival_name:
        console.print(f"[bold]Primary Rival:[/] [yellow]{report.rival_name}[/]")
    if report.ceded_terms:
        console.print(f"[bold]Ceded Terms (Rival Pull):[/] [red]{', '.join(report.ceded_terms)}[/]")
    if report.unclaimed_terms:
        console.print(
            f"[bold]Unclaimed Distinctive Terms:[/] [green]{', '.join(report.unclaimed_terms)}[/]",
        )

    if report.has_probes:
        console.print(
            f"[bold]Baseline Metrics:[/] "
            f"Recall: {report.baseline_recall:.1%} | "
            f"Accuracy: {report.baseline_accuracy:.1%} | "
            f"Misroutes: {report.baseline_misroute:.1%}\n",
        )
    else:
        console.print(
            "[bold]Baseline Metrics:[/] "
            "[dim]Not evaluated (provide --queries to run empirical probes)[/]\n",
        )


def _format_candidate_delta(delta: float) -> str:
    """Format change in recall with directional indicator and color styling."""
    if delta > 0:
        return f"[green]+{delta:.1%} ▲[/]"
    if delta < 0:
        return f"[red]{delta:.1%} ▼[/]"
    return "0.0%"


def _format_candidate_row(
    cand: OptimizationCandidate,
    has_probes: bool,
    idx: int,
) -> tuple[str, str, str, str, str, str]:
    """Format single candidate row values for optimization comparison table."""
    if has_probes:
        delta_str = _format_candidate_delta(cand.delta_recall)
        rec_str = f"{cand.recall:.1%}"
        mis_str = f"{cand.misroute_rate:.1%}"
    else:
        delta_str = "[dim]—[/]"
        rec_str = "[dim]—[/]"
        mis_str = "[dim]—[/]"

    lint_str = "[green]✓ CLEAN[/]" if cand.lint_clean else "[yellow]✗ WARN[/]"
    return f"#{idx}", cand.description, delta_str, rec_str, mis_str, lint_str


def _print_optimization_footer(console: Console, report: OptimizationReport) -> None:
    """Print next-step application recommendation or confirmation notice."""
    if report.applied:
        console.print(
            f"[bold green]✓ Successfully updated {report.skill_name}/SKILL.md "
            "with candidate #1![/]",
        )
    elif report.best_candidate:
        console.print(
            "[bold cyan]Recommendation:[/] To apply candidate #1 to disk, re-run with "
            "[bold]--auto-apply[/].",
        )


def print_optimization(console: Console, report: OptimizationReport) -> None:
    """Render closed-loop optimization report and candidate comparison table to console."""
    console.print()
    console.rule(f"[bold cyan]Reach Closed-Loop Optimizer: {report.skill_name}[/]")
    console.print()

    _print_optimization_baseline(console, report)

    if not report.candidates:
        console.print("[yellow]No candidates were generated.[/]\n")
        return

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Candidate Description", style="cyan")
    table.add_column("Δ Recall", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Misroutes", justify="right")
    table.add_column("Linter", justify="center")

    for idx, cand in enumerate(report.candidates, start=1):
        table.add_row(*_format_candidate_row(cand, report.has_probes, idx))

    console.print(table)
    console.print()
    _print_optimization_footer(console, report)
    console.print()
