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

"""Format and render environment and runtime diagnostic check reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from reach.views.base import Console


def render_doctor_table(
    console: Console,
    results: list[tuple[str, str, str, str, str]],
    *,
    verbose: bool = False,
) -> int:
    """Render diagnostic check results formatted in Rich tables.

    Each result item is a tuple: (category, name, status, detail, remedy).
    """
    table = Table(title="Reach Environment & Runtime Diagnostics", border_style="dim")
    table.add_column("Category", style="bold cyan", no_wrap=True)
    table.add_column("Item", style="bold white")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    status_symbols = {
        "ok": "[green]✓ OK[/green]",
        "warn": "[yellow]! WARN[/yellow]",
        "fail": "[red]✗ FAIL[/red]",
    }

    has_failures = False
    remedies: list[tuple[str, str]] = []

    for cat, name, status, detail, remedy in results:
        sym = status_symbols.get(status, status)
        table.add_row(cat, name, sym, detail)
        if status == "fail":
            has_failures = True
        if remedy:
            remedies.append((name, remedy))

    console.print(table)

    if remedies and (verbose or has_failures):
        console.print()
        remedy_lines = [f"[bold]{name}:[/bold] {rem}" for name, rem in remedies]
        console.print(
            Panel(
                "\n".join(remedy_lines),
                title="Recommended Actions",
                border_style="yellow" if not has_failures else "red",
            ),
        )

    return 1 if has_failures else 0
