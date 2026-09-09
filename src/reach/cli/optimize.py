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

"""Optimize skill descriptions with closed-loop candidate generation and empirical probes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

from cyclopts import Parameter

if TYPE_CHECKING:
    from reach.optimize import OptimizationReport
    from reach.views import Console

from .app import LOOP, app
from .flags import POSITIVE_INT, SWITCH, AgentName, Global, agent_help_text

type OptimizeFormat = Literal["text", "json", "diff"]

#: Default number of candidate descriptions to synthesize during optimization.
DEFAULT_CANDIDATES: Final = 3

#: Default probe budget allocated across candidate evaluations.
DEFAULT_OPTIMIZE_BUDGET: Final = 30


@app.command(name="optimize", group=LOOP)
def _optimize(
    skill: Annotated[
        str,
        Parameter(
            name=["skill", "--skill"],
            help="Name of the target skill to optimize",
        ),
    ],
    *,
    skills: Annotated[
        Path | None,
        Parameter(
            help=(
                "Path to a skill directory, SKILL.md file, or catalog tree (discovered if omitted)"
            ),
        ),
    ] = None,
    queries: Annotated[
        Path | None,
        Parameter(
            help="Path to labeled evaluation queries JSON file for empirical validation",
        ),
    ] = None,
    candidates: Annotated[
        int,
        POSITIVE_INT,
        Parameter(
            help="Number of candidate descriptions to synthesize (default: 3)",
        ),
    ] = DEFAULT_CANDIDATES,
    budget: Annotated[
        int,
        POSITIVE_INT,
        Parameter(
            help="Maximum empirical probes to execute across candidate evaluations (default: 30)",
        ),
    ] = DEFAULT_OPTIMIZE_BUDGET,
    agent: Annotated[
        AgentName | None,
        Parameter(
            name="--agent",
            show_choices=False,
            help=agent_help_text(
                "Agent runtime for candidate empirical probing (default: from reach.toml)",
            ),
        ),
    ] = None,
    global_: Global = False,
    auto_apply: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--auto-apply",
            help="Automatically write the highest-ranking candidate description to SKILL.md",
        ),
    ] = False,
    format: Annotated[
        OptimizeFormat,
        Parameter(
            help="Output format: text, json, diff (default: text)",
        ),
    ] = "text",
    config: Annotated[
        Path | None,
        Parameter(
            name="--config",
            help="Path to reach.toml configuration file",
        ),
    ] = None,
) -> int:
    """Optimize a skill's description using candidate synthesis and empirical probes."""
    from reach.optimize import optimize_skill
    from reach.views import (
        build_console,
        print_optimization,
        render_optimization_diff,
    )

    console = build_console()

    from reach.config import OptimizeSettings, RunConfig, default_agent

    run_config = None
    if config is not None:
        run_config = RunConfig.from_toml(config)

    resolved_agent = agent or (
        run_config.runtime.agent if run_config is not None else default_agent()
    )
    eff_settings = RunConfig.resolve(
        OptimizeSettings,
        config=run_config,
        budget=budget if budget != DEFAULT_OPTIMIZE_BUDGET or run_config is None else None,
    )

    if eff_settings.budget < 1:
        console.print("[red]Error:[/] Probe budget must be at least 1")
        return 2

    try:
        report = optimize_skill(
            skill_name=skill,
            skills_path=skills,
            queries_path=queries,
            agent=resolved_agent,
            candidates_count=candidates,
            budget=eff_settings.budget,
            auto_apply=auto_apply,
            config=config,
            global_scope=global_,
            settings=eff_settings,
        )
    except ValueError as err:
        console.print(f"[red]Error:[/] {err}")
        return 2
    except (OSError, RuntimeError) as err:
        console.print(f"[red]Runtime Error:[/] {err}")

        return 3

    match format:
        case "json":
            print(report.model_dump_json(indent=2))
        case "diff":
            diff_text = render_optimization_diff(report)
            if diff_text:
                print(diff_text, end="")
            else:
                console.print("[dim]No modifications recommended or diff unavailable.[/]")
        case _:
            print_optimization(console, report)
            if (
                not auto_apply
                and report.best_candidate is not None
                and sys.stdin.isatty()
                and sys.stdout.isatty()
            ):
                _prompt_interactive_apply(console, report)

    return 0


def _prompt_interactive_apply(console: Console, report: OptimizationReport) -> None:
    """Interactively prompt user in a TTY to inspect diff or apply candidate description."""
    if report.best_candidate is None or not report.manifest_path:
        return
    best = report.best_candidate

    from reach.optimize import update_skill_description
    from reach.views import render_optimization_diff

    delta_pct = best.delta_recall * 100
    delta_str = f"+{delta_pct:.1f}%" if delta_pct > 0 else f"{delta_pct:.1f}%"
    console.print()
    try:
        prompt_msg = (
            f"Apply candidate #1 ({delta_str} recall) to "
            f"{report.manifest_path.name}? [y/N/d(iff)]: "
        )
        response = input(prompt_msg).strip().lower()
        if response in ("d", "diff"):
            diff_text = render_optimization_diff(report)
            if diff_text:
                console.print(diff_text)
            response = (
                input(f"Apply candidate #1 to {report.manifest_path.name}? [y/N]: ").strip().lower()
            )

        if response in ("y", "yes"):
            if update_skill_description(report.manifest_path, best.description):
                console.print(
                    f"[green]✓[/green] Applied candidate #1 description to "
                    f"[bold]{report.manifest_path}[/bold]"
                )
            else:
                console.print(f"[red]Error:[/] Failed to update {report.manifest_path}")
        else:
            console.print("[dim]Skipped applying candidate.[/dim]")
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Skipped applying candidate.[/dim]")
