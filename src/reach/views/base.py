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

"""Provide core console setup, theme definitions, base widgets, and progress bars."""

from __future__ import annotations

import math
import sys
from collections import Counter
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from itertools import batched
from pathlib import Path
from typing import IO, TYPE_CHECKING

from cyclopts.help import DefaultFormatter
from cyclopts.help.specs import (
    AsteriskColumn,
    ColumnSpec,
    DescriptionRenderer,
    NameRenderer,
    PanelSpec,
)
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.measure import Measurement
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from reach.models import ProbeResult
from reach.uncertainty import required_probes

__all__ = [
    "ERROR",
    "ETA",
    "HIT",
    "MISROUTE",
    "NAME_COLUMN_SHARE",
    "NON_SELECTION",
    "REACH_THEME",
    "TALLIED",
    "TOOK",
    "Console",
    "Recorder",
    "badge",
    "build_console",
    "error_panel",
    "help_console",
    "help_formatter",
    "is_live",
    "middle_truncate",
    "print_discovery",
    "print_draft_preview",
    "print_drafted",
    "print_generation",
    "print_generation_spend",
    "print_plan",
    "print_query_set",
    "print_query_view",
    "print_quick_scope",
    "print_resuming",
    "print_wrote",
    "probe_progress",
]

if TYPE_CHECKING:
    from cyclopts.help import HelpEntry
    from rich.console import ConsoleOptions

    from reach.difficulty import LexicalRank
    from reach.discovery import Discovery
    from reach.leak import Leak
    from reach.queries import QuerySet
    from reach.run import Plan


#: Color and style definitions for Rich terminal rendering.
REACH_THEME = Theme(
    {
        "reach.catalog": "bold cyan",
        "reach.count": "bold",
        "reach.digest": "dim",
        "reach.label": "dim",
        "reach.bar": "cyan",
        "reach.bar.done": "dim cyan",
        "reach.bar.track": "grey23",
        "reach.spinner": "cyan",
        "reach.hit": "green",
        "reach.misroute": "yellow",
        "reach.non-selection": "bright_blue",
        "reach.error": "bold red",
        "progress.download": "bold",
        "progress.elapsed": "dim",
        "progress.remaining": "dim",
        "reach.help.name": "cyan",
        "reach.help.description": "default",
        "reach.help.required": "bold",
        "reach.help.border": "grey23",
        "reach.error.border": "red",
        "reach.error.detail": "default",
    },
)

#: Standard probe outcome classification constants.
HIT = "hit"
MISROUTE = "misroute"
NON_SELECTION = "non-selection"
ERROR = "error"

#: Standard outcome categories displayed in progress counters.
TALLIED = (HIT, MISROUTE, NON_SELECTION)

#: Progress clock mode labels.
ETA = "eta"
TOOK = "took"


def build_console(
    *,
    quiet: bool = False,
    file: IO[str] | None = None,
    width: int | None = None,
    force_terminal: bool | None = None,
    record: bool = False,
) -> Console:
    """Construct a configured Rich Console instance directing output to stderr."""
    return _console(
        stderr=True,
        quiet=quiet,
        file=file,
        width=width,
        force_terminal=force_terminal,
        record=record,
    )


def help_console(
    *,
    file: IO[str] | None = None,
    width: int | None = None,
    force_terminal: bool | None = None,
) -> Console:
    """Construct a Rich Console instance for rendering command-line help to stdout."""
    return _console(
        stderr=False,
        quiet=False,
        file=file,
        width=width,
        force_terminal=force_terminal,
    )


def _console(
    *,
    stderr: bool,
    quiet: bool,
    file: IO[str] | None,
    width: int | None,
    force_terminal: bool | None,
    record: bool = False,
) -> Console:
    """Instantiate a themed Rich Console with unified settings."""
    if force_terminal is None and not sys.stdout.isatty():
        force_terminal = False
    return Console(
        file=file,
        stderr=stderr and file is None,
        theme=REACH_THEME,
        quiet=quiet,
        width=width,
        force_terminal=force_terminal,
        highlight=False,
        record=record,
    )


def is_live(console: Console) -> bool:
    """Return True if the console is attached to an interactive, non-quiet terminal."""
    return console.is_terminal and not console.quiet


def error_panel(lines: Sequence[str]) -> Panel:
    """Construct a standardized error panel wrapping the provided detail lines."""
    return Panel(
        Text("\n".join(lines), style="reach.error.detail"),
        title="Error",
        title_align="left",
        border_style="reach.error.border",
        box=box.ROUNDED,
        expand=True,
    )


#: Maximum share of console width allocated to parameter name columns in help panels.
NAME_COLUMN_SHARE = 0.35


def help_formatter() -> DefaultFormatter:
    """Construct a custom Cyclopts DefaultFormatter styled using the Reach theme."""
    return DefaultFormatter(
        panel_spec=PanelSpec(border_style="reach.help.border"),
        column_specs=_help_columns,
    )


def _help_columns(
    console: Console,
    options: ConsoleOptions,  # noqa: ARG001
    entries: Sequence[HelpEntry],
) -> tuple[ColumnSpec, ...]:
    """Calculate column layout specs for CLI help parameter panels."""
    cap = math.ceil(console.width * NAME_COLUMN_SHARE)
    named = ColumnSpec(
        renderer=NameRenderer(max_width=cap),
        max_width=cap,
        style="reach.help.name",
    )
    described = ColumnSpec(
        renderer=DescriptionRenderer(),
        overflow="fold",
        style="reach.help.description",
    )
    marker = AsteriskColumn.copy(style="reach.help.required")  # type: ignore[no-untyped-call]
    if any(entry.required for entry in entries):
        return (marker, named, described)
    return (named, described)


_CONSONANTS = "bdfghjklmnprstvz"
_VOWELS = "aiou"
_HEX = frozenset("0123456789abcdefABCDEF")


def badge(digest: str) -> str:
    """Convert a hexadecimal digest string into pronounceable proquint syllables."""
    if not digest or len(digest) % 4 or not set(digest) <= _HEX:
        return digest
    return "-".join(_quint(int("".join(chunk), 16)) for chunk in batched(digest, 4))


def _quint(word: int) -> str:
    """Encode a 16-bit integer into a 5-character C-V-C-V-C proquint string."""
    return (
        _CONSONANTS[word >> 12 & 0xF]
        + _VOWELS[word >> 10 & 0x3]
        + _CONSONANTS[word >> 6 & 0xF]
        + _VOWELS[word >> 4 & 0x3]
        + _CONSONANTS[word & 0xF]
    )


def _digest(value: str, *, verbose: bool) -> str:
    """Render a digest as a proquint badge, appending raw hex if verbose is True."""
    said = badge(value)
    return f"{said} {value}" if verbose and said != value else said


def print_plan(
    console: Console,
    plan: Plan,
    *,
    agent: str,
    model: str,
    fingerprint: str,
    max_turns: int | None = None,
    early_exit: bool | None = None,
    verbose: bool = False,
) -> None:
    """Print the execution plan summary and statistical resolution estimates."""
    info_parts = [f" probes on {agent}/{model}"]
    if max_turns is not None:
        exit_str = "early-exit=on" if early_exit is not False else "early-exit=off"
        info_parts.append(f" [turns={max_turns}, {exit_str}]")
    info_text = "".join(info_parts)
    console.print(
        Text.assemble(
            (plan.catalog_id, "reach.catalog"),
            f": {plan.catalog_size} skills, ",
            (str(plan.probes), "reach.count"),
            f"{info_text} ",
            (
                (
                    f"[config {_digest(fingerprint, verbose=verbose)}"
                    f" corpus {_digest(plan.corpus_digest, verbose=verbose)}]"
                ),
                "reach.digest",
            ),
        ),
        soft_wrap=True,
    )
    if plan.per_query_resolution is not None:
        console.print(
            Text.assemble(
                "  resolves a per-query rate to ",
                (f"+-{plan.per_query_resolution:.2f}", "reach.count"),
                f" at {plan.attempts} attempts; ",
                f"+-0.20 would take {required_probes(0.20)} per query",
            ),
            soft_wrap=True,
        )


type Recorder = Callable[[int, int, ProbeResult], None]


@contextmanager
def probe_progress(
    console: Console,
    *,
    catalog_id: str,
    total: int,
    truth: Mapping[str, str],
) -> Generator[Recorder]:
    """Provide a context manager displaying live probe progress."""
    tally: Counter[str] = Counter()
    if not is_live(console):
        yield _line_per_probe(console, tally, truth)
        return

    progress = Progress(
        SpinnerColumn(style="reach.spinner"),
        BarColumn(
            style="reach.bar.track",
            complete_style="reach.bar",
            finished_style="reach.bar.done",
        ),
        MofNCompleteColumn(),
        TextColumn("{task.fields[tally]}"),
        TextColumn("[reach.label]{task.fields[clock]:>4}[/]"),
        TimeRemainingColumn(compact=True, elapsed_when_finished=True),
        console=console,
    )
    task = progress.add_task("probing", total=total, tally=_tally(tally), clock=ETA)
    resident = Text.assemble(
        ("probing  ", "reach.label"),
        (catalog_id, "reach.catalog"),
    )

    def record(index: int, of: int, result: ProbeResult) -> None:
        """Update live progress bar with observed probe outcome."""
        tally[_outcome(result, truth.get(result.query_id))] += 1
        progress.update(
            task,
            completed=index,
            total=of,
            tally=_tally(tally),
            clock=TOOK if index >= of else ETA,
        )

    with Live(Group(resident, progress), console=console):
        yield record


def _line_per_probe(
    console: Console,
    tally: Counter[str],
    truth: Mapping[str, str],
) -> Recorder:
    """Construct a recorder callback that logs individual probe results line-by-line."""

    def record(index: int, of: int, result: ProbeResult) -> None:
        """Print a single probe result line to the console."""
        tally[_outcome(result, truth.get(result.query_id))] += 1
        if result.error:
            mark = result.error
        elif result.invoked_skills:
            if len(result.invoked_skills) > 1:
                traj = " -> ".join(result.invoked_skills)
                exit_str = ", exit ✓" if result.early_exit else ""
                mark = f"{traj} ({result.turns_taken} turns{exit_str})"
            elif result.early_exit:
                mark = f"{result.invoked_skills[0]} (exit ✓)"
            else:
                mark = result.invoked_skills[0]
        else:
            mark = "(no selection)"
        console.print(
            f"[{index}/{of}] {result.query_id} -> {mark}",
            markup=False,
            soft_wrap=True,
        )

    return record


def _outcome(result: ProbeResult, expected: str | None) -> str:
    """Categorize a probe result as a hit, misroute, non-selection, or error."""
    if result.error:
        return ERROR
    if result.predicted_label == expected:
        return HIT
    if not result.selected:
        return NON_SELECTION
    return MISROUTE


def _tally(counts: Counter[str]) -> str:
    """Format outcome counts into styled markup spans for progress bars."""
    errored = bool(counts[ERROR])
    shown = [*TALLIED, *([ERROR] if errored else [])]
    tally = "  ".join(f"[reach.{name}]{counts[name]} {name}[/]" for name in shown)
    return tally if errored else tally + " " * len(f"  {counts[ERROR]} {ERROR}")


def print_discovery(console: Console, discovery: Discovery) -> None:
    """Print discovered skill directory roots and any warning messages."""
    for root in discovery.roots:
        console.print(
            Text.assemble(
                ("  discovered  ", "reach.label"),
                (f"{root.scope:<10}", "reach.label"),
                str(root.path),
            ),
            soft_wrap=True,
        )
    for warning in discovery.warnings:
        console.print(
            Text.assemble(("  warning  ", "reach.misroute"), warning),
            soft_wrap=True,
        )


def print_generation(
    console: Console,
    *,
    catalog_id: str,
    residents: int,
    targets: int,
    count: int,
    agent: str,
    model: str,
    rivals: int | None = None,
    adversarial: bool = False,
    adversarial_count: int = 1,
) -> None:
    """Print prompt sizing and target breakdown before generating synthetic queries."""
    capped = (
        f", {rivals} of them shown as rivals"
        if rivals is not None and rivals < residents - 1
        else ""
    )
    adv_msg = f" (+{adversarial_count * targets} adversarial)" if adversarial else ""
    console.print(
        Text.assemble(
            (catalog_id, "reach.catalog"),
            f": {residents} skills resident{capped}, drafting ",
            (str(count * targets), "reach.count"),
            f"{adv_msg} queries for {targets} targets on {agent}/{model}",
        ),
        soft_wrap=True,
    )


def print_quick_scope(
    console: Console,
    *,
    catalog_id: str,
    residents: int,
    corpus: int,
    attempts: int,
    rivals: int | None,
    authored: int,
) -> None:
    """Print scope summary and ground truth configuration for quick evaluations."""
    console.print(
        Text.assemble(
            ("quick  ", "reach.label"),
            (catalog_id, "reach.catalog"),
            f": {residents} of {corpus} skills resident, ",
            (str(attempts), "reach.count"),
            f" attempt{'s' if attempts != 1 else ''} per query",
        ),
        soft_wrap=True,
    )
    ground = (
        f"the {authored} question{'s' if authored != 1 else ''} you typed"
        if authored
        else (
            "drafted here and probed unreviewed"
            + (f", against {rivals} rivals" if rivals is not None else "")
        )
    )
    console.print(
        Text.assemble(("       ground truth  ", "reach.label"), ground),
        soft_wrap=True,
    )
    console.print(
        Text(
            "       formal evaluation evaluates routing against all resident skills in the catalog",
            style="reach.label",
        ),
        soft_wrap=True,
    )


def print_resuming(console: Console, recovered: int, owed: int, path: Path) -> None:
    """Print status of recovered targets when resuming a partially completed draft."""
    console.print(
        Text.assemble(
            "resuming ",
            (str(path), "reach.label"),
            f": {recovered} targets already drafted, {owed} still owed",
        ),
        soft_wrap=True,
    )


def print_drafted(console: Console, target: str, drafted: int) -> None:
    """Print completion notice for a single target skill's drafted queries."""
    console.print(
        Text.assemble("  ", (target, "reach.label"), f": {drafted} grounded drafts"),
        soft_wrap=True,
    )


def print_generation_spend(console: Console, cost_usd: float, calls: int) -> None:
    """Print total dollar cost and API call count for synthetic query drafting."""
    if not calls:
        return
    console.print(
        Text.assemble(
            ("spent  ", "reach.label"),
            f"${cost_usd:.4f} over {calls} generation calls",
        ),
        soft_wrap=True,
    )


def print_draft_preview(
    console: Console,
    *,
    destination: Path | None,
    longest_prompt: int,
    budget: int | None,
) -> None:
    """Print dry-run preview of prompt sizes and output destination."""
    if destination is not None:
        console.print(
            Text.assemble(
                ("would write  ", "reach.label"),
                str(destination),
                " and stop there; probing is a second invocation",
            ),
            soft_wrap=True,
        )
    if budget:
        console.print(
            Text.assemble(
                ("longest prompt  ", "reach.label"),
                f"{longest_prompt:,} chars of the {budget:,} this runtime takes "
                f"({longest_prompt / budget:.0%})",
            ),
            soft_wrap=True,
        )


def print_wrote(console: Console, path: Path, *, then: str = "") -> None:
    """Print notification of written file path with optional next-step instructions."""
    console.print(Text.assemble(("wrote  ", "reach.label"), str(path)), soft_wrap=True)
    if then:
        console.print(Text(then, style="reach.label"), soft_wrap=True)


def _cell(text: str, style: str = "") -> Text:
    """Format single-line text cell with ellipsis overflow truncation."""
    return Text(text, style=style, no_wrap=True, overflow="ellipsis")


def middle_truncate(value: str, width: int) -> str:
    """Truncate text by replacing middle characters with an ellipsis."""
    if width <= 0 or len(value) <= width:
        return value
    if width == 1:
        return "…"
    keep = width - 1
    tail = (keep + 1) // 2
    return f"{value[: keep - tail]}…{value[len(value) - tail :]}"


class _TruncatedName:
    """Renderable cell that middle-truncates long skill names."""

    def __init__(self, value: str, style: str = "") -> None:
        self.value = value
        self.style = style

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> Generator[Text]:
        """Draw the middle-truncated name text within available max_width."""
        yield Text(
            middle_truncate(self.value, options.max_width),
            style=self.style,
            no_wrap=True,
            overflow="ellipsis",
        )

    def __rich_measure__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> Measurement:
        """Measure the minimum and maximum width required for this cell."""
        return Measurement(min(1, len(self.value)), len(self.value))


def _leak(leak: Leak | None) -> Text:
    """Format leak check results into a styled Text cell."""
    if leak is None:
        return _cell("")
    if not leak.leaked:
        return _cell("clean", style="reach.hit")
    return _cell("; ".join(leak.routes), style="reach.misroute")


def print_query_set(
    console: Console,
    query_set: QuerySet,
    *,
    path: Path | None,
    catalog_id: str,
    residents: int,
    ranks: Mapping[str, LexicalRank],
    flags: Mapping[str, Leak],
    then: str = "",
) -> None:
    """Render table of drafted queries."""
    console.print(
        Text.assemble(
            (catalog_id, "reach.catalog"),
            f": {residents} skills, ",
            (str(len(query_set.queries)), "reach.count"),
            " queries drafted",
        ),
        soft_wrap=True,
    )
    table = Table(box=box.SIMPLE, pad_edge=False, header_style="reach.label")
    table.add_column("query")
    table.add_column("expects")
    table.add_column("rank", justify="right")
    table.add_column("leak")
    table.add_column("text")
    for query in query_set.queries:
        rank = ranks.get(query.id)
        table.add_row(
            _cell(query.id),
            _cell(query.truth_label),
            _cell(str(rank) if rank is not None else ""),
            _leak(flags.get(query.id)),
            _cell(query.text, style="reach.digest"),
        )
    console.print(table)
    if path is not None:
        print_wrote(console, path, then=then)
    elif then:
        console.print(Text(then, style="reach.label"), soft_wrap=True)


def print_query_view(
    console: Console,
    query_set: QuerySet,
    *,
    ranks: Mapping[str, LexicalRank],
    flags: Mapping[str, Leak] | None = None,
    citations: Mapping[tuple[str, str], str] | None = None,
) -> None:
    """Render query set details as an inspection table."""
    table = Table(box=box.SIMPLE, pad_edge=False, header_style="reach.label")
    table.add_column("query")
    table.add_column("expects")
    table.add_column("rank", justify="right")
    if flags is not None:
        table.add_column("leak")
    if citations is not None:
        table.add_column("citation")
    table.add_column("text")
    for query in query_set.queries:
        rank = ranks.get(query.id)
        row = [
            _cell(query.id),
            _cell(query.truth_label),
            _cell(str(rank) if rank is not None else ""),
        ]
        if flags is not None:
            row.append(_leak(flags.get(query.id)))
        if citations is not None:
            key = (query.expected_skill or "", query.text)
            row.append(_cell(citations.get(key, ""), style="reach.digest"))
        row.append(_cell(query.text, style="reach.digest"))
        table.add_row(*row)
    console.print(table)
