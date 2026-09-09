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

"""Define CLI application root, command groupings, and error formatting."""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING, Annotated

from cyclopts import App, Group, Parameter
from pydantic import ValidationError as PydanticValidationError

from reach.views import build_console, help_formatter

if TYPE_CHECKING:
    from cyclopts.exceptions import CycloptsError

#: Command group for core evaluation verbs.
LOOP = Group("The loop", sort_key=0)

#: Command group for setup, environment health checks, and autocomplete.
SETUP = Group("Setup & Diagnostics", sort_key=1)

#: Command group for tool metadata and help commands.
ABOUT = Group("About", sort_key=2)

#: Parameter group for top-level options.
OPTIONS = Group("Options", sort_key=0)

app = App(
    name="reach",
    help="Measure whether a skill is reachable when its rivals are resident.",
    version=metadata.version("skill-reach"),
    help_formatter=help_formatter(),
    group_commands=ABOUT,
    group_parameters=OPTIONS,
    default_parameter=Parameter(allow_repeating=True),
    result_action="return_value",
)


def _verbs() -> list[str]:
    """Return the list of registered command names in the CLI application."""
    return [name for name in app if not name.startswith("-")]


@app.default
def _no_verb(*tokens: Annotated[str, Parameter(show=False)]) -> int:
    """Print error message and usage instructions when no valid command is supplied."""
    complaint = build_console()
    if tokens:
        complaint.print(f"reach: unknown command {tokens[0]!r}")
        complaint.print(f"       choose from: {', '.join(_verbs())}")
    app.help_print([], console=complaint)
    return 2


def _explain(error: CycloptsError) -> list[str]:
    """Format Cyclopts and Pydantic validation errors into human-readable messages."""
    cause = error.__cause__
    if not isinstance(cause, PydanticValidationError):
        return [str(error)]
    refusals = []
    for failure in cause.errors():
        loc = failure.get("loc", ())
        field = str(loc[-1]).replace("_", "-") if loc else "input"
        refusals.append(
            f"Invalid value {failure['input']!r} for --{field}. {failure['msg']}.",
        )
    return refusals
