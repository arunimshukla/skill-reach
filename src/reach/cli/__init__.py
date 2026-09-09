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

"""Provide command-line interface entry point and command registration for Reach."""

from __future__ import annotations

import sys

from cyclopts.exceptions import CycloptsError

from reach.views import build_console, error_panel, help_console

from . import (  # noqa: F401
    check,
    clean,
    cluster,
    completion,
    diff,
    doctor,
    eval,
    init,
    lint,
    optimize,
    overlap,
    query,
    sweep,
    view,
)
from .app import _explain, _verbs, app
from .eval import EVAL_REQUIRED
from .flags import (
    AgentName,
    Format,
    StudyFlags,
    Vary,
    build_config,
    parse_agent_options,
)

__all__ = [
    "EVAL_REQUIRED",
    "AgentName",
    "Format",
    "StudyFlags",
    "Vary",
    "_verbs",
    "app",
    "build_config",
    "main",
    "parse_agent_options",
]


def _reorder_argv(argv: list[str] | None) -> list[str]:
    """Normalize argv by positioning recognized subcommands before leading options."""
    raw = sys.argv[1:] if argv is None else list(argv)
    if not raw:
        return []

    verbs = set(_verbs())
    verb_idx: int | None = None
    for i, token in enumerate(raw):
        if token in verbs:
            verb_idx = i
            break

    if verb_idx is None or verb_idx == 0:
        return raw

    before = raw[:verb_idx]
    verb = raw[verb_idx]
    after = raw[verb_idx + 1 :]
    return [verb, *after, *before]


def main(argv: list[str] | None = None) -> int:
    """Execute the CLI application and translate exceptions into exit codes."""
    effective_argv = _reorder_argv(argv)
    try:
        outcome = app(
            effective_argv,
            console=help_console(),
            error_console=build_console(),
            exit_on_error=False,
            error_formatter=lambda error: error_panel(_explain(error)),
        )
    except CycloptsError:
        return 2
    except (
        ValueError,
        KeyError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
    ) as error:
        said = error.args[0] if isinstance(error, KeyError) else str(error)
        build_console().print(error_panel([str(said)]))
        return 2
    return 0 if outcome is None else outcome


if __name__ == "__main__":
    raise SystemExit(main())
