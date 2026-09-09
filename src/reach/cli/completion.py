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

"""Generate or install shell tab completion scripts for bash, zsh, and fish."""

from __future__ import annotations

import functools
import os
from typing import Annotated, Literal

from cyclopts import App, Parameter
from cyclopts.completion import _base as _cyclopts_completion_base
from cyclopts.completion._base import CompletionData

from reach.views import build_console

from .app import SETUP, app
from .flags import SWITCH

_orig_extract_completion_data = _cyclopts_completion_base.extract_completion_data
_completion_data_cache: dict[int, dict[tuple[str, ...], CompletionData]] = {}


def _cached_extract_completion_data(app: App) -> dict[tuple[str, ...], CompletionData]:
    """Cache completion data extraction across shell types."""
    key = id(app)
    if key not in _completion_data_cache:
        _completion_data_cache[key] = _orig_extract_completion_data(app)
    return _completion_data_cache[key]


_cyclopts_completion_base.extract_completion_data = _cached_extract_completion_data  # type: ignore  # noqa: PGH003


@functools.lru_cache(maxsize=4)
def _generate_completion_script(target_shell: Literal["bash", "zsh", "fish"] | None) -> str:
    """Generate shell completion script with caching."""
    return app.generate_completion(shell=target_shell)


def _detect_shell() -> Literal["bash", "zsh", "fish"]:
    """Detect current shell from SHELL environment variable."""
    shell_env = os.environ.get("SHELL", "").lower()
    if "zsh" in shell_env:
        return "zsh"
    if "bash" in shell_env:
        return "bash"
    if "fish" in shell_env:
        return "fish"
    return "zsh"


@app.command(name="completion", group=SETUP)
def _completion(
    shell: Annotated[
        Literal["bash", "zsh", "fish"] | None,
        Parameter(
            help="Shell type for completion script generation (defaults to current $SHELL)",
        ),
    ] = None,
    *,
    install: Annotated[
        bool,
        SWITCH,
        Parameter(
            name=["--install", "-i"],
            help="Install completion script directly to user shell startup configuration",
        ),
    ] = False,
) -> int:
    """Generate or install shell tab-completion scripts for Reach."""
    target_shell = shell or _detect_shell()

    if install:
        console = build_console()
        try:
            installed_path = app.install_completion(shell=target_shell)
        except Exception as err:  # noqa: BLE001
            console.print(f"[red]Error installing completion:[/] {err}")
            return 1
        else:
            console.print(
                f"[green]✓[/green] Shell completion for [bold]{target_shell}[/bold] "
                f"installed to [bold]{installed_path}[/bold].",
            )
            console.print("Restart your shell session to activate autocomplete.")
            return 0

    script = _generate_completion_script(target_shell)
    print(script, end="")
    return 0
