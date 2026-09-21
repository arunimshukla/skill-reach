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

"""Provide a deterministic keyword-matching runtime for offline CI evaluation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Self, override

from pydantic import Field

from reach.config import RuntimeSettings, resolve_path
from reach.runtime import (
    AgentOptions,
    AgentRuntime,
    SelectionOutcome,
    SessionStatus,
    SessionSummary,
    SkillRoot,
)
from reach.runtime.generator import BaseTextGenerator

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "KeywordGenerator",
    "KeywordOptions",
    "KeywordRuntime",
]


class KeywordOptions(AgentOptions):
    """Specify configuration options for keyword heuristic routing."""

    model: str = "keyword"
    scope: str = Field(default="user")


class KeywordRuntime(AgentRuntime[KeywordOptions]):
    """Implement a deterministic keyword matching agent for offline evaluation."""

    name = "keyword"
    _skills_subpath = ".agents/skills"

    def __init__(
        self,
        settings_or_options: RuntimeSettings | KeywordOptions | None = None,
        options: KeywordOptions | None = None,
    ) -> None:
        """Initialize keyword heuristic runtime with settings or options."""
        resolved_settings: RuntimeSettings | None = None
        resolved_options: KeywordOptions | None = options
        if isinstance(settings_or_options, KeywordOptions):
            resolved_options = settings_or_options
        elif isinstance(settings_or_options, RuntimeSettings):
            resolved_settings = settings_or_options

        super().__init__(settings=resolved_settings, options=resolved_options)
        self._pattern: re.Pattern[str] | None = None
        self._term_to_skill: dict[str, str] = {}

    def _set_resident(self, resident: Sequence[str]) -> None:
        """Precompute normalized term mapping and compiled regex for fast matching."""
        self._resident = tuple(resident)
        self._pattern, self._term_to_skill = _compile_term_matcher(self._resident)

    @override
    def clone_isolated(self) -> Self:
        """Create a thread-local isolated clone with fresh regex and term map state."""
        clone = super().clone_isolated()
        clone._pattern = None  # noqa: SLF001
        clone._term_to_skill = {}  # noqa: SLF001
        return clone

    def match_skill(self, text: str, resident: Sequence[str] = ()) -> str | None:
        """Find the first matching resident skill mentioned in the given text."""
        active = tuple(resident) if resident else self._resident
        if not active:
            return None
        pattern: re.Pattern[str] | None
        if active == self._resident and self._pattern is not None:
            pattern, term_map = self._pattern, self._term_to_skill
        else:
            pattern, term_map = _compile_term_matcher(active)

        if pattern is None:
            return None
        match = pattern.search(text)
        return term_map.get(match.group(1).lower()) if match else None

    @override
    def _post_install(self, workdir: Path) -> None:
        """Precompute normalized term mapping and compiled regex for fast matching."""
        del workdir
        self._set_resident(self._resident)

    @override
    def skill_roots(self, workdir: Path) -> tuple[SkillRoot, ...]:
        """Return default user skill directory for workspace."""
        here = self.skills_dir(resolve_path(workdir))
        return (SkillRoot(path=here, scope=self.options.scope),) if here.is_dir() else ()

    @override
    def select(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Route query to the first resident skill mentioned in query text."""
        try:
            if self._pattern is None and self._resident:
                self._set_resident(self._resident)

            matched = self.match_skill(query_text)
            return self.make_tracker(target_skill).apply_to_outcome(
                SelectionOutcome(
                    invoked_skills=(matched,) if matched is not None else (),
                    observed_catalog=self._resident,
                    observed_tools=("keyword",),
                    cost_usd=0.0,
                    duration_ms=1,
                ),
            )
        finally:
            self.post_probe(workdir)

    @override
    def parse_stream(
        self,
        lines: Iterable[str],
        resident: Sequence[str] = (),
        early_exit: bool = False,
    ) -> SessionSummary:
        """Extract invoked skills mentioned in stream lines."""
        line_list = list(lines)
        invoked = self.match_skill("\n".join(line_list), resident)
        invoked_skills = (invoked,) if invoked is not None else ()
        return SessionSummary(
            invoked_skills=invoked_skills,
            early_exit=early_exit,
            status=SessionStatus.SUCCESS if line_list else None,
        )


def _compile_term_matcher(
    resident: Sequence[str],
) -> tuple[re.Pattern[str] | None, dict[str, str]]:
    """Compile a regex pattern and term lookup mapping from a sequence of resident skills."""
    if not resident:
        return None, {}
    term_map: dict[str, str] = {}
    for name in resident:
        term_map[name.lower()] = name
        term_map[name.replace("-", " ").lower()] = name

    sorted_terms = sorted(term_map.keys(), key=len, reverse=True)
    escaped = "|".join(re.escape(t) for t in sorted_terms)
    pattern = re.compile(rf"(?<![\w-])({escaped})(?![\w-])", re.IGNORECASE)
    return pattern, term_map


class KeywordGenerator(BaseTextGenerator[KeywordOptions]):
    """Stub text generator for keyword search drivers."""

    name: str = "keyword"

    def __init__(
        self,
        model: str = "keyword",
        *,
        timeout_s: int = 300,
        options: KeywordOptions | None = None,
    ) -> None:
        """Initialize keyword text generator with model, timeout, and options."""
        super().__init__(
            model=model,
            timeout_s=timeout_s,
            options=options if options is not None else KeywordOptions(model=model),
        )

    @override
    def complete(self, prompt: str, *, schema: str | Mapping[str, Any] | None = None) -> str:
        """Return empty string as keyword drivers do not generate text."""
        del prompt, schema
        self.completions += 1
        return ""
