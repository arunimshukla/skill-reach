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
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, override

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


class KeywordRuntime(AgentRuntime):
    """Implement a deterministic keyword matching agent for offline evaluation."""

    name = "keyword"
    _skills_subpath = ".agents/skills"

    def __init__(
        self,
        settings_or_options: RuntimeSettings | KeywordOptions | None = None,
        options: KeywordOptions | None = None,
    ) -> None:
        """Initialize keyword heuristic runtime with settings or options."""
        opts: KeywordOptions
        if isinstance(settings_or_options, KeywordOptions):
            self.settings = RuntimeSettings(agent="keyword")
            opts = settings_or_options
        elif isinstance(settings_or_options, RuntimeSettings):
            self.settings = settings_or_options
            opts = options or (
                KeywordOptions.model_validate(dict(settings_or_options.options or {}))
                if isinstance(settings_or_options.options, dict)
                else KeywordOptions()
            )
        else:
            self.settings = RuntimeSettings(agent="keyword")
            opts = options or KeywordOptions()
        self.options: KeywordOptions = opts
        self._resident: tuple[str, ...] = ()
        self._pattern: re.Pattern[str] | None = None
        self._term_to_skill: dict[str, str] = {}

    def _set_resident(self, resident: Sequence[str]) -> None:
        """Precompute normalized term mapping and compiled regex for fast matching."""
        self._resident = tuple(resident)
        term_map: dict[str, str] = {}
        for name in resident:
            term_map[name.lower()] = name
            term_map[name.replace("-", " ").lower()] = name

        sorted_terms = sorted(term_map.keys(), key=len, reverse=True)
        if sorted_terms:
            escaped = "|".join(re.escape(t) for t in sorted_terms)
            self._pattern = re.compile(rf"(?<![\w-])({escaped})(?![\w-])", re.IGNORECASE)
            self._term_to_skill = term_map
        else:
            self._pattern = None
            self._term_to_skill = {}

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

            invoked: str | None = None
            if self._pattern is not None:
                match = self._pattern.search(query_text)
                if match:
                    matched_term = match.group(1).lower()
                    invoked = self._term_to_skill.get(matched_term)

            invoked_skills = (invoked,) if invoked is not None else ()
            early_exit_hit = bool(
                self.options.early_exit and invoked is not None and invoked == target_skill,
            )
            return SelectionOutcome(
                invoked_skills=invoked_skills,
                early_exit=early_exit_hit,
                turns_taken=1,
                observed_catalog=self._resident,
                observed_tools=("keyword",),
                cost_usd=0.0,
                duration_ms=1,
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
        text = "\n".join(line_list)
        active_resident = resident or self._resident
        invoked: str | None = None

        if active_resident and active_resident == self._resident and self._pattern is not None:
            match = self._pattern.search(text)
            if match:
                invoked = self._term_to_skill.get(match.group(1).lower())
        elif active_resident:
            term_map: dict[str, str] = {}
            for name in active_resident:
                term_map[name.lower()] = name
                term_map[name.replace("-", " ").lower()] = name
            sorted_terms = sorted(term_map.keys(), key=len, reverse=True)
            escaped = "|".join(re.escape(t) for t in sorted_terms)
            pat = re.compile(rf"(?<![\w-])({escaped})(?![\w-])", re.IGNORECASE)
            match = pat.search(text)
            if match:
                invoked = term_map.get(match.group(1).lower())

        invoked_skills = (invoked,) if invoked is not None else ()
        return SessionSummary(
            invoked_skills=invoked_skills,
            early_exit=early_exit,
            status=SessionStatus.SUCCESS if line_list else None,
        )


class KeywordGenerator(BaseTextGenerator[None]):
    """Stub text generator for keyword search drivers."""

    name: str = "keyword"

    @override
    def complete(self, prompt: str) -> str:
        """Return empty string as keyword drivers do not generate text."""
        del prompt
        self.completions += 1
        return ""
