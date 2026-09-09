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

"""Model and calculate Claude Code skill listing budgets, display widths, and fitting."""

from __future__ import annotations

import unicodedata
from math import ceil, floor
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field

from reach.config import DEFAULT_CLAUDE_MODEL
from reach.runtime.profiles import ModelProfile, model_profile

#: Ordinal bounds for standard ASCII printable character glyphs.
ASCII_PRINTABLE_MIN: Final = ord(" ")
ASCII_PRINTABLE_MAX: Final = ord("~")

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from reach.models import Skill

__all__ = [
    "BUDGET_FRACTION_PLACES",
    "DEFAULT_CHARS_PER_TOKEN",
    "DEFAULT_COMPLETION_WINDOW",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_LISTING_BUDGET_CHARS",
    "WIDE",
    "ZERO_WIDTH",
    "ListingEntry",
    "SkillListing",
    "budget_fraction_for",
    "display_width",
    "fit_skill_listing",
    "listing_budget_chars",
    "listing_chars",
    "parse_fraction",
]

_SONNET_LISTING = model_profile(DEFAULT_CLAUDE_MODEL)
if _SONNET_LISTING is None or _SONNET_LISTING.completion_window is None:
    msg = f"reach.toml is missing {DEFAULT_CLAUDE_MODEL}'s verified listing profile"
    raise RuntimeError(msg)

DEFAULT_CONTEXT_WINDOW: int = _SONNET_LISTING.context_window
DEFAULT_CHARS_PER_TOKEN: int = int(_SONNET_LISTING.chars_per_token)

DEFAULT_COMPLETION_WINDOW: int = _SONNET_LISTING.completion_window
BUDGET_FRACTION_PLACES: int = _SONNET_LISTING.budget_fraction_places
DEFAULT_LISTING_BUDGET_CHARS: int = floor(
    DEFAULT_CONTEXT_WINDOW * DEFAULT_CHARS_PER_TOKEN * _SONNET_LISTING.listing_budget_fraction,
)


def _listing_profile(model: str) -> ModelProfile:
    """Return context window and token constants for the target model."""
    return model_profile(model)


def listing_budget_chars(
    fraction: float | None = None,
    model: str = DEFAULT_CLAUDE_MODEL,
) -> int:
    """Calculate character listing budget permitted by the CLI for a model."""
    profile = _listing_profile(model)
    resolved_fraction = profile.listing_budget_fraction if fraction is None else fraction
    window = profile.context_window * profile.chars_per_token
    return floor(window * resolved_fraction)


def budget_fraction_for(chars: int, model: str = DEFAULT_CLAUDE_MODEL) -> float | None:
    """Calculate minimum budget fraction required to fit character length."""
    profile = _listing_profile(model)
    window = profile.context_window * profile.chars_per_token

    scale = 10**profile.budget_fraction_places
    fraction = ceil(chars / window * scale) / scale

    while fraction <= 1.0 and listing_budget_chars(fraction, model) < chars:
        fraction = (round(fraction * scale) + 1) / scale
    return fraction if fraction <= 1.0 else None


def parse_fraction(value: str) -> float | None:
    """Parse budget fraction string enforcing bounds (0.0, 1.0]."""
    try:
        fraction = float(value)
    except ValueError as exc:
        msg = f"fraction must be a number: {value!r}"
        raise ValueError(msg) from exc
    if fraction <= 0.0:
        msg = f"fraction must be positive: {fraction}"
        raise ValueError(msg)
    return fraction if fraction <= 1.0 else None


def display_width(text: str) -> int:
    """Calculate terminal column display width for a given text string."""
    if text.isascii():
        if text.isprintable():
            return len(text)
        return sum(1 for char in text if ASCII_PRINTABLE_MIN <= ord(char) <= ASCII_PRINTABLE_MAX)
    return sum(
        (
            0
            if unicodedata.combining(char) or unicodedata.category(char) in ZERO_WIDTH
            else 2
            if unicodedata.east_asian_width(char) in WIDE
            else 1
        )
        for char in text
    )


ZERO_WIDTH = frozenset({"Cc", "Cf", "Mn", "Me"})
WIDE = frozenset({"W", "F"})


class ListingEntry(BaseModel):
    """Represent character and column cost metrics for a single resident skill."""

    model_config = ConfigDict(frozen=True)

    skill: str
    full_chars: int = Field(ge=0)
    chars: int = Field(ge=0)
    protected: bool = False

    @property
    def described(self) -> bool:
        """Return True if the skill includes its full description in the listing."""
        return self.chars >= self.full_chars


class SkillListing(BaseModel):
    """Represent column costs and description retention for a catalog listing."""

    model_config = ConfigDict(frozen=True)

    budget_chars: int = Field(ge=0)
    full_chars: int = Field(ge=0)
    chars: int = Field(ge=0)
    entries: tuple[ListingEntry, ...] = ()

    @property
    def over_budget(self) -> bool:
        """Return True if the complete listing exceeds the character budget."""
        return self.full_chars > self.budget_chars

    @property
    def fits(self) -> bool:
        """Return True if the fitted listing is within allowable budget boundaries."""
        return self.chars <= self.budget_chars

    @property
    def described(self) -> tuple[str, ...]:
        """Return names of all skills whose full descriptions are retained."""
        return tuple(entry.skill for entry in self.entries if entry.described)

    @property
    def name_only(self) -> tuple[str, ...]:
        """Return names of skills truncated to bare name strings."""
        return tuple(entry.skill for entry in self.entries if not entry.described)


def _entry_chars(
    skills: Sequence[Skill],
    max_desc_chars: int | None,
) -> tuple[tuple[str, int, int], ...]:
    """Calculate full and bare display column widths for a collection of skills."""
    priced = []
    for skill in skills:
        description = skill.description
        if max_desc_chars is not None and len(description) > max_desc_chars:
            description = description[: max_desc_chars - 1] + "\N{HORIZONTAL ELLIPSIS}"
        priced.append(
            (
                skill.name,
                display_width(f"- {skill.name}: {description}"),
                display_width(f"- {skill.name}"),
            ),
        )
    return tuple(priced)


def _joined(costs: Iterable[int], count: int) -> int:
    """Calculate total character cost including newline separators."""
    return sum(costs) + max(0, count - 1)


def listing_chars(skills: Sequence[Skill], *, max_desc_chars: int | None = None) -> int:
    """Return total display column cost of formatting skills without budget limits."""
    priced = _entry_chars(skills, max_desc_chars)
    return _joined((full for _, full, _ in priced), len(priced))


def _select_budget_entries(
    priced: Sequence[tuple[str, int, int]],
    shielded: frozenset[str],
    scores: Mapping[str, float],
    budget_chars: int,
    count: int,
) -> set[str]:
    """Greedily allocate remaining character headroom to candidate skills by priority."""
    floor = _joined(
        (entry if name in shielded else bare for name, entry, bare in priced),
        count,
    )
    headroom = budget_chars - floor
    kept = set(shielded)
    candidates = sorted(
        (row for row in priced if row[0] not in shielded),
        key=lambda row: scores.get(row[0], 0.0),
        reverse=True,
    )
    for name, entry, bare in candidates:
        if (cost := entry - bare) <= headroom:
            headroom -= cost
            kept.add(name)
    return kept


def fit_skill_listing(
    skills: Sequence[Skill],
    *,
    budget_chars: int = DEFAULT_LISTING_BUDGET_CHARS,
    max_desc_chars: int | None = None,
    protected: Iterable[str] = (),
    priority: Mapping[str, float] | None = None,
) -> SkillListing:
    """Compute description retention under listing budget constraints."""
    priced = _entry_chars(skills, max_desc_chars)
    shielded = frozenset(protected)
    scores = priority or {}
    count = len(priced)
    full = _joined((entry for _, entry, _ in priced), count)

    if full <= budget_chars:
        kept = {name for name, _, _ in priced}
    else:
        kept = _select_budget_entries(priced, shielded, scores, budget_chars, count)

    return SkillListing(
        budget_chars=budget_chars,
        full_chars=full,
        chars=_joined(
            (entry if name in kept else bare for name, entry, bare in priced),
            count,
        ),
        entries=tuple(
            ListingEntry(
                skill=name,
                full_chars=entry,
                chars=entry if name in kept else bare,
                protected=name in shielded,
            )
            for name, entry, bare in priced
        ),
    )
