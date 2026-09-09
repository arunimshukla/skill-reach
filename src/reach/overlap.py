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

"""Rank and analyze lexical overlap between skill descriptions using BM25."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

from pydantic import BaseModel, ConfigDict, PrivateAttr

from reach.retrieval import (
    Bm25Scorer,
    skill_text,
    tokenize,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reach.models import Skill

__all__ = [
    "Competition",
    "CorpusOverlap",
    "Rival",
    "Standing",
    "compete",
    "rank_corpus",
]


class Rival(BaseModel):
    """Represent a competitor skill and its lexical overlap score."""

    model_config = ConfigDict(frozen=True)

    name: str
    score: float


class Standing(BaseModel):
    """Represent a skill's ranked standing in a competition field."""

    model_config = ConfigDict(frozen=True)

    is_target: bool
    name: str
    rank: int
    score: float


class Competition(BaseModel):
    """Summarize lexical competition against a target skill across the corpus."""

    model_config = ConfigDict(frozen=True)

    rivals: tuple[Rival, ...] = ()
    self_score: float
    skill: str

    _ranked: tuple[Rival, ...] = PrivateAttr()

    @override
    def model_post_init(self, _context: object) -> None:
        """Cache pre-sorted rivals after initialization."""
        object.__setattr__(
            self,
            "_ranked",
            tuple(sorted(self.rivals, key=lambda r: (-r.score, r.name))),
        )

    @property
    def ranked_rivals(self) -> tuple[Rival, ...]:
        """Return rivals sorted by descending score, then ascending name."""
        return self._ranked

    @property
    def outranked_by(self) -> int:
        """Count the number of rivals with a score strictly higher than self_score."""
        return sum(1 for rival in self.rivals if rival.score > self.self_score)

    @property
    def nearest_rival(self) -> Rival | None:
        """Return the top-scoring rival, or None if no rivals exist."""
        return self._ranked[0] if self._ranked else None

    @property
    def scoring_rival(self) -> Rival | None:
        """Return the top rival only if it has a non-zero overlap score."""
        nearest = self.nearest_rival
        return nearest if nearest is not None and nearest.score > 0 else None

    @property
    def rival_ratio(self) -> float:
        """Calculate the nearest rival's score relative to the target's self score."""
        nearest = self.nearest_rival
        if nearest is None:
            return 0.0
        if self.self_score > 0:
            return nearest.score / self.self_score
        return math.inf if nearest.score > 0 else 0.0

    @property
    def standings(self) -> tuple[Standing, ...]:
        """Return full competition standings including target skill placement."""
        seats = list(self.ranked_rivals)
        placed: list[tuple[str, float, bool]] = [(r.name, r.score, False) for r in seats]
        placed.insert(self.outranked_by, (self.skill, self.self_score, True))
        return tuple(
            Standing(rank=i, name=name, score=score, is_target=is_target)
            for i, (name, score, is_target) in enumerate(placed, start=1)
        )


class CorpusOverlap(BaseModel):
    """Hold competition rankings for all skills across the corpus."""

    model_config = ConfigDict(frozen=True)

    competitions: tuple[Competition, ...] = ()

    def find(self, skill: str) -> Competition:
        """Retrieve the Competition model for a named skill."""
        for competition in self.competitions:
            if competition.skill == skill:
                return competition
        msg = f"no skill named {skill!r} in this corpus"
        raise ValueError(msg)


def compete(skill: Skill, corpus: Sequence[Skill], scorer: Bm25Scorer) -> Competition:
    """Score a target skill against itself and all other corpus skills."""
    query = tokenize(skill_text(skill))
    return Competition(
        skill=skill.name,
        self_score=scorer.score(query, skill.name),
        rivals=tuple(
            Rival(name=c.name, score=scorer.score(query, c.name))
            for c in corpus
            if c.name != skill.name
        ),
    )


def rank_corpus(skills: Sequence[Skill]) -> CorpusOverlap:
    """Compute and rank lexical overlap competition across a corpus of skills."""
    scorer = Bm25Scorer.from_skills(skills)
    competitions = [compete(skill, skills, scorer) for skill in skills]
    ordered = sorted(competitions, key=lambda c: (-c.rival_ratio, c.skill))
    return CorpusOverlap(competitions=tuple(ordered))


#: Standard user-facing advisory note regarding description overlap vs runtime routing.
OVERLAP_CAVEAT = (
    "Description overlap reflects lexical similarity, not agent routing behavior.",
    (
        "Static overlap does not predict which skills misroute; "
        "use `reach eval` for empirical testing."
    ),
    "Static analysis only: no probe was issued.",
)
