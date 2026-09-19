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

"""Define validated boundary data models and schemas shared across the harness."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from reach.runtime import CatalogFit, SelectionOutcome

__all__ = [
    "NO_SKILL",
    "Catalog",
    "CatalogMode",
    "DisclosureState",
    "InvocationPattern",
    "ProbeResult",
    "Provenance",
    "Query",
    "QueryKind",
    "Skill",
]

#: Sentinel label representing abstention (no skill invoked).
NO_SKILL = "(no skill)"


class QueryKind(StrEnum):
    """Enumerate structural categories of authored/generated evaluation queries."""

    CONTEXTUAL = "contextual"
    IMPLICIT = "implicit"
    NEIGHBOR_NEGATIVE = "neighbor_negative"
    OUT_OF_SCOPE = "out_of_scope"


class CatalogMode(StrEnum):
    """Enumerate strategies for composing resident skill catalogs."""

    ALL = "all"
    NEIGHBORHOOD = "neighborhood"
    SINGLETON = "singleton"
    SWEEP = "sweep"


class InvocationPattern(StrEnum):
    """Categorize observed probe trajectory invocation behavior."""

    ABANDONED = "abandoned"
    CORRECT_ABSTENTION = "correct_abstention"
    DISTRACTOR_HIJACK = "distractor_hijack"
    MIXED_ORACLE = "mixed_oracle"
    ORACLE_ONLY = "oracle_only"
    UNWANTED_TRIGGER = "unwanted_trigger"


class DisclosureState(StrEnum):
    """Reflect how much of a skill's selection surface reached the model context."""

    ABSENT = "absent"
    FULL = "full"
    NAME_ONLY_ELIDED = "name_only_elided"
    WITHHELD = "withheld"


class Skill(BaseModel):
    """Represent a skill's selection surface and metadata from SKILL.md."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Unique identifier and directory name of the skill.")
    description: str = Field(
        description="Selection surface text presented to the model in the system prompt.",
    )
    path: Path = Field(description="Filesystem path to the skill directory.")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata parsed from frontmatter.",
    )
    allowed_tools: tuple[str, ...] = Field(
        default=(),
        description="Explicit tools authorized by allowed-tools frontmatter.",
    )
    declared_dependencies: tuple[str, ...] = Field(
        default=(),
        description="Union of dependencies extracted from allowed-tools and metadata.",
    )
    manifest_source: str | None = Field(
        default=None,
        description="Upstream package source URL or identifier from lockfile.",
    )
    model_invocable: bool = Field(
        default=True,
        description="Whether the skill allows model invocation.",
    )

    @field_validator("name")
    @classmethod
    def _require_name(cls, value: str) -> str:
        """Validate that the skill name contains non-whitespace text."""
        clean = value.strip()
        if not clean:
            msg = "name must be non-empty"
            raise ValueError(msg)
        return clean

    @field_validator("description")
    @classmethod
    def _require_description(cls, value: str) -> str:
        """Validate that the skill description contains non-whitespace text."""
        if not value.strip():
            msg = "description must be non-empty"
            raise ValueError(msg)
        return value


class Query(BaseModel):
    """Represent a single evaluation probe query and expected target skill."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Unique identifier for the evaluation query.")
    text: str = Field(description="Realistic user request text presented to the agent.")
    kind: QueryKind | None = Field(
        default=None,
        description="Structural category of query (implicit, contextual, negative, out of scope).",
    )
    expected_skill: str | None = Field(
        default=None,
        description="Ground truth skill name expected to be invoked, or None if out of scope.",
    )
    acceptable_skills: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Optional secondary skill names considered valid matches for this query.",
    )
    notes: str = Field(default="", description="Author notes, rationale, or difficulty context.")

    @model_validator(mode="after")
    def _out_of_scope_is_consistent(self) -> Self:
        """Validate out-of-scope query kinds align with absence of expected skills."""
        if (self.kind is QueryKind.OUT_OF_SCOPE) != (self.expected_skill is None):
            msg = (
                f"{self.id}: kind={self.kind} disagrees with expected_skill={self.expected_skill!r}"
            )
            raise ValueError(
                msg,
            )
        return self

    @property
    def is_out_of_scope(self) -> bool:
        """Check whether the query represents an out-of-scope intent."""
        return self.expected_skill is None

    @property
    def truth_label(self) -> str:
        """Return expected skill name or NO_SKILL sentinel for out-of-scope queries."""
        return self.expected_skill if self.expected_skill is not None else NO_SKILL

    @property
    def valid_skills(self) -> frozenset[str]:
        """Return all valid target skill names (expected_skill plus acceptable_skills)."""
        if self.expected_skill is None:
            return frozenset()
        return frozenset((self.expected_skill, *self.acceptable_skills))

    def matches_skill(self, invoked: str | None) -> bool:
        """Check whether an invoked skill satisfies expected_skill or acceptable_skills."""
        if self.expected_skill is None:
            return invoked is None
        return invoked is not None and invoked in self.valid_skills

    def effective_predicted_label(self, predicted_label: str) -> str:
        """Normalize an acceptable secondary skill selection to truth_label for scoring."""
        if self.expected_skill is not None and predicted_label in self.acceptable_skills:
            return self.truth_label
        return predicted_label


class Catalog(BaseModel):
    """Represent a collection of resident skills presented during a probe execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    mode: CatalogMode
    skills: tuple[str, ...]
    target: str | None = None

    @model_validator(mode="after")
    def _target_is_resident(self) -> Self:
        """Validate that catalogs contain unique skills and neighborhood targets are resident."""
        if len(self.skills) != len(set(self.skills)):
            msg = f"{self.id}: catalog contains duplicate skill names"
            raise ValueError(msg)
        is_corpus_sweep = self.mode is CatalogMode.SWEEP and self.id.startswith("sweep:corpus")
        if (
            self.mode in (CatalogMode.NEIGHBORHOOD, CatalogMode.SWEEP)
            and not is_corpus_sweep
            and self.target is None
        ):
            msg = f"{self.id}: a {self.mode.value} catalog must name its target"
            raise ValueError(msg)
        if self.target is not None and self.target not in self.skills:
            msg = f"{self.id}: target {self.target!r} is not resident"
            raise ValueError(msg)
        return self

    @property
    def size(self) -> int:
        """Return the number of resident skills in the catalog."""
        return len(self.skills)


class Provenance(BaseModel):
    """Record experiment fingerprints, digest identifiers, and run labels."""

    model_config = ConfigDict(frozen=True)

    config_fingerprint: str = ""
    condition_digest: str = ""
    corpus_digest: str = ""
    queries_digest: str = ""
    tag: str = ""


class ProbeResult(BaseModel):
    """Stage 2 telemetry: record execution telemetry and observed selections for a single probe."""

    query_id: str
    catalog_id: str
    catalog_mode: CatalogMode
    catalog_size: int
    model: str
    runtime: str
    resolved_model: str = ""
    config_fingerprint: str = ""

    condition_digest: str = ""
    corpus_digest: str = ""
    queries_digest: str = ""
    attempt: int = 1
    invoked_skills: tuple[str, ...] = ()
    early_exit: bool = False
    turns_taken: int = Field(default=1, ge=1)
    reasoning: tuple[str, ...] = ()
    observed_catalog: tuple[str, ...] = ()
    observed_tools: tuple[str, ...] = ()
    cost_usd: float | None = None
    duration_ms: int | None = None
    prompt_tokens: int | None = None
    error: str | None = None
    disclosure_state: DisclosureState = DisclosureState.FULL
    invocation_pattern: InvocationPattern | None = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def invoked_skill(self) -> str | None:
        """Return the first invoked skill name, or None if none was invoked."""
        return self.invoked_skills[0] if self.invoked_skills else None

    @property
    def selected(self) -> bool:
        """Check whether any skill was invoked during the probe."""
        return bool(self.invoked_skills)

    @property
    def predicted_label(self) -> str:
        """Return invoked skill name or NO_SKILL sentinel if no skill was selected."""
        return self.invoked_skills[0] if self.invoked_skills else NO_SKILL

    @classmethod
    def from_outcome(
        cls,
        outcome: SelectionOutcome,
        query: Query,
        catalog: Catalog,
        runtime_name: str,
        model: str = "",
        attempt: int = 1,
        provenance: Provenance | None = None,
        error: str | None = None,
        fit: CatalogFit | None = None,
        is_dynamic: bool = False,
    ) -> ProbeResult:
        """Construct a ProbeResult (Stage 2) from a runtime SelectionOutcome (Stage 1)."""
        from reach.metrics import classify_invocation_pattern

        prov = provenance or Provenance()
        final_error = error or outcome.error
        pattern = classify_invocation_pattern(query, outcome.invoked_skills)

        if is_dynamic:
            if query.expected_skill and query.expected_skill not in outcome.observed_catalog:
                disc_state = DisclosureState.WITHHELD
            else:
                disc_state = DisclosureState.FULL
        elif fit and query.expected_skill and query.expected_skill in fit.elided_skills:
            disc_state = DisclosureState.NAME_ONLY_ELIDED
        else:
            disc_state = DisclosureState.FULL

        return cls(
            query_id=query.id,
            catalog_id=catalog.id,
            catalog_mode=catalog.mode,
            catalog_size=catalog.size,
            model=model,
            resolved_model=outcome.resolved_model,
            runtime=runtime_name,
            config_fingerprint=prov.config_fingerprint,
            condition_digest=prov.condition_digest,
            corpus_digest=prov.corpus_digest,
            queries_digest=prov.queries_digest,
            attempt=attempt,
            invoked_skills=outcome.invoked_skills,
            early_exit=outcome.early_exit,
            turns_taken=outcome.turns_taken,
            reasoning=outcome.reasoning,
            observed_catalog=outcome.observed_catalog,
            observed_tools=outcome.observed_tools,
            cost_usd=outcome.cost_usd,
            duration_ms=outcome.duration_ms,
            prompt_tokens=outcome.prompt_tokens,
            error=final_error,
            disclosure_state=disc_state,
            invocation_pattern=pattern,
        )
