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

"""Load, save, and validate labeled evaluation query sets and calculate digests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from reach._io import write_model
from reach.models import Query, QueryKind

if TYPE_CHECKING:
    from collections.abc import Iterable

    from reach.exchange import FieldMap

__all__ = [
    "Origin",
    "QuerySet",
    "QuerySetProvenance",
    "load_query_set",
    "query_set_digest",
    "save_query_set",
]


class Origin(StrEnum):
    """Enumerate origins of query sets."""

    AUTHORED = "authored"
    GENERATED = "generated"
    IMPORTED = "imported"


class QuerySetProvenance(BaseModel):
    """Record creation metadata, generator parameters, and review status."""

    model_config = ConfigDict(frozen=True)

    origin: Origin
    recorded_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    tool_version: str = ""
    generator_model: str = ""
    generator_arm: str = ""
    queries_per_target: int | None = None
    rivals_in_view: int | None = None
    bodies_digest: str = ""
    config_fingerprint: str = ""
    source: str = ""
    reviewed: bool | None = None
    adversarial: bool | None = None
    adversarial_per_target: int | None = None


class QuerySet(BaseModel):
    """Represent a collection of labeled evaluation queries targeting a catalog."""

    model_config = ConfigDict(frozen=True)

    catalog_id: str
    queries: tuple[Query, ...]
    notes: str = ""
    provenance: QuerySetProvenance

    @model_validator(mode="after")
    def _assert_unique_ids(self) -> Self:
        """Validate that all query IDs within the query set are unique."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for q in self.queries:
            if q.id in seen:
                duplicates.add(q.id)
            seen.add(q.id)
        if duplicates:
            msg = f"duplicate query ids: {sorted(duplicates)}"
            raise ValueError(msg)
        return self

    def for_skill(self, name: str) -> tuple[Query, ...]:
        """Return queries whose expected skill or truth label matches the specified name."""
        return tuple(q for q in self.queries if name in (q.expected_skill, q.truth_label))


class _LegacyQueryEntry(BaseModel):
    """Validate and normalize a legacy or skill-creator JSON query entry."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str | None = None
    text: str | None = None
    query: str | None = None
    kind: QueryKind | None = None
    should_trigger: bool | None = None
    expected_skill: str | None = None
    neighbor_skill: str | None = None
    true_skill: str | None = None
    acceptable_skills: tuple[str, ...] = ()
    notes: str = ""

    def to_query(self, index: int) -> Query:
        """Convert normalized legacy entry into a canonical Query model."""
        prompt = self.text if self.text is not None else self.query
        if prompt is None or not prompt.strip():
            msg = f"Legacy query at index {index} must provide non-empty 'text' or 'query'"
            raise ValueError(msg)

        q_id = self.id.strip() if self.id and self.id.strip() else f"q-{index:03d}"

        if self.should_trigger is False:
            positive_rival = self.neighbor_skill or self.true_skill
            if positive_rival:
                expected = positive_rival
                q_kind = self.kind or QueryKind.NEIGHBOR_NEGATIVE
            else:
                expected = None
                q_kind = QueryKind.OUT_OF_SCOPE
        else:
            expected = self.expected_skill
            if self.kind is not None:
                q_kind = self.kind
            elif expected is None:
                q_kind = QueryKind.OUT_OF_SCOPE
            else:
                q_kind = QueryKind.IMPLICIT

        return Query(
            id=q_id,
            text=prompt,
            kind=q_kind,
            expected_skill=expected,
            acceptable_skills=tuple(s for s in self.acceptable_skills if s),
            notes=self.notes,
        )


class _LegacyQueryEnvelope(BaseModel):
    """Validate a dict-wrapped skill-creator query payload containing 'queries' or 'evals'."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    catalog_id: str | None = None
    queries: tuple[_LegacyQueryEntry, ...] | None = None
    evals: tuple[_LegacyQueryEntry, ...] | None = None

    @model_validator(mode="after")
    def _require_skill_creator_markers(self) -> Self:
        """Require 'evals' key or entries with 'query' or 'should_trigger' fields."""
        items = self.queries if self.queries is not None else self.evals
        if items is None or not items:
            msg = "Legacy query envelope requires non-empty queries or evals"
            raise ValueError(msg)
        if self.evals is None and not any(
            item.query is not None or item.should_trigger is not None for item in items
        ):
            msg = "Canonical QuerySet dict must validate via QuerySet schema"
            raise ValueError(msg)
        return self


def _parse_json_query_set(
    content: str,
    *,
    catalog_id: str,
    source: str,
) -> QuerySet:
    """Parse canonical QuerySet JSON or normalize top-level array / skill-creator JSON."""
    try:
        return QuerySet.model_validate_json(content)
    except (ValueError, ValidationError) as primary_err:
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            raise primary_err from None

        resolved_catalog_id = catalog_id
        try:
            if isinstance(raw, list):
                entries = [_LegacyQueryEntry.model_validate(item) for item in raw]
            elif isinstance(raw, dict):
                envelope = _LegacyQueryEnvelope.model_validate(raw)
                if envelope.catalog_id:
                    resolved_catalog_id = envelope.catalog_id
                raw_items = envelope.queries if envelope.queries is not None else envelope.evals
                entries = list(raw_items or ())
            else:
                raise primary_err

            normalized_queries = tuple(
                entry.to_query(idx) for idx, entry in enumerate(entries, start=1)
            )
        except (ValueError, ValidationError):
            raise primary_err from None

        return QuerySet(
            catalog_id=resolved_catalog_id,
            queries=normalized_queries,
            provenance=QuerySetProvenance(origin=Origin.AUTHORED, source=source),
        )


def load_query_set(
    path: Path | str,
    *,
    catalog_id: str = "all",
) -> QuerySet:
    """Load and validate a QuerySet from a JSON, JSONL, or CSV file."""
    resolved = Path(path).expanduser().resolve()
    content = resolved.read_text(encoding="utf-8")
    suffix = resolved.suffix.lower()
    if suffix == ".jsonl":
        from reach.exchange import Exchange, import_query_set

        return import_query_set(
            content, Exchange.JSONL, catalog_id=catalog_id, source=str(resolved)
        )
    if suffix == ".csv":
        from reach.exchange import Exchange, import_query_set

        return import_query_set(content, Exchange.CSV, catalog_id=catalog_id, source=str(resolved))
    if suffix == ".json":
        return _parse_json_query_set(content, catalog_id=catalog_id, source=str(resolved))
    try:
        return _parse_json_query_set(content, catalog_id=catalog_id, source=str(resolved))
    except (ValueError, ValidationError):
        from reach.exchange import Exchange, import_query_set

        return import_query_set(
            content, Exchange.JSONL, catalog_id=catalog_id, source=str(resolved)
        )


def save_query_set(
    query_set: QuerySet,
    path: Path | str,
    *,
    fmt: str | None = None,
    mapping: FieldMap | None = None,
) -> Path:
    """Serialize a QuerySet instance to disk in JSON, JSONL, or CSV format."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    target_fmt = fmt or (
        "jsonl"
        if resolved.suffix.lower() == ".jsonl"
        else "csv"
        if resolved.suffix.lower() == ".csv"
        else "json"
    )
    if target_fmt == "jsonl":
        from reach.exchange import Exchange, export_query_set

        resolved.write_text(
            export_query_set(query_set, Exchange.JSONL, mapping=mapping),
            encoding="utf-8",
        )
        return resolved
    if target_fmt == "csv":
        from reach.exchange import Exchange, export_query_set

        resolved.write_text(
            export_query_set(query_set, Exchange.CSV, mapping=mapping),
            encoding="utf-8",
        )
        return resolved
    return write_model(query_set, resolved)


def query_set_digest(query_set: QuerySet) -> str:
    """Compute a deterministic 12-character SHA-256 digest of query set content."""
    return _digest_queries(query_set.catalog_id, _query_rows(query_set))


def _query_rows(query_set: QuerySet) -> list[tuple[str, ...]]:
    """Convert a query set into canonical row tuples for digest calculation."""
    rows: list[tuple[str, ...]] = []
    for query in query_set.queries:
        row: tuple[str, ...] = (
            query.id,
            query.text,
            query.kind or "",
            query.expected_skill or "",
        )
        if query.acceptable_skills:
            acceptable = json.dumps(
                sorted(query.acceptable_skills),
                separators=(",", ":"),
            )
            row += (acceptable,)
        rows.append(row)
    return rows


def _digest_queries(catalog_id: str, rows: Iterable[tuple[str, ...]]) -> str:
    """Compute a 12-character SHA-256 digest over sorted query rows and catalog ID."""
    material = "\n".join("\t".join(row) for row in sorted(rows, key=lambda row: row[0]))
    canonical = f"{catalog_id}\n{material}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
