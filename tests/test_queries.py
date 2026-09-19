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

"""Verify QuerySet loading, schema validation, digests, and serialization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from reach.models import NO_SKILL, Query, QueryKind
from reach.queries import (
    Origin,
    QuerySet,
    QuerySetProvenance,
    load_query_set,
    query_set_digest,
    save_query_set,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_duplicate_ids_are_rejected(
    write_queries: Callable[..., Path],
) -> None:
    """Verify loading fails with ValueError if query set contains duplicate IDs."""
    rows = [
        {"id": "dupe", "text": "a", "kind": "implicit", "expected_skill": "s1"},
        {"id": "dupe", "text": "b", "kind": "implicit", "expected_skill": "s2"},
    ]
    with pytest.raises(ValueError, match="duplicate query ids"):
        load_query_set(write_queries(queries=rows, catalog_id="c"))


@pytest.mark.parametrize(
    "bad_kind",
    ["explicit", "", "NEGATIVE"],
    ids=["dropped", "empty", "wrong-case"],
)
def test_unknown_kind_is_rejected(
    write_queries: Callable[..., Path],
    bad_kind: str,
) -> None:
    """Verify unrecognized query kind string raises ValueError."""
    rows = [{"id": "q", "text": "a", "kind": bad_kind, "expected_skill": "s1"}]
    with pytest.raises(ValueError, match="kind"):
        load_query_set(write_queries(queries=rows, catalog_id="c"))


def test_a_query_may_go_unlabeled(
    write_queries: Callable[..., Path],
) -> None:
    """Verify query with omitted kind parses with kind=None."""
    rows = [{"id": "q", "text": "a", "expected_skill": "s1"}]
    assert load_query_set(write_queries(queries=rows, catalog_id="c")).queries[0].kind is None


def test_an_unlabeled_query_still_needs_ground_truth(
    write_queries: Callable[..., Path],
) -> None:
    """Verify query lacking expected_skill and kind raises ValueError."""
    rows = [{"id": "q", "text": "a"}]
    with pytest.raises(ValueError, match="expected_skill"):
        load_query_set(write_queries(queries=rows, catalog_id="c"))


def test_an_absent_kind_does_not_digest_as_a_kind(
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify query_set_digest distinguishes between unlabeled kind and explicit kind."""
    unlabeled = [{"id": "q", "text": "a", "expected_skill": "s1"}]
    labeled = [{**unlabeled[0], "kind": "neighbor_negative"}]
    before = load_query_set(write_queries(queries=unlabeled, root=tmp_path / "a", catalog_id="c"))
    after = load_query_set(write_queries(queries=labeled, root=tmp_path / "b", catalog_id="c"))
    assert query_set_digest(after) != query_set_digest(before)


def test_a_set_names_the_catalog_it_was_written_for(
    write_queries: Callable[..., Path],
) -> None:
    """Verify loaded QuerySet preserves catalog_id from file."""
    rows = [{"id": "q", "text": "a", "kind": "implicit", "expected_skill": "s1"}]
    assert load_query_set(write_queries(queries=rows, catalog_id="c")).catalog_id == "c"


ROWS = [
    {"id": "q1", "text": "a", "kind": "implicit", "expected_skill": "s1"},
    {"id": "q2", "text": "b", "kind": "out_of_scope"},
]


def test_a_digest_ignores_where_the_set_sits(
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify query_set_digest is independent of file path location."""
    here = load_query_set(write_queries(queries=ROWS, root=tmp_path / "a", catalog_id="c"))
    there = load_query_set(write_queries(queries=ROWS, root=tmp_path / "b", catalog_id="c"))
    assert query_set_digest(here) == query_set_digest(there)


def test_a_digest_ignores_the_order_rows_were_written_in(
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify query_set_digest is invariant under row reordering."""
    forwards = load_query_set(write_queries(queries=ROWS, root=tmp_path / "a", catalog_id="c"))
    backwards = load_query_set(
        write_queries(queries=list(reversed(ROWS)), root=tmp_path / "b", catalog_id="c")
    )
    assert query_set_digest(forwards) == query_set_digest(backwards)


def test_a_digest_ignores_notes_nobody_is_scored_against(
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify query_set_digest ignores editorial notes."""
    plain = load_query_set(write_queries(queries=ROWS, root=tmp_path / "a", catalog_id="c"))
    annotated = plain.model_copy(update={"notes": "reviewed 2026-08-21"})
    assert query_set_digest(annotated) == query_set_digest(plain)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text", "rewritten"),
        ("kind", "contextual"),
        ("expected_skill", "s2"),
        ("id", "q9"),
    ],
)
def test_a_digest_moves_when_a_query_or_its_truth_moves(
    write_queries: Callable[..., Path],
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    """Verify modifying query fields alters query_set_digest."""
    before = load_query_set(write_queries(queries=ROWS, root=tmp_path / "a", catalog_id="c"))
    edited = [{**ROWS[0], field: value}, ROWS[1]]
    after = load_query_set(write_queries(queries=edited, root=tmp_path / "b", catalog_id="c"))
    assert query_set_digest(after) != query_set_digest(before)


def test_a_digest_moves_when_the_labeling_catalog_changes(
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify changing catalog_id alters query_set_digest."""
    before = load_query_set(write_queries(queries=ROWS, root=tmp_path / "a", catalog_id="c"))
    rescoped = before.model_copy(update={"catalog_id": "neighborhood:s1"})
    assert query_set_digest(rescoped) != query_set_digest(before)


def provenance(**overrides) -> QuerySetProvenance:
    """Build a sample QuerySetProvenance instance with optional field overrides."""
    return QuerySetProvenance.model_validate(
        {
            "origin": Origin.GENERATED,
            "recorded_at": datetime(2026, 8, 22, 12, 0, tzinfo=UTC),
            "generator_model": "opus",
            "generator_arm": "content",
            "queries_per_target": 2,
            "rivals_in_view": 110,
            "bodies_digest": "b664ddebc7c9",
            "config_fingerprint": "9aadef3d65c9",
            **overrides,
        },
    )


def test_a_digest_ignores_how_the_set_was_made(
    write_queries: Callable[..., Path],
) -> None:
    """Verify query_set_digest is unaffected by provenance metadata."""
    plain = load_query_set(write_queries(queries=ROWS, catalog_id="c"))
    recorded = plain.model_copy(update={"provenance": provenance()})
    assert query_set_digest(recorded) == query_set_digest(plain)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("origin", Origin.IMPORTED),
        ("generator_model", "sonnet"),
        ("generator_arm", "framing"),
        ("rivals_in_view", 10),
        ("queries_per_target", 5),
    ],
)
def test_no_provenance_field_can_move_a_digest(
    write_queries: Callable[..., Path],
    field: str,
    value,
) -> None:
    """Verify changing individual provenance fields leaves query_set_digest unchanged."""
    plain = load_query_set(write_queries(queries=ROWS, catalog_id="c"))
    recorded = plain.model_copy(update={"provenance": provenance(**{field: value})})
    assert query_set_digest(recorded) == query_set_digest(plain)


def test_a_set_recorded_without_provenance_is_refused(tmp_path: Path) -> None:
    """Verify legacy query sets without provenance fail schema validation."""
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"catalog_id": "c", "queries": []}), encoding="utf-8")
    with pytest.raises(ValidationError, match="provenance"):
        load_query_set(legacy)


def test_provenance_survives_a_trip_through_disk(tmp_path: Path) -> None:
    """Verify provenance metadata round-trips through disk save and load."""
    written = QuerySet(catalog_id="c", queries=(), provenance=provenance())
    reloaded = load_query_set(save_query_set(written, tmp_path / "out" / "set.json"))
    assert reloaded.provenance == written.provenance


def test_an_unrecognized_origin_is_refused() -> None:
    """Verify invalid origin string raises ValueError."""
    with pytest.raises(ValueError, match="origin"):
        QuerySetProvenance.model_validate({"origin": "invented"})


def test_a_recorded_time_must_carry_its_zone() -> None:
    """Verify naive datetime timestamps raise ValueError on QuerySetProvenance creation."""
    naive = datetime(2026, 8, 22, 12, 0, tzinfo=UTC).replace(tzinfo=None)
    with pytest.raises(ValueError, match=r"timezone.*info|timezone[-_ ]?aware"):
        QuerySetProvenance(origin=Origin.AUTHORED, recorded_at=naive)


def test_saving_a_set_writes_what_the_loader_reads(tmp_path: Path) -> None:
    """Verify QuerySet saved to disk equals loaded QuerySet."""
    written = QuerySet(catalog_id="c", notes="n", queries=(), provenance=provenance())
    assert load_query_set(save_query_set(written, tmp_path / "set.json")) == written


@pytest.mark.parametrize("filename", ["queries.jsonl", "queries.csv"])
def test_save_and_load_query_set_tabular(tmp_path: Path, filename: str) -> None:
    """Verify QuerySet saved to JSONL or CSV round-trips correctly through load_query_set."""
    initial = QuerySet(
        catalog_id="c",
        queries=(
            Query(
                id="q1",
                text="tier cold objects",
                expected_skill="gcs-lifecycle-rules",
            ),
        ),
        provenance=provenance(),
    )
    saved = save_query_set(initial, tmp_path / filename)
    loaded = load_query_set(saved)
    assert len(loaded.queries) == 1
    assert loaded.queries[0].id == "q1"
    assert loaded.queries[0].text == "tier cold objects"
    assert loaded.queries[0].expected_skill == "gcs-lifecycle-rules"


@pytest.mark.parametrize(
    ("target", "expected_ids"),
    [
        ("s1", ("q1",)),
        ("s2", ("q2",)),
        ("s3", ()),
        (NO_SKILL, ("q3",)),
        ("(no skill)", ("q3",)),
    ],
)
def test_query_set_for_skill_lookup(target: str, expected_ids: tuple[str, ...]) -> None:
    """Verify QuerySet.for_skill matches expected skill names and out-of-scope sentinels."""
    q1 = Query(id="q1", text="text 1", expected_skill="s1")
    q2 = Query(id="q2", text="text 2", expected_skill="s2")
    q3 = Query(id="q3", text="out of scope", kind=QueryKind.OUT_OF_SCOPE)
    qs = QuerySet(catalog_id="c", queries=(q1, q2, q3), provenance=provenance())
    assert tuple(q.id for q in qs.for_skill(target)) == expected_ids


@pytest.mark.parametrize(
    ("payload", "expected_first", "expected_second"),
    [
        pytest.param(
            [
                {
                    "query": "deploy service to cloud run",
                    "should_trigger": True,
                    "expected_skill": "cloud-run-basics",
                },
                {
                    "query": "book a flight to tokyo",
                    "should_trigger": False,
                    "expected_skill": "cloud-run-basics",
                },
            ],
            ("q-001", "deploy service to cloud run", "cloud-run-basics", QueryKind.IMPLICIT),
            ("q-002", "book a flight to tokyo", None, QueryKind.OUT_OF_SCOPE),
            id="top-level-array-clears-negative-expected-skill",
        ),
        pytest.param(
            [
                {
                    "query": "deploy service to cloud run",
                    "should_trigger": True,
                    "expected_skill": "cloud-run-basics",
                },
                {
                    "query": "configure gke node pool",
                    "should_trigger": False,
                    "expected_skill": "cloud-run-basics",
                    "neighbor_skill": "gke-basics",
                },
            ],
            ("q-001", "deploy service to cloud run", "cloud-run-basics", QueryKind.IMPLICIT),
            ("q-002", "configure gke node pool", "gke-basics", QueryKind.NEIGHBOR_NEGATIVE),
            id="top-level-array-preserves-neighbor-skill-on-negative",
        ),
        pytest.param(
            {
                "catalog_id": "custom:cloud-run-basics",
                "queries": [
                    {
                        "query": "deploy service to cloud run",
                        "should_trigger": True,
                        "expected_skill": "cloud-run-basics",
                    },
                    {
                        "query": "book a flight to tokyo",
                        "should_trigger": False,
                        "expected_skill": None,
                    },
                ],
            },
            ("q-001", "deploy service to cloud run", "cloud-run-basics", QueryKind.IMPLICIT),
            ("q-002", "book a flight to tokyo", None, QueryKind.OUT_OF_SCOPE),
            id="dict-wrapper-queries-key",
        ),
        pytest.param(
            {
                "evals": [
                    {
                        "query": "deploy service to cloud run",
                        "should_trigger": True,
                        "expected_skill": "cloud-run-basics",
                    },
                    {
                        "query": "book a flight to tokyo",
                        "should_trigger": False,
                    },
                ],
            },
            ("q-001", "deploy service to cloud run", "cloud-run-basics", QueryKind.IMPLICIT),
            ("q-002", "book a flight to tokyo", None, QueryKind.OUT_OF_SCOPE),
            id="dict-wrapper-evals-key",
        ),
    ],
)
def test_load_query_set_normalizes_legacy_json_formats(
    tmp_path: Path,
    payload: object,
    expected_first: tuple[str, str, str | None, QueryKind],
    expected_second: tuple[str, str, str | None, QueryKind],
) -> None:
    """Verify load_query_set normalizes array and dict-wrapped skill-creator JSON via Pydantic."""
    p = tmp_path / "eval_set.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_query_set(p, catalog_id="neighborhood:cloud-run-basics")
    assert len(loaded.queries) == 2
    q0, q1 = loaded.queries
    assert (q0.id, q0.text, q0.expected_skill, q0.kind) == expected_first
    assert (q1.id, q1.text, q1.expected_skill, q1.kind) == expected_second
