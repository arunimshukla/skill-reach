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

"""Validate two-stage dynamic retriever runtime with slot workspace isolation."""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

from reach.models import Catalog, CatalogMode, DisclosureState, Query, QueryKind, Skill
from reach.run import ProbeHarness, validate_residency
from reach.runtime.fake import FakeRuntime
from reach.runtime.retriever import TwoStageRetrieverRuntime

if TYPE_CHECKING:
    from pathlib import Path


def _make_skills(root: Path, count: int) -> list[Skill]:
    skills = []
    for i in range(count):
        name = f"cloud-tool-{i:02d}"
        s_dir = root / name
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Tool {i} for cloud resources.\n---\n# Body\n",
            encoding="utf-8",
        )
        skills.append(
            Skill(
                name=name,
                description=f"Tool {i} for managing cloud resources.",
                path=s_dir,
            )
        )
    return skills


def test_retriever_runtime_filters_to_top_k(tmp_path: Path) -> None:
    """Verify retriever runtime pre-filters corpus down to top_k candidate skills."""
    skills = _make_skills(tmp_path / "skills", 20)
    catalog = Catalog(id="all-20", mode=CatalogMode.ALL, skills=tuple(s.name for s in skills))

    inner = FakeRuntime({"manage cloud resources": "cloud-tool-00"}, model="fake-model")
    runtime = TwoStageRetrieverRuntime(inner, top_k=5)

    assert runtime.is_dynamic is True

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    runtime.install(catalog, skills, workdir)

    outcome = runtime.select("manage cloud resources", workdir)
    assert len(outcome.observed_catalog) == 5
    assert set(outcome.observed_catalog) <= set(catalog.skills)
    assert validate_residency(catalog, outcome.observed_catalog, dynamic=runtime.is_dynamic) is None


def test_retriever_runtime_telemetry_withheld_state(tmp_path: Path) -> None:
    """Verify probe telemetry marks WITHHELD when expected skill is not retrieved."""
    skills = _make_skills(tmp_path / "skills", 20)
    catalog = Catalog(id="all-20", mode=CatalogMode.ALL, skills=tuple(s.name for s in skills))

    inner = FakeRuntime(default="cloud-tool-00", model="fake-model")
    runtime = TwoStageRetrieverRuntime(inner, top_k=3)

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    runtime.install(catalog, skills, workdir)

    # Query expects cloud-tool-19, but query text only matches cloud-tool-00/01/02
    query = Query(
        id="q-rare",
        text="cloud-tool-00 cloud-tool-01 cloud-tool-02",
        kind=QueryKind.IMPLICIT,
        expected_skill="cloud-tool-19",
    )

    result = ProbeHarness(runtime).probe(query, catalog, workdir)
    assert result.disclosure_state is DisclosureState.WITHHELD


def test_retriever_runtime_concurrent_slot_isolation(tmp_path: Path) -> None:
    """Verify parallel workers execute in distinct slot directories without collisions."""
    skills = _make_skills(tmp_path / "skills", 10)
    catalog = Catalog(id="all-10", mode=CatalogMode.ALL, skills=tuple(s.name for s in skills))

    inner = FakeRuntime(default="cloud-tool-00", model="fake-model")
    runtime = TwoStageRetrieverRuntime(inner, top_k=4)

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    runtime.install(catalog, skills, workdir)

    def _worker(idx: int) -> tuple[str, ...]:
        q_text = f"manage cloud resources task {idx}"
        outcome = runtime.select(q_text, workdir)
        return outcome.observed_catalog

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker, i) for i in range(8)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 8
    for catalog_subset in results:
        assert len(catalog_subset) == 4


def test_retriever_runtime_with_dense_scorer(tmp_path: Path) -> None:
    """Verify retriever runtime functions with DenseScorer as ranker."""
    from reach.retrieval import DenseScorer

    skills = _make_skills(tmp_path / "skills", 5)
    catalog = Catalog(id="all-5", mode=CatalogMode.ALL, skills=tuple(s.name for s in skills))

    inner = FakeRuntime(default="cloud-tool-00", model="fake-model")
    dense_scorer = DenseScorer(
        vectors={
            "query": [1.0, 0.0],
            "cloud-tool-00": [0.99, 0.01],
            "cloud-tool-01": [0.5, 0.5],
            "cloud-tool-02": [0.0, 1.0],
            "cloud-tool-03": [0.1, 0.9],
            "cloud-tool-04": [0.2, 0.8],
        }
    )
    runtime = TwoStageRetrieverRuntime(inner, top_k=2, scorer=dense_scorer)

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    runtime.install(catalog, skills, workdir)

    outcome = runtime.select("query", workdir)
    assert len(outcome.observed_catalog) == 2
    assert outcome.observed_catalog[0] == "cloud-tool-00"
