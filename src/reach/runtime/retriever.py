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

"""Filter large skill catalogs via dynamic two-stage BM25 retrieval."""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING, override

from reach.catalog import corpus_digest
from reach.config import RuntimeSettings
from reach.models import Catalog, CatalogMode, Skill
from reach.retrieval import Bm25Scorer, TextScorer
from reach.runtime import AgentRuntime, CatalogFit, SelectionOutcome, SkillRoot
from reach.runtime._fs import probe_slot_dir, probe_slot_id

if TYPE_CHECKING:
    from collections.abc import Iterable


from pydantic import BaseModel, ConfigDict


class _RetrieverStage2CacheKey(BaseModel):
    """Identify a content-addressed Stage-2 LLM evaluation for a top-k retrieved subset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    query_text: str
    subset_digest: str
    top_names: tuple[str, ...]
    target_skill: str | None = None
    replicate_index: int = 0


class TwoStageRetrieverRuntime(AgentRuntime):
    """Wrap an AgentRuntime with a Stage-1 lexical pre-filter to top-k candidate skills."""

    is_dynamic: bool = True

    def __init__(
        self,
        inner: AgentRuntime,
        top_k: int = 8,
        scorer: TextScorer | None = None,
        *,
        cache_outcomes: bool = True,
    ) -> None:
        """Initialize the retriever runtime wrapping an underlying agent runtime."""
        self.inner = inner
        self.top_k = top_k
        self.scorer = scorer
        self.cache_outcomes = cache_outcomes
        self.name = f"retriever-{inner.name}"
        self.settings: RuntimeSettings | None = (
            getattr(inner, "settings", None) or RuntimeSettings()
        )
        self.options = inner.options
        self._all_skills: list[Skill] = []
        self._ranker: TextScorer | None = None
        self._catalog: Catalog | None = None
        self._resident: tuple[str, ...] = ()
        self._thread_local = threading.local()
        self._clone_lock = threading.Lock()
        self._worker_clones: list[AgentRuntime] = []
        self._cache_lock = threading.Lock()
        self._outcome_cache: dict[_RetrieverStage2CacheKey, SelectionOutcome] = {}
        self._scale_call_counts: dict[_RetrieverStage2CacheKey, int] = {}

    def _worker_inner(self) -> AgentRuntime:
        """Return a thread-isolated clone of the underlying inner runtime."""
        worker_rt = getattr(self._thread_local, "inner", None)
        if worker_rt is None:
            worker_rt = self.inner.clone_isolated()
            self._thread_local.inner = worker_rt
            with self._clone_lock:
                self._worker_clones.append(worker_rt)
        return worker_rt

    @override
    def cleanup(self) -> None:
        """Clean up inner runtime and all thread-local worker clones."""
        with self._clone_lock:
            clones = list(self._worker_clones)
            self._worker_clones.clear()
        for clone in clones:
            clone.cleanup()
        self.inner.cleanup()

    @property
    @override
    def model(self) -> str:
        """Return the model name configured on the underlying runtime."""
        return self.inner.model

    @model.setter
    @override
    def model(self, value: str) -> None:
        """Update the model identifier on the underlying runtime."""
        self.inner.model = value
        with self._clone_lock:
            for clone in self._worker_clones:
                clone.model = value

    @property
    @override
    def skills_subpath(self) -> str:
        """Return relative skill directory subpath from the underlying runtime."""
        return self.inner.skills_subpath

    @override
    def fit(self, catalog: Catalog, skills: Iterable[Skill]) -> CatalogFit:
        """Forward catalog fit evaluation to the underlying runtime."""
        return self.inner.fit(catalog, skills)

    @override
    def skill_roots(self, workdir: Path) -> tuple[SkillRoot, ...]:
        """Forward skill roots resolution to the underlying runtime."""
        return self.inner.skill_roots(workdir)

    @override
    def install(self, catalog: Catalog, skills: Iterable[Skill], workdir: Path) -> Path:
        """Index skills for pre-filtering, record residency, and reset per-scale call counts."""
        self._all_skills = list(skills)
        self._catalog = catalog
        self._ranker = self.scorer or Bm25Scorer.from_skills(self._all_skills)
        self._resident = catalog.skills
        with self._cache_lock:
            self._scale_call_counts.clear()
        return workdir

    @override
    def select(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Pre-filter catalog to top-k skills and execute probe in an isolated slot."""
        if not self._all_skills:
            return self.inner.select(query_text, workdir, target_skill=target_skill)

        ranker = self._ranker or Bm25Scorer.from_skills(self._all_skills)
        ranked = ranker.rank_text(query_text, self._all_skills)
        top_names = tuple(name for name, _ in ranked[: self.top_k])
        if not top_names and self._all_skills:
            top_names = tuple(s.name for s in self._all_skills[: self.top_k])

        subset_skills = [s for s in self._all_skills if s.name in top_names]
        cache_key: _RetrieverStage2CacheKey | None = None
        if self.cache_outcomes:
            subset_digest = corpus_digest(subset_skills)
            base_key = _RetrieverStage2CacheKey(
                model=self.model,
                query_text=query_text,
                subset_digest=subset_digest,
                top_names=top_names,
                target_skill=target_skill,
            )
            with self._cache_lock:
                rep_idx = self._scale_call_counts.get(base_key, 0) + 1
                self._scale_call_counts[base_key] = rep_idx
                cache_key = base_key.model_copy(update={"replicate_index": rep_idx})
                cached = self._outcome_cache.get(cache_key)
            if cached is not None:
                return cached

        thread_id = probe_slot_id()
        slot_dir = probe_slot_dir(workdir)
        slot_dir.mkdir(parents=True, exist_ok=True)

        sub_catalog = Catalog(
            id=f"retrieved:{thread_id}",
            mode=CatalogMode.ALL,
            skills=top_names,
        )

        worker_rt = self._worker_inner()
        try:
            worker_rt.install(sub_catalog, subset_skills, slot_dir)
            outcome = worker_rt.select(query_text, slot_dir, target_skill=target_skill)
            final_outcome = outcome.model_copy(update={"observed_catalog": top_names})
            if self.cache_outcomes and cache_key is not None and not final_outcome.error:
                with self._cache_lock:
                    self._outcome_cache[cache_key] = final_outcome
            return final_outcome
        finally:
            if slot_dir.exists():
                shutil.rmtree(slot_dir, ignore_errors=True)
