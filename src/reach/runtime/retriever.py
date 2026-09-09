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

from reach.config import RuntimeSettings
from reach.models import Catalog, CatalogMode, Skill
from reach.retrieval import Bm25Scorer, TextScorer
from reach.runtime import AgentRuntime, CatalogFit, SelectionOutcome, SkillRoot

if TYPE_CHECKING:
    from collections.abc import Iterable


class TwoStageRetrieverRuntime(AgentRuntime):
    """Execute two-stage skill runtime with BM25 pre-filtering and isolated slot execution."""

    is_dynamic: bool = True

    def __init__(
        self,
        inner: AgentRuntime,
        top_k: int = 8,
        scorer: TextScorer | None = None,
    ) -> None:
        """Initialize the retriever runtime wrapping an underlying agent runtime."""
        self.inner = inner
        self.top_k = top_k
        self.scorer = scorer
        self.name = f"retriever-{inner.name}"
        self.settings: RuntimeSettings | None = (
            getattr(inner, "settings", None) or RuntimeSettings()
        )
        self.options = inner.options
        self._all_skills: list[Skill] = []
        self._ranker: TextScorer | None = None
        self._catalog: Catalog | None = None
        self._resident: tuple[str, ...] = ()

    @property
    @override
    def model(self) -> str:
        """Return the model name configured on the underlying runtime."""
        return self.inner.model

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
        """Index skills for pre-filtering and record residency."""
        self._all_skills = list(skills)
        self._catalog = catalog
        self._ranker = self.scorer or Bm25Scorer.from_skills(self._all_skills)
        self._resident = catalog.skills
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

        thread_id = getattr(threading, "get_native_id", threading.get_ident)()
        slot_dir = Path(workdir) / f"slot_{thread_id}"
        slot_dir.mkdir(parents=True, exist_ok=True)

        subset_skills = [s for s in self._all_skills if s.name in top_names]
        sub_catalog = Catalog(
            id=f"retrieved:{thread_id}",
            mode=CatalogMode.ALL,
            skills=top_names,
        )

        try:
            self.inner.install(sub_catalog, subset_skills, slot_dir)
            outcome = self.inner.select(query_text, slot_dir, target_skill=target_skill)
            return outcome.model_copy(update={"observed_catalog": top_names})
        finally:
            if slot_dir.exists():
                shutil.rmtree(slot_dir, ignore_errors=True)
