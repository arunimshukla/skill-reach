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

"""Verify KeywordRuntime matching performance, specificity, word boundaries, and symlinks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from reach.models import Catalog, CatalogMode, Skill
from reach.runtime.keyword import KeywordOptions, KeywordRuntime

if TYPE_CHECKING:
    from pathlib import Path


def _mock_skills(root: Path, names: tuple[str, ...]) -> list[Skill]:
    skills = []
    for name in names:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        (p / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
        skills.append(Skill(name=name, description=f"Description for {name}", path=p))
    return skills


def test_keyword_runtime_select_exact_and_space_separated(tmp_path: Path) -> None:
    """Verify matching works for both kebab-case and space-separated skill names."""
    names = ("cloud-sql", "cloud-storage")
    skills = _mock_skills(tmp_path / "src", names)
    catalog = Catalog(id="cat1", mode=CatalogMode.ALL, skills=names)
    runtime = KeywordRuntime()
    runtime.install(catalog, skills, tmp_path / "work")

    # Kebab-case
    outcome1 = runtime.select("Please query database with cloud-sql", tmp_path)
    assert outcome1.invoked_skill == "cloud-sql"

    # Space-separated
    outcome2 = runtime.select("Upload objects to cloud storage bucket", tmp_path)
    assert outcome2.invoked_skill == "cloud-storage"


def test_keyword_runtime_specificity_priority(tmp_path: Path) -> None:
    """Verify longer compound skill names take precedence over generic prefix rivals."""
    # Alphabetically, 'cloud-run' comes before 'cloud-run-jobs'
    names = ("cloud-run", "cloud-run-jobs")
    skills = _mock_skills(tmp_path / "src", names)
    catalog = Catalog(id="cat1", mode=CatalogMode.ALL, skills=names)
    runtime = KeywordRuntime()
    runtime.install(catalog, skills, tmp_path / "work")

    # Query mentioning cloud-run-jobs should match cloud-run-jobs, not cloud-run
    outcome = runtime.select("Deploy batch task on cloud-run-jobs", tmp_path)
    assert outcome.invoked_skill == "cloud-run-jobs"

    # Query mentioning only cloud-run should still match cloud-run
    outcome_run = runtime.select("Deploy web service on cloud-run", tmp_path)
    assert outcome_run.invoked_skill == "cloud-run"


def test_keyword_runtime_word_boundary_prevents_partial_word_matches(tmp_path: Path) -> None:
    """Verify short skill names do not falsely match substrings inside unrelated words."""
    names = ("log", "ai", "sql", "go")
    skills = _mock_skills(tmp_path / "src", names)
    catalog = Catalog(id="cat1", mode=CatalogMode.ALL, skills=names)
    runtime = KeywordRuntime()
    runtime.install(catalog, skills, tmp_path / "work")

    # "catalog" contains "log", "obtain" contains "ai", "algorithm" contains "go"
    outcome1 = runtime.select("Look at the catalog to obtain an algorithm", tmp_path)
    assert outcome1.invoked_skill is None
    assert outcome1.invoked_skills == ()

    # Distinct tokens should match correctly
    outcome2 = runtime.select("View the system log output", tmp_path)
    assert outcome2.invoked_skill == "log"

    outcome3 = runtime.select("Ask the AI model for help", tmp_path)
    assert outcome3.invoked_skill == "ai"


def test_keyword_runtime_early_exit(tmp_path: Path) -> None:
    """Verify early exit flag is triggered when matched skill matches target_skill."""
    names = ("target-tool", "other-tool")
    skills = _mock_skills(tmp_path / "src", names)
    catalog = Catalog(id="cat1", mode=CatalogMode.ALL, skills=names)
    runtime = KeywordRuntime(options=KeywordOptions(early_exit=True))
    runtime.install(catalog, skills, tmp_path / "work")

    outcome_hit = runtime.select("Execute target-tool now", tmp_path, target_skill="target-tool")
    assert outcome_hit.early_exit is True

    outcome_miss = runtime.select("Execute other-tool now", tmp_path, target_skill="target-tool")
    assert outcome_miss.early_exit is False


def test_keyword_runtime_empty_query_and_catalog(tmp_path: Path) -> None:
    """Verify edge cases with empty queries, empty lines, or uninstalled runtime."""
    runtime = KeywordRuntime()
    outcome = runtime.select("", tmp_path)
    assert outcome.invoked_skill is None
    assert outcome.invoked_skills == ()

    summary = runtime.parse_stream([])
    assert summary.invoked_skill is None
    assert summary.invoked_skills == ()


def test_keyword_runtime_parse_stream_extracts_skill() -> None:
    """Verify KeywordRuntime extracts invoked skill from stream lines."""
    runtime = KeywordRuntime()
    summary = runtime.parse_stream(
        ["Model chose tool cloud-sql to proceed"],
        resident=("cloud-sql", "gcloud"),
    )
    assert summary.invoked_skill == "cloud-sql"
    assert summary.invoked_skills == ("cloud-sql",)
    assert summary.saw_result
