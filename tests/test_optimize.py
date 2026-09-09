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

"""Verify closed-loop description optimizer domain engine (`reach optimize`)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import yaml

from reach.catalog import split_frontmatter
from reach.models import Query, QueryKind, Skill
from reach.optimize import (
    OptimizationCandidate,
    OptimizationReport,
    _synthesize_via_heuristics,
    build_optimization_prompt,
    evaluate_candidate,
    filter_candidates,
    optimize_skill,
    synthesize_candidates,
    update_skill_description,
)
from reach.queries import load_query_set
from reach.rewrite import synthesize_directional_disclaimer

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_build_optimization_prompt_contains_all_context(
    write_skill: Callable[..., Path],
) -> None:
    """Verify build_optimization_prompt includes target, rivals, ceded, and unclaimed terms."""
    target_dir = write_skill(
        name="cloud-deployer",
        description="Deploy applications and services to cloud platforms.",
        body="# Cloud Deployer\nFeatures automated rollout and canary deployments with metrics.",
    )
    target = Skill(
        name="cloud-deployer",
        description="Deploy applications and services to cloud platforms.",
        path=target_dir,
    )
    rival_dir = write_skill(
        name="container-builder",
        description="Build and package container images with Dockerfiles.",
    )
    rival = Skill(
        name="container-builder",
        description="Build and package container images with Dockerfiles.",
        path=rival_dir,
    )

    prompt = build_optimization_prompt(
        target=target,
        rivals=[rival],
        ceded_terms=("container", "docker"),
        unclaimed_terms=("canary", "rollout", "metrics"),
        min_length=20,
        max_length=500,
    )

    assert "cloud-deployer" in prompt
    assert "container-builder" in prompt
    assert "container" in prompt
    assert "docker" in prompt
    assert "canary" in prompt
    assert "rollout" in prompt
    assert "min_length: 20" in prompt or "20" in prompt


def test_update_skill_description_updates_frontmatter_and_preserves_body(
    write_skill: Callable[..., Path],
) -> None:
    """Verify update_skill_description cleanly modifies YAML frontmatter and keeps body."""
    skill_dir = write_skill(
        name="my-skill",
        description="Old initial description that needs updating.",
        body="# Heading\n\nPreserve this exact body content.\n- Item 1\n- Item 2",
    )

    new_desc = "New improved description highlighting specialized tools."
    success = update_skill_description(skill_dir / "SKILL.md", new_desc)
    assert success is True

    manifest_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    split = split_frontmatter(manifest_text)
    assert split is not None
    frontmatter_yaml, body = split

    data = yaml.safe_load(frontmatter_yaml)
    assert data["name"] == "my-skill"
    assert data["description"] == new_desc
    assert "# Heading" in body
    assert "- Item 1" in body


def test_update_skill_description_non_existent_file_returns_false(tmp_path: Path) -> None:
    """Verify update_skill_description returns False if path does not exist."""
    assert update_skill_description(tmp_path / "non_existent.md", "New description") is False


def test_filter_candidates_with_linter() -> None:
    """Verify filter_candidates flags candidates violating length or format rules."""
    raw_candidates = [
        OptimizationCandidate(
            description="A perfectly valid description that satisfies all static constraints.",
            rationale="Adds distinctive terms.",
        ),
        OptimizationCandidate(
            description="Too short",
            rationale="Too brief description.",
        ),
        OptimizationCandidate(
            description="A" * 1200,
            rationale="Exceeds maximum allowable description length.",
        ),
    ]

    filtered = filter_candidates(raw_candidates, skill_name="my-tool")
    assert len(filtered) == 3
    # Candidate 0 is clean
    assert filtered[0].lint_clean is True
    # Candidate 1 is too short
    assert filtered[1].lint_clean is False
    # Candidate 2 is too long
    assert filtered[2].lint_clean is False


def test_evaluate_candidate_measures_delta_recall(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
) -> None:
    """Verify evaluate_candidate evaluates candidate against queries and computes deltas."""
    target_dir = write_skill(
        name="calc-tool",
        description="Evaluate mathematical expressions.",
    )
    target = Skill(
        name="calc-tool",
        description="Evaluate mathematical expressions.",
        path=target_dir,
    )
    query_file = write_queries(target="calc-tool", count=4)
    queries = load_query_set(query_file).queries

    candidate = OptimizationCandidate(
        description="Perform arithmetic calculations, matrix algebra, and equations.",
        rationale="Includes algebra keywords.",
    )

    evaluated = evaluate_candidate(
        candidate=candidate,
        target=target,
        rivals=[],
        queries=queries,
        agent="keyword",
        baseline_recall=0.50,
        baseline_accuracy=0.50,
        budget=10,
    )

    assert evaluated.recall >= 0.0
    assert isinstance(evaluated.delta_recall, float)


def test_run_candidate_probes_scores_rival_and_out_of_scope_queries(tmp_path: Path) -> None:
    """Verify candidate probe runner rewards correct rival and abstention choices."""
    from reach.optimize import _run_candidate_probes
    from reach.runtime.fake import FakeRuntime

    queries = [
        Query(
            id="q1",
            text="target query 1",
            expected_skill="target-tool",
            kind=QueryKind.IMPLICIT,
        ),
        Query(
            id="q2",
            text="target query 2",
            expected_skill="target-tool",
            kind=QueryKind.IMPLICIT,
        ),
        Query(
            id="q3",
            text="rival query 1",
            expected_skill="rival-tool",
            kind=QueryKind.IMPLICIT,
        ),
        Query(
            id="q4",
            text="rival query 2",
            expected_skill="rival-tool",
            kind=QueryKind.IMPLICIT,
        ),
        Query(
            id="q5",
            text="unrelated query 1",
            expected_skill=None,
            kind=QueryKind.OUT_OF_SCOPE,
        ),
        Query(
            id="q6",
            text="unrelated query 2",
            expected_skill=None,
            kind=QueryKind.OUT_OF_SCOPE,
        ),
    ]
    selections = {
        "target query 1": "target-tool",
        "target query 2": "rival-tool",
        "rival query 1": "rival-tool",
        "rival query 2": "target-tool",
        "unrelated query 1": None,
        "unrelated query 2": "target-tool",
    }
    runtime = FakeRuntime(selections)
    triggers, positive_queries, correct_count, misroutes = _run_candidate_probes(
        runtime,
        queries,
        "target-tool",
        tmp_path,
    )
    assert positive_queries == 2
    assert triggers == 1
    assert correct_count == 3
    assert misroutes == 3


def test_evaluate_candidate_with_rival_queries_computes_accuracy(
    write_skill: Callable[..., Path],
) -> None:
    """Verify evaluate_candidate calculates accuracy correctly when rival queries are present."""
    target_dir = write_skill(name="target-tool", description="Target calculation tool.")
    rival_dir = write_skill(name="rival-tool", description="Rival regex matching tool.")
    target = Skill(name="target-tool", description="Target calculation tool.", path=target_dir)
    rival = Skill(name="rival-tool", description="Rival regex matching tool.", path=rival_dir)

    queries = [
        Query(
            id="q1",
            text="Please run target tool",
            expected_skill="target-tool",
            kind=QueryKind.IMPLICIT,
        ),
        Query(
            id="q2",
            text="Please run rival tool",
            expected_skill="rival-tool",
            kind=QueryKind.IMPLICIT,
        ),
    ]
    candidate = OptimizationCandidate(
        description="Target calculation tool with algebra.",
        rationale="Candidate test.",
    )
    evaluated = evaluate_candidate(
        candidate=candidate,
        target=target,
        rivals=[rival],
        queries=queries,
        agent="keyword",
        budget=10,
    )
    assert evaluated.accuracy == 1.0
    assert evaluated.recall == 1.0
    assert evaluated.misroute_rate == 0.0


def test_optimize_skill_end_to_end_with_fake_agent(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify optimize_skill executes end-to-end and returns ranked candidates."""
    write_skill(
        name="text-tool",
        description="Format and manipulate strings.",
        body="# Text Tool\nSupports regex tokenization, casing, and unicode sanitization.",
    )
    write_skill(
        name="regex-tool",
        description="Match pattern strings using regular expressions.",
    )
    query_file = write_queries(target="text-tool", count=4)

    mock_candidates = [
        OptimizationCandidate(
            description="Format text strings, handle casing transforms, and unicode sanitization.",
            rationale="Incorporates unclaimed body terms casing and sanitization.",
        ),
        OptimizationCandidate(
            description="Process strings and text data without complex regex patterns.",
            rationale="Differentiates from regex rival.",
        ),
    ]

    with patch("reach.optimize.synthesize_candidates", return_value=mock_candidates):
        report = optimize_skill(
            skill_name="text-tool",
            skills_path=tmp_path,
            queries_path=query_file,
            agent="keyword",
            budget=10,
            auto_apply=False,
        )

        assert isinstance(report, OptimizationReport)
        assert report.skill_name == "text-tool"
        assert len(report.candidates) == 2
        assert report.applied is False
        assert report.best_candidate is not None


def test_optimize_skill_auto_apply_writes_to_disk(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify optimize_skill with auto_apply=True updates SKILL.md on disk."""
    target_dir = write_skill(
        name="disk-tool",
        description="Initial description.",
        body="# Disk Tool\nFeatures mount operations and partition formatting.",
    )

    query_file = write_queries(target="disk-tool", count=4)

    mock_candidates = [
        OptimizationCandidate(
            description="Perform partition formatting and disk mount operations.",
            rationale="Incorporates body keywords partition and mount.",
            recall=1.0,
            delta_recall=0.5,
        ),
    ]

    with patch("reach.optimize.synthesize_candidates", return_value=mock_candidates):
        report = optimize_skill(
            skill_name="disk-tool",
            skills_path=tmp_path,
            queries_path=query_file,
            agent="keyword",
            budget=10,
            auto_apply=True,
        )

        assert report.applied is True
        manifest_text = (target_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "Perform partition formatting and disk mount operations." in manifest_text


def test_optimize_skill_unknown_skill_raises_value_error(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify optimize_skill raises ValueError when requested skill is not in catalog."""
    write_skill(name="known-tool", description="A known utility.")
    with pytest.raises(ValueError, match="Skill 'non-existent' not found"):
        optimize_skill(
            skill_name="non-existent",
            skills_path=tmp_path,
        )


def test_optimize_skill_without_queries_generates_and_ranks_candidates(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify optimize_skill succeeds and produces candidates when no query set is provided."""
    write_skill(
        name="offline-tool",
        description="An offline utility tool.",
        body="# Offline Tool\nProvides localized processing.",
    )
    report = optimize_skill(
        skill_name="offline-tool",
        skills_path=tmp_path,
        queries_path=None,
        agent="keyword",
    )
    assert report.skill_name == "offline-tool"
    assert len(report.candidates) >= 1
    assert report.best_candidate is not None


def test_update_skill_description_corrupted_yaml_returns_false(tmp_path: Path) -> None:
    """Verify update_skill_description gracefully returns False on corrupted YAML."""
    corrupted_file = tmp_path / "corrupted" / "SKILL.md"
    corrupted_file.parent.mkdir(parents=True, exist_ok=True)
    corrupted_file.write_text("---\n[invalid: yaml: :\n---\n# Body", encoding="utf-8")

    success = update_skill_description(corrupted_file, "New description")
    assert success is False


def test_filter_candidates_empty_list_returns_empty() -> None:
    """Verify filter_candidates handles empty candidate sequences cleanly."""
    assert filter_candidates([], skill_name="any-tool") == []


def test_optimize_skill_misspelled_name_suggests_close_matches(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify optimize_skill suggests closest matching skill name on typos."""
    write_skill(name="python-designer", description="A tool for Python design.")
    with pytest.raises(ValueError, match=r"Did you mean: python-designer\?"):
        optimize_skill(
            skill_name="python-design",
            skills_path=tmp_path,
        )


def test_synthesize_candidates_returns_requested_count_even_without_rivals(
    write_skill: Callable[..., Path],
) -> None:
    """Verify synthesize_candidates produces requested number of candidates with 0 rivals."""
    solo_dir = write_skill(
        name="solo-tool",
        description="A standalone tool without any rivals in catalog.",
    )
    solo_skill = Skill(
        name="solo-tool",
        description="A standalone tool without any rivals in catalog.",
        path=solo_dir,
    )

    candidates = synthesize_candidates(
        target=solo_skill,
        rivals=[],
        count=3,
    )
    assert len(candidates) == 3
    # Verify all 3 descriptions are distinct
    descriptions = {c.description for c in candidates}
    assert len(descriptions) == 3


def test_optimize_skill_includes_semantic_rival_when_available(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify optimize_skill identifies both lexical and semantic rivals."""
    from unittest.mock import patch

    from reach.retrieval import DenseScorer

    write_skill(
        name="target-tool",
        description="Target tool for managing relational databases.",
    )
    write_skill(
        name="lexical-rival",
        description="Another tool for managing relational databases.",
    )
    write_skill(
        name="semantic-rival",
        description="SQL query optimizer and table inspector.",
    )

    mock_vectors = {
        "target-tool": [1.0, 0.0],
        "semantic-rival": [0.99, 0.0],
        "lexical-rival": [0.5, 0.0],
    }
    with patch.object(DenseScorer, "from_skills", return_value=DenseScorer(vectors=mock_vectors)):
        report = optimize_skill(
            skill_name="target-tool",
            skills_path=tmp_path,
            agent="fake",
        )
        assert len(report.candidates) >= 1


def test_optimize_skill_resolves_default_agent_from_config(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify optimize_skill resolves default agent from reach.toml configuration."""
    from unittest.mock import patch

    write_skill(
        name="cfg-tool",
        description="A tool for testing default agent resolution.",
    )
    custom_toml = tmp_path / "reach.toml"
    custom_toml.write_text("[general]\ndefault_agent = 'fake'\n", encoding="utf-8")

    with patch("reach.optimize._setup_driver") as mock_setup:
        from reach.runtime.fake import FakeGenerator

        mock_setup.return_value = FakeGenerator()

        optimize_skill(
            skill_name="cfg-tool",
            skills_path=tmp_path,
            config=custom_toml,
        )
        mock_setup.assert_called_once_with(None, None, config=custom_toml)


def test_build_optimization_prompt_resolves_limits_from_reach_toml(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify build_optimization_prompt reads min and max length from reach.toml."""
    target_dir = write_skill(
        name="custom-tool",
        description="Custom description for prompt testing.",
    )
    target = Skill(
        name="custom-tool",
        description="Custom description for prompt testing.",
        path=target_dir,
    )
    custom_toml = tmp_path / "reach.toml"
    custom_toml.write_text(
        "[lint]\nmin_description_length = 42\nmax_description_length = 650\n",
        encoding="utf-8",
    )

    prompt = build_optimization_prompt(
        target=target,
        rivals=[],
        config=custom_toml,
    )
    assert "between 42 and 650 characters" in prompt


def test_optimize_skill_propagates_lint_config(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify optimize_skill passes loaded LintSettings to filtering and synthesis."""
    from unittest.mock import patch

    write_skill(
        name="cfg-tool",
        description="A tool for testing lint config propagation.",
    )
    custom_toml = tmp_path / "reach.toml"
    custom_toml.write_text(
        "[general]\ndefault_agent = 'fake'\n[lint]\nmin_description_length = 150\n",
        encoding="utf-8",
    )

    with patch("reach.optimize._setup_driver") as mock_setup:
        from reach.runtime.fake import FakeGenerator

        mock_setup.return_value = FakeGenerator()
        report = optimize_skill(
            skill_name="cfg-tool",
            skills_path=tmp_path,
            config=custom_toml,
        )
        assert report.candidates
        for cand in report.candidates:
            if len(cand.description) < 150:
                assert cand.lint_clean is False


def test_synthesize_directional_disclaimer() -> None:
    """Verify directional disclaimer synthesizer produces valid bounded phrasing."""
    # With ceded terms
    res = synthesize_directional_disclaimer(
        "Deploy containerized apps.",
        "container-builder",
        ceded_terms=("docker", "image"),
    )
    assert res == "Deploy containerized apps. For docker, image, use container-builder instead."

    # Without trailing punctuation
    res_no_punct = synthesize_directional_disclaimer(
        "Deploy containerized apps",
        "container-builder",
    )
    assert res_no_punct == (
        "Deploy containerized apps. For container-builder-related tasks, "
        "use container-builder instead."
    )


def test_heuristic_candidates_includes_directional_disclaimer(tmp_path: Path) -> None:
    """Verify _heuristic_candidates generates zero-cost directional disclaimer candidate."""
    target = Skill(
        name="cloud-deployer",
        description="Deploy applications to cloud platforms.",
        path=tmp_path / "s1",
    )
    rival = Skill(
        name="container-builder",
        description="Build container images.",
        path=tmp_path / "s2",
    )
    candidates = _synthesize_via_heuristics(
        target=target,
        rivals=[rival],
        ceded_terms=("docker", "build"),
    )
    # Check that a candidate names container-builder with directional disclaimer
    matching = [c for c in candidates if "container-builder" in c.description]
    assert len(matching) >= 1
    assert "use container-builder instead" in matching[0].description


def test_synthesize_candidates_injects_directional_candidate_with_llm(tmp_path: Path) -> None:
    """Verify synthesize_candidates injects directional candidate alongside LLM results."""
    from unittest.mock import MagicMock

    target = Skill(
        name="cloud-deployer",
        description="Deploy applications to cloud platforms.",
        path=tmp_path / "s1",
    )
    rival = Skill(
        name="container-builder",
        description="Build container images.",
        path=tmp_path / "s2",
    )
    driver = MagicMock()
    driver.name = "mock-llm"
    driver.complete.return_value = (
        '{"candidates": [{"description": "LLM description for cloud deployments.", '
        '"rationale": "Optimized phrasing."}]}'
    )
    candidates = synthesize_candidates(
        target=target,
        rivals=[rival],
        driver=driver,
        ceded_terms=("docker", "build"),
    )
    assert len(candidates) == 2
    assert "use container-builder instead" in candidates[0].description
    assert candidates[1].description == "LLM description for cloud deployments."
