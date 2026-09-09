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

"""Validate multi-scale catalog scaling sweep, knee detection, and decomposition telemetry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from reach.catalog import resolve_sweep_scales
from reach.config import CatalogSettings, PlanSettings, RunConfig, StudySettings
from reach.models import CatalogMode, Query, QueryKind, Skill
from reach.queries import Origin, QuerySet, QuerySetProvenance, save_query_set
from reach.runtime.fake import FakeRuntime
from reach.sweep import (
    ScalingPoint,
    ScalingStudy,
    compute_scaling_noise_floor,
    find_kneedle_knee,
    run_scaling_sweep,
)

if TYPE_CHECKING:
    from pathlib import Path


def _create_mock_skills(root: Path, count: int) -> list[Skill]:
    """Create synthetic skill directories and Skill models."""
    skills = []
    for i in range(count):
        name = f"skill-{i:02d}"
        s_dir = root / name
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Description for {name}\n---\n# Body\n",
            encoding="utf-8",
        )
        skills.append(Skill(name=name, description=f"Description for {name}", path=s_dir))
    return skills


def test_find_kneedle_knee_sharp_drop() -> None:
    """Verify log-scale knee curvature identifies inflection point k*."""
    scales = (1, 5, 10, 20, 50, 100)
    pass_rates = (1.0, 0.95, 0.60, 0.55, 0.52, 0.50)
    knee = find_kneedle_knee(scales, pass_rates, noise_floor=0.05)
    assert knee == 10


def test_find_kneedle_knee_flat_returns_none() -> None:
    """Verify flat or within-noise curves return None."""
    scales = (1, 5, 10, 20)
    pass_rates = (0.95, 0.94, 0.95, 0.93)
    knee = find_kneedle_knee(scales, pass_rates, noise_floor=0.05)
    assert knee is None


def test_find_kneedle_knee_too_few_points() -> None:
    """Verify fewer than 3 points returns None."""
    assert find_kneedle_knee((1, 5), (1.0, 0.5)) is None


def test_compute_scaling_noise_floor_reuses_diff() -> None:
    """Verify compute_scaling_noise_floor calculates threshold using diff noise floor."""
    floor = compute_scaling_noise_floor(1.0, 0.5, sample_size=20)
    assert floor > 0.0


def test_run_scaling_sweep_insufficient_corpus(tmp_path: Path) -> None:
    """Verify sweeping 0 or 1 skill corpus raises descriptive ValueError."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _create_mock_skills(skills_dir, 1)

    q_file = tmp_path / "queries.json"
    qs = QuerySet(
        catalog_id="all",
        queries=(Query(id="q1", text="text", kind=QueryKind.IMPLICIT, expected_skill="skill-00"),),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
        ),
    )
    with pytest.raises(ValueError, match="Scaling sweep requires at least 2 skills in corpus"):
        run_scaling_sweep(config, target_skill="skill-00")


def test_sweep_scales_clamping() -> None:
    """Verify corpus smaller than requested scales clamps cleanly and gold standard default."""
    scales = resolve_sweep_scales(total_skills=14, requested=(1, 5, 10, 20, 50, 100))
    assert scales == (1, 5, 10, 14)

    # Test default scales with gold standard progression
    scales_default_large = resolve_sweep_scales(total_skills=300)
    assert scales_default_large == (10, 25, 50, 100, 200, 300)

    scales_default_clamped = resolve_sweep_scales(total_skills=150)
    assert scales_default_clamped == (10, 25, 50, 100, 150)


def test_scaling_sweep_happy_path(tmp_path: Path) -> None:
    """Verify scaling sweep runs end-to-end with FakeRuntime and outputs ScalingStudy."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _create_mock_skills(skills_dir, 20)

    q_file = tmp_path / "queries.yaml"
    qs = QuerySet(
        catalog_id="all",
        queries=(
            Query(id="q0", text="run skill 0", kind=QueryKind.IMPLICIT, expected_skill="skill-00"),
            Query(
                id="q1", text="deploy skill 0", kind=QueryKind.IMPLICIT, expected_skill="skill-00"
            ),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    study_settings = StudySettings(
        skills=skills_dir,
        queries=q_file,
        workdir=tmp_path / "ws",
        rescope=True,
        partial=True,
    )
    config = RunConfig(
        study=study_settings,
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )

    # FakeRuntime selects target skill for scale 1 and 5, but for scale 10 and 20 selects a rival
    selections = {
        "run skill 0": "skill-00",
        "deploy skill 0": "skill-00",
    }
    runtime = FakeRuntime(selections, model="mock-model")

    study = run_scaling_sweep(
        config=config,
        target_skill="skill-00",
        scales=(1, 5, 10, 20),
        runtime=runtime,
    )

    assert isinstance(study, ScalingStudy)
    assert study.target_skill == "skill-00"
    assert study.scales == (1, 5, 10, 20)
    assert len(study.points) == 4
    for pt in study.points:
        assert isinstance(pt, ScalingPoint)
        assert pt.probes_executed == 2
        assert pt.scale in (1, 5, 10, 20)


def test_run_scaling_sweep_with_duplicate_skills_in_corpus(tmp_path: Path) -> None:
    """Verify run_scaling_sweep executes cleanly when corpus contains duplicate skill names."""
    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 5)
    dup_dir = skills_dir / "nested" / "skill-01"
    dup_dir.mkdir(parents=True, exist_ok=True)
    (dup_dir / "SKILL.md").write_text(
        "---\nname: skill-01\ndescription: Duplicate copy\n---\n# Body\n",
        encoding="utf-8",
    )

    q_file = tmp_path / "queries.json"
    qs = QuerySet(
        catalog_id="all",
        queries=(
            Query(id="q0", text="run skill 0", kind=QueryKind.IMPLICIT, expected_skill="skill-00"),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            rescope=True,
            partial=True,
        ),
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )

    runtime = FakeRuntime({"run skill 0": "skill-00"}, model="mock-model", materialize=True)
    study = run_scaling_sweep(
        config=config,
        target_skill="skill-00",
        scales=(1, 3, 5),
        runtime=runtime,
    )
    assert len(study.points) == 3
    for pt in study.points:
        assert pt.probes_failed == 0


def test_run_scaling_sweep_in_memory(tmp_path: Path) -> None:
    """Verify run_scaling_sweep executes using in-memory skills and query_set without disk files."""
    skills = [
        Skill(name=f"skill-{i:02d}", description=f"Skill {i} description", path=tmp_path / f"s{i}")
        for i in range(5)
    ]
    qs = QuerySet(
        catalog_id="in-memory",
        queries=(
            Query(id="q0", text="run skill 0", kind=QueryKind.IMPLICIT, expected_skill="skill-00"),
        ),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    runtime = FakeRuntime({"run skill 0": "skill-00"}, model="mock-model", materialize=False)
    study = run_scaling_sweep(
        skills=skills,
        query_set=qs,
        target_skill="skill-00",
        scales=(1, 3, 5),
        runtime=runtime,
    )
    assert study.target_skill == "skill-00"
    assert study.is_corpus_sweep is False
    assert len(study.points) == 3
    assert study.points[0].pass_rate == 1.0


def test_corpus_scaling_sweep_happy_path(tmp_path: Path) -> None:
    """Verify whole-corpus capacity scaling sweep produces multi-class F1 and SLA thresholds."""
    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 6)

    queries = [
        Query(
            id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(6)
    ]
    q_file = tmp_path / "queries.json"
    qs = QuerySet(
        catalog_id="synthetic",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            early_stop=False,
        ),
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )

    # Runtime selects the correct skill for all queries
    selections = {f"Requesting task number {i:02d}": f"skill-{i:02d}" for i in range(6)}
    runtime = FakeRuntime(selections, model="mock-model")

    study = run_scaling_sweep(
        config=config,
        target_skill=None,
        scales=(1, 3, 6),
        runtime=runtime,
    )

    assert study.is_corpus_sweep is True
    assert study.target_skill is None
    assert study.total_corpus_skills == 6
    assert len(study.points) == 3
    assert study.sla_90_scale == 6
    assert study.sla_85_scale == 6

    for pt in study.points:
        assert pt.recall == 1.0
        assert pt.precision == 1.0
        assert pt.f1_score == 1.0
        assert pt.negative_probes == 0


def test_corpus_scaling_sweep_early_stopping(tmp_path: Path) -> None:
    """Verify corpus scaling sweep early stops when F1 drops below threshold."""
    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 10)

    queries = [
        Query(
            id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(10)
    ]
    q_file = tmp_path / "queries.json"
    qs = QuerySet(
        catalog_id="synthetic",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            early_stop=True,
        ),
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )

    # Runtime returns wrong skill for everything, dropping F1 to 0.0
    runtime = FakeRuntime({}, model="mock-model")

    study = run_scaling_sweep(
        config=config,
        target_skill=None,
        scales=(1, 3, 5, 8, 10),
        runtime=runtime,
    )

    assert study.is_corpus_sweep is True
    # Should evaluate scale 1 and scale 3, then early-stop before scale 5
    assert len(study.points) == 2
    assert study.points[-1].f1_score == 0.0


def test_corpus_scaling_sweep_anchor_default(tmp_path: Path) -> None:
    """Verify default anchor cohort uses cluster medoids from the first scale step."""
    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 8)

    queries = [
        Query(
            id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(8)
    ]
    q_file = tmp_path / "queries.json"
    qs = QuerySet(
        catalog_id="synthetic",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            early_stop=False,
        ),
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )
    selections = {f"Requesting task number {i:02d}": f"skill-{i:02d}" for i in range(8)}
    runtime = FakeRuntime(selections, model="mock-model")

    study = run_scaling_sweep(
        config=config,
        target_skill=None,
        scales=(2, 4, 8),
        runtime=runtime,
    )

    assert study.is_corpus_sweep is True
    assert study.anchor_skills is not None
    assert len(study.anchor_skills) == 2
    # In an anchor sweep, only anchor skill queries are probed at every scale
    for pt in study.points:
        assert pt.in_scope_probes == 2
        assert pt.recall == 1.0


def test_corpus_scaling_sweep_anchor_explicit_and_all(tmp_path: Path) -> None:
    """Verify explicit anchor cohort and anchor='all' full-corpus expansion."""
    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 8)

    queries = [
        Query(
            id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(8)
    ]
    q_file = tmp_path / "queries.json"
    qs = QuerySet(
        catalog_id="synthetic",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    save_query_set(qs, q_file)

    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            early_stop=False,
        ),
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )
    selections = {f"Requesting task number {i:02d}": f"skill-{i:02d}" for i in range(8)}
    runtime = FakeRuntime(selections, model="mock-model")

    # Explicit anchor cohort by name
    study_explicit = run_scaling_sweep(
        config=config,
        target_skill=None,
        scales=(3, 6, 8),
        anchor=("skill-01", "skill-03", "skill-05"),
        runtime=runtime,
    )
    assert study_explicit.anchor_skills == ("skill-01", "skill-03", "skill-05")
    for pt in study_explicit.points:
        assert pt.in_scope_probes == 3

    # anchor="all" dynamic expansion
    study_all = run_scaling_sweep(
        config=config,
        target_skill=None,
        scales=(2, 4, 8),
        anchor="all",
        runtime=runtime,
    )
    assert study_all.anchor_skills is None
    # In 'all' mode, in-scope probe count expands with catalog scale
    assert [pt.in_scope_probes for pt in study_all.points] == [2, 4, 8]

    # Nonexistent anchor raises ValueError
    with pytest.raises(ValueError, match=r"anchor skill.*not found in corpus"):
        run_scaling_sweep(
            config=config,
            target_skill=None,
            scales=(2, 4),
            anchor=("nonexistent-skill",),
            runtime=runtime,
        )


def test_compute_sla_crossings() -> None:
    """Verify compute_sla_crossings calculates conservative scale and log-linear interpolation."""
    from reach.sweep import compute_sla_crossings

    points = [
        ScalingPoint(
            scale=5,
            catalog_id="sweep:corpus:5",
            pass_rate=0.95,
            pass_rate_interval=(0.90, 1.0),
            recall=0.95,
            recall_interval=(0.90, 1.0),
            precision=0.95,
            precision_interval=(0.90, 1.0),
            internal_precision=0.95,
            abstention_rate=1.0,
            abstention_interval=(1.0, 1.0),
            f1_score=0.95,
            f1_interval=(0.90, 1.0),
            in_scope_probes=10,
            negative_probes=0,
            delta_vs_baseline=0.0,
            delta_context=0.0,
            delta_shadowing=0.0,
            probes_executed=10,
            probes_failed=0,
        ),
        ScalingPoint(
            scale=10,
            catalog_id="sweep:corpus:10",
            pass_rate=0.85,
            pass_rate_interval=(0.80, 0.90),
            recall=0.85,
            recall_interval=(0.80, 0.90),
            precision=0.85,
            precision_interval=(0.80, 0.90),
            internal_precision=0.85,
            abstention_rate=1.0,
            abstention_interval=(1.0, 1.0),
            f1_score=0.85,
            f1_interval=(0.80, 0.90),
            in_scope_probes=20,
            negative_probes=0,
            delta_vs_baseline=-0.10,
            delta_context=-0.05,
            delta_shadowing=-0.05,
            probes_executed=20,
            probes_failed=0,
        ),
    ]

    sla_scale, sla_interp = compute_sla_crossings(points, threshold=0.90)
    assert sla_scale == 5
    assert sla_interp is not None
    assert 5.0 < sla_interp < 10.0


def test_bootstrap_f1_ci() -> None:
    """Verify bootstrap_f1_ci computes bounded empirical confidence intervals."""
    from reach.models import CatalogMode, ProbeResult
    from reach.sweep import bootstrap_f1_ci

    results = [
        ProbeResult(
            query_id="q1",
            catalog_id="c",
            invoked_skills=("s1",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=4,
            model="mock-model",
            runtime="mock-runtime",
        ),
        ProbeResult(
            query_id="q2",
            catalog_id="c",
            invoked_skills=("s2",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=4,
            model="mock-model",
            runtime="mock-runtime",
        ),
        ProbeResult(
            query_id="q3",
            catalog_id="c",
            invoked_skills=("s1",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=4,
            model="mock-model",
            runtime="mock-runtime",
        ),  # FP
        ProbeResult(
            query_id="q4",
            catalog_id="c",
            invoked_skills=(),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=4,
            model="mock-model",
            runtime="mock-runtime",
        ),  # abstention / FN
    ]

    truth: dict[str, str | None] = {"q1": "s1", "q2": "s2", "q3": "s3", "q4": "s4"}
    installed = {"s1", "s2", "s3", "s4"}

    ci_low, ci_high = bootstrap_f1_ci(results, truth, installed, iterations=200, seed=42)
    assert 0.0 <= ci_low <= ci_high <= 1.0


def test_build_scaling_point_abstention_none_when_no_negatives() -> None:
    """Verify abstention_rate is None when there are no negative probes in evaluation."""
    from reach.models import CatalogMode, ProbeResult, Query, QueryKind
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _build_scaling_point

    queries = [
        Query(id="q1", text="q1", expected_skill="s1", kind=QueryKind.IMPLICIT),
        Query(id="q2", text="q2", expected_skill="s2", kind=QueryKind.IMPLICIT),
    ]
    query_set = QuerySet(
        catalog_id="c",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    results = [
        ProbeResult(
            query_id="q1",
            catalog_id="c",
            invoked_skills=("s1",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=2,
            model="mock",
            runtime="mock",
        ),
        ProbeResult(
            query_id="q2",
            catalog_id="c",
            invoked_skills=("s2",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=2,
            model="mock",
            runtime="mock",
        ),
    ]
    pt, _decomp = _build_scaling_point(
        scale=2,
        catalog_id="c",
        results=tuple(results),
        resolved_query_set=query_set,
        baseline_results=(),
        installed_skills={"s1", "s2"},
    )
    assert pt.in_scope_probes == 2
    assert pt.negative_probes == 0
    assert pt.abstention_rate is None
    assert pt.abstention_interval is None
    assert pt.prompt_tokens_mean is None


def test_build_scaling_point_abstention_calculated_when_negatives_present() -> None:
    """Verify abstention_rate is calculated when negative probes are present in evaluation."""
    from reach.models import CatalogMode, ProbeResult, Query, QueryKind
    from reach.queries import Origin, QuerySet, QuerySetProvenance
    from reach.sweep import _build_scaling_point

    queries = [
        Query(id="q1", text="in-scope", expected_skill="s1", kind=QueryKind.IMPLICIT),
        Query(id="q2", text="out-of-scope", expected_skill=None, kind=QueryKind.OUT_OF_SCOPE),
    ]
    query_set = QuerySet(
        catalog_id="c",
        queries=tuple(queries),
        provenance=QuerySetProvenance(origin=Origin.AUTHORED),
    )
    results = [
        ProbeResult(
            query_id="q1",
            catalog_id="c",
            invoked_skills=("s1",),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=2,
            model="mock",
            runtime="mock",
        ),
        ProbeResult(
            query_id="q2",
            catalog_id="c",
            invoked_skills=(),
            catalog_mode=CatalogMode.SWEEP,
            catalog_size=2,
            model="mock",
            runtime="mock",
        ),
    ]
    pt, _decomp = _build_scaling_point(
        scale=2,
        catalog_id="c",
        results=tuple(results),
        resolved_query_set=query_set,
        baseline_results=(),
        installed_skills={"s1", "s2"},
    )
    assert pt.in_scope_probes == 1
    assert pt.negative_probes == 1
    assert pt.abstention_rate == 1.0
    assert pt.abstention_interval is not None
    assert pt.abstention_interval[0] > 0.0


def test_scaling_sweep_does_not_skip_probes_when_out_path_configured(tmp_path: Path) -> None:
    """Verify scaling sweep evaluates all probes across scales even when study.out is configured."""
    skills_dir = tmp_path / "skills"
    _create_mock_skills(skills_dir, 4)
    queries = [
        Query(
            id=f"q-{i}",
            text=f"Requesting task number {i:02d}",
            expected_skill=f"skill-{i:02d}",
            kind=QueryKind.IMPLICIT,
        )
        for i in range(4)
    ]
    q_file = tmp_path / "queries.json"
    save_query_set(
        QuerySet(
            catalog_id="synthetic",
            queries=tuple(queries),
            provenance=QuerySetProvenance(origin=Origin.AUTHORED),
        ),
        q_file,
    )
    out_file = tmp_path / "sweep_results.jsonl"
    config = RunConfig(
        study=StudySettings(
            skills=skills_dir,
            queries=q_file,
            workdir=tmp_path / "ws",
            out=out_file,
        ),
        plan=PlanSettings(attempts=1),
        catalog=CatalogSettings(mode=CatalogMode.SWEEP),
    )
    selections = {f"Requesting task number {i:02d}": f"skill-{i:02d}" for i in range(4)}
    runtime = FakeRuntime(selections, model="mock-model", materialize=False)

    study = run_scaling_sweep(config=config, scales=(2, 4), runtime=runtime)

    assert len(runtime.queries) == 4
    assert len(study.points) == 2
    assert study.points[0].probes_executed == 2
    assert study.points[1].probes_executed == 2
