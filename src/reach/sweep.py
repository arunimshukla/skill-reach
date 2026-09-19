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

"""Execute catalog scaling sweeps and detect capacity knees."""

from __future__ import annotations

import math
import random
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reach.catalog import (
    CorpusScalingPlan,
    build_scaling_catalogs,
    find_cluster_medoids,
    load_skills,
    resolve_sweep_scales,
)
from reach.config import RunConfig
from reach.diff import DEFAULT_CONFIDENCE, NOISE_INFLATION
from reach.diff import noise_floor as diff_noise_floor
from reach.metrics import DecompositionResult, decompose_pass_rate_drop
from reach.models import NO_SKILL, Catalog, CatalogMode, ProbeResult, Query, Skill
from reach.queries import QuerySet, load_query_set
from reach.run import Composition, conduct
from reach.runtime import AgentRuntime, build_runtime
from reach.uncertainty import wilson_interval

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "ScalingPoint",
    "ScalingStudy",
    "bootstrap_f1_ci",
    "compute_scaling_noise_floor",
    "compute_sla_crossings",
    "find_kneedle_knee",
    "run_scaling_sweep",
]


class ScalingPoint(BaseModel):
    """Represent evaluation outcomes and decomposition for a single catalog scale point."""

    model_config = ConfigDict(frozen=True)

    scale: int
    catalog_id: str
    pass_rate: float
    pass_rate_interval: tuple[float, float]
    recall: float = 0.0
    recall_interval: tuple[float, float] = (0.0, 1.0)
    precision: float = 0.0
    precision_interval: tuple[float, float] = (0.0, 1.0)
    internal_precision: float = 0.0
    external_distractor_precision: float | None = None
    abstention_rate: float | None = None
    abstention_interval: tuple[float, float] | None = None
    f1_score: float = 0.0
    f1_interval: tuple[float, float] = (0.0, 1.0)
    in_scope_probes: int = 0
    negative_probes: int = 0
    disclosure_states: dict[str, int] = Field(default_factory=dict)
    delta_vs_baseline: float
    delta_context: float
    delta_shadowing: float
    probes_executed: int
    probes_failed: int = 0
    prompt_tokens_mean: float | None = None
    duration_ms_mean: float = 0.0


class ScalingStudy(BaseModel):
    """Represent multi-scale catalog scaling study and knee curvature analysis."""

    model_config = ConfigDict(frozen=True)

    target_skill: str | None = None
    is_corpus_sweep: bool = False
    scales: tuple[int, ...]
    points: tuple[ScalingPoint, ...]
    knee_scale: int | None = None
    sla_90_scale: int | None = None
    sla_90_interpolated: float | None = None
    sla_85_scale: int | None = None
    sla_85_interpolated: float | None = None
    baseline_pass_rate: float
    final_pass_rate: float
    total_delta: float
    total_context_loss: float
    total_shadowing_loss: float
    noise_floor: float = 0.05
    total_corpus_skills: int = 0
    decomposition: DecompositionResult | None = None
    anchor_skills: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _validate_target_skill_for_mode(self) -> ScalingStudy:
        """Ensure targeted sweeps specify a target skill."""
        if not self.is_corpus_sweep and self.target_skill is None:
            msg = "Targeted scaling sweep requires target_skill to be specified."
            raise ValueError(msg)
        return self


_MIN_KNEE_POINTS: int = 3
_MIN_DIFF_POINTS: int = 2
_MIN_EARLY_STOP_POINTS: int = 2
_EARLY_STOP_F1_THRESHOLD: float = 0.80
_SLA_90_THRESHOLD: float = 0.90
_SLA_85_THRESHOLD: float = 0.85


def compute_scaling_noise_floor(
    baseline_pass_rate: float,
    scaled_pass_rate: float,
    sample_size: int,
    confidence: float = DEFAULT_CONFIDENCE,
    noise_inflation: float = NOISE_INFLATION,
) -> float:
    """Calculate minimum scaling pass-rate drop distinguishable from noise using diff."""
    n = max(1, sample_size)
    control_se = math.sqrt(max(0.0, baseline_pass_rate * (1.0 - baseline_pass_rate)) / n)
    treatment_se = math.sqrt(max(0.0, scaled_pass_rate * (1.0 - scaled_pass_rate)) / n)
    floor = diff_noise_floor(control_se, treatment_se, confidence, noise_inflation)
    return max(0.01, floor)


def _compute_effective_noise_floor(
    noise_floor: float | None,
    points: Sequence[ScalingPoint],
    baseline_count: int,
) -> float:
    """Resolve user-configured or diff-derived scaling noise floor."""
    if noise_floor is not None:
        return noise_floor
    if len(points) >= _MIN_DIFF_POINTS:
        return round(
            compute_scaling_noise_floor(
                points[0].pass_rate,
                points[-1].pass_rate,
                sample_size=max(1, baseline_count),
            ),
            4,
        )
    return 0.05


def find_kneedle_knee(
    scales: Sequence[int],
    pass_rates: Sequence[float],
    noise_floor: float = 0.10,
) -> int | None:
    """Identify the inflection knee scale k* using normalized log-scale Kneedle curvature."""
    if (
        len(scales) < _MIN_KNEE_POINTS
        or len(scales) != len(pass_rates)
        or (max(pass_rates) - min(pass_rates)) <= noise_floor
    ):
        return None

    points = sorted(zip(scales, pass_rates, strict=True), key=lambda p: p[0])
    k_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    if (y_vals[0] - y_vals[-1]) <= noise_floor:
        return None

    log_k = [math.log(k) for k in k_vals]
    min_log = log_k[0]
    max_log = log_k[-1]
    log_range = max_log - min_log
    min_y = min(y_vals)
    max_y = max(y_vals)
    y_range = max_y - min_y

    if log_range <= 0.0 or y_range <= 0.0:
        return None

    norm_x = [(lk - min_log) / log_range for lk in log_k]
    norm_y = [(y - min_y) / y_range for y in y_vals]

    y0 = norm_y[0]
    y_end = norm_y[-1]
    diffs_below = [(y0 + x * (y_end - y0)) - y for x, y in zip(norm_x, norm_y, strict=True)]
    diffs_above = [y - (y0 + x * (y_end - y0)) for x, y in zip(norm_x, norm_y, strict=True)]

    max_below = max(diffs_below[1:-1])
    max_above = max(diffs_above[1:-1])
    min_prominence = max(0.05, (noise_floor * 0.5) / y_range)

    if max_above >= max_below and max_above > min_prominence:
        knee_idx = 1 + diffs_above[1:-1].index(max_above)
        return k_vals[knee_idx]
    if max_below > min_prominence:
        knee_idx = 1 + diffs_below[1:-1].index(max_below)
        return k_vals[knee_idx]
    return None


def bootstrap_f1_ci(
    results: Sequence[ProbeResult],
    truth: Mapping[str, str | None],
    installed_skills: set[str],
    iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute empirical bootstrap confidence interval for micro F1 score."""
    m = len(results)
    if m <= 0 or iterations <= 0:
        return (0.0, 1.0)

    tp_arr = [
        1
        if not r.error
        and r.predicted_label == truth.get(r.query_id)
        and truth.get(r.query_id) is not None
        else 0
        for r in results
    ]
    fn_arr = [
        1 if truth.get(r.query_id) is not None and not tp_arr[i] else 0
        for i, r in enumerate(results)
    ]
    fp_arr = [
        1
        if not r.error
        and r.predicted_label in installed_skills
        and r.predicted_label != truth.get(r.query_id)
        else 0
        for r in results
    ]

    rng = random.Random(seed)  # noqa: S311
    f1_boots: list[float] = []
    indices = list(range(m))

    for _ in range(iterations):
        sample_idx = [rng.choice(indices) for _ in range(m)]
        tp_s = sum(tp_arr[i] for i in sample_idx)
        fp_s = sum(fp_arr[i] for i in sample_idx)
        fn_s = sum(fn_arr[i] for i in sample_idx)
        denom = 2 * tp_s + fp_s + fn_s
        f1_boots.append(2.0 * tp_s / denom if denom > 0 else 0.0)

    f1_boots.sort()
    low_idx = max(0, int(iterations * 0.025))
    high_idx = min(int(iterations * 0.975), iterations - 1)
    return (round(f1_boots[low_idx], 4), round(f1_boots[high_idx], 4))


def _log_interpolate_scale(
    p1: ScalingPoint,
    p2: ScalingPoint,
    threshold: float,
) -> float | None:
    """Interpolate log-linear scale crossing between two adjacent scaling points."""
    denom = p2.f1_score - p1.f1_score
    if denom == 0.0:
        return None
    t = (threshold - p1.f1_score) / denom
    log_k1 = math.log(max(1, p1.scale))
    log_k2 = math.log(max(1, p2.scale))
    return round(math.exp(log_k1 + t * (log_k2 - log_k1)), 1)


def compute_sla_crossings(
    points: Sequence[ScalingPoint],
    threshold: float,
) -> tuple[int | None, float | None]:
    """Compute discrete conservative scale and log-linear continuous crossing for an SLA target."""
    if not points:
        return None, None

    ordered = sorted(points, key=lambda p: p.scale)
    discrete_k: int | None = None
    interp_k: float | None = None

    for i in range(len(ordered) - 1):
        p1, p2 = ordered[i], ordered[i + 1]
        if p1.f1_score >= threshold > p2.f1_score:
            discrete_k = p1.scale
            interp_k = _log_interpolate_scale(p1, p2, threshold)
            break

    if discrete_k is None:
        qualifying = [p.scale for p in ordered if p.f1_score >= threshold]
        discrete_k = max(qualifying) if qualifying else None
        for i in range(len(ordered) - 1):
            p1, p2 = ordered[i], ordered[i + 1]
            if p1.f1_score <= threshold < p2.f1_score:
                interp_k = _log_interpolate_scale(p1, p2, threshold)
                break

    if interp_k is None and discrete_k is not None:
        interp_k = float(discrete_k)

    return discrete_k, interp_k


def _resolve_sweep_target_and_queries(
    skills: Sequence[Skill],
    raw_query_set: QuerySet,
    target_skill: str | None,
) -> tuple[str, QuerySet]:
    """Resolve target skill and filter query set to relevant target queries."""
    target = target_skill
    if not target:
        target = next(
            (q.expected_skill for q in raw_query_set.queries if q.expected_skill),
            skills[0].name,
        )

    by_name = {s.name: s for s in skills}
    if target not in by_name:
        msg = f"target skill {target!r} not found in loaded skills"
        raise KeyError(msg)

    target_queries = tuple(
        q for q in raw_query_set.queries if q.expected_skill == target or q.is_out_of_scope
    )
    if not target_queries:
        target_queries = raw_query_set.queries

    query_set = QuerySet(
        catalog_id="all",
        queries=target_queries,
        provenance=raw_query_set.provenance,
    )
    return target, query_set


class _ScalePassRate(NamedTuple):
    """Represent point pass rate and Wilson confidence interval."""

    executed: int
    fails: int
    pass_rate: float
    pass_interval: tuple[float, float]


class _ScaleClassificationMetrics(NamedTuple):
    """Represent classification and confusion matrix outcomes for a scaling point."""

    in_scope_probes: int
    negative_probes: int
    recall: float
    recall_interval: tuple[float, float]
    precision: float
    precision_interval: tuple[float, float]
    internal_precision: float
    external_distractor_precision: float | None
    abstention_rate: float | None
    abstention_interval: tuple[float, float] | None
    f1_score: float
    f1_interval: tuple[float, float]


class _ScaleTelemetry(NamedTuple):
    """Represent aggregate execution telemetry and resource utilization."""

    disclosure_states: dict[str, int]
    prompt_tokens_mean: float | None
    duration_ms_mean: float


class _ScaleDecomposition(NamedTuple):
    """Represent comparative degradation components against baseline."""

    delta_vs_baseline: float
    delta_context: float
    delta_shadowing: float
    decomposition: DecompositionResult | None


def _calculate_scale_pass_rate(
    results: Sequence[ProbeResult],
    truth_labels: Mapping[str, str],
) -> _ScalePassRate:
    """Calculate overall pass rate, failure count, and Wilson confidence interval."""
    executed = len(results)
    hits = sum(
        1 for r in results if not r.error and r.predicted_label == truth_labels.get(r.query_id)
    )
    fails = executed - hits
    pass_rate = hits / executed if executed else 0.0
    interval_obj = wilson_interval(hits, executed)
    pass_interval = (interval_obj.low, interval_obj.high) if interval_obj else (0.0, 1.0)
    return _ScalePassRate(
        executed=executed,
        fails=fails,
        pass_rate=pass_rate,
        pass_interval=pass_interval,
    )


def _compute_scope_counts(
    in_scope: Sequence[ProbeResult],
    truth_expected: Mapping[str, str | None],
    installed: set[str],
) -> tuple[int, int, float, tuple[float, float]]:
    """Compute true positives, routing false negatives, recall, and Wilson interval."""
    tp = sum(1 for r in in_scope if not r.error and r.predicted_label == truth_expected[r.query_id])
    fn_misroute = sum(
        1
        for r in in_scope
        if not r.error
        and r.predicted_label != truth_expected[r.query_id]
        and r.predicted_label in installed
    )
    recall = tp / len(in_scope) if in_scope else 0.0
    rec_int = wilson_interval(tp, len(in_scope))
    recall_interval = (rec_int.low, rec_int.high) if rec_int else (0.0, 1.0)
    return tp, fn_misroute, recall, recall_interval


def _compute_negative_counts(
    negative: Sequence[ProbeResult],
    installed: set[str],
) -> tuple[int, int, float | None, tuple[float, float] | None]:
    """Compute true negatives, distractor false positives, and abstention intervals."""
    tn = sum(1 for r in negative if not r.error and r.predicted_label == NO_SKILL)
    fp_distractor = sum(1 for r in negative if not r.error and r.predicted_label in installed)
    if not negative:
        return tn, fp_distractor, None, None
    abstention_rate = round(tn / len(negative), 4)
    abst_int = wilson_interval(tn, len(negative))
    abstention_interval = (round(abst_int.low, 4), round(abst_int.high, 4)) if abst_int else None
    return tn, fp_distractor, abstention_rate, abstention_interval


def _compute_precision_metrics(
    tp: int,
    fn_misroute: int,
    fp_distractor: int,
    has_negatives: bool,
) -> tuple[float, float | None, float, tuple[float, float]]:
    """Compute internal precision, external distractor precision, and overall precision."""
    internal_fp = fn_misroute
    internal_precision = tp / (tp + internal_fp) if (tp + internal_fp) else 1.0
    ext_prec = (
        round(tp / (tp + fp_distractor), 4)
        if (has_negatives and (tp + fp_distractor) > 0)
        else None
    )
    overall_fp = internal_fp + fp_distractor
    precision = tp / (tp + overall_fp) if (tp + overall_fp) else 1.0
    prec_int = wilson_interval(tp, tp + overall_fp) if (tp + overall_fp) else None
    precision_interval = (prec_int.low, prec_int.high) if prec_int else (1.0, 1.0)
    return internal_precision, ext_prec, precision, precision_interval


def _compute_f1_score(precision: float, recall: float) -> float:
    """Compute standard harmonic mean F1 score from precision and recall."""
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom > 0 else 0.0


def _calculate_scale_classification(
    results: Sequence[ProbeResult],
    truth_expected: Mapping[str, str | None],
    installed_skills: set[str] | None,
    seed: int,
) -> _ScaleClassificationMetrics:
    """Calculate precision, recall, abstention rate, and F1 confidence intervals."""
    installed = installed_skills if installed_skills is not None else set()
    in_scope = [r for r in results if truth_expected.get(r.query_id) is not None]
    negative = [r for r in results if truth_expected.get(r.query_id) is None]

    tp, fn_misroute, recall, recall_interval = _compute_scope_counts(
        in_scope, truth_expected, installed
    )
    _tn, fp_distractor, abstention_rate, abstention_interval = _compute_negative_counts(
        negative, installed
    )
    internal_prec, ext_prec, precision, precision_interval = _compute_precision_metrics(
        tp, fn_misroute, fp_distractor, bool(negative)
    )

    f1 = _compute_f1_score(precision, recall)
    f1_ci = bootstrap_f1_ci(results, truth_expected, installed, iterations=1000, seed=seed)

    return _ScaleClassificationMetrics(
        in_scope_probes=len(in_scope),
        negative_probes=len(negative),
        recall=recall,
        recall_interval=recall_interval,
        precision=precision,
        precision_interval=precision_interval,
        internal_precision=internal_prec,
        external_distractor_precision=ext_prec,
        abstention_rate=abstention_rate,
        abstention_interval=abstention_interval,
        f1_score=f1,
        f1_interval=(f1_ci[0], f1_ci[1]),
    )


def _aggregate_scale_telemetry(results: Sequence[ProbeResult]) -> _ScaleTelemetry:
    """Aggregate disclosure states, prompt tokens, and duration across probes."""
    disclosure_counts = dict(
        Counter(
            r.disclosure_state.value
            for r in results
            if getattr(r, "disclosure_state", None) is not None
        )
    )
    tokens: list[float] = [
        float(tok) for r in results if (tok := getattr(r, "prompt_tokens", None)) is not None
    ]
    avg_tokens = statistics.fmean(tokens) if tokens else None
    prompt_tokens_mean = round(avg_tokens, 2) if avg_tokens is not None else None

    durations = [r.duration_ms for r in results if r.duration_ms is not None]
    avg_duration = round(statistics.fmean(durations), 2) if durations else 0.0

    return _ScaleTelemetry(
        disclosure_states=disclosure_counts,
        prompt_tokens_mean=prompt_tokens_mean,
        duration_ms_mean=avg_duration,
    )


def _evaluate_baseline_decomposition(
    results: Sequence[ProbeResult],
    baseline_results: Sequence[ProbeResult],
    queries: Sequence[Query],
    seed: int,
) -> _ScaleDecomposition:
    """Evaluate loss decomposition against baseline results if baseline is provided."""
    if not baseline_results:
        return _ScaleDecomposition(
            delta_vs_baseline=0.0,
            delta_context=0.0,
            delta_shadowing=0.0,
            decomposition=None,
        )
    decomp = decompose_pass_rate_drop(
        baseline_results=baseline_results,
        scaled_results=results,
        queries=queries,
        seed=seed,
    )
    return _ScaleDecomposition(
        delta_vs_baseline=decomp.delta_total,
        delta_context=decomp.delta_context,
        delta_shadowing=decomp.delta_shadowing,
        decomposition=decomp,
    )


def _build_scaling_point(
    scale: int,
    catalog_id: str,
    results: tuple[ProbeResult, ...],
    resolved_query_set: QuerySet,
    baseline_results: tuple[ProbeResult, ...],
    installed_skills: set[str] | None = None,
    seed: int = 42,
) -> tuple[ScalingPoint, DecompositionResult | None]:
    """Calculate point metrics, Wilson confidence intervals, and pass-rate decomposition."""
    queries = resolved_query_set.queries
    truth_labels = {q.id: q.truth_label for q in queries}
    truth_expected = {q.id: q.expected_skill for q in queries}

    pass_stats = _calculate_scale_pass_rate(results, truth_labels)
    class_stats = _calculate_scale_classification(results, truth_expected, installed_skills, seed)
    telemetry = _aggregate_scale_telemetry(results)
    decomp_stats = _evaluate_baseline_decomposition(results, baseline_results, queries, seed)

    point = ScalingPoint(
        scale=scale,
        catalog_id=catalog_id,
        pass_rate=round(pass_stats.pass_rate, 4),
        pass_rate_interval=(
            round(pass_stats.pass_interval[0], 4),
            round(pass_stats.pass_interval[1], 4),
        ),
        recall=round(class_stats.recall, 4),
        recall_interval=(
            round(class_stats.recall_interval[0], 4),
            round(class_stats.recall_interval[1], 4),
        ),
        precision=round(class_stats.precision, 4),
        precision_interval=(
            round(class_stats.precision_interval[0], 4),
            round(class_stats.precision_interval[1], 4),
        ),
        internal_precision=round(class_stats.internal_precision, 4),
        external_distractor_precision=class_stats.external_distractor_precision,
        abstention_rate=class_stats.abstention_rate,
        abstention_interval=class_stats.abstention_interval,
        f1_score=round(class_stats.f1_score, 4),
        f1_interval=(
            round(class_stats.f1_interval[0], 4),
            round(class_stats.f1_interval[1], 4),
        ),
        in_scope_probes=class_stats.in_scope_probes,
        negative_probes=class_stats.negative_probes,
        disclosure_states=telemetry.disclosure_states,
        delta_vs_baseline=round(decomp_stats.delta_vs_baseline, 4),
        delta_context=round(decomp_stats.delta_context, 4),
        delta_shadowing=round(decomp_stats.delta_shadowing, 4),
        probes_executed=pass_stats.executed,
        probes_failed=pass_stats.fails,
        prompt_tokens_mean=telemetry.prompt_tokens_mean,
        duration_ms_mean=telemetry.duration_ms_mean,
    )
    return point, decomp_stats.decomposition


def _prepare_sweep_config(
    config: RunConfig | None,
    attempts: int | None,
    early_stop: bool | None,
) -> RunConfig:
    """Apply CLI overrides to execution configuration."""
    cfg = config or RunConfig()
    plan_update = {"attempts": attempts} if attempts is not None else {}
    study_update: dict[str, object] = {}
    if early_stop is not None:
        study_update["early_stop"] = early_stop

    if plan_update:
        cfg = cfg.model_copy(update={"plan": cfg.plan.model_copy(update=plan_update)})
    if study_update:
        cfg = cfg.model_copy(update={"study": cfg.study.model_copy(update=study_update)})
    return cfg


def _build_study_result(
    target: str | None,
    is_corpus: bool,
    evaluated_scales: Sequence[int],
    points: Sequence[ScalingPoint],
    knee: int | None,
    noise_floor: float,
    total_skills: int,
    decomp: DecompositionResult | None,
    anchor_skills: Sequence[str] | None = None,
) -> ScalingStudy:
    """Construct finished ScalingStudy data model."""
    sla_90_scale: int | None = None
    sla_90_interp: float | None = None
    sla_85_scale: int | None = None
    sla_85_interp: float | None = None

    if is_corpus:
        sla_90_scale, sla_90_interp = compute_sla_crossings(points, _SLA_90_THRESHOLD)
        sla_85_scale, sla_85_interp = compute_sla_crossings(points, _SLA_85_THRESHOLD)

    b_rate = points[0].pass_rate if points else 0.0
    f_rate = points[-1].pass_rate if points else 0.0
    t_delta = points[-1].delta_vs_baseline if len(points) > 1 else 0.0
    t_ctx = points[-1].delta_context if len(points) > 1 else 0.0
    t_shd = points[-1].delta_shadowing if len(points) > 1 else 0.0

    return ScalingStudy(
        target_skill=target,
        is_corpus_sweep=is_corpus,
        scales=tuple(evaluated_scales),
        points=tuple(points),
        knee_scale=knee,
        sla_90_scale=sla_90_scale,
        sla_90_interpolated=sla_90_interp,
        sla_85_scale=sla_85_scale,
        sla_85_interpolated=sla_85_interp,
        baseline_pass_rate=b_rate,
        final_pass_rate=f_rate,
        total_delta=t_delta,
        total_context_loss=t_ctx,
        total_shadowing_loss=t_shd,
        noise_floor=noise_floor,
        total_corpus_skills=total_skills,
        decomposition=decomp,
        anchor_skills=tuple(anchor_skills) if anchor_skills is not None else None,
    )


def _resolve_anchor_skills(
    requested_anchor: object,
    resolved_skills: Sequence[Skill],
    actual_scales: Sequence[int],
) -> tuple[str, ...] | None:
    """Resolve anchor skills cohort from configuration or initial scale medoids."""
    if isinstance(requested_anchor, str) and requested_anchor.lower() == "all":
        return None
    if isinstance(requested_anchor, int):
        return find_cluster_medoids(resolved_skills, requested_anchor)
    if isinstance(requested_anchor, str):
        if requested_anchor.isdigit():
            return find_cluster_medoids(resolved_skills, int(requested_anchor))
        raw_names = tuple(p.strip() for p in requested_anchor.split(",") if p.strip())
    elif isinstance(requested_anchor, (list, tuple)):
        raw_names = tuple(str(s).strip() for s in requested_anchor if str(s).strip())
    else:
        return find_cluster_medoids(resolved_skills, actual_scales[0])

    corpus_names = {s.name for s in resolved_skills}
    missing = [a for a in raw_names if a not in corpus_names]
    if missing:
        msg = f"anchor skill(s) not found in corpus: {', '.join(missing)}"
        raise ValueError(msg)
    return raw_names


def _setup_sweep_execution(
    is_corpus: bool,
    target_skill: str | None,
    anchor: int | Sequence[str] | str | None,
    resolved_skills: Sequence[Skill],
    actual_scales: Sequence[int],
    raw_query_set: QuerySet,
    effective_config: RunConfig,
    rivals_share: float,
) -> tuple[str | None, list[Catalog], QuerySet, CorpusScalingPlan | None, tuple[str, ...] | None]:
    """Configure catalogs, query sets, and scaling plan for sweep execution."""
    if is_corpus:
        requested_anchor = anchor if anchor is not None else effective_config.study.anchor
        resolved_anchors = _resolve_anchor_skills(requested_anchor, resolved_skills, actual_scales)
        plan = CorpusScalingPlan.create(
            skills=resolved_skills,
            scales=actual_scales,
            anchor_skills=resolved_anchors,
        )
        return None, list(plan.catalogs), raw_query_set, plan, resolved_anchors

    target, query_set = _resolve_sweep_target_and_queries(
        resolved_skills, raw_query_set, target_skill
    )
    catalogs = build_scaling_catalogs(
        skills=resolved_skills,
        target_skill=target,
        scales=actual_scales,
        rivals_share=rivals_share,
        seed=effective_config.catalog.seed,
    )
    return target, catalogs, query_set, None, None


def run_scaling_sweep(
    config: RunConfig | None = None,
    target_skill: str | None = None,
    scales: Sequence[int] | None = None,
    anchor: int | Sequence[str] | str | None = None,
    runtime: AgentRuntime | None = None,
    rivals_share: float = 0.5,
    noise_floor: float | None = None,
    workers: int = 1,
    skills: Sequence[Skill] | None = None,
    query_set: QuerySet | None = None,
    attempts: int | None = None,
    early_stop: bool | None = None,
) -> ScalingStudy:
    """Execute multi-scale catalog evaluation sweep and return scaling analysis."""
    effective_config = _prepare_sweep_config(config, attempts, early_stop)
    resolved_skills = (
        list(skills) if skills is not None else load_skills(effective_config.require_skills())
    )
    if len(resolved_skills) <= 1:
        msg = f"Scaling sweep requires at least 2 skills in corpus, got {len(resolved_skills)}"
        raise ValueError(msg)

    raw_query_set = (
        query_set if query_set is not None else load_query_set(effective_config.require_queries())
    )

    is_corpus = target_skill is None
    requested_scales = scales if scales is not None else effective_config.study.scales
    actual_scales = resolve_sweep_scales(len(resolved_skills), requested_scales)

    target, catalogs, resolved_query_set, corpus_plan, resolved_anchors = _setup_sweep_execution(
        is_corpus=is_corpus,
        target_skill=target_skill,
        anchor=anchor,
        resolved_skills=resolved_skills,
        actual_scales=actual_scales,
        raw_query_set=raw_query_set,
        effective_config=effective_config,
        rivals_share=rivals_share,
    )

    resolved_runtime = runtime or build_runtime(effective_config.runtime)
    baseline_results: tuple[ProbeResult, ...] = ()
    points: list[ScalingPoint] = []
    final_decomp: DecompositionResult | None = None

    work_dir = effective_config.study.workdir
    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="reach_sweep_")
        work_dir = Path(temp_dir_obj.name)

    shared_outcome_cache: dict[Any, Any] = {}
    try:
        for scale, catalog in zip(actual_scales, catalogs, strict=True):
            safe_cat_id = catalog.id.replace(":", "_").replace("/", "_")
            scale_out = work_dir / f"sweep_{safe_cat_id}.jsonl"
            scale_config = effective_config.model_copy(
                update={
                    "study": effective_config.study.model_copy(
                        update={
                            "catalog": catalog.id,
                            "rescope": True,
                            "partial": True,
                            "workdir": work_dir,
                            "out": scale_out,
                        }
                    ),
                    "catalog": effective_config.catalog.model_copy(
                        update={"mode": CatalogMode.SWEEP}
                    ),
                }
            )

            scale_query_set = (
                corpus_plan.queries_for_scale(
                    catalog=catalog,
                    raw_query_set=raw_query_set,
                )
                if is_corpus and corpus_plan is not None
                else resolved_query_set
            )

            composed = Composition(
                config=scale_config,
                query_set=scale_query_set,
                catalog=catalog,
                skills=tuple(resolved_skills),
            )

            outcome = conduct(
                config=scale_config,
                runtime=resolved_runtime,
                composed=composed,
                allow_truncation=True,
                append_across_arms=True,
                workers=workers,
                outcome_cache=shared_outcome_cache,
            )

            point, decomp = _build_scaling_point(
                scale=scale,
                catalog_id=catalog.id,
                results=outcome.results,
                resolved_query_set=scale_query_set,
                baseline_results=baseline_results,
                installed_skills=set(catalog.skills),
                seed=effective_config.catalog.seed,
            )
            points.append(point)

            if scale == actual_scales[0]:
                baseline_results = outcome.results
            elif decomp is not None:
                final_decomp = decomp

            if (
                effective_config.study.early_stop
                and is_corpus
                and len(points) >= _MIN_EARLY_STOP_POINTS
                and point.f1_interval[1] < _EARLY_STOP_F1_THRESHOLD
            ):
                break

        effective_noise_floor = _compute_effective_noise_floor(
            noise_floor, points, len(baseline_results)
        )
        evaluated_scales = actual_scales[: len(points)]
        rate_curve = [p.f1_score for p in points] if is_corpus else [p.pass_rate for p in points]
        knee = find_kneedle_knee(evaluated_scales, rate_curve, noise_floor=effective_noise_floor)

        return _build_study_result(
            target=target,
            is_corpus=is_corpus,
            evaluated_scales=evaluated_scales,
            points=points,
            knee=knee,
            noise_floor=effective_noise_floor,
            total_skills=len(resolved_skills),
            decomp=final_decomp,
            anchor_skills=resolved_anchors,
        )
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()
