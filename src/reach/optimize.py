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

"""Closed-loop skill description optimization using diagnostic findings and empirical probes."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from reach.catalog import load_skills, split_frontmatter
from reach.config import OptimizeSettings, load_config, resolve_path
from reach.lint import LintSettings
from reach.models import Catalog, CatalogMode, Query, Skill
from reach.overlap import rank_corpus
from reach.queries import load_query_set
from reach.rewrite import skill_body, suggest_rewrite, synthesize_directional_disclaimer
from reach.runtime import FAKE_AGENT, TextGenerator, build_runtime, build_text_generator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reach.runtime import AgentRuntime

__all__ = [
    "OptimizationCandidate",
    "OptimizationReport",
    "build_optimization_prompt",
    "evaluate_candidate",
    "filter_candidates",
    "optimize_skill",
    "synthesize_candidates",
    "update_skill_description",
]

_DEFAULT_OPTIMIZE = OptimizeSettings()
DEFAULT_BUDGET = _DEFAULT_OPTIMIZE.budget
DEFAULT_TEMPERATURE = _DEFAULT_OPTIMIZE.temperature


class OptimizationCandidate(BaseModel):
    """Represent a generated description rewrite and its empirical performance."""

    model_config = ConfigDict(frozen=True)

    accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    delta_recall: float = Field(default=0.0, ge=-1.0, le=1.0)
    description: str
    lint_clean: bool = True
    misroute_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    recall: float = Field(default=0.0, ge=0.0, le=1.0)


class OptimizationReport(BaseModel):
    """Represent the full results of closed-loop skill description optimization."""

    model_config = ConfigDict(frozen=True)

    applied: bool = False
    baseline_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    baseline_description: str
    baseline_misroute: float = Field(default=0.0, ge=0.0, le=1.0)
    baseline_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    candidates: tuple[OptimizationCandidate, ...] = ()
    ceded_terms: tuple[str, ...] = ()
    has_probes: bool = False
    manifest_path: Path | None = None
    rival_name: str = ""
    skill_name: str
    unclaimed_terms: tuple[str, ...] = ()

    @property
    def best_candidate(self) -> OptimizationCandidate | None:
        """Return top-ranked candidate if any candidates exist."""
        return self.candidates[0] if self.candidates else None


def update_skill_description(manifest_path: Path, new_description: str) -> bool:
    """Update the description field in SKILL.md frontmatter while preserving file contents."""
    if not manifest_path.is_file():
        return False
    try:
        text = manifest_path.read_text(encoding="utf-8")
        split = split_frontmatter(text)
        if split is None:
            return False
        raw_frontmatter, body = split
        data = yaml.safe_load(raw_frontmatter)
        if not isinstance(data, dict):
            return False
        data["description"] = new_description
        new_yaml = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
        manifest_path.write_text(f"---\n{new_yaml}\n---{body}", encoding="utf-8")
    except (OSError, yaml.YAMLError):
        return False
    return True


def _resolve_lint_settings(config: LintSettings | Path | None = None) -> LintSettings:
    """Resolve and return LintSettings from an existing instance or config path."""
    if isinstance(config, LintSettings):
        return config
    return LintSettings.from_settings(load_config(config))


def build_optimization_prompt(
    target: Skill,
    rivals: Sequence[Skill],
    ceded_terms: Sequence[str] = (),
    unclaimed_terms: Sequence[str] = (),
    count: int = 3,
    min_length: int | None = None,
    max_length: int | None = None,
    config: LintSettings | Path | None = None,
) -> str:
    """Construct an LLM prompt to synthesize differentiated skill description candidates."""
    lint_config = _resolve_lint_settings(config)
    effective_min = min_length if min_length is not None else lint_config.min_description_length
    effective_max = max_length if max_length is not None else lint_config.max_description_length

    target_body = skill_body(target)
    rival_info = (
        "\n".join(f"- {r.name}: {r.description}" for r in rivals)
        if rivals
        else "No immediate rivals identified."
    )

    ceded_str = ", ".join(f"'{t}'" for t in ceded_terms) if ceded_terms else "None"
    unclaimed_str = ", ".join(f"'{t}'" for t in unclaimed_terms) if unclaimed_terms else "None"

    return f"""You are an expert AI agent skill engineer optimizing a skill's catalog description.
An AI agent uses the description to decide whether to invoke this skill when solving user tasks.

Target Skill Name: {target.name}
Current Description: {target.description}

Target Skill Body:
{target_body[:1500]}

Competing Rival Skills:
{rival_info}

Diagnostic Vocabulary Analysis:
- Ceded Terms (words currently in description that attract rival skills instead): {ceded_str}
- Unclaimed Terms (distinctive keywords from body absent from rivals): {unclaimed_str}

Task:
Generate {count} distinct candidate descriptions for '{target.name}'.
Each candidate should:
1. Be strictly between {effective_min} and {effective_max} characters.
2. Distinctly claim the user tasks and intents this skill solves.
3. Incorporate distinctive unclaimed terms where natural.
4. Avoid or disclaim ceded terms that cause confusing misroutes to rivals.

Format your output as a JSON object with a 'candidates' array:
{{
  "candidates": [
    {{
      "description": "...",
      "rationale": "Explanation of strategy used..."
    }}
  ]
}}
"""


def filter_candidates(
    candidates: Sequence[OptimizationCandidate],
    skill_name: str,  # noqa: ARG001
    config: LintSettings | None = None,
) -> list[OptimizationCandidate]:
    """Validate candidates with static linter rules, marking non-compliant candidates."""
    lint_config = config or LintSettings()
    results: list[OptimizationCandidate] = []

    for candidate in candidates:
        desc_len = len(candidate.description)
        clean = not (
            desc_len < lint_config.min_description_length
            or desc_len > lint_config.max_description_length
        )

        results.append(
            OptimizationCandidate(
                description=candidate.description,
                rationale=candidate.rationale,
                lint_clean=clean,
                recall=candidate.recall,
                accuracy=candidate.accuracy,
                misroute_rate=candidate.misroute_rate,
                delta_recall=candidate.delta_recall,
            ),
        )
    return results


class _CandidatePayload(BaseModel):
    """Represent an individual description rewrite proposal from generator output."""

    model_config = ConfigDict(frozen=True)

    description: str
    rationale: str = ""


class _OptimizationResponse(BaseModel):
    """Represent the structured collection of candidate rewrites from generator output."""

    model_config = ConfigDict(frozen=True)

    candidates: tuple[_CandidatePayload, ...] = ()


def _synthesize_via_llm(
    driver: TextGenerator,
    target: Skill,
    rivals: Sequence[Skill],
    ceded_terms: Sequence[str],
    unclaimed_terms: Sequence[str],
    count: int,
    lint_config: LintSettings | None = None,
) -> list[OptimizationCandidate] | None:
    """Attempt candidate generation using active language model runtime."""
    prompt = build_optimization_prompt(
        target=target,
        rivals=rivals,
        ceded_terms=ceded_terms,
        unclaimed_terms=unclaimed_terms,
        count=count,
        config=lint_config,
    )
    try:
        raw_text = driver.complete(prompt)
        if not raw_text:
            return None
        text = raw_text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        response = _OptimizationResponse.model_validate_json(text.strip())
        if response.candidates:
            return [
                OptimizationCandidate(
                    description=item.description.strip(),
                    rationale=item.rationale.strip(),
                )
                for item in response.candidates
                if item.description.strip()
            ]
    except (ValueError, OSError):
        pass
    return None


def _synthesize_via_heuristics(
    target: Skill,
    rivals: Sequence[Skill],
    unclaimed_terms: Sequence[str] = (),
    count: int = 3,
    ceded_terms: Sequence[str] = (),
) -> list[OptimizationCandidate]:
    """Synthesize rewrite candidates using vocabulary heuristics and contrastive differentiation."""
    results: list[OptimizationCandidate] = []
    base_desc = target.description.strip().rstrip(".")

    # Strategy 1: Incorporate unclaimed distinctive terms cleanly
    if unclaimed_terms:
        added = ", ".join(unclaimed_terms[:3])
        results.append(
            OptimizationCandidate(
                description=f"{base_desc}, featuring {added}.",
                rationale=f"Incorporated distinctive unclaimed terms: {added}",
            ),
        )
    else:
        domain_name = target.name.replace("-", " ")
        results.append(
            OptimizationCandidate(
                description=f"{base_desc}. Handles dedicated workflows for {domain_name}.",
                rationale="Added explicit domain task scope clause.",
            ),
        )

    # Strategy 2: Contrastive differentiation against primary rival
    if rivals:
        primary_rival = rivals[0].name
        disc_desc = synthesize_directional_disclaimer(
            target.description,
            primary_rival,
            ceded_terms=ceded_terms,
        )
        results.append(
            OptimizationCandidate(
                description=disc_desc,
                rationale=f"Sharpened contrastive boundaries against rival {primary_rival}",
            ),
        )
    else:
        results.append(
            OptimizationCandidate(
                description=f"Use when the user needs to {base_desc[0].lower() + base_desc[1:]}.",
                rationale="Framed description as actionable user invocation trigger.",
            ),
        )

    # Strategy 3: Action-oriented verb focus
    results.append(
        OptimizationCandidate(
            description=(
                f"Execute specialized operations for {target.name.replace('-', ' ')}: "
                f"{base_desc[0].lower() + base_desc[1:]}."
            ),
            rationale="Recast description with action-oriented imperative prefix.",
        ),
    )

    # Additional variants if count > 3 requested
    while len(results) < count:
        idx = len(results) + 1
        cand_extra = (
            f"Specialized {target.name.replace('-', ' ')} utility (variant #{idx}): {base_desc}."
        )
        results.append(
            OptimizationCandidate(
                description=cand_extra,
                rationale=f"Synthesized variant #{idx}.",
            ),
        )

    return results[:count]


def synthesize_candidates(
    target: Skill,
    rivals: Sequence[Skill],
    ceded_terms: Sequence[str] = (),
    unclaimed_terms: Sequence[str] = (),
    count: int = 3,
    driver: TextGenerator | None = None,
    config: LintSettings | Path | None = None,
) -> list[OptimizationCandidate]:
    """Synthesize candidate descriptions using LLM generation or vocabulary heuristics."""
    lint_config = _resolve_lint_settings(config)
    if driver is not None and getattr(driver, "name", "") != FAKE_AGENT:
        llm_results = _synthesize_via_llm(
            driver,
            target,
            rivals,
            ceded_terms,
            unclaimed_terms,
            count,
            lint_config=lint_config,
        )
        if llm_results:
            if rivals:
                primary_rival = rivals[0].name
                disc_desc = synthesize_directional_disclaimer(
                    target.description,
                    primary_rival,
                    ceded_terms=ceded_terms,
                )
                zero_cost_cand = OptimizationCandidate(
                    description=disc_desc,
                    rationale=f"Sharpened contrastive boundaries against rival {primary_rival}",
                )
                return [zero_cost_cand, *llm_results]
            return llm_results

    return _synthesize_via_heuristics(
        target, rivals, unclaimed_terms, count=count, ceded_terms=ceded_terms
    )


def _setup_driver(
    agent: str | None = None,
    runtime_options: dict[str, Any] | None = None,
    config: Path | None = None,
) -> TextGenerator:
    """Initialize and configure the designated TextGenerator driver."""
    from reach.config import default_agent

    resolved_agent = agent or default_agent(config)
    return build_text_generator(agent=resolved_agent, options=dict(runtime_options or {}))


def _setup_runtime(
    agent: str | None = None,
    runtime_options: dict[str, Any] | None = None,
    config: Path | None = None,
) -> AgentRuntime:
    """Initialize and configure the designated AgentRuntime driver."""
    from reach.config import RuntimeSettings, default_agent

    resolved_agent = agent or default_agent(config)
    settings = RuntimeSettings(agent=resolved_agent, options=dict(runtime_options or {}))
    return build_runtime(settings)


def _run_candidate_probes(
    runtime: AgentRuntime,
    queries_to_run: Sequence[Query],
    target_name: str,
    workdir: Path,
) -> tuple[int, int, int, int]:
    """Execute empirical queries in isolated workspace and tally routing counts."""
    triggers = 0
    positive_queries = 0
    correct_count = 0
    misroutes = 0

    for query in queries_to_run:
        is_positive = query.expected_skill == target_name
        if is_positive:
            positive_queries += 1

        try:
            try:
                outcome = runtime.select(query.text, workdir, target_skill=query.expected_skill)
            except TypeError:
                outcome = runtime.select(query.text, workdir)
            invoked = outcome.invoked_skill
        except (OSError, RuntimeError, ValueError):
            invoked = None

        if is_positive:
            if invoked == target_name:
                triggers += 1
                correct_count += 1
            elif invoked is not None:
                misroutes += 1
        elif invoked == query.expected_skill:
            correct_count += 1
        elif invoked == target_name:
            misroutes += 1

    return triggers, positive_queries, correct_count, misroutes


def evaluate_candidate(
    candidate: OptimizationCandidate,
    target: Skill,
    rivals: Sequence[Skill],
    queries: Sequence[Query],
    agent: str | None = None,
    baseline_recall: float = 0.0,
    baseline_accuracy: float = 0.0,  # noqa: ARG001
    budget: int = 20,
    config: Path | None = None,
) -> OptimizationCandidate:
    """Empirically evaluate a candidate description against queries within a probe budget."""
    if not queries or budget < 1:
        return candidate

    queries_to_run = list(queries)[:budget]
    candidate_skill = Skill(
        name=target.name,
        description=candidate.description,
        path=target.path,
    )
    all_skills = [candidate_skill, *rivals]
    catalog = Catalog(
        id="opt-catalog",
        skills=tuple(s.name for s in all_skills),
        mode=CatalogMode.ALL,
    )

    runtime = _setup_runtime(agent, config=config)

    with tempfile.TemporaryDirectory() as temp_dir:
        workdir = Path(temp_dir)
        runtime.install(catalog, all_skills, workdir)
        triggers, positive_queries, correct_count, misroutes = _run_candidate_probes(
            runtime,
            queries_to_run,
            target.name,
            workdir,
        )

    probes_executed = len(queries_to_run)
    recall = (triggers / positive_queries) if positive_queries > 0 else 1.0
    accuracy = (correct_count / probes_executed) if probes_executed > 0 else 1.0
    misroute_rate = (misroutes / probes_executed) if probes_executed > 0 else 0.0
    delta_recall = round(recall - baseline_recall, 4)

    return OptimizationCandidate(
        description=candidate.description,
        rationale=candidate.rationale,
        lint_clean=candidate.lint_clean,
        recall=round(recall, 4),
        accuracy=round(accuracy, 4),
        misroute_rate=round(misroute_rate, 4),
        delta_recall=delta_recall,
    )


def _find_target_skill(all_skills: Sequence[Skill], skill_name: str, resolved_root: Path) -> Skill:
    """Locate the target skill in the catalog or raise ValueError with close matches."""
    target_skill = next((s for s in all_skills if s.name == skill_name), None)
    if target_skill is not None:
        return target_skill

    import difflib

    names = [s.name for s in all_skills]
    close = difflib.get_close_matches(skill_name, names, n=3)
    suggestion = f" Did you mean: {', '.join(close)}?" if close else ""
    msg = f"Skill '{skill_name}' not found in {resolved_root}.{suggestion}"
    raise ValueError(msg)


def _find_dense_semantic_rival(all_skills: Sequence[Skill], target_skill: Skill) -> str:
    """Find top semantic rival via dense embeddings if available."""
    try:
        from reach.retrieval import DenseScorer

        dense_scorer = DenseScorer.from_skills(all_skills)
        candidates_pool = [s for s in all_skills if s.name != target_skill.name]
        semantic_ranked = dense_scorer.rank(target_skill, candidates_pool)
        if semantic_ranked:
            return semantic_ranked[0][0]
    except (RuntimeError, ValueError, OSError):
        pass
    return ""


def _extract_lexical_rewrite_info(
    all_skills: Sequence[Skill],
    target_skill: Skill,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Extract rival name, ceded terms, and unclaimed terms from lexical overlap."""
    overlap = rank_corpus(all_skills)
    target_comp = overlap.find(target_skill.name)
    target_rewrite = suggest_rewrite(target_comp, all_skills) if target_comp else None
    if not target_rewrite:
        return "", (), ()
    rival_name = target_rewrite.rival or ""
    ceded_terms = tuple(t.term for t in target_rewrite.ceded)
    return rival_name, ceded_terms, target_rewrite.unclaimed


def _resolve_named_skills(all_skills: Sequence[Skill], names: Sequence[str]) -> list[Skill]:
    """Look up skill instances by name, preserving uniqueness and order."""
    skill_map = {s.name: s for s in all_skills}
    result: list[Skill] = []
    seen = set()
    for name in names:
        if name and name not in seen and name in skill_map:
            result.append(skill_map[name])
            seen.add(name)
    return result


def _identify_rivals(
    all_skills: Sequence[Skill],
    target_skill: Skill,
) -> tuple[str, tuple[str, ...], tuple[str, ...], list[Skill]]:
    """Analyze lexical rewrite opportunities and top dense semantic rivals."""
    rival_name, ceded_terms, unclaimed = _extract_lexical_rewrite_info(all_skills, target_skill)
    semantic_rival_name = _find_dense_semantic_rival(all_skills, target_skill)
    rival_skills = _resolve_named_skills(all_skills, [rival_name, semantic_rival_name])
    return rival_name, ceded_terms, unclaimed, rival_skills


def _load_optimization_queries(
    queries_path: Path | str | None,
    target_skill_name: str,
) -> list[Query]:
    """Load query set and filter for queries targeting the skill."""
    if queries_path is None:
        return []
    resolved_queries_path = resolve_path(queries_path)
    query_set = load_query_set(resolved_queries_path)
    queries = [q for q in query_set.queries if q.expected_skill == target_skill_name]
    return queries or list(query_set.queries)


def _evaluate_all_candidates(
    candidates: Sequence[OptimizationCandidate],
    target_skill: Skill,
    rivals: Sequence[Skill],
    queries: Sequence[Query],
    agent: str | None,
    baseline_recall: float,
    baseline_accuracy: float,
    budget: int,
    config: Path | None,
) -> list[OptimizationCandidate]:
    """Empirically evaluate all lint-clean candidates and rank them."""
    eval_budget_per_candidate = max(budget // (len(candidates) or 1), 5)
    evaluated_candidates: list[OptimizationCandidate] = []

    for cand in candidates:
        if cand.lint_clean and queries:
            evaluated = evaluate_candidate(
                candidate=cand,
                target=target_skill,
                rivals=rivals,
                queries=queries,
                agent=agent,
                baseline_recall=baseline_recall,
                baseline_accuracy=baseline_accuracy,
                budget=eval_budget_per_candidate,
                config=config,
            )
            evaluated_candidates.append(evaluated)
        else:
            evaluated_candidates.append(cand)

    evaluated_candidates.sort(key=lambda c: (-c.delta_recall, c.misroute_rate, -c.accuracy))
    return evaluated_candidates


def optimize_skill(
    skill_name: str,
    skills_path: Path | str | None = None,
    queries_path: Path | str | None = None,
    agent: str | None = None,
    candidates_count: int = 3,
    budget: int = DEFAULT_BUDGET,
    auto_apply: bool = False,
    runtime_options: dict[str, Any] | None = None,
    config: Path | None = None,
    global_scope: bool = False,
    settings: OptimizeSettings | None = None,
) -> OptimizationReport:
    """Orchestrate closed-loop skill description optimization and candidate evaluation.

    Args:
        skill_name: Target skill identifier to optimize.
        skills_path: Directory path containing the skill catalog.
        queries_path: Optional path to labeled queries JSON file.
        agent: Agent runtime identifier (e.g. "claude-code", "antigravity-cli").
        candidates_count: Number of description rewrite candidates to synthesize.
        budget: Maximum probe budget allowed across candidate evaluations.
        auto_apply: If True, automatically overwrite SKILL.md with the top candidate.
        runtime_options: Additional key-value configuration options passed to runtime.
        config: Optional path to custom reach.toml configuration file.
        global_scope: If True, discovers skills from user global configuration (~/).
        settings: Optional typed OptimizeSettings model.

    Returns:
        An OptimizationReport recording baseline scores, evaluated candidates, and rewrite diffs.
    """
    if settings is not None:
        budget = settings.budget

    if skills_path is not None:
        resolved_root = resolve_path(skills_path)
    else:
        from reach.config import resolve_discovery_candidates

        workdir = Path.home() if global_scope else Path.cwd().resolve()
        candidates = resolve_discovery_candidates(
            workdir,
            agent=agent,
            config_path=config,
            global_scope=global_scope,
        )
        resolved_root = candidates[0] if candidates else workdir

    all_skills = load_skills(resolved_root)
    target_skill = _find_target_skill(all_skills, skill_name, resolved_root)

    rival_name, ceded_terms, unclaimed, rival_skills = _identify_rivals(all_skills, target_skill)
    queries = _load_optimization_queries(queries_path, target_skill.name)

    baseline_recall = 0.0
    baseline_accuracy = 0.0
    baseline_misroute = 0.0
    if queries:
        baseline_cand = OptimizationCandidate(description=target_skill.description)
        eval_base = evaluate_candidate(
            candidate=baseline_cand,
            target=target_skill,
            rivals=rival_skills,
            queries=queries,
            agent=agent,
            budget=min(budget // 2, 10),
            config=config,
        )
        baseline_recall = eval_base.recall
        baseline_accuracy = eval_base.accuracy
        baseline_misroute = eval_base.misroute_rate

    lint_config = _resolve_lint_settings(config)
    driver = _setup_driver(agent, runtime_options, config=config)
    raw_candidates = synthesize_candidates(
        target=target_skill,
        rivals=rival_skills,
        ceded_terms=ceded_terms,
        unclaimed_terms=unclaimed,
        count=candidates_count,
        driver=driver,
        config=lint_config,
    )
    linted_candidates = filter_candidates(
        raw_candidates,
        skill_name=skill_name,
        config=lint_config,
    )

    evaluated_candidates = _evaluate_all_candidates(
        candidates=linted_candidates,
        target_skill=target_skill,
        rivals=rival_skills,
        queries=queries,
        agent=agent,
        baseline_recall=baseline_recall,
        baseline_accuracy=baseline_accuracy,
        budget=budget,
        config=config,
    )

    applied = False
    best = evaluated_candidates[0] if evaluated_candidates else None
    if auto_apply and best is not None:
        manifest_file = target_skill.path / "SKILL.md"
        applied = update_skill_description(manifest_file, best.description)

    return OptimizationReport(
        skill_name=skill_name,
        manifest_path=target_skill.path / "SKILL.md",
        baseline_description=target_skill.description,
        baseline_recall=baseline_recall,
        baseline_accuracy=baseline_accuracy,
        baseline_misroute=baseline_misroute,
        rival_name=rival_name,
        ceded_terms=ceded_terms,
        unclaimed_terms=unclaimed,
        candidates=tuple(evaluated_candidates),
        applied=applied,
        has_probes=bool(queries),
    )
