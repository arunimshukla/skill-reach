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

"""Execute catalog scaling sweeps across geometric scale steps and detect capacity knees."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter

from reach.config import (
    RegistrySettings,
    RunConfig,
    RuntimeSettings,
    StudySettings,
    resolve_sub_settings,
)
from reach.runtime import AgentRuntime, build_runtime
from reach.sweep import run_scaling_sweep
from reach.views import Console, build_console, print_sweep, print_wrote, render_sweep

from .app import LOOP, app
from .discovery import _corpus, _no_skills
from .flags import (
    POSITIVE_INT,
    RATE,
    REGISTRY_GROUP,
    AgentName,
    ConfigFlag,
    EarlyStopFlag,
    Format,
    Global,
    RegistryFlags,
    YesFlag,
    agent_help_text,
)
from .safety import confirm_skill_execution

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reach.models import Skill
    from reach.sweep import ScalingStudy


def _resolve_sweep_queries(
    queries: Path | None,
    configured: Path | None,
    skills_path: Path | None = None,
) -> Path:
    """Resolve queries file from explicit argument, configuration, or .reach fallback."""
    resolved = queries if queries is not None else configured
    if resolved is None:
        candidates: list[Path] = [
            Path(".reach/queries.json"),
            Path(".reach/queries.jsonl"),
            Path(".reach/queries.csv"),
        ]
        if skills_path is not None:
            sp = Path(skills_path).resolve()
            search_roots = [sp] if sp.is_dir() else [sp.parent]
            if sp.parent != sp and sp.parent not in search_roots:
                search_roots.append(sp.parent)
            for root in search_roots:
                candidates.append(root / ".reach" / "queries.json")
                candidates.append(root / ".reach" / "queries.jsonl")
                candidates.append(root / ".reach" / "queries.csv")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        msg = (
            "scaling sweep requires a labeled query benchmark to test reachability across "
            "catalog scales.\n\n"
            "• Pass an existing queries file:\n"
            "    reach sweep ./skills --queries .reach/queries.json\n\n"
            "• Or draft benchmark queries first:\n"
            "    reach query draft --skills ./skills --out .reach/queries.json"
        )
        raise ValueError(msg)
    return resolved


def _resolve_sweep_out_path(
    out: Path | None,
    configured: Path | None,
    queries_path: Path | None = None,
) -> Path:
    """Determine the file path where the sweep artifact should be written."""
    if out is not None:
        return out
    if configured is not None:
        return configured
    if queries_path is not None:
        qp = Path(queries_path)
        if qp.parent.name == ".reach" and qp.parent.is_dir():
            return qp.parent / "sweep.json"
    return Path(".reach/sweep.json")


def _output_sweep(
    console: Console,
    study: ScalingStudy,
    *,
    format: Format,
    out: Path,
) -> None:
    """Render and write scaling study results to disk."""
    out_format = format
    if format == "text":
        if out.suffix == ".json":
            out_format = "json"
        elif out.suffix == ".csv":
            out_format = "csv"

    out_content = (
        render_sweep(study, out_format)
        if out_format in ("json", "csv")
        else render_sweep(study, "json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(out_content, encoding="utf-8")

    if format == "text":
        print_sweep(console, study)
        print_wrote(console, out)
    else:
        rendered = render_sweep(study, format)
        print(rendered)


@app.command(name="sweep", group=LOOP)
def _sweep(
    skills: Annotated[
        Path | None,
        Parameter(
            name=["skills", "--skills"],
            help="Path to the skill directory or corpus to sweep (discovered if omitted)",
        ),
    ] = None,
    *,
    target: Annotated[
        str | None,
        Parameter(
            name=["target", "--target"],
            help=(
                "Target skill to evaluate across scaling steps "
                "(omitted for whole-corpus capacity evaluation)"
            ),
        ),
    ] = None,
    queries: Annotated[
        Path | None,
        Parameter(
            name="--queries",
            help="Path to labeled evaluation queries JSON file",
        ),
    ] = None,
    scales: Annotated[
        str | None,
        Parameter(
            name="--scales",
            help="Comma-separated list of catalog sizes to evaluate",
        ),
    ] = None,
    anchor: Annotated[
        str | None,
        Parameter(
            name="--anchor",
            help=(
                "Anchor skills cohort evaluated across all scales. "
                "Defaults to cluster medoids of the initial scale step. "
                "Accepts integer count (e.g. 10), comma-separated skill names, "
                "or 'all' for full-corpus expansion."
            ),
        ),
    ] = None,
    rivals_share: Annotated[
        float,
        RATE,
        Parameter(
            name="--rivals-share",
            help="Proportion of distractor skills selected as nearest rivals",
        ),
    ] = 0.5,
    concurrency: Annotated[
        int,
        POSITIVE_INT,
        Parameter(
            name=["--concurrency", "-j"],
            help="Number of concurrent probe execution workers",
        ),
    ] = 1,
    attempts: Annotated[
        int | None,
        POSITIVE_INT,
        Parameter(
            name=["--attempts", "-a"],
            help="Number of probe execution attempts per query at each scale step",
        ),
    ] = None,
    early_stop: EarlyStopFlag = True,
    noise_floor: Annotated[
        float,
        RATE,
        Parameter(
            name="--noise-floor",
            help="Minimum pass rate drop to trigger knee detection",
        ),
    ] = 0.05,
    agent: Annotated[
        AgentName | None,
        Parameter(
            name="--agent",
            show_choices=False,
            help=agent_help_text("Agent runtime to execute scaling probes"),
        ),
    ] = None,
    model: Annotated[
        str | None,
        Parameter(
            name=["--model", "-m"],
            help="Target model identifier",
        ),
    ] = None,
    global_: Global = False,
    registry: Annotated[
        RegistryFlags | None,
        Parameter(group=REGISTRY_GROUP),
    ] = None,
    format: Annotated[Format, Parameter(help="Output format: text, json, csv")] = "text",
    out: Annotated[
        Path | None,
        Parameter(
            name=["--out", "-o"],
            help="Where to write sweep output (default: .reach/sweep.json)",
        ),
    ] = None,
    workdir: Annotated[
        Path | None,
        Parameter(
            name="--workdir",
            help="Working directory for probe execution",
        ),
    ] = None,
    yes: YesFlag = False,
    config: ConfigFlag = None,
) -> int:
    """Execute multi-scale catalog evaluation sweeps to measure reachability decay."""
    console = build_console()
    run_config: RunConfig | None = RunConfig.from_toml(config) if config is not None else None

    effective_config, driver = _resolve_sweep_effective_config(
        run_config=run_config,
        agent=agent,
        model=model,
        registry=registry,
        skills=skills,
        queries=queries,
        workdir=workdir,
        out=out,
    )

    found = _load_sweep_corpus(
        console=console,
        driver=driver,
        effective_config=effective_config,
        skills=skills,
        run_config=run_config,
        global_scope=global_,
    )

    try:
        parsed_scales = _parse_scales_cli(scales, effective_config.study.scales)
    except ValueError as err:
        console.print(f"[red]Error:[/] {err}")
        return 2

    effective_config, resolved_queries = _finalize_sweep_study_config(
        effective_config, found, skills, queries, workdir, early_stop
    )
    if format == "text":
        console.print(f"[dim]Using benchmark queries from:[/] [cyan]{resolved_queries}[/]\n")

    if code := confirm_skill_execution(
        console,
        runtime_name=driver.name,
        skills=found,
        action="scaling sweep",
        yes=yes,
        trusted=effective_config.study.trusted,
    ):
        return code

    try:
        study = run_scaling_sweep(
            config=effective_config,
            target_skill=target,
            scales=parsed_scales,
            anchor=anchor,
            runtime=driver,
            rivals_share=rivals_share,
            noise_floor=noise_floor,
            workers=concurrency,
            attempts=attempts,
            early_stop=early_stop,
        )
    except ValueError as err:
        console.print(f"[red]Error:[/] {err}")
        return 2
    except (OSError, RuntimeError) as err:
        console.print(f"[red]Runtime Error:[/] {err}")
        return 3

    destination = _resolve_sweep_out_path(
        out,
        configured=effective_config.study.out,
        queries_path=resolved_queries,
    )
    _output_sweep(console, study, format=format, out=destination)
    return 0


def _parse_scales_cli(
    scales: str | None,
    configured_scales: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    """Parse comma-separated scale integers, falling back to configured scales."""
    if not scales:
        return configured_scales
    try:
        return tuple(int(s.strip()) for s in scales.split(",") if s.strip())
    except ValueError as err:
        msg = "Invalid scales format. Use comma-separated integers, e.g. 10,25,50,100"
        raise ValueError(msg) from err


def _resolve_sweep_effective_config(
    run_config: RunConfig | None,
    agent: str | None,
    model: str | None,
    registry: RegistryFlags | None,
    skills: Path | None,
    queries: Path | None,
    workdir: Path | None,
    out: Path | None,
) -> tuple[RunConfig, AgentRuntime]:
    """Resolve layered runtime, registry, and study settings across CLI flags and configs."""
    eff_runtime = RunConfig.resolve(
        RuntimeSettings,
        run_config,
        agent=agent,
        model=model,
    )
    if not eff_runtime.agent:
        eff_runtime = eff_runtime.model_copy(update={"agent": "keyword"})

    eff_registry = resolve_sub_settings(
        RegistrySettings,
        run_config.registry if run_config is not None else None,
        **(registry.overrides() if registry is not None else {}),
    )

    eff_study = RunConfig.resolve(
        StudySettings,
        run_config,
        skills=skills,
        queries=queries,
        workdir=workdir,
        out=out,
    )

    effective_config = (run_config or RunConfig()).model_copy(
        update={
            "runtime": eff_runtime,
            "registry": eff_registry,
            "study": eff_study,
        }
    )
    driver = build_runtime(eff_runtime)
    return effective_config, driver


def _load_sweep_corpus(
    console: Console,
    driver: AgentRuntime,
    effective_config: RunConfig,
    skills: Path | None,
    run_config: RunConfig | None,
    global_scope: bool,
) -> tuple[Skill, ...]:
    """Load or discover candidate skills for sweep execution."""
    if skills is not None or run_config is None:
        eff_runtime = effective_config.runtime
        found, _roots, _discovered = _corpus(
            console,
            driver,
            settings=effective_config,
            global_scope=global_scope,
            agent=eff_runtime.agent,
        )
    else:
        from reach.catalog import load_skills

        found = load_skills(effective_config.require_skills())

    if not found:
        raise _no_skills(skills, global_scope=global_scope)
    return tuple(found)


def _finalize_sweep_study_config(
    effective_config: RunConfig,
    found: Sequence[Skill],
    skills: Path | None,
    queries: Path | None,
    workdir: Path | None,
    early_stop: bool,
) -> tuple[RunConfig, Path]:
    """Determine working directory and resolved benchmark queries path."""
    import tempfile

    work_dir = (
        workdir or effective_config.study.workdir or Path(tempfile.mkdtemp(prefix="reach_sweep_"))
    )
    resolved_skills = (
        found[0].path.parent
        if effective_config.study.skills is None and found
        else effective_config.study.skills
    )
    resolved_queries = _resolve_sweep_queries(
        queries,
        effective_config.study.queries,
        skills_path=resolved_skills or skills,
    )
    updated_study = effective_config.study.model_copy(
        update={
            "workdir": work_dir,
            "skills": resolved_skills,
            "queries": resolved_queries,
            "early_stop": early_stop,
        }
    )
    return effective_config.model_copy(update={"study": updated_study}), resolved_queries
