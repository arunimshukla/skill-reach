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

"""Execute skill reachability evaluations, drafting query sets and probing catalogs."""

from __future__ import annotations

import shutil
import tempfile
from difflib import get_close_matches
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict

from reach.artifact import ContestedSkill, artifact_path, write_artifact
from reach.catalog import parse_frontmatter
from reach.config import RunConfig
from reach.generate import citations_path
from reach.models import CatalogMode, Query, Skill
from reach.queries import Origin, QuerySet, QuerySetProvenance, save_query_set
from reach.report import render
from reach.run import (
    Composition,
    Plan,
    compose,
    conduct,
    validate_appendable,
    validate_catalog_fit,
    write_sidecar,
)
from reach.runtime import AgentRuntime, build_runtime
from reach.views import (
    Console,
    build_console,
    print_plan,
    print_quick_scope,
    print_scorecard,
    print_wrote,
    probe_progress,
)

from .app import LOOP, app
from .discovery import _asks_for_a_mode, _corpus, _run_dir_defaults
from .drafting import DRAFTED_THEN, _draft_query_set, _rivals_in_view, _sole_catalog
from .flags import (
    CATALOG_GROUP,
    GENERATE_GROUP,
    LIST,
    PLAN_GROUP,
    RECORD_GROUP,
    REGISTRY_GROUP,
    RUNTIME_GROUP,
    STUDY_GROUP,
    SWITCH,
    CatalogFlags,
    Format,
    GenerateFlags,
    Global,
    PlanFlags,
    Quiet,
    RecordFlags,
    RegistryFlags,
    RuntimeFlags,
    StudyFlags,
    Verbose,
    YesFlag,
    build_config,
)
from .safety import confirm_skill_execution

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reach.discovery import Discovery

#: Required configuration fields when running eval without a pre-built config file.
EVAL_REQUIRED = ("queries",)

#: Provenance notes recorded for manually typed CLI queries.
TYPED_NOTES = (
    "Typed at the prompt as `reach eval --query`. Authored ground truth: the "
    "user specified both the query and the expected target skill."
)

#: Maximum number of skill names to preview in cli diagnostic hints.
PREVIEW_SKILL_COUNT: Final = 3

#: Default probe attempts per query in quick evaluation mode.
QUICK_ATTEMPTS = 3


class QuickEval(BaseModel):
    """Encapsulate resolved parameters for a quick evaluation run."""

    model_config = ConfigDict(frozen=True)

    target: str
    texts: tuple[str, ...] = ()
    corpus: Path | None = None


def _validate_quick_flags(
    target: str | None,
    query: Sequence[str],
    expected: str | None,
    config: Path | None,
    study: StudyFlags,
) -> bool:
    """Validate quick evaluation CLI arguments, returning True if quick mode is active."""
    if target is None and not query:
        if expected is not None:
            msg = (
                "--expected labels a --query; pass one, or name the skill "
                "positionally to synthesize benchmark queries"
            )
            raise ValueError(msg)
        return False
    if config is not None:
        msg = (
            "--config cannot be combined with quick evaluation targets or --query; "
            "use either full configuration via --config or quick evaluation mode"
        )
        raise ValueError(msg)
    if study.queries is not None:
        msg = (
            "--queries cannot be combined with quick evaluation mode; "
            "omit --queries for quick evaluation, or omit positional "
            "skill/--query for formal evaluation"
        )
        raise ValueError(msg)
    if expected is not None and not query:
        msg = "--expected labels a --query; pass one, or drop --expected"
        raise ValueError(msg)
    return True


def _resolve_manifest_target(
    target: str | None,
    study: StudyFlags,
) -> tuple[str | None, Path | None]:
    """Parse skill name from manifest file if target path contains SKILL.md."""
    if target is None:
        return None, None
    named = Path(target).expanduser()
    manifest = named / "SKILL.md"
    if manifest.is_file():
        skill = parse_frontmatter(manifest.read_text(encoding="utf-8"), manifest)
        if skill is None:
            msg = f"{manifest} has no frontmatter to take a name from"
            raise ValueError(msg)
        corpus = named.parent if study.skills is None else None
        return skill.name, corpus
    if named.is_dir():
        contained_skills = sorted(
            d.name for d in named.iterdir() if d.is_dir() and (d / "SKILL.md").is_file()
        )
        if contained_skills:
            preview = ", ".join(f"'{s}'" for s in contained_skills[:PREVIEW_SKILL_COUNT])
            remainder = len(contained_skills) - PREVIEW_SKILL_COUNT
            more = f" (and {remainder} more)" if len(contained_skills) > PREVIEW_SKILL_COUNT else ""
            msg = (
                f"'{target}' is a directory containing {len(contained_skills)} skills "
                f"({preview}{more}), not a single skill.\n\n"
                "• To evaluate a single skill immediately (auto-drafts queries):\n"
                f"    reach eval {target.rstrip('/')}/{contained_skills[0]}\n\n"
                "• To evaluate the full catalog against a benchmark:\n"
                f"    1. Draft queries:  reach query --skills {target} --out .reach/queries.json\n"
                f"    2. Run evaluation: reach eval --skills {target}"
            )
            raise ValueError(msg)
    return target, None


def _quick(
    target: str | None,
    query: Sequence[str],
    expected: str | None,
    config: Path | None,
    study: StudyFlags,
) -> QuickEval | None:
    """Resolve target skill, queries, and corpus path for quick eval runs."""
    if not _validate_quick_flags(target, query, expected, config, study):
        return None

    resolved_target, corpus = _resolve_manifest_target(target, study)
    label = expected or resolved_target
    if label is None:
        msg = "--query needs ground truth: pass --expected, or name the skill positionally"
        raise ValueError(msg)
    return QuickEval(target=label, texts=tuple(query), corpus=corpus)


def _quick_defaults(
    quick: QuickEval,
    scratch: Path,
    *,
    study: StudyFlags,
    plan: PlanFlags,
) -> tuple[StudyFlags, PlanFlags]:
    """Populate default study and plan parameters for temporary quick eval runs."""
    return (
        study.model_copy(
            update={
                "skills": study.skills if study.skills is not None else quick.corpus,
                "queries": scratch / "queries.json",
                "workdir": study.workdir if study.workdir is not None else scratch,
                "partial": True if study.partial is None else study.partial,
            },
        ),
        plan.model_copy(
            update={
                "attempts": QUICK_ATTEMPTS if plan.attempts is None else plan.attempts,
            },
        ),
    )


def _assert_present(target: str, skills: Sequence[Skill]) -> None:
    """Verify that the target skill exists in the resolved skill corpus."""
    names = [s.name for s in skills]
    if target in names:
        return
    near = get_close_matches(target, names, n=3)
    hint = f"; did you mean {', '.join(near)}?" if near else ""
    msg = (
        f"no skill named {target!r} among the {len(names)} found{hint}. `reach overlap` lists them"
    )
    raise ValueError(
        msg,
    )


def _typed_query_set(quick: QuickEval, catalog_id: str) -> QuerySet:
    """Convert typed CLI query strings into a structured QuerySet object."""
    return QuerySet(
        catalog_id=catalog_id,
        notes=TYPED_NOTES,
        provenance=QuerySetProvenance(
            origin=Origin.AUTHORED,
            tool_version=metadata.version("skill-reach"),
        ),
        queries=tuple(
            Query(id=f"typed-{n}", text=text, expected_skill=quick.target)
            for n, text in enumerate(quick.texts, start=1)
        ),
    )


def _validate_quick_save(save: Path | None, quick: QuickEval | None) -> None:
    """Validate that --save is only used in quick evaluation mode."""
    if save is not None and quick is None:
        msg = (
            "--save promotes a quick run's draft out of its scratch directory; "
            "name a skill or pass --query to select quick mode, or drop --save"
        )
        raise ValueError(
            msg,
        )


def _adjust_eval_catalog_mode(
    settings: RunConfig,
    config: Path | None,
    catalog: CatalogFlags | None,
    quick: QuickEval | None,
) -> RunConfig:
    """Apply catalog mode defaults and neighborhood catalog ID for quick evaluation."""
    if quick is None and not _asks_for_a_mode(config, catalog):
        return settings.with_overrides(catalog={"mode": CatalogMode.ALL})
    if (
        quick is not None
        and (settings.study.catalog is None or settings.study.catalog in ("auto", ""))
        and settings.catalog.mode is CatalogMode.NEIGHBORHOOD
    ):
        return settings.with_overrides(
            study={"catalog": f"neighborhood:{quick.target}"},
        )
    return settings


def _apply_execution_mode_defaults(
    *,
    quick: QuickEval | None,
    auto: bool,
    scratch: Path | None,
    run_dir: Path | None,
    config: Path | None,
    study: StudyFlags,
    plan: PlanFlags,
    record: RecordFlags,
    generate: GenerateFlags,
    dry_run: bool,
) -> tuple[StudyFlags, PlanFlags, RecordFlags, GenerateFlags]:
    """Apply execution mode defaults (quick, auto, run_dir, dry_run) to flags."""
    if quick is not None and scratch is not None:
        study, plan = _quick_defaults(quick, scratch, study=study, plan=plan)
        if not generate.targets:
            generate = generate.model_copy(update={"targets": (quick.target,)})
    elif auto and scratch is not None:
        default_queries = Path(".reach/queries.json")
        if study.queries is None:
            if default_queries.is_file():
                study = study.model_copy(update={"queries": default_queries})
            else:
                study = study.model_copy(update={"queries": scratch / "queries.json"})
        if study.workdir is None:
            study = study.model_copy(update={"workdir": scratch / "workspace"})
    elif run_dir is not None:
        study, record = _run_dir_defaults(run_dir, study=study, record=record)
    elif (
        dry_run
        and study.workdir is None
        and (config is None or not RunConfig.declared(config, "study", "workdir"))
    ):
        study = study.model_copy(update={"workdir": Path("work")})
    elif study.workdir is None and scratch is not None:
        study = study.model_copy(update={"workdir": scratch / "workspace"})
    return study, plan, record, generate


def _validate_queries_available(
    study: StudyFlags,
    *,
    quick: QuickEval | None,
    auto: bool,
    config: Path | None,
) -> StudyFlags:
    """Ensure benchmark queries exist or raise a user-friendly instructional error."""
    if quick is not None or auto or config is not None or study.queries is not None:
        return study
    default_queries = Path(".reach/queries.json")
    if default_queries.is_file():
        return study.model_copy(update={"queries": default_queries})

    corpus_hint = f" --skills {study.skills}" if study.skills is not None else ""
    msg = (
        "catalog evaluation requires a labeled query benchmark to test routing.\n\n"
        "• To evaluate your full catalog automatically (auto-drafts queries):\n"
        f"    reach eval{corpus_hint} --auto\n\n"
        "• To evaluate a single skill immediately:\n"
        "    reach eval <skill-name>\n\n"
        "• To curate an authored benchmark first:\n"
        f"    1. Draft queries:  reach query draft{corpus_hint} --out .reach/queries.json\n"
        f"    2. Run evaluation: reach eval{corpus_hint} --queries .reach/queries.json"
    )
    raise ValueError(msg)


def _resolve_eval_settings(
    *,
    quick: QuickEval | None,
    auto: bool = False,
    scratch: Path | None,
    run_dir: Path | None,
    config: Path | None,
    catalog: CatalogFlags | None,
    runtime: RuntimeFlags | None,
    plan: PlanFlags,
    study: StudyFlags,
    record: RecordFlags,
    generate: GenerateFlags,
    registry: RegistryFlags | None = None,
    dry_run: bool = False,
) -> tuple[RunConfig, GenerateFlags]:
    """Build and refine the run configuration and generation flags for evaluation."""
    study, plan, record, generate = _apply_execution_mode_defaults(
        quick=quick,
        auto=auto,
        scratch=scratch,
        run_dir=run_dir,
        config=config,
        study=study,
        plan=plan,
        record=record,
        generate=generate,
        dry_run=dry_run,
    )
    study = _validate_queries_available(study, quick=quick, auto=auto, config=config)

    settings = build_config(
        config,
        catalog=catalog,
        runtime=runtime,
        plan=plan,
        study=study,
        record=record,
        registry=registry,
        required=() if (quick is not None or auto) else EVAL_REQUIRED,
    )
    adjusted = _adjust_eval_catalog_mode(settings, config, catalog, quick)
    return adjusted, generate


def _setup_quick_scope(
    console: Console,
    settings: RunConfig,
    skills: Sequence[Skill],
    quick: QuickEval,
    generate: GenerateFlags,
) -> None:
    """Validate quick mode targets, print scope details, and persist typed queries."""
    _assert_present(quick.target, skills)
    resident = _sole_catalog(settings, skills)
    print_quick_scope(
        console,
        catalog_id=resident.id,
        residents=len(resident.skills),
        corpus=len(skills),
        attempts=settings.plan.attempts,
        rivals=None if quick.texts else _rivals_in_view(resident, generate),
        authored=len(quick.texts),
    )
    if quick.texts:
        save_query_set(
            _typed_query_set(quick, resident.id),
            settings.require_queries(),
        )


def _draft_then_message(*, quick: QuickEval | None, auto: bool) -> str:
    """Format the explanatory message displayed when query drafting completes."""
    if quick is not None:
        return (
            "probing it now; `reach query draft --queries <path>` is how "
            "to keep a set and review it first"
        )
    if auto:
        return "probing it now because --auto was passed; nothing has reviewed this set"
    return DRAFTED_THEN


def _draft_missing_queries(
    console: Console,
    settings: RunConfig,
    skills: Sequence[Skill],
    generate: GenerateFlags,
    *,
    quick: QuickEval | None,
    auto: bool,
    dry_run: bool,
) -> int | None:
    """Draft query set if not found on disk, returning an exit code if stopping early."""
    if settings.study.queries is not None and settings.study.queries.exists():
        return None

    same_invocation_probe = quick is not None or auto
    drafted = _draft_query_set(
        console,
        settings,
        skills,
        generate,
        dry_run=dry_run,
        then=_draft_then_message(
            quick=quick,
            auto=auto,
        ),
        keep=quick is None and not auto,
        same_invocation_probe=same_invocation_probe,
    )
    if drafted != 0 or not same_invocation_probe or dry_run:
        return drafted
    return None


def _probe_and_record(
    *,
    console: Console,
    settings: RunConfig,
    skills: Sequence[Skill],
    driver: AgentRuntime,
    roots: Sequence[Path],
    discovered: Discovery | None,
    quick: QuickEval | None,
    scratch: Path | None,
    save: Path | None,
    out: Path | None,
    format: Format,
    dry_run: bool,
    no_resume: bool,
    append_across_arms: bool,
    allow_truncation: bool,
    concurrency: int,
    verbose: bool,
    reasoning: bool = False,
) -> int:
    """Execute probe suite, render report, and promote quick draft if requested."""
    bank = quick is None or settings.study.out is not None or save is not None
    contested = discovered.contested if discovered is not None else ()
    outcome = _probe_query_set(
        console,
        settings,
        skills,
        driver,
        roots,
        contested=contested,
        out=out,
        format=format,
        dry_run=dry_run,
        resume=not no_resume,
        append_across_arms=append_across_arms,
        allow_truncation=allow_truncation,
        bank=bank,
        verbose=verbose,
        reasoning=reasoning,
        workers=concurrency,
    )
    if (
        quick is not None
        and scratch is not None
        and save is not None
        and outcome == 0
        and not dry_run
    ):
        _promote_quick_draft(
            scratch=scratch,
            settings=settings,
            artifact_destination=_artifact_destination(
                settings,
                out,
                bank=bank,
            ),
            save=save,
        )
    return outcome


@app.command(
    name="eval",
    group=LOOP,
    help="Measure whether a catalog's skills are reachable, writing a set if none exists.",
)
def _eval(
    target: Annotated[
        str | None,
        Parameter(
            help="One skill to evaluate, by name or by directory. Naming one "
            "asks for a quick run: questions drafted, probed and summarized",
        ),
    ] = None,
    *,
    query: Annotated[
        tuple[str, ...],
        LIST,
        Parameter(
            help="Probe this exact question rather than drafting one; repeatable",
        ),
    ] = (),
    expected: Annotated[
        str | None,
        Parameter(
            help="The skill every --query should reach; defaults to the skill named",
        ),
    ] = None,
    save: Annotated[
        Path | None,
        Parameter(
            help="Save a quick run's query set, citations, and artifact to DIR "
            "before scratch directory cleanup (requires quick evaluation mode)",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        Parameter(help="TOML run configuration; flags override it"),
    ] = None,
    catalog: Annotated[CatalogFlags | None, Parameter(group=CATALOG_GROUP)] = None,
    runtime: Annotated[RuntimeFlags | None, Parameter(group=RUNTIME_GROUP)] = None,
    plan: Annotated[PlanFlags | None, Parameter(group=PLAN_GROUP)] = None,
    study: Annotated[StudyFlags | None, Parameter(group=STUDY_GROUP)] = None,
    record: Annotated[RecordFlags | None, Parameter(group=RECORD_GROUP)] = None,
    registry: Annotated[RegistryFlags | None, Parameter(group=REGISTRY_GROUP)] = None,
    run_dir: Annotated[
        Path | None,
        Parameter(
            group=STUDY_GROUP,
            help="Directory to contain default --queries, --workdir, and --records "
            "paths (explicit flags override this)",
        ),
    ] = None,
    generate: Annotated[GenerateFlags | None, Parameter(group=GENERATE_GROUP)] = None,
    out: Annotated[
        Path | None,
        Parameter(
            name=["--out", "-o"],
            group=RECORD_GROUP,
            help="Where to write the evaluation artifact (default: .reach/eval.json)",
        ),
    ] = None,
    format: Annotated[
        Format,
        Parameter(help="How to render the finished evaluation"),
    ] = "text",
    dry_run: Annotated[
        bool,
        SWITCH,
        Parameter(
            help="Simulate drafting and probing without executing model calls or saving results",
        ),
    ] = False,
    no_resume: Annotated[
        bool,
        SWITCH,
        Parameter(help="Re-probe everything, ignoring results already in --out"),
    ] = False,
    append_across_arms: Annotated[
        bool,
        SWITCH,
        Parameter(
            help="Add probes to an --out recorded under a different configuration",
        ),
    ] = False,
    allow_truncation: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--allow-truncation",
            help="Probe a catalog too wide for the runtime's skill listing, "
            "measuring it with the descriptions it will really show",
        ),
    ] = False,
    concurrency: Annotated[
        int,
        Parameter(
            name=["--concurrency", "-j"],
            help="Number of concurrent probes to run (defaults to 1 for sequential execution)",
        ),
    ] = 1,
    auto: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--auto",
            help="Automatically draft queries for all skills and probe the catalog in one step",
        ),
    ] = False,
    global_: Global = False,
    reasoning: Annotated[
        bool,
        SWITCH,
        Parameter(
            name="--reasoning",
            help="Display model reasoning / thought traces for misrouted queries",
        ),
    ] = False,
    yes: YesFlag = False,
    quiet: Quiet = False,
    verbose: Verbose = False,
) -> int:
    """Measure whether a catalog's skills are reachable."""
    study = study or StudyFlags()
    plan = plan or PlanFlags()
    generate = generate or GenerateFlags()
    record = record or RecordFlags()
    registry = registry or RegistryFlags()
    artifact_out: Path | None = None
    if out is not None:
        if out.suffix == ".jsonl":
            if record.records is None:
                record = record.model_copy(update={"records": out})
        else:
            artifact_out = out
    quick = _quick(target, query, expected, config, study)
    _validate_quick_save(save, quick)
    has_declared_workdir = study.workdir is not None or (
        config is not None and RunConfig.declared(config, "study", "workdir")
    )
    needs_scratch = (
        quick is not None or auto or (not has_declared_workdir and run_dir is None and not dry_run)
    )
    scratch = Path(tempfile.mkdtemp(prefix="reach-")) if needs_scratch else None
    try:
        settings, generate = _resolve_eval_settings(
            quick=quick,
            auto=auto,
            scratch=scratch,
            run_dir=run_dir,
            config=config,
            catalog=catalog,
            runtime=runtime,
            plan=plan,
            study=study,
            record=record,
            generate=generate,
            registry=registry,
            dry_run=dry_run,
        )

        console = build_console(quiet=quiet)
        driver = build_runtime(settings.runtime)
        skills, roots, discovered = _corpus(console, driver, settings, global_scope=global_)

        if quick is not None:
            _setup_quick_scope(console, settings, skills, quick, generate)

        draft_outcome = _draft_missing_queries(
            console,
            settings,
            skills,
            generate,
            quick=quick,
            auto=auto,
            dry_run=dry_run,
        )
        if draft_outcome is not None:
            return draft_outcome

        if code := confirm_skill_execution(
            console,
            runtime_name=driver.name,
            skills=skills,
            roots=roots,
            action="evaluation",
            yes=yes,
            trusted=settings.study.trusted,
            dry_run=dry_run,
        ):
            return code

        return _probe_and_record(
            console=console,
            settings=settings,
            skills=skills,
            driver=driver,
            roots=roots,
            discovered=discovered,
            quick=quick,
            scratch=scratch,
            save=save,
            out=artifact_out,
            format=format,
            dry_run=dry_run,
            no_resume=no_resume,
            append_across_arms=append_across_arms,
            allow_truncation=allow_truncation,
            concurrency=concurrency,
            verbose=verbose,
            reasoning=reasoning,
        )
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def _announce_plan(
    console: Console,
    composed: Composition,
    driver: AgentRuntime,
    *,
    verbose: bool = False,
) -> Plan:
    """Print the plan summary to the console and return the planned object."""
    planned = composed.plan
    settings = composed.config
    print_plan(
        console,
        planned,
        agent=settings.runtime.agent,
        model=driver.model,
        fingerprint=settings.fingerprint,
        max_turns=driver.max_turns,
        early_exit=driver.early_exit,
        verbose=verbose,
    )
    return planned


def _open_record(
    settings: RunConfig,
    composed: Composition,
    *,
    guard: bool = True,
) -> None:
    """Verify append compatibility and write the run sidecar metadata file."""
    if settings.study.out is None:
        return
    out_path = settings.study.out.resolve()
    if guard:
        validate_appendable(
            out_path,
            settings.fingerprint,
            settings.condition,
            composed.provenance.queries_digest,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_sidecar(settings, out_path)


def _probe_query_set(
    console: Console,
    settings: RunConfig,
    skills: Sequence[Skill],
    driver: AgentRuntime,
    roots: Sequence[Path],
    *,
    contested: Sequence[ContestedSkill],
    out: Path | None,
    format: str,
    dry_run: bool,
    resume: bool = True,
    append_across_arms: bool = False,
    allow_truncation: bool = False,
    bank: bool = True,
    verbose: bool = False,
    reasoning: bool = False,
    workers: int = 1,
) -> int:
    """Execute evaluation probes across target queries and format output artifacts."""
    composed = compose(settings, skills)
    validate_catalog_fit(driver, composed.catalog, skills, allow_truncation=allow_truncation)
    planned = _announce_plan(console, composed, driver, verbose=verbose)
    if dry_run:
        return 0

    _open_record(settings, composed, guard=not append_across_arms)
    with probe_progress(
        console,
        catalog_id=planned.catalog_id,
        total=planned.probes,
        truth=composed.truth,
    ) as progress:
        outcome = conduct(
            settings,
            driver,
            progress=progress,
            composed=composed,
            resume=resume,
            append_across_arms=append_across_arms,
            allow_truncation=allow_truncation,
            workers=workers,
        )

    measured = outcome.artifact(roots=roots, contested=contested)
    destination = _artifact_destination(settings, out, bank=bank)
    written = write_artifact(measured, destination) if destination is not None else None
    if format in ("json", "jsonl", "csv"):
        print(render(measured, format), end="" if format == "jsonl" else "\n")
        return 0

    print_scorecard(console, measured, verbose=verbose or reasoning)
    if written is not None:
        print_wrote(console, written)
    return 0


def _artifact_destination(
    settings: RunConfig,
    out: Path | None,
    *,
    bank: bool,
) -> Path | None:
    """Determine the file path where the evaluation artifact should be written."""
    if out is not None:
        return out
    if settings.study.out is not None:
        return artifact_path(settings.study.out)
    if settings.study.queries is not None and bank:
        return artifact_path(settings.study.queries)
    return Path(".reach/eval.json")


def _promote_quick_draft(
    *,
    scratch: Path,
    settings: RunConfig,
    artifact_destination: Path | None,
    save: Path,
) -> None:
    """Copy quick evaluation query sets, citations, and artifacts to save path."""
    save.mkdir(parents=True, exist_ok=True)
    queries_path = settings.study.queries
    if queries_path is not None and queries_path.exists():
        shutil.copy2(queries_path, save / queries_path.name)
        trail = citations_path(queries_path)
        if trail.exists():
            shutil.copy2(trail, save / trail.name)
    if artifact_destination is None:
        return
    if artifact_destination.is_relative_to(scratch):
        shutil.copy2(artifact_destination, save / artifact_destination.name)
