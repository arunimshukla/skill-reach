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

"""Provide static pre-flight linting for skill definitions and catalogs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

import yaml
from pydantic import BaseModel, ConfigDict

from reach.catalog import _skill_files, find_skill_manifest, parse_frontmatter, split_frontmatter
from reach.config import LintSettings, resolve_path
from reach.runtime.claude_code import DEFAULT_DENIED_TOOLS, SKILL_TOOL_NAME

if TYPE_CHECKING:
    from reach.models import Skill

__all__ = [
    "RULES",
    "LintIssue",
    "LintReport",
    "LintSettings",
    "RuleDefinition",
    "Severity",
    "explain_rule",
    "lint_file",
    "lint_skills",
    "lint_tree",
]

#: Minimum skills required for pairwise semantic similarity comparisons.
MIN_PAIRWISE_SKILLS: Final = 2


#: Pattern validating kebab-case naming syntax: lowercase alphanumeric with single hyphens.
_KEBAB_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

#: Pattern matching unresolved template placeholders.
_PLACEHOLDER = re.compile(r"\b(?:TODO|FIXME|XXX)\b|<FILL_IN>|<TODO>|\[TODO\]", re.IGNORECASE)

#: Pattern matching overly broad, unbounded attractor claims that hijack queries.
_UNBOUNDED_ATTRACTOR = re.compile(
    r"\b(?:"
    r"(?:assist|help|handle|support|do|solve)\s+(?:with\s+)?(?:any|all)\b|"
    r"general[\s-]purpose\b|"
    r"universal\s+assistant\b|"
    r"all[\s-]in[\s-]one\b|"
    r"manage\s+files\s+and\s+run\s+commands\b|"
    r"for\s+any\s+(?:task|problem|request|query|coding)\b|"
    r"anything\s+(?:related|coding|code)\b"
    r")",
    re.IGNORECASE,
)

#: Built-in tool primitives and reserved agent commands across supported runtimes.
RESERVED_TOOL_NAMES = frozenset(
    {t.lower() for t in DEFAULT_DENIED_TOOLS}
    | {
        SKILL_TOOL_NAME.lower(),
        "ask_question",
        "finish",
        "replace_file_content",
        "run_command",
        "view_file",
        "write_to_file",
    },
)


class Severity(StrEnum):
    """Specify the severity level for a lint diagnostic."""

    ERROR = "error"
    IGNORE = "ignore"
    WARN = "warn"


class RuleDefinition(BaseModel):
    """Describe a static lint rule, its rationale, and recommended remediation."""

    model_config = ConfigDict(frozen=True)

    rule: str
    default_severity: Severity
    summary: str
    explanation: str
    remedy: str


#: Canonical registry of static lint rules and their documentation.
RULES: dict[str, RuleDefinition] = {
    "invalid-yaml": RuleDefinition(
        rule="invalid-yaml",
        default_severity=Severity.ERROR,
        summary="SKILL.md contains missing or unparseable YAML frontmatter",
        explanation=(
            "Agent runtimes parse frontmatter metadata to discover skills. "
            "Malformed YAML prevents the skill from being indexed or loaded."
        ),
        remedy="Ensure the file begins with '---' delimiters and contains valid YAML syntax.",
    ),
    "missing-name": RuleDefinition(
        rule="missing-name",
        default_severity=Severity.ERROR,
        summary="Frontmatter does not declare a skill 'name'",
        explanation=(
            "A skill must have an explicit identifier for agent catalog "
            "registration and invocation dispatch."
        ),
        remedy="Add a 'name' field to the frontmatter matching the skill's directory name.",
    ),
    "missing-description": RuleDefinition(
        rule="missing-description",
        default_severity=Severity.ERROR,
        summary="Frontmatter has no 'description' or the description is empty",
        explanation=(
            "Agent models read descriptions to determine whether a skill applies to a user query. "
            "A missing description renders the skill unselectable."
        ),
        remedy=(
            "Add a descriptive 'description' field stating what the skill does and when to use it."
        ),
    ),
    "invalid-name-format": RuleDefinition(
        rule="invalid-name-format",
        default_severity=Severity.ERROR,
        summary="Skill name does not adhere to lowercase kebab-case convention",
        explanation=(
            "Standard skill runtimes expect lowercase alphanumeric identifiers "
            "separated by single hyphens (max 64 chars)."
        ),
        remedy=(
            "Rename the skill to use only lowercase letters, digits, and hyphens "
            "(e.g. 'git-workflow')."
        ),
    ),
    "name-mismatch": RuleDefinition(
        rule="name-mismatch",
        default_severity=Severity.ERROR,
        summary="Frontmatter 'name' differs from the parent directory name",
        explanation=(
            "Mismatched directory and manifest names cause discovery anomalies "
            "when runtimes load skills by folder name."
        ),
        remedy="Align the frontmatter 'name' with the enclosing directory name.",
    ),
    "duplicate-name": RuleDefinition(
        rule="duplicate-name",
        default_severity=Severity.ERROR,
        summary="Multiple skills in the corpus declare the same name",
        explanation=(
            "Duplicate skill names cause nondeterministic catalog collisions "
            "and directory shadowing."
        ),
        remedy="Rename conflicting skills so every skill in the corpus has a distinct name.",
    ),
    "duplicate-capability": RuleDefinition(
        rule="duplicate-capability",
        default_severity=Severity.WARN,
        summary="Skill description has high semantic overlap (> 92%) with another skill",
        explanation=(
            "Descriptions with near-identical semantic vectors create ambiguous attractor "
            "basins that lead to misroutes and non-deterministic skill selection."
        ),
        remedy=(
            "Differentiate the skill descriptions by clarifying distinct trigger boundaries "
            "or consolidating redundant skills."
        ),
    ),
    "description-too-short": RuleDefinition(
        rule="description-too-short",
        default_severity=Severity.WARN,
        summary="Description is too brief to provide actionable routing criteria",
        explanation=(
            "Descriptions under 20 characters lack the context and trigger conditions "
            "agent models need to reliably route queries."
        ),
        remedy=(
            "Expand the description to clearly describe the skill's capabilities "
            "and trigger scenarios."
        ),
    ),
    "unresolved-placeholder": RuleDefinition(
        rule="unresolved-placeholder",
        default_severity=Severity.WARN,
        summary="Description contains unresolved template placeholders",
        explanation=(
            "Markers like TODO, FIXME, or <FILL_IN> in descriptions distract models "
            "and degrade selection accuracy."
        ),
        remedy=(
            "Replace template markers with concrete guidance describing actual skill capabilities."
        ),
    ),
    "reserved-name-collision": RuleDefinition(
        rule="reserved-name-collision",
        default_severity=Severity.WARN,
        summary="Skill name collides with a built-in agent tool or primitive",
        explanation=(
            "Naming a skill after a built-in command (e.g. 'bash', 'edit', 'read') "
            "confuses tool selection routing."
        ),
        remedy="Rename the skill to describe the specific domain task (e.g. 'bash-script-runner').",
    ),
    "listing-overflow": RuleDefinition(
        rule="listing-overflow",
        default_severity=Severity.WARN,
        summary="Skill description is unusually long and risks runtime truncation",
        explanation=(
            "Agent runtimes enforce strict listing budgets on catalog context. "
            "Excessively verbose descriptions risk truncation."
        ),
        remedy=(
            "Condense the description to highlight key triggers, moving extensive "
            "documentation into the markdown body."
        ),
    ),
    "unresolved-declared-dependency": RuleDefinition(
        rule="unresolved-declared-dependency",
        default_severity=Severity.WARN,
        summary="Declared dependency skill is missing from catalog",
        explanation=(
            "A skill declared in metadata.requires_skill or allowed-tools: Skill(X) "
            "does not exist in the resident catalog or discovery roots."
        ),
        remedy="Ensure the required skill is installed or update the dependency declaration.",
    ),
    "lockfile-drift": RuleDefinition(
        rule="lockfile-drift",
        default_severity=Severity.WARN,
        summary="SKILL.md digest does not match lockfile computedHash",
        explanation=(
            "The skill contents have changed locally since being pinned in skills-lock.json."
        ),
        remedy="Re-run npx skills update or refresh the lockfile hash.",
    ),
    "unbounded-attractor": RuleDefinition(
        rule="unbounded-attractor",
        default_severity=Severity.WARN,
        summary="Description uses greedy or universal phrasing that hijacks queries",
        explanation=(
            "Descriptions claiming unbounded scope (e.g. 'assist with any task' or "
            "'manage files and run commands') act as greedy attractor sinks in "
            "multi-skill catalogs, causing distractor hijacking."
        ),
        remedy=(
            "Narrow the description to specific domains, tools, and trigger conditions, "
            "and add directional disclaimers specifying when not to invoke the skill."
        ),
    ),
}


def explain_rule(rule_name: str) -> RuleDefinition | None:
    """Retrieve documentation and remediation advice for a named lint rule."""
    return RULES.get(rule_name)


class LintIssue(BaseModel):
    """Represent a single diagnostic finding for a skill file or catalog."""

    model_config = ConfigDict(frozen=True)

    rule: str
    severity: Severity
    skill: str
    path: Path | None = None
    message: str
    remedy: str = ""
    line: int | None = None


class LintReport(BaseModel):
    """Aggregate lint issues across all evaluated skills."""

    model_config = ConfigDict(frozen=True)

    issues: tuple[LintIssue, ...] = ()
    skills_checked: int = 0
    skill_name: str | None = None

    @property
    def errors(self) -> tuple[LintIssue, ...]:
        """Filter report issues to return only those with ERROR severity."""
        return tuple(issue for issue in self.issues if issue.severity == Severity.ERROR)

    @property
    def warnings(self) -> tuple[LintIssue, ...]:
        """Filter report issues to return only those with WARN severity."""
        return tuple(issue for issue in self.issues if issue.severity == Severity.WARN)

    @property
    def clean(self) -> bool:
        """Return True if no lint issues of any severity were found."""
        return not self.issues

    @property
    def has_errors(self) -> bool:
        """Return True if one or more errors were recorded."""
        return bool(self.errors)


def _resolve_severity(rule_name: str, config: LintSettings) -> Severity | None:
    """Determine effective severity for a rule based on configuration overrides."""
    configured = config.rules.get(rule_name)
    if configured is not None:
        if isinstance(configured, Severity):
            return None if configured == Severity.IGNORE else configured
        if isinstance(configured, str):
            try:
                sev = Severity(configured.lower())
            except ValueError:
                return None
            else:
                return None if sev is Severity.IGNORE else sev
        return None if configured == Severity.IGNORE else configured
    definition = RULES.get(rule_name)
    return definition.default_severity if definition is not None else Severity.WARN


def _create_issue(
    rule_name: str,
    skill_name: str,
    path: Path,
    message: str,
    config: LintSettings,
    line: int | None = None,
) -> LintIssue | None:
    """Construct a LintIssue if the rule is not ignored under active configuration."""
    severity = _resolve_severity(rule_name, config)
    if severity is None:
        return None
    definition = RULES.get(rule_name)
    remedy = definition.remedy if definition is not None else ""
    return LintIssue(
        rule=rule_name,
        severity=severity,
        skill=skill_name,
        path=path,
        message=message,
        remedy=remedy,
        line=line,
    )


def _record_issue(
    issues: list[LintIssue],
    rule_name: str,
    skill_name: str,
    path: Path,
    message: str,
    config: LintSettings,
    line: int | None = None,
) -> None:
    """Construct and append a LintIssue if not ignored under active configuration."""
    issue = _create_issue(rule_name, skill_name, path, message, config, line=line)
    if issue is not None:
        issues.append(issue)


def _lint_name(
    declared_name: object,
    skill_dir_name: str,
    skill_file: Path,
    config: LintSettings,
    issues: list[LintIssue],
) -> str:
    """Validate skill name presence, kebab-case formatting, and collision constraints."""
    effective_name = skill_dir_name
    if declared_name is None:
        _record_issue(
            issues,
            "missing-name",
            effective_name,
            skill_file,
            "SKILL.md frontmatter does not define a 'name' field.",
            config,
        )
    else:
        name_str = str(declared_name)
        effective_name = name_str
        if not _KEBAB_NAME.match(name_str) or len(name_str) > config.max_name_length:
            msg = (
                f"Skill name {name_str!r} must be lowercase kebab-case "
                f"(max {config.max_name_length} characters)."
            )
            _record_issue(issues, "invalid-name-format", effective_name, skill_file, msg, config)
        if name_str != skill_dir_name:
            msg = f"Skill name {name_str!r} does not match directory name {skill_dir_name!r}."
            _record_issue(issues, "name-mismatch", effective_name, skill_file, msg, config)

    lowered_name = effective_name.lower().replace("-", "").replace("_", "")
    if lowered_name in RESERVED_TOOL_NAMES or effective_name.lower() in RESERVED_TOOL_NAMES:
        msg = f"Skill name {effective_name!r} collides with built-in agent tool or primitive."
        _record_issue(issues, "reserved-name-collision", effective_name, skill_file, msg, config)

    return effective_name


def _lint_description(
    raw_desc: object,
    effective_name: str,
    skill_file: Path,
    config: LintSettings,
    issues: list[LintIssue],
) -> None:
    """Validate description presence, minimum/maximum length, and placeholders."""
    if raw_desc is None or not str(raw_desc).strip():
        _record_issue(
            issues,
            "missing-description",
            effective_name,
            skill_file,
            "SKILL.md frontmatter does not define a non-empty 'description' field.",
            config,
        )
        return

    desc_str = str(raw_desc).strip()
    if len(desc_str) < config.min_description_length:
        msg = (
            f"Description length ({len(desc_str)} chars) is below minimum recommended "
            f"threshold ({config.min_description_length} chars)."
        )
        _record_issue(issues, "description-too-short", effective_name, skill_file, msg, config)

    if len(desc_str) > config.max_description_length:
        msg = (
            f"Description length ({len(desc_str)} chars) exceeds warning threshold "
            f"({config.max_description_length} chars) and may be truncated by "
            "runtime listing budgets."
        )
        _record_issue(issues, "listing-overflow", effective_name, skill_file, msg, config)

    if match := _PLACEHOLDER.search(desc_str):
        msg = f"Description contains unresolved template placeholder {match.group(0)!r}."
        _record_issue(issues, "unresolved-placeholder", effective_name, skill_file, msg, config)

    if match := _UNBOUNDED_ATTRACTOR.search(desc_str):
        msg = (
            f"Description contains overly broad attractor phrasing {match.group(0)!r} "
            "which causes distractor hijacking in multi-skill catalogs."
        )
        _record_issue(issues, "unbounded-attractor", effective_name, skill_file, msg, config)


def _lint_frontmatter_dict(
    data: dict[object, object],
    skill_dir_name: str,
    skill_file: Path,
    config: LintSettings,
) -> tuple[list[LintIssue], str]:
    """Validate parsed frontmatter dictionary fields and semantic properties."""
    issues: list[LintIssue] = []
    effective_name = _lint_name(data.get("name"), skill_dir_name, skill_file, config, issues)
    _lint_description(data.get("description"), effective_name, skill_file, config, issues)
    return issues, effective_name


def _check_lockfile_drift(
    path: Path,
    skill_name: str,
    config: LintSettings,
    issues: list[LintIssue],
) -> None:
    """Check whether SKILL.md content has drifted from its pinned lockfile hash."""
    manifest = find_skill_manifest(path)
    if manifest is None or manifest.name != "skills-lock.json":
        return
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        skills = data.get("skills")
        if not isinstance(skills, dict):
            return
        entry = skills.get(skill_name)
        if not isinstance(entry, dict):
            return
        expected_hash = entry.get("computedHash")
        if not isinstance(expected_hash, str) or not expected_hash:
            return

        raw_bytes = path.read_bytes()
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        try:
            norm_bytes = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
            norm_hash = hashlib.sha256(norm_bytes).hexdigest()
        except (UnicodeDecodeError, OSError):
            norm_hash = raw_hash

        if expected_hash not in (raw_hash, norm_hash):
            msg = (
                f"Skill {skill_name!r} content hash ({raw_hash[:8]}...) diverges from "
                f"pinned lockfile hash ({expected_hash[:8]}...)."
            )
            _record_issue(issues, "lockfile-drift", skill_name, path, msg, config)
    except (OSError, json.JSONDecodeError):
        return


def lint_file(skill_file: Path | str, config: LintSettings | None = None) -> LintReport:
    """Inspect a single SKILL.md manifest file for structural and authoring issues."""
    path = resolve_path(skill_file)
    cfg = config if config is not None else LintSettings.from_settings()
    dir_name = path.parent.name
    issues: list[LintIssue] = []

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Failed to read SKILL.md: {exc}"
        _record_issue(issues, "invalid-yaml", dir_name, path, msg, cfg)
        return LintReport(issues=tuple(issues), skills_checked=1, skill_name=dir_name)

    split = split_frontmatter(content)
    if split is None:
        _record_issue(
            issues,
            "invalid-yaml",
            dir_name,
            path,
            "File lacks valid YAML frontmatter surrounded by '---' delimiters.",
            cfg,
        )
        return LintReport(issues=tuple(issues), skills_checked=1, skill_name=dir_name)

    frontmatter_text, _body = split
    try:
        loaded = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        _record_issue(
            issues,
            "invalid-yaml",
            dir_name,
            path,
            f"YAML parsing error in frontmatter: {exc}",
            cfg,
        )
        return LintReport(issues=tuple(issues), skills_checked=1, skill_name=dir_name)

    if not isinstance(loaded, dict):
        _record_issue(
            issues,
            "invalid-yaml",
            dir_name,
            path,
            "Frontmatter YAML must be a mapping/dictionary of key-value pairs.",
            cfg,
        )
        return LintReport(issues=tuple(issues), skills_checked=1, skill_name=dir_name)

    field_issues, skill_name = _lint_frontmatter_dict(loaded, dir_name, path, cfg)
    issues.extend(field_issues)
    _check_lockfile_drift(path, skill_name, cfg, issues)

    return LintReport(issues=tuple(issues), skills_checked=1, skill_name=skill_name)


def _check_duplicates(
    names_seen: Mapping[str, int],
    paths_by_name: Mapping[str, Sequence[Path]],
    cfg: LintSettings,
) -> list[LintIssue]:
    """Identify and record duplicate skill name collision issues across a corpus."""
    issues: list[LintIssue] = []
    for name, count in names_seen.items():
        if count > 1:
            for colliding_path in paths_by_name[name]:
                msg = f"Duplicate skill name {name!r} declared across {count} locations in corpus."
                _record_issue(issues, "duplicate-name", name, colliding_path, msg, cfg)
    return issues


def _check_duplicate_capabilities(
    skills: Sequence[Skill],
    paths_by_name: Mapping[str, Sequence[Path]],
    cfg: LintSettings,
) -> list[LintIssue]:
    """Identify and record near-duplicate capability semantic collisions across a corpus."""
    if _resolve_severity("duplicate-capability", cfg) is None or len(skills) < MIN_PAIRWISE_SKILLS:
        return []

    try:
        from reach.retrieval import DenseScorer

        scorer = DenseScorer.from_skills(skills)
        pairs = scorer.pairwise_similarity(skills)
    except (RuntimeError, ValueError, OSError):
        return []

    issues: list[LintIssue] = []
    threshold = cfg.similarity_threshold
    for s1_name, s2_name, sim in pairs:
        if sim >= threshold:
            for s1_path in paths_by_name.get(s1_name, ()):
                msg = (
                    f"Near-duplicate capability: '{s1_name}' shares {sim:.1%} semantic "
                    f"similarity with '{s2_name}' (threshold: {threshold:.1%})."
                )
                _record_issue(issues, "duplicate-capability", s1_name, s1_path, msg, cfg)
    return issues


def _check_declared_dependencies(
    skills: Sequence[Skill],
    names_seen: Mapping[str, int],
    paths_by_name: Mapping[str, Sequence[Path]],
    cfg: LintSettings,
) -> list[LintIssue]:
    """Identify and record declared dependencies missing from the corpus."""
    issues: list[LintIssue] = []
    known_names = set(names_seen.keys())
    for skill in skills:
        for dep in skill.declared_dependencies:
            if dep not in known_names:
                for skill_path in paths_by_name.get(skill.name, ()):
                    msg = (
                        f"Skill '{skill.name}' declares dependency '{dep}' which is "
                        "missing from the catalog or discovery roots."
                    )
                    _record_issue(
                        issues,
                        "unresolved-declared-dependency",
                        skill.name,
                        skill_path,
                        msg,
                        cfg,
                    )
    return issues


def _lint_paths(
    file_paths: Sequence[Path],
    cfg: LintSettings,
    skill_filter: str | None = None,
) -> LintReport:
    """Lint a sequence of SKILL.md file paths and aggregate diagnostics with duplicate checks."""
    all_issues: list[LintIssue] = []
    names_seen: Counter[str] = Counter()
    paths_by_name: dict[str, list[Path]] = {}
    valid_skills: list[Skill] = []
    skills_checked = 0

    for file_path in file_paths:
        file_report = lint_file(file_path, config=cfg)
        skill_name = file_report.skill_name or file_path.parent.name

        if skill_filter is not None and skill_name != skill_filter:
            continue

        skills_checked += 1
        all_issues.extend(file_report.issues)
        names_seen[skill_name] += 1
        paths_by_name.setdefault(skill_name, []).append(file_path)

        try:
            parsed = parse_frontmatter(file_path.read_text(encoding="utf-8"), file_path)
            if parsed is not None and parsed.description:
                valid_skills.append(parsed)
        except (OSError, yaml.YAMLError, UnicodeDecodeError):
            pass

    all_issues.extend(_check_duplicates(names_seen, paths_by_name, cfg))
    all_issues.extend(_check_duplicate_capabilities(valid_skills, paths_by_name, cfg))
    all_issues.extend(_check_declared_dependencies(valid_skills, names_seen, paths_by_name, cfg))
    all_issues.sort(key=lambda i: (str(i.path or ""), i.line or 0, i.rule))
    return LintReport(issues=tuple(all_issues), skills_checked=skills_checked)


def lint_tree(
    root: Path | str,
    config: LintSettings | None = None,
    skill_filter: str | None = None,
) -> LintReport:
    """Recursively search for and lint all SKILL.md files under a directory root.

    Args:
        root: Directory root to search for skills.
        config: Optional LintSettings overrides; loads from reach.toml settings if omitted.
        skill_filter: Optional skill name filter to limit reported diagnostics.

    Returns:
        A LintReport summarizing all issues found across discovered skills.
    """
    resolved_root = resolve_path(root)
    cfg = config if config is not None else LintSettings.from_settings()
    return _lint_paths(_skill_files(resolved_root), cfg, skill_filter=skill_filter)


def lint_skills(
    skills: Sequence[Path | str],
    config: LintSettings | None = None,
) -> LintReport:
    """Lint a specific collection of skill directories or SKILL.md file paths.

    Args:
        skills: Sequence of paths to skill directories or SKILL.md files.
        config: Optional LintSettings overrides; loads from reach.toml settings if omitted.

    Returns:
        A LintReport containing all detected issues, severities, and skill counts.
    """
    cfg = config if config is not None else LintSettings.from_settings()
    resolved_files: list[Path] = []
    for item in skills:
        path = resolve_path(item)
        target = path / "SKILL.md" if path.is_dir() else path
        if target.is_file():
            resolved_files.append(target)
    return _lint_paths(resolved_files, cfg)
