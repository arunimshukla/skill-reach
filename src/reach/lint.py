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
from collections.abc import Iterable, Mapping, Sequence
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
    from reach.retrieval import Bm25Scorer

__all__ = [
    "RULES",
    "LintIssue",
    "LintReport",
    "LintSettings",
    "RuleDefinition",
    "Severity",
    "explain_rule",
    "extract_skill_references",
    "find_competing_neighbors",
    "find_unknown_skill_references",
    "hands_off_to_skill",
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
    "unknown-skill-reference": RuleDefinition(
        rule="unknown-skill-reference",
        default_severity=Severity.ERROR,
        summary="Description hands off to a skill name that does not exist in the catalog",
        explanation=(
            "Negative routing instructions (e.g. 'Don't use for X — use <other-skill>') "
            "that reference a missing or unmerged skill actively repel the router away "
            "from the resident skill while the target skill is absent, creating a 0% "
            "recall sinkhole."
        ),
        remedy=(
            "Remove the handoff reference until the target skill is added to the catalog, "
            "or correct the referenced skill name."
        ),
    ),
    "missing-mutual-handoff": RuleDefinition(
        rule="missing-mutual-handoff",
        default_severity=Severity.WARN,
        summary="Overlapping neighbor skills lack mutual routing handoffs ('use <other-skill>')",
        explanation=(
            "When closely related skills share domain vocabulary or one skill defines a "
            "one-way boundary without a reciprocal handoff on the neighbor, the unguarded "
            "skill acts as a one-way attractor sink and hijacks queries."
        ),
        remedy=(
            "Add reciprocal 'Don't use for X (use <neighbor-skill>)' handoff clauses to "
            "both overlapping skills so each carves out the other's territory."
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


#: Common hyphenated compound adjectives and technical terms that are not skill identifiers.
_NON_SKILL_HYPHENATED_TERMS: Final[frozenset[str]] = frozenset(
    {
        "all-in-one",
        "apt-get",
        "auto-scaling",
        "built-in",
        "ci-cd",
        "client-side",
        "command-line",
        "cross-origin",
        "cross-platform",
        "cross-project",
        "cross-region",
        "day-to-day",
        "docker-compose",
        "dry-run",
        "end-to-end",
        "event-driven",
        "fail-fast",
        "fine-grained",
        "first-party",
        "flat-rate",
        "full-text",
        "general-purpose",
        "git-lfs",
        "high-availability",
        "high-level",
        "high-performance",
        "high-throughput",
        "huggingface-hub",
        "in-memory",
        "in-place",
        "key-value",
        "kebab-case",
        "least-privilege",
        "long-lived",
        "long-running",
        "low-latency",
        "low-level",
        "machine-learning",
        "multi-agent",
        "multi-channel",
        "multi-cloud",
        "multi-cluster",
        "multi-region",
        "multi-stage",
        "multi-step",
        "multi-tenant",
        "multi-turn",
        "near-duplicate",
        "non-empty",
        "non-interactive",
        "non-null",
        "non-zero",
        "object-oriented",
        "on-demand",
        "on-prem",
        "on-premises",
        "one-off",
        "one-shot",
        "one-way",
        "open-source",
        "out-of-scope",
        "out-of-the-box",
        "pay-as-you-go",
        "pip-compile",
        "point-in-time",
        "pre-built",
        "pre-commit",
        "pre-configured",
        "pre-flight",
        "production-ready",
        "pull-based",
        "push-based",
        "read-only",
        "read-write",
        "real-time",
        "red-green-refactor",
        "role-based",
        "root-cause",
        "round-robin",
        "rule-based",
        "run-to-run",
        "scikit-learn",
        "self-contained",
        "self-hosted",
        "self-managed",
        "self-repulsion",
        "self-service",
        "self-signed",
        "server-side",
        "sha-256",
        "short-form",
        "short-lived",
        "side-by-side",
        "single-node",
        "single-page",
        "source-to-image",
        "split-horizon",
        "stage-level",
        "state-of-the-art",
        "step-by-step",
        "sub-agent",
        "sub-agents",
        "test-driven",
        "test-first",
        "third-party",
        "time-series",
        "token-based",
        "top-1",
        "top-k",
        "top-level",
        "two-stage",
        "type-safe",
        "utf-8",
        "user-defined",
        "user-facing",
        "well-architected",
        "well-formed",
        "well-structured",
        "write-in",
        "zero-downtime",
        "zero-trust",
    }
)

#: Generic platform/domain prefixes excluded when extracting distinctive skill name runs.
_GENERIC_NAME_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "agent",
        "agents",
        "api",
        "aws",
        "azure",
        "basics",
        "cli",
        "cloud",
        "common",
        "core",
        "default",
        "gcp",
        "general",
        "google",
        "helper",
        "sdk",
        "skill",
        "skills",
        "tool",
        "tools",
    }
)

_HANDOFF_VERB = r"(?:use|see|prefer|refer\s+to|defer\s+to|delegate\s+to|hand\s+off\s+to)"
_KEBAB_ID = r"[a-z0-9]+(?:-[a-z0-9]+)+"
_KEBAB_ID_RE = re.compile(_KEBAB_ID, re.IGNORECASE)
_ANY_SKILL_ID = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_KEBAB_LIST = (
    rf"`?{_KEBAB_ID}`?(?:\s+(?:first|instead|skill))?"
    rf"(?:\s*(?:,\s*(?:or|and)\b|,|\bor\b|\band\b)\s*(?:use\s+|see\s+|prefer\s+|the\s+)?`?{_KEBAB_ID}`?(?:\s+(?:first|instead|skill))?)*"
)

_PAREN_HANDOFF_RE = re.compile(
    rf"\([^)]*?\b{_HANDOFF_VERB}\s+(?:the\s+)?(?P<targets>{_KEBAB_LIST})[^)]*\)",
    re.IGNORECASE,
)

_QUALIFIED_HANDOFF_RE = re.compile(
    rf"\b{_HANDOFF_VERB}\s+(?:the\s+)?`?({_ANY_SKILL_ID})`?\s+(?:instead|first|skill)\b",
    re.IGNORECASE,
)

_BACKTICK_HANDOFF_RE = re.compile(
    rf"\b{_HANDOFF_VERB}\s+(?:the\s+)?`({_KEBAB_ID})`",
    re.IGNORECASE,
)

_NEGATIVE_CLAUSE_MARKER_RE = re.compile(
    r"\b(?:don't\s+use|do\s+not\s+use|not\s+for\b|never\s+use|avoid\s+using|"
    r"instead\s+of\b|rather\s+than\b|for\s+[^.!?;]+,\s*(?:use|prefer|defer\s+to|see)\b)",
    re.IGNORECASE,
)

_VERB_TARGET_IN_CLAUSE_RE = re.compile(
    rf"\b{_HANDOFF_VERB}\s+(?:the\s+)?(?P<targets>{_KEBAB_LIST})",
    re.IGNORECASE,
)


#: Minimum distinctive token count in a neighbor skill name to detect phrase encroachment.
_MIN_DISTINCTIVE_NAME_TOKENS: Final = 2

#: Minimum character length for shared trigger terms reported in mutual handoff diagnostics.
_MIN_SHARED_TRIGGER_LENGTH: Final = 3


def _is_valid_skill_ref(
    candidate: str,
    self_lower: str | None,
    require_hyphen: bool = True,
) -> bool:
    """Check whether an extracted token is a plausible skill reference rather than prose."""
    cleaned = candidate.strip().lower()
    if not cleaned or (self_lower is not None and cleaned == self_lower):
        return False
    if require_hyphen and "-" not in cleaned:
        return False
    if not _KEBAB_NAME.match(cleaned):
        return False
    return cleaned not in _NON_SKILL_HYPHENATED_TERMS and cleaned not in RESERVED_TOOL_NAMES


def _add_if_valid_ref(
    found: set[str],
    candidate: str | None,
    self_lower: str | None,
    *,
    require_hyphen: bool = True,
) -> None:
    """Add candidate to found set if it passes skill reference validation."""
    if candidate and _is_valid_skill_ref(candidate, self_lower, require_hyphen=require_hyphen):
        found.add(candidate.lower())


def extract_skill_references(
    description: str,
    *,
    self_name: str | None = None,
) -> tuple[str, ...]:
    """Extract skill identifiers referenced in routing handoff or disclaimer clauses.

    Args:
        description: Raw skill description text from SKILL.md frontmatter.
        self_name: Optional name of the skill itself to exclude self-references.

    Returns:
        Sorted tuple of unique referenced skill names in kebab-case.
    """
    if not description or not description.strip():
        return ()

    self_lower = self_name.strip().lower() if self_name else None
    found: set[str] = set()

    for match in _PAREN_HANDOFF_RE.finditer(description):
        for token in _KEBAB_ID_RE.findall(match.group("targets")):
            _add_if_valid_ref(found, token, self_lower, require_hyphen=True)

    for match in _QUALIFIED_HANDOFF_RE.finditer(description):
        qualifier_is_skill = match.group(0).strip().lower().endswith("skill")
        _add_if_valid_ref(
            found,
            match.group(1),
            self_lower,
            require_hyphen=not qualifier_is_skill,
        )

    for match in _BACKTICK_HANDOFF_RE.finditer(description):
        _add_if_valid_ref(found, match.group(1), self_lower, require_hyphen=True)

    for sentence in re.split(r"[.!?]+", description):
        if not _NEGATIVE_CLAUSE_MARKER_RE.search(sentence):
            continue
        for match in _VERB_TARGET_IN_CLAUSE_RE.finditer(sentence):
            for token in _KEBAB_ID_RE.findall(match.group("targets")):
                _add_if_valid_ref(found, token, self_lower, require_hyphen=True)

    return tuple(sorted(found))


def find_unknown_skill_references(
    description: str,
    known_skills: Iterable[str],
    *,
    self_name: str | None = None,
    settings: LintSettings | None = None,
) -> tuple[str, ...]:
    """Return referenced skill names in description that are absent from known_skills.

    Respects the configured severity for ``unknown-skill-reference`` and returns
    an empty tuple when the rule is set to ``ignore``.
    """
    cfg = settings if settings is not None else LintSettings.from_settings()
    if _resolve_severity("unknown-skill-reference", cfg) is None:
        return ()
    known_lower = {name.lower() for name in known_skills}
    refs = extract_skill_references(description, self_name=self_name)
    return tuple(ref for ref in refs if ref not in known_lower)


def _check_unknown_skill_references(
    skills: Sequence[Skill],
    names_seen: Mapping[str, int],
    paths_by_name: Mapping[str, Sequence[Path]],
    cfg: LintSettings,
) -> list[LintIssue]:
    """Identify routing handoffs in descriptions that point to non-existent skills."""
    if _resolve_severity("unknown-skill-reference", cfg) is None:
        return []

    issues: list[LintIssue] = []
    known_names = {name.lower() for name in names_seen}
    for skill in skills:
        unknown_refs = find_unknown_skill_references(
            skill.description,
            known_names,
            self_name=skill.name,
            settings=cfg,
        )
        for ref in unknown_refs:
            for skill_path in paths_by_name.get(skill.name, ()):
                msg = (
                    f"Skill '{skill.name}' hands off to unknown skill '{ref}' in its "
                    "description, which is missing from the catalog and causes routing "
                    "self-repulsion."
                )
                _record_issue(
                    issues,
                    "unknown-skill-reference",
                    skill.name,
                    skill_path,
                    msg,
                    cfg,
                )
    return issues


def hands_off_to_skill(
    source_description: str,
    target_name: str,
    extracted_refs: frozenset[str] | None = None,
) -> bool:
    """Return True if source_description explicitly hands off to or disclaims target_name."""
    target_lower = target_name.lower()
    refs = (
        extracted_refs
        if extracted_refs is not None
        else frozenset(extract_skill_references(source_description))
    )
    if target_lower in refs:
        return True

    from reach.leak import contains_run
    from reach.retrieval import tokenize

    wanted = tokenize(target_lower)
    if not wanted:
        return False

    handoff_sentence_re = re.compile(
        r"\b(?:use|see|prefer|refer|defer|delegate|instead|don't|do\s+not|not\s+for|avoid)\b",
        re.IGNORECASE,
    )
    for sentence in re.split(r"[.!?]+", source_description):
        if handoff_sentence_re.search(sentence) and contains_run(tokenize(sentence), wanted):
            return True
    return False


_hands_off_to_skill = hands_off_to_skill


def _claims_neighbor_name_phrase(source: Skill, neighbor: Skill) -> bool:
    """Check whether source description contains neighbor's distinctive multi-token name."""
    from reach.leak import contains_run
    from reach.retrieval import tokenize

    neighbor_tokens = [t for t in tokenize(neighbor.name) if t not in _GENERIC_NAME_TOKENS]
    if len(neighbor_tokens) < _MIN_DISTINCTIVE_NAME_TOKENS:
        return False
    source_desc_tokens = tokenize(source.description)
    return contains_run(source_desc_tokens, neighbor_tokens)


def _max_lexical_ratio(
    s1_name: str,
    s2_name: str,
    comp_by_name: Mapping[str, object],
) -> float:
    """Compute maximum directional BM25 rival score ratio between two skills."""
    from reach.overlap import Competition

    c1 = comp_by_name.get(s1_name)
    c2 = comp_by_name.get(s2_name)
    r12 = (
        next((r.score for r in c1.rivals if r.name == s2_name), 0.0) / c1.self_score
        if isinstance(c1, Competition) and c1.self_score > 0
        else 0.0
    )
    r21 = (
        next((r.score for r in c2.rivals if r.name == s1_name), 0.0) / c2.self_score
        if isinstance(c2, Competition) and c2.self_score > 0
        else 0.0
    )
    return max(r12, r21)


def find_competing_neighbors(
    modified: set[str],
    skills: Sequence[Skill],
    *,
    settings: LintSettings | None = None,
    dense_similarities: Mapping[tuple[str, str], float] | None = None,
) -> set[str]:
    """Identify competing neighbor skills that could be hijacked by modified skills."""
    if not modified or len(skills) < MIN_PAIRWISE_SKILLS:
        return set()

    from reach.overlap import rank_corpus

    cfg = settings if settings is not None else LintSettings.from_settings()
    lex_thresh = cfg.mutual_handoff_lexical_threshold
    sem_thresh = cfg.mutual_handoff_similarity_threshold
    sim_map = (
        dense_similarities
        if dense_similarities is not None
        else _compute_dense_similarities(skills)
    )
    overlap = rank_corpus(skills)
    comp_by_name = {c.skill: c for c in overlap.competitions}
    by_name = {s.name: s for s in skills}
    refs_by_name = {
        s.name: frozenset(extract_skill_references(s.description, self_name=s.name))
        for s in skills
    }
    neighbors: set[str] = set()

    for mod_name in modified:
        mod_skill = by_name.get(mod_name)
        if mod_skill is None:
            continue
        neighbors.update(refs_by_name.get(mod_name, frozenset()) & (set(by_name) - modified))
        for candidate in skills:
            if candidate.name in modified:
                continue
            sem_sim = max(
                sim_map.get((mod_name, candidate.name), 0.0),
                sim_map.get((candidate.name, mod_name), 0.0),
            )
            lex_ratio = _max_lexical_ratio(mod_name, candidate.name, comp_by_name)
            if (
                lex_ratio >= lex_thresh
                or sem_sim >= sem_thresh
                or mod_name in refs_by_name.get(candidate.name, ())
                or _claims_neighbor_name_phrase(mod_skill, candidate)
                or _claims_neighbor_name_phrase(candidate, mod_skill)
            ):
                neighbors.add(candidate.name)

    return neighbors


def _shared_trigger_terms(
    s1: Skill,
    s2: Skill,
    scorer: Bm25Scorer | None = None,
    limit: int = 3,
) -> tuple[str, ...]:
    """Extract top shared high-IDF domain vocabulary terms between two skills."""
    from reach.leak import FUNCTION_WORDS
    from reach.retrieval import tokenize

    ignored = FUNCTION_WORDS | _GENERIC_NAME_TOKENS | {"and"}
    t1 = set(tokenize(s1.description)) - ignored
    t2 = set(tokenize(s2.description)) - ignored
    shared = [t for t in (t1 & t2) if len(t) >= _MIN_SHARED_TRIGGER_LENGTH]
    if scorer is not None:
        shared.sort(key=lambda term: (-scorer.idf(term), term))
    else:
        shared.sort()
    return tuple(shared[:limit])


def _check_missing_mutual_handoffs(
    skills: Sequence[Skill],
    paths_by_name: Mapping[str, Sequence[Path]],
    cfg: LintSettings,
    dense_similarities: Mapping[tuple[str, str], float] | None = None,
) -> list[LintIssue]:
    """Identify overlapping neighbor pairs with one-way or missing reciprocal routing handoffs."""
    if (
        _resolve_severity("missing-mutual-handoff", cfg) is None
        or len(skills) < MIN_PAIRWISE_SKILLS
    ):
        return []

    from reach.overlap import rank_corpus
    from reach.retrieval import Bm25Scorer

    scorer = Bm25Scorer.from_skills(skills)
    overlap = rank_corpus(skills)
    comp_by_name = {c.skill: c for c in overlap.competitions}
    refs_by_name = {
        s.name: frozenset(extract_skill_references(s.description, self_name=s.name))
        for s in skills
    }

    issues: list[LintIssue] = []
    lex_thresh = cfg.mutual_handoff_lexical_threshold
    sem_thresh = cfg.mutual_handoff_similarity_threshold

    for i, s1 in enumerate(skills):
        for s2 in skills[i + 1 :]:
            s1_to_s2 = hands_off_to_skill(s1.description, s2.name, refs_by_name[s1.name])
            s2_to_s1 = hands_off_to_skill(s2.description, s1.name, refs_by_name[s2.name])
            if s1_to_s2 and s2_to_s1:
                continue

            max_lex_ratio = _max_lexical_ratio(s1.name, s2.name, comp_by_name)
            sem_sim = 0.0
            if dense_similarities is not None:
                sem_sim = max(
                    dense_similarities.get((s1.name, s2.name), 0.0),
                    dense_similarities.get((s2.name, s1.name), 0.0),
                )

            one_way_handoff = (s1_to_s2 != s2_to_s1) and (
                max_lex_ratio > 0.0 or sem_sim >= sem_thresh
            )
            either_has_boundaries = bool(refs_by_name[s1.name] or refs_by_name[s2.name])
            high_neighbor_contention = (
                _claims_neighbor_name_phrase(s1, s2)
                or _claims_neighbor_name_phrase(s2, s1)
                or (sem_sim >= sem_thresh and max_lex_ratio >= lex_thresh)
                or (
                    either_has_boundaries
                    and (max_lex_ratio >= lex_thresh or sem_sim >= sem_thresh)
                )
            )

            if not (one_way_handoff or high_neighbor_contention):
                continue

            shared_terms = _shared_trigger_terms(s1, s2, scorer)
            shared_str = (
                f" (shared triggers: {', '.join(repr(t) for t in shared_terms)})"
                if shared_terms
                else ""
            )

            for subject, partner, subj_hands_to_partner in (
                (s1, s2, s1_to_s2),
                (s2, s1, s2_to_s1),
            ):
                if subj_hands_to_partner:
                    msg = (
                        f"Skill '{subject.name}' hands off to '{partner.name}', but "
                        f"'{partner.name}' has no reciprocal handoff back to '{subject.name}'"
                        f"{shared_str}."
                    )
                else:
                    msg = (
                        f"Skill '{subject.name}' overlaps with '{partner.name}'{shared_str} "
                        f"but does not include a mutual routing handoff "
                        f"(e.g. \"Don't use for ... (use {partner.name})\")."
                    )
                for skill_path in paths_by_name.get(subject.name, ()):
                    _record_issue(
                        issues,
                        "missing-mutual-handoff",
                        subject.name,
                        skill_path,
                        msg,
                        cfg,
                    )

    return issues


def _compute_dense_similarities(skills: Sequence[Skill]) -> dict[tuple[str, str], float]:
    """Compute pairwise semantic similarities when DenseScorer is available."""
    if len(skills) < MIN_PAIRWISE_SKILLS:
        return {}
    try:
        from reach.retrieval import DenseScorer

        scorer = DenseScorer.from_skills(skills)
        pairs = scorer.pairwise_similarity(skills)
        return {(s1, s2): sim for s1, s2, sim in pairs}
    except (RuntimeError, ValueError, OSError):
        return {}


def _check_duplicate_capabilities(
    skills: Sequence[Skill],
    paths_by_name: Mapping[str, Sequence[Path]],
    cfg: LintSettings,
    dense_similarities: Mapping[tuple[str, str], float] | None = None,
) -> list[LintIssue]:
    """Identify and record near-duplicate capability semantic collisions across a corpus."""
    if _resolve_severity("duplicate-capability", cfg) is None or len(skills) < MIN_PAIRWISE_SKILLS:
        return []

    sim_map = (
        dense_similarities
        if dense_similarities is not None
        else _compute_dense_similarities(skills)
    )
    if not sim_map:
        return []

    issues: list[LintIssue] = []
    threshold = cfg.similarity_threshold
    for (s1_name, s2_name), sim in sim_map.items():
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

        names_seen[skill_name] += 1
        paths_by_name.setdefault(skill_name, []).append(file_path)

        try:
            parsed = parse_frontmatter(file_path.read_text(encoding="utf-8"), file_path)
            if parsed is not None and parsed.description:
                valid_skills.append(parsed)
        except (OSError, yaml.YAMLError, UnicodeDecodeError):
            pass

        if skill_filter is not None and skill_name != skill_filter:
            continue

        skills_checked += 1
        all_issues.extend(file_report.issues)

    need_dense = (
        _resolve_severity("duplicate-capability", cfg) is not None
        or _resolve_severity("missing-mutual-handoff", cfg) is not None
    )
    dense_sims = _compute_dense_similarities(valid_skills) if need_dense else {}

    corpus_issues: list[LintIssue] = []
    corpus_issues.extend(_check_duplicates(names_seen, paths_by_name, cfg))
    corpus_issues.extend(
        _check_duplicate_capabilities(
            valid_skills,
            paths_by_name,
            cfg,
            dense_similarities=dense_sims,
        )
    )
    corpus_issues.extend(_check_declared_dependencies(valid_skills, names_seen, paths_by_name, cfg))
    corpus_issues.extend(
        _check_unknown_skill_references(valid_skills, names_seen, paths_by_name, cfg)
    )
    corpus_issues.extend(
        _check_missing_mutual_handoffs(
            valid_skills,
            paths_by_name,
            cfg,
            dense_similarities=dense_sims,
        )
    )

    if skill_filter is not None:
        corpus_issues = [i for i in corpus_issues if i.skill == skill_filter]

    all_issues.extend(corpus_issues)
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
