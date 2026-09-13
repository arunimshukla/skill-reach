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

"""Verify static skill linting rules, severity configuration, and report generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reach.lint import (
    LintSettings,
    RuleDefinition,
    Severity,
    explain_rule,
    lint_file,
    lint_tree,
)
from reach.rendering import format_github_annotation

if TYPE_CHECKING:
    from collections.abc import Callable


def test_valid_skill_passes_clean(write_skill: Callable[..., Path]) -> None:
    """Verify that a well-formed skill produces zero lint issues."""
    manifest = (
        write_skill(
            name="valid-skill",
            description="Extract and parse tabular data from PDF invoices accurately.",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest)
    assert report.clean
    assert len(report.issues) == 0


@pytest.mark.parametrize(
    ("name", "raw_yaml", "expected_rule", "expected_severity"),
    [
        (
            "bad-yaml",
            "---\nname: [unclosed list\n---\n",
            "invalid-yaml",
            Severity.ERROR,
        ),
        (
            "no-frontmatter",
            "# Just markdown\nNo frontmatter here.\n",
            "invalid-yaml",
            Severity.ERROR,
        ),
        (
            "no-name",
            "---\ndescription: Has a description but no name.\n---\n",
            "missing-name",
            Severity.ERROR,
        ),
        (
            "no-desc",
            "---\nname: no-desc\ndescription: ''\n---\n",
            "missing-description",
            Severity.ERROR,
        ),
    ],
)
def test_invalid_frontmatter_rules(
    write_skill: Callable[..., Path],
    name: str,
    raw_yaml: str,
    expected_rule: str,
    expected_severity: Severity,
) -> None:
    """Verify malformed frontmatter triggers corresponding error rules."""
    manifest = write_skill(name=name, raw_yaml=raw_yaml) / "SKILL.md"
    report = lint_file(manifest)
    assert report.has_errors
    assert any(i.rule == expected_rule and i.severity == expected_severity for i in report.issues)


def test_lint_file_accepts_unicode_bom(tmp_path: Path) -> None:
    """Verify lint_file successfully parses and lints files with leading Unicode BOM."""
    skill_dir = tmp_path / "bom-skill"
    skill_dir.mkdir(parents=True)
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        "\ufeff---\nname: bom-skill\ndescription: Skill saved with Unicode BOM.\n---\nBody\n",
        encoding="utf-8",
    )
    report = lint_file(manifest)
    assert not report.has_errors
    assert report.skills_checked == 1


@pytest.mark.parametrize(
    "invalid_name",
    [
        "CamelCaseName",
        "name_with_underscores",
        "name with spaces",
        "-leading-hyphen",
        "trailing-hyphen-",
        "double--hyphen",
        "special@chars!",
    ],
)
def test_invalid_name_format(write_skill: Callable[..., Path], invalid_name: str) -> None:
    """Verify that names not matching kebab-case produce invalid-name-format error."""
    manifest = (
        write_skill(
            name=invalid_name,
            description="A sufficiently long description for testing invalid names.",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest)
    assert any(
        i.rule == "invalid-name-format" and i.severity == Severity.ERROR for i in report.issues
    )


def test_name_mismatch(write_skill: Callable[..., Path]) -> None:
    """Verify that a name differing from the directory name produces name-mismatch error."""
    manifest = (
        write_skill(
            name="declared-name",
            description="A sufficiently long description for testing directory name mismatch.",
            dir_name="dir-name",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest)
    assert any(i.rule == "name-mismatch" and i.severity == Severity.ERROR for i in report.issues)


def test_description_too_short(write_skill: Callable[..., Path]) -> None:
    """Verify that descriptions under threshold produce description-too-short warning."""
    manifest = (
        write_skill(
            name="terse-skill",
            description="Does stuff.",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest, config=LintSettings(min_description_length=20))
    assert any(
        i.rule == "description-too-short" and i.severity == Severity.WARN for i in report.issues
    )


@pytest.mark.parametrize(
    "placeholder",
    ["TODO: write this", "FIXME later", "Run <FILL_IN> here", "Check [TODO] item"],
)
def test_unresolved_placeholder(write_skill: Callable[..., Path], placeholder: str) -> None:
    """Verify that placeholder strings produce unresolved-placeholder warning."""
    manifest = (
        write_skill(
            name="placeholder-skill",
            description=f"Automate cloud deployments {placeholder} for clusters.",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest)
    assert any(
        i.rule == "unresolved-placeholder" and i.severity == Severity.WARN for i in report.issues
    )


@pytest.mark.parametrize(
    "reserved",
    ["bash", "edit", "grep", "read", "skill", "task", "view_file"],
)
def test_reserved_name_collision(write_skill: Callable[..., Path], reserved: str) -> None:
    """Verify that skill names colliding with built-in primitives produce a warning."""
    manifest = (
        write_skill(
            name=reserved,
            description="Execute shell commands and manage background processes.",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest)
    assert any(
        i.rule == "reserved-name-collision" and i.severity == Severity.WARN for i in report.issues
    )


def test_listing_overflow(write_skill: Callable[..., Path]) -> None:
    """Verify that descriptions exceeding listing threshold produce listing-overflow warning."""
    huge_desc = "A" * 1200
    manifest = (
        write_skill(
            name="huge-skill",
            description=huge_desc,
        )
        / "SKILL.md"
    )
    report = lint_file(manifest, config=LintSettings(max_description_length=1000))
    assert any(i.rule == "listing-overflow" and i.severity == Severity.WARN for i in report.issues)


def test_duplicate_name_across_corpus(write_skill: Callable[..., Path], tmp_path: Path) -> None:
    """Verify that duplicate skill names across the tree produce duplicate-name error."""
    write_skill(
        name="shared-name",
        description="First skill implementation with this specific name.",
        root=tmp_path / "repo1",
    )
    write_skill(
        name="shared-name",
        description="Second skill implementation colliding with the first.",
        root=tmp_path / "repo2",
    )
    report = lint_tree(tmp_path)
    assert report.has_errors
    duplicates = [i for i in report.issues if i.rule == "duplicate-name"]
    assert len(duplicates) >= 1
    assert duplicates[0].severity == Severity.ERROR


def test_rule_severity_override_ignore(write_skill: Callable[..., Path]) -> None:
    """Verify that setting rule severity to ignore silences the issue."""
    manifest = (
        write_skill(
            name="terse-skill",
            description="Does stuff.",
        )
        / "SKILL.md"
    )
    config = LintSettings(rules={"description-too-short": Severity.IGNORE})
    report = lint_file(manifest, config=config)
    assert not any(i.rule == "description-too-short" for i in report.issues)


def test_rule_severity_override_elevate_to_error(write_skill: Callable[..., Path]) -> None:
    """Verify that overriding a warning to error promotes its severity."""
    manifest = (
        write_skill(
            name="terse-skill",
            description="Does stuff.",
        )
        / "SKILL.md"
    )
    config = LintSettings(rules={"description-too-short": Severity.ERROR})
    report = lint_file(manifest, config=config)
    issue = next(i for i in report.issues if i.rule == "description-too-short")
    assert issue.severity == Severity.ERROR


def test_explain_rule() -> None:
    """Verify explain_rule returns complete metadata for known rules and None for unknown."""
    rule = explain_rule("description-too-short")
    assert rule is not None
    assert isinstance(rule, RuleDefinition)
    assert rule.rule == "description-too-short"
    assert rule.default_severity == Severity.WARN
    assert rule.summary
    assert rule.explanation
    assert rule.remedy

    assert explain_rule("non-existent-rule") is None


def test_lint_config_from_settings_loads_thresholds_and_rules() -> None:
    """Verify LintSettings.from_settings parses custom thresholds and rule severities."""
    settings = {
        "lint": {
            "max_description_length": 500,
            "max_name_length": 32,
            "min_description_length": 40,
            "rules": {
                "description-too-short": "error",
                "invalid-name-format": "warn",
            },
        },
    }
    config = LintSettings.from_settings(settings)
    assert config.max_description_length == 500
    assert config.max_name_length == 32
    assert config.min_description_length == 40
    assert config.rules["description-too-short"] == Severity.ERROR
    assert config.rules["invalid-name-format"] == Severity.WARN


def test_custom_max_name_length_threshold(write_skill: Callable[..., Path]) -> None:
    """Verify that skill names exceeding a custom max_name_length are rejected."""
    long_name = "this-is-a-moderately-long-name"
    assert len(long_name) == 30
    manifest = (
        write_skill(
            name=long_name,
            description="A sufficiently long description for testing custom max name length.",
        )
        / "SKILL.md"
    )
    # Default 64 chars allows 30 chars
    report_default = lint_file(manifest)
    assert report_default.clean

    # Custom 25 chars rejects 30 chars
    config_custom = LintSettings(max_name_length=25)
    report_custom = lint_file(manifest, config=config_custom)
    assert not report_custom.clean
    issue = next(i for i in report_custom.issues if i.rule == "invalid-name-format")
    assert "max 25 characters" in issue.message


def test_custom_min_description_length_threshold(write_skill: Callable[..., Path]) -> None:
    """Verify that descriptions below a custom min_description_length are flagged."""
    desc = "Twenty-five char desc...."
    assert len(desc) == 25
    manifest = (
        write_skill(
            name="custom-len-skill",
            description=desc,
        )
        / "SKILL.md"
    )
    # Default 20 chars allows 25 chars
    report_default = lint_file(manifest)
    assert not any(i.rule == "description-too-short" for i in report_default.issues)

    # Custom 30 chars flags 25 chars
    config_custom = LintSettings(min_description_length=30)
    report_custom = lint_file(manifest, config=config_custom)
    issue = next(i for i in report_custom.issues if i.rule == "description-too-short")
    assert "30 chars" in issue.message


def test_format_github_annotation_basic() -> None:
    """Verify format_github_annotation formats severity, properties, and message."""
    from reach.rendering import format_github_annotation

    result = format_github_annotation(
        "error",
        "Invalid schema in manifest",
        title="invalid-yaml",
        file="skills/demo/SKILL.md",
        line=1,
    )
    expected = (
        "::error file=skills/demo/SKILL.md,line=1,title=invalid-yaml::Invalid schema in manifest"
    )
    assert result == expected


def test_format_github_annotation_escapes_newlines_and_percent() -> None:
    """Verify format_github_annotation encodes newlines and percent signs in messages."""
    result = format_github_annotation(
        "warn",
        "First line\nSecond line has 50% rate",
    )
    assert result == "::warning::First line%0ASecond line has 50%25 rate"


def test_format_github_annotation_escapes_parameter_delimiters() -> None:
    """Verify format_github_annotation escapes %, CRLF, colons, and commas in parameter values."""
    result = format_github_annotation(
        "error",
        "Found issue",
        title="bad:rule,v100%\r\n::workflow-cmd",
        file="skills/demo:special,v1%0A/SKILL.md",
    )
    assert "title=bad%3Arule%2Cv100%25%0D%0A%3A%3Aworkflow-cmd" in result
    assert "file=skills/demo%3Aspecial%2Cv1%250A/SKILL.md" in result
    # Ensure raw unescaped delimiters do not split parameters or commands
    assert "\r" not in result
    assert "\n" not in result


def test_format_github_annotation_converts_absolute_workspace_path_to_relative(
    tmp_path: Path,
) -> None:
    """Verify format_github_annotation converts absolute file paths under root to relative paths."""
    workspace = tmp_path / "repo"
    skill_file = workspace / "skills" / "demo" / "SKILL.md"
    result = format_github_annotation(
        "error",
        "Invalid schema",
        file=skill_file,
        root=workspace,
    )
    assert "file=skills/demo/SKILL.md" in result
    assert str(skill_file) not in result


def test_format_github_annotation_converts_cwd_path_to_relative() -> None:
    """Verify format_github_annotation relativizes paths under current working directory."""
    from reach.rendering import format_github_annotation

    abs_path = Path.cwd() / "skills" / "demo" / "SKILL.md"
    result = format_github_annotation(
        "error",
        "Invalid schema",
        file=abs_path,
    )
    assert "file=skills/demo/SKILL.md" in result


def test_render_lint_github_formats_all_issues(write_skill: Callable[..., Path]) -> None:
    """Verify render_lint_github formats all report issues as workflow commands."""
    from reach.lint import lint_file
    from reach.views.lint import render_lint_github

    manifest = (
        write_skill(
            name="bad_naming",
            description="A short description for testing github workflow formatting.",
        )
        / "SKILL.md"
    )
    report = lint_file(manifest)
    rendered = render_lint_github(report)

    assert "::error" in rendered
    assert "title=invalid-name-format" in rendered
    assert str(manifest) in rendered


def test_lint_tree_flags_near_duplicate_capability(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify lint_tree identifies semantically redundant skill descriptions."""
    from unittest.mock import patch

    from reach.retrieval import DenseScorer

    write_skill(
        name="pdf-parser",
        description="Extract structured tables from PDF files into spreadsheets.",
    )
    write_skill(
        name="pdf-extractor",
        description="Extract tabular information from PDF documents into spreadsheets.",
    )
    write_skill(
        name="image-editor",
        description="Crop and resize bitmap images and photographs.",
    )

    mock_vectors = {
        "pdf-parser": [1.0, 0.95, 0.0],
        "pdf-extractor": [0.99, 0.94, 0.0],
        "image-editor": [0.0, 0.0, 1.0],
    }
    with patch.object(DenseScorer, "from_skills", return_value=DenseScorer(vectors=mock_vectors)):
        report = lint_tree(tmp_path)
        dup_issues = [i for i in report.issues if i.rule == "duplicate-capability"]
        assert len(dup_issues) >= 1
        assert "pdf-parser" in dup_issues[0].message
        assert "pdf-extractor" in dup_issues[0].message


def test_unresolved_declared_dependency_warns_when_missing_from_corpus(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that a declared dependency missing from the corpus produces a warning."""
    write_skill(
        name="caller-skill",
        description="A skill that requires an absent dependency.",
        raw_yaml=(
            "---\n"
            "name: caller-skill\n"
            "description: A skill that requires an absent dependency.\n"
            "metadata:\n"
            "  requires_skill: missing-helper\n"
            "---\n"
        ),
    )
    report = lint_tree(tmp_path)
    issue = next((i for i in report.issues if i.rule == "unresolved-declared-dependency"), None)
    assert issue is not None
    assert issue.severity == Severity.WARN
    assert "missing-helper" in issue.message


def test_unresolved_declared_dependency_passes_when_present(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that a declared dependency present in the corpus passes cleanly."""
    write_skill(
        name="caller-skill",
        description="A skill that requires a present dependency.",
        raw_yaml=(
            "---\n"
            "name: caller-skill\n"
            "description: A skill that requires a present dependency.\n"
            "metadata:\n"
            "  requires_skill: helper-skill\n"
            "---\n"
        ),
    )
    write_skill(
        name="helper-skill",
        description="The helper skill present in the corpus.",
    )
    report = lint_tree(tmp_path)
    dep_issues = [i for i in report.issues if i.rule == "unresolved-declared-dependency"]
    assert len(dep_issues) == 0


def test_lockfile_drift_warns_on_hash_mismatch(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that lockfile-drift emits a warning when SKILL.md hash diverges from lockfile."""
    skill_dir = write_skill(
        name="drift-skill",
        description="A skill whose content diverges from skills-lock.json.",
        root=tmp_path / ".agents" / "skills",
    )
    manifest = skill_dir / "SKILL.md"
    lockfile = tmp_path / "skills-lock.json"
    lockfile.write_text(
        '{"version": 1, "skills": {"drift-skill": {"computedHash": "expected-old-hash"}}}',
        encoding="utf-8",
    )
    report = lint_file(manifest)
    issue = next((i for i in report.issues if i.rule == "lockfile-drift"), None)
    assert issue is not None
    assert issue.severity == Severity.WARN
    assert "drift-skill" in issue.message


def test_lockfile_clean_when_hash_matches(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify that lockfile-drift does not fire when SKILL.md hash matches lockfile."""
    import hashlib

    skill_dir = write_skill(
        name="matching-skill",
        description="A skill whose content matches skills-lock.json perfectly.",
        root=tmp_path / ".agents" / "skills",
    )
    manifest = skill_dir / "SKILL.md"
    content_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    lockfile = tmp_path / "skills-lock.json"
    lockfile.write_text(
        f'{{"version": 1, "skills": {{"matching-skill": {{"computedHash": "{content_hash}"}}}}}}',
        encoding="utf-8",
    )
    report = lint_file(manifest)
    drift_issues = [i for i in report.issues if i.rule == "lockfile-drift"]
    assert len(drift_issues) == 0


def test_lint_settings_from_settings() -> None:
    """Verify LintSettings.from_settings parses settings and overrides correctly."""
    from reach.config import LintSettings

    cfg = LintSettings.from_settings(
        settings={
            "lint": {"max_name_length": 32, "rules": {"no-description": "error"}},
            "retrieval": {"similarity_threshold": 0.85},
        },
        overrides={"kebab-case-name": Severity.WARN},
    )
    assert cfg.max_name_length == 32
    assert cfg.similarity_threshold == 0.85
    assert cfg.rules["no-description"] == "error"
    assert cfg.rules["kebab-case-name"] == Severity.WARN
