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

"""Validate unbounded-attractor lint rule against greedy and bounded frontmatter descriptions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from reach.config import LintSettings
from reach.lint import Severity, explain_rule, lint_file

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "greedy_desc",
    [
        "Assist with any coding task and debug issues.",
        "A tool to help with any programming problem.",
        "Manage files and run commands in the terminal.",
        "General-purpose developer assistant.",
        "Universal assistant for your workflows.",
        "All-in-one helper for software engineering.",
        "Handle any request given by the user.",
    ],
)
def test_unbounded_attractor_flags_greedy_descriptions(tmp_path: Path, greedy_desc: str) -> None:
    """Verify unbounded-attractor rule flags descriptions with greedy universal triggers."""
    skill_dir = tmp_path / "greedy-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: greedy-skill\ndescription: {greedy_desc}\n---\n# Body\n",
        encoding="utf-8",
    )

    report = lint_file(skill_file, LintSettings())
    attractor_issues = [i for i in report.issues if i.rule == "unbounded-attractor"]
    assert len(attractor_issues) == 1
    assert attractor_issues[0].severity == Severity.WARN
    assert "greedy-skill" in attractor_issues[0].skill


@pytest.mark.parametrize(
    "bounded_desc",
    [
        "Deploy containerized microservices to Google Cloud Run with gcloud.",
        "Parse and validate JSON schemas against Draft 7 specifications.",
        "Generate Terraform templates for Google Kubernetes Engine clusters.",
        "Format Python code using the Black code formatter.",
    ],
)
def test_unbounded_attractor_permits_bounded_descriptions(
    tmp_path: Path, bounded_desc: str
) -> None:
    """Verify unbounded-attractor rule passes domain-specific bounded descriptions."""
    skill_dir = tmp_path / "bounded-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: bounded-skill\ndescription: {bounded_desc}\n---\n# Body\n",
        encoding="utf-8",
    )

    report = lint_file(skill_file, LintSettings())
    attractor_issues = [i for i in report.issues if i.rule == "unbounded-attractor"]
    assert len(attractor_issues) == 0


def test_explain_unbounded_attractor_rule() -> None:
    """Verify explain_rule returns definition and remediation for unbounded-attractor."""
    definition = explain_rule("unbounded-attractor")
    assert definition is not None
    assert definition.rule == "unbounded-attractor"
    assert definition.default_severity == Severity.WARN
    assert "remedy" in definition.remedy.lower() or len(definition.remedy) > 0
