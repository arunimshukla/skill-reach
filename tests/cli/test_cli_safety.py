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

"""Verify CLI command-level safety confirmation wiring, options, and bypass behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from reach.check import CheckStage, run_check
from reach.cli.app import app
from reach.config import RunConfig, StudySettings


def test_yes_flag_registered_on_target_commands() -> None:
    """Verify --yes and -y flags are registered on eval, sweep, check, and optimize."""
    for cmd_name in ("eval", "sweep", "check", "optimize"):
        cmd = app[cmd_name]
        arg_names = {
            name
            for arg in cmd.assemble_argument_collection()
            if arg.parameter.name
            for name in arg.parameter.name
        }
        assert "--yes" in arg_names, f"--yes missing from 'reach {cmd_name}'"
        assert "-y" in arg_names, f"-y missing from 'reach {cmd_name}'"


def test_fingerprint_unaffected_by_trusted_setting() -> None:
    """Verify toggling trusted=True does not change the configuration digest fingerprint."""
    base = RunConfig(study=StudySettings(tag="test", trusted=False))
    trusted = RunConfig(study=StudySettings(tag="test", trusted=True))
    assert base.fingerprint == trusted.fingerprint


def test_check_fails_fast_on_static_lint_before_prompting(tmp_path: Path) -> None:
    """Verify check exits with lint failure before reaching empirical safety confirmation."""
    bad_skill = tmp_path / "skills" / "bad-skill"
    bad_skill.mkdir(parents=True)
    # Write invalid frontmatter to trigger Stage 1 static lint error
    (bad_skill / "SKILL.md").write_text("Invalid content without frontmatter", encoding="utf-8")

    queries_file = tmp_path / "queries.json"
    queries_file.write_text(
        '{"catalog_id":"cat","provenance":{"origin":"authored","tool_version":"0.1.0"},"queries":[{"id":"q1","text":"query","expected_skill":"bad-skill"}]}',
        encoding="utf-8",
    )

    # Even in non-TTY with a live agent ('pi'), Stage 1 must fail fast with code 1
    # without failing closed on the confirmation gate (code 2)
    with (
        patch("sys.stdin.isatty", return_value=False),
        patch("sys.stdout.isatty", return_value=False),
    ):
        outcome = run_check(
            skills_paths=[bad_skill.parent],
            queries_path=queries_file,
            agent="pi",
            strict=True,
            yes=False,
        )

    assert outcome.exit_code == 1
    assert outcome.stage_failed == CheckStage.STATIC
