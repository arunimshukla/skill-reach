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

"""Verify reach optimize CLI command operations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

from reach.cli import main
from reach.optimize import OptimizationCandidate

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest


def test_optimize_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify reach optimize --help displays command options and exits 0."""
    assert main(["optimize", "--help"]) == 0
    captured = capsys.readouterr()
    assert "Usage: reach optimize" in captured.out
    assert "--skill" in captured.out
    assert "--auto-apply" in captured.out
    assert "--budget" in captured.out


def test_optimize_missing_skill_flag_fails() -> None:
    """Verify reach optimize without --skill fails with exit code 2."""
    assert main(["optimize"]) == 2


def test_optimize_budget_less_than_one_fails(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach optimize with --budget 0 fails with exit code 2."""
    write_skill(name="my-tool", description="Valid description.")
    assert main(["optimize", "--skill", "my-tool", "--skills", str(tmp_path), "--budget", "0"]) == 2


def test_optimize_unknown_skill_fails(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach optimize on missing skill fails with exit code 2."""
    write_skill(name="other-tool", description="Other description.")
    assert main(["optimize", "--skill", "missing-tool", "--skills", str(tmp_path)]) == 2


def test_optimize_positional_unknown_skill_fails(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach optimize with positional missing skill fails with exit code 2."""
    write_skill(name="other-tool", description="Other description.")
    assert main(["optimize", "missing-tool", "--skills", str(tmp_path)]) == 2


def test_optimize_text_output(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach optimize runs with default text format and prints candidate scorecard."""
    write_skill(
        name="opt-tool",
        description="Old basic description.",
        body="# Opt Tool\nProvides tokenization and string formatting.",
    )
    query_file = write_queries(target="opt-tool", count=3)

    ret = main(
        [
            "optimize",
            "--skill",
            "opt-tool",
            "--skills",
            str(tmp_path),
            "--queries",
            str(query_file),
            "--agent",
            "fake",
            "--budget",
            "6",
        ],
    )
    assert ret == 0
    captured = capsys.readouterr()
    output = captured.err + captured.out
    assert "Reach Closed-Loop Optimizer: opt-tool" in output
    assert "Candidate Description" in output
    assert "#1" in output


def test_optimize_positional_skill_text_output(
    write_skill: Callable[..., Path],
    write_queries: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach optimize runs with skill provided as a positional argument."""
    write_skill(
        name="opt-tool",
        description="Old basic description.",
        body="# Opt Tool\nProvides tokenization and string formatting.",
    )
    query_file = write_queries(target="opt-tool", count=3)

    ret = main(
        [
            "optimize",
            "opt-tool",
            "--skills",
            str(tmp_path),
            "--queries",
            str(query_file),
            "--agent",
            "fake",
            "--budget",
            "6",
        ],
    )
    assert ret == 0
    captured = capsys.readouterr()
    output = captured.err + captured.out
    assert "Reach Closed-Loop Optimizer: opt-tool" in output
    assert "Candidate Description" in output
    assert "#1" in output


def test_optimize_json_output(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach optimize --format json emits valid machine-readable JSON."""
    write_skill(name="json-tool", description="Initial description.")
    ret = main(
        [
            "optimize",
            "--skill",
            "json-tool",
            "--skills",
            str(tmp_path),
            "--agent",
            "fake",
            "--format",
            "json",
        ],
    )
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["skill_name"] == "json-tool"
    assert "candidates" in data
    assert len(data["candidates"]) >= 1


def test_optimize_diff_output(
    write_skill: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach optimize --format diff emits unified diff against best candidate."""
    write_skill(name="diff-tool", description="Initial description.")
    ret = main(
        [
            "optimize",
            "--skill",
            "diff-tool",
            "--skills",
            str(tmp_path),
            "--agent",
            "fake",
            "--format",
            "diff",
        ],
    )
    assert ret == 0

    captured = capsys.readouterr()
    assert "--- a/diff-tool/SKILL.md" in captured.out
    assert "+++ b/diff-tool/SKILL.md" in captured.out
    assert "-  Initial description." in captured.out


def test_optimize_auto_apply_writes_to_disk(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify reach optimize --auto-apply modifies the SKILL.md file on disk."""
    skill_dir = write_skill(
        name="apply-tool",
        description="Description before optimization.",
        body="# Body\nDistinct content to preserve.",
    )
    manifest = skill_dir / "SKILL.md"

    mock_candidates = [
        OptimizationCandidate(
            description="Optimized candidate description written to disk.",
            rationale="Auto-apply test.",
            recall=1.0,
            delta_recall=0.5,
        ),
    ]

    with patch("reach.optimize.synthesize_candidates", return_value=mock_candidates):
        ret = main(
            [
                "optimize",
                "--skill",
                "apply-tool",
                "--skills",
                str(tmp_path),
                "--auto-apply",
            ],
        )
        assert ret == 0

    content = manifest.read_text(encoding="utf-8")
    assert "Optimized candidate description written to disk." in content
    assert "Distinct content to preserve." in content


def test_optimize_interactive_prompt_yes_applies_candidate(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify interactive TTY prompt applies candidate when user inputs 'y'."""
    skill_dir = write_skill(
        name="prompt-tool",
        description="Original description.",
    )
    manifest = skill_dir / "SKILL.md"

    mock_candidates = [
        OptimizationCandidate(
            description="Interactively applied description.",
            recall=0.9,
            delta_recall=0.3,
        ),
    ]

    with (
        patch("reach.optimize.synthesize_candidates", return_value=mock_candidates),
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", return_value="y"),
    ):
        ret = main(["optimize", "prompt-tool", "--skills", str(tmp_path)])
        assert ret == 0

    assert "Interactively applied description." in manifest.read_text(encoding="utf-8")


def test_optimize_interactive_prompt_no_skips_candidate(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify interactive TTY prompt leaves file unchanged when user inputs 'n'."""
    skill_dir = write_skill(
        name="prompt-tool",
        description="Original description.",
    )
    manifest = skill_dir / "SKILL.md"

    mock_candidates = [
        OptimizationCandidate(
            description="Declined description.",
            recall=0.9,
            delta_recall=0.3,
        ),
    ]

    with (
        patch("reach.optimize.synthesize_candidates", return_value=mock_candidates),
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", return_value="n"),
    ):
        ret = main(["optimize", "prompt-tool", "--skills", str(tmp_path)])
        assert ret == 0

    assert "Original description." in manifest.read_text(encoding="utf-8")


def test_optimize_interactive_prompt_diff_then_yes(
    write_skill: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Verify user can request diff before applying candidate in interactive prompt."""
    skill_dir = write_skill(
        name="prompt-tool",
        description="Original description.",
    )
    manifest = skill_dir / "SKILL.md"

    mock_candidates = [
        OptimizationCandidate(
            description="Diff inspected then applied description.",
            recall=0.9,
            delta_recall=0.3,
        ),
    ]

    inputs = iter(["d", "y"])
    with (
        patch("reach.optimize.synthesize_candidates", return_value=mock_candidates),
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", side_effect=lambda _: next(inputs)),
    ):
        ret = main(["optimize", "prompt-tool", "--skills", str(tmp_path)])
        assert ret == 0

    assert "Diff inspected then applied description." in manifest.read_text(encoding="utf-8")


def test_prompt_interactive_apply_noop_when_best_candidate_is_none(tmp_path: Path) -> None:
    """Verify _prompt_interactive_apply returns cleanly when best_candidate is None."""
    from reach.cli.optimize import _prompt_interactive_apply
    from reach.optimize import OptimizationReport
    from reach.views import build_console

    console = build_console()
    report = OptimizationReport(
        skill_name="test-tool",
        baseline_description="test",
        manifest_path=tmp_path / "SKILL.md",
        candidates=(),
    )
    # Calling this should return immediately without raising AttributeError or prompt
    _prompt_interactive_apply(console, report)
