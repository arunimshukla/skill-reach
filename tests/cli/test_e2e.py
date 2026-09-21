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

"""Verify end-to-end execution of all Reach CLI commands and parameters on a synthetic catalog."""

from __future__ import annotations

import csv
import io
import json
from importlib import metadata
from typing import TYPE_CHECKING

import pytest

from reach.cli import main
from reach.queries import load_query_set

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def synthetic_two_arms(
    synthetic_skills_repo: Path,
    synthetic_query_file: Path,
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Produce two comparison evaluation arms under different catalog sizes for diff tests."""
    control = tmp_path / "control.jsonl"
    treatment = tmp_path / "treatment.jsonl"

    for out, size, rivals in ((control, "3", "2"), (treatment, "5", "4")):
        status = main(
            [
                "eval",
                "--skills",
                str(synthetic_skills_repo),
                "--queries",
                str(synthetic_query_file),
                "--workdir",
                str(tmp_path / f"work_{size}"),
                "--out",
                str(out),
                "--agent",
                "fake",
                "--mode",
                "neighborhood",
                "--catalog",
                "neighborhood:cloud-run-basics",
                "--catalog-size",
                size,
                "--rivals",
                rivals,
                "--attempts",
                "1",
                "--partial",
                "--quiet",
            ],
        )
        assert status == 0, f"Failed producing arm at size {size}"

    return control, treatment


# ---------------------------------------------------------------------------
# Top-Level CLI Verbs & Help
# ---------------------------------------------------------------------------


def test_top_level_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify top-level --help displays available verbs and exits with 0."""
    status = main(["--help"])
    assert status == 0
    captured = capsys.readouterr().out
    for verb in ("overlap", "eval", "query", "diff", "view"):
        assert verb in captured


def test_top_level_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify top-level --version prints application version and exits with 0."""
    status = main(["--version"])
    assert status == 0
    assert metadata.version("skill-reach") in capsys.readouterr().out


def test_unknown_command_reports_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify unrecognized verb displays error to stderr and exits with code 2."""
    status = main(["invalid-command-xyz"])
    assert status == 2
    err = capsys.readouterr().err
    assert "unknown command" in err


# ---------------------------------------------------------------------------
# Command: reach overlap
# ---------------------------------------------------------------------------


def test_overlap_baseline_text(
    synthetic_skills_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap ranks competitors across the synthetic catalog in text."""
    status = main(["overlap", "--skills", str(synthetic_skills_repo)])
    assert status == 0
    err = capsys.readouterr().err
    assert "Description overlap reflects lexical similarity" in err


def test_overlap_json_format(
    synthetic_skills_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap --format json outputs valid corpus analysis payload."""
    status = main(["overlap", "--skills", str(synthetic_skills_repo), "--format", "json"])
    assert status == 0
    data = json.loads(capsys.readouterr().out)
    assert data["corpus_size"] == 6
    assert "skills" in data
    assert len(data["skills"]) == 6


def test_overlap_csv_format(
    synthetic_skills_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap --format csv outputs parsable CSV rows."""
    status = main(["overlap", "--skills", str(synthetic_skills_repo), "--format", "csv"])
    assert status == 0
    reader = csv.DictReader(io.StringIO(capsys.readouterr().out))
    rows = list(reader)
    assert len(rows) == 6
    assert "skill" in rows[0]
    assert "nearest_rival" in rows[0]


def test_overlap_single_and_multiple_skill_filters(
    synthetic_skills_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap with single and repeatable --skill flags."""
    status = main(
        [
            "overlap",
            "--skills",
            str(synthetic_skills_repo),
            "--skill",
            "cloud-run-basics",
            "--skill",
            "gke-basics",
        ],
    )
    assert status == 0
    err = capsys.readouterr().err
    assert "cloud-run-basics" in err
    assert "gke-basics" in err


def test_overlap_suggest_rewrites(
    synthetic_skills_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap --suggest generates rewrite recommendations for overlapping skills."""
    status = main(
        [
            "overlap",
            "--skills",
            str(synthetic_skills_repo),
            "--skill",
            "gke-basics",
            "--suggest",
        ],
    )
    assert status == 0
    err = capsys.readouterr().err
    assert "gke-basics" in err


def test_overlap_suggest_without_skill_fails(
    synthetic_skills_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap --suggest without --skill exits with code 2."""
    status = main(["overlap", "--skills", str(synthetic_skills_repo), "--suggest"])
    assert status == 2
    assert "--suggest needs --skill" in capsys.readouterr().err


def test_overlap_nonexistent_skill_fails(
    synthetic_skills_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap with an unknown skill name exits with code 2."""
    status = main(
        [
            "overlap",
            "--skills",
            str(synthetic_skills_repo),
            "--skill",
            "nonexistent-skill",
        ],
    )
    assert status == 2
    assert "no skill named 'nonexistent-skill'" in capsys.readouterr().err


def test_overlap_agent_fake(
    synthetic_skills_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach overlap accepts alternate agent flag."""
    status = main(
        [
            "overlap",
            "--skills",
            str(synthetic_skills_repo),
            "--agent",
            "fake",
            "--skill",
            "cloud-run-basics",
        ],
    )
    assert status == 0


# ---------------------------------------------------------------------------
# Command: reach query (draft, export, import, view)
# ---------------------------------------------------------------------------


def test_query_draft_dry_run(
    synthetic_skills_repo: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach query draft with --dry-run previews drafting on synthetic skills."""
    destination = tmp_path / "drafted.json"
    status = main(
        [
            "query",
            "draft",
            "--skills",
            str(synthetic_skills_repo),
            "--queries",
            str(destination),
            "--skill",
            "cloud-run-basics",
            "--count",
            "2",
            "--generator-model",
            "sonnet",
            "--top-rivals",
            "3",
            "--dry-run",
        ],
    )
    assert status == 0
    err = capsys.readouterr().err
    assert "would write" in err
    assert not destination.exists()


def test_query_draft_run_dir_and_quiet(synthetic_skills_repo: Path, tmp_path: Path) -> None:
    """Verify reach query draft accepts --run-dir and mutes output with --quiet."""
    run_dir = tmp_path / "run_scope"
    status = main(
        [
            "query",
            "draft",
            "--skills",
            str(synthetic_skills_repo),
            "--run-dir",
            str(run_dir),
            "--skill",
            "cloud-run-basics",
            "--count",
            "1",
            "--generator-model",
            "sonnet",
            "--top-rivals",
            "2",
            "--dry-run",
            "--quiet",
        ],
    )
    assert status == 0


def test_query_draft_refuses_to_overwrite_existing(
    synthetic_skills_repo: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach query draft fails fast when target query file already exists."""
    dest = tmp_path / "already_exists.json"
    dest.write_text("{}", encoding="utf-8")

    status = main(
        [
            "query",
            "draft",
            "--skills",
            str(synthetic_skills_repo),
            "--queries",
            str(dest),
            "--skill",
            "cloud-run-basics",
            "--generator-model",
            "sonnet",
            "--dry-run",
        ],
    )
    assert status == 2
    assert "Query set already exists" in capsys.readouterr().err


def test_query_export_formats(
    synthetic_query_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify query export to CSV, JSONL, and stdout formats."""
    csv_out = tmp_path / "exported.csv"
    jsonl_out = tmp_path / "exported.jsonl"

    assert main(["query", str(synthetic_query_file), "--out", str(csv_out)]) == 0
    assert csv_out.exists()

    assert main(["query", str(synthetic_query_file), "--out", str(jsonl_out)]) == 0
    assert jsonl_out.exists()

    assert main(["query", str(synthetic_query_file), "--format", "csv"]) == 0
    out_text = capsys.readouterr().out
    assert "How do I deploy a containerized service to Cloud Run?" in out_text


def _assert_imported_queryset(
    path: Path,
    expected_catalog: str,
    expected_notes: str | None = None,
) -> None:
    """Assert imported query set has expected length, catalog, and optional notes."""
    qs = load_query_set(path)
    assert len(qs.queries) == 2
    assert qs.catalog_id == expected_catalog
    if expected_notes:
        assert qs.notes == expected_notes


def test_query_import_roundtrip(synthetic_query_file: Path, tmp_path: Path) -> None:
    """Verify re-importing CSV and JSONL files into JSON query sets with validation."""
    csv_out = tmp_path / "exported.csv"
    jsonl_out = tmp_path / "exported.jsonl"
    assert main(["query", str(synthetic_query_file), "--out", str(csv_out)]) == 0
    assert main(["query", str(synthetic_query_file), "--out", str(jsonl_out)]) == 0

    imported_csv = tmp_path / "imported_from_csv.json"
    assert (
        main(
            [
                "query",
                str(csv_out),
                "--out",
                str(imported_csv),
                "--catalog",
                "synthetic-catalog",
                "--notes",
                "Imported from CSV round-trip",
            ],
        )
        == 0
    )
    _assert_imported_queryset(imported_csv, "synthetic-catalog", "Imported from CSV round-trip")

    imported_jsonl = tmp_path / "imported_from_jsonl.json"
    assert (
        main(
            [
                "query",
                str(jsonl_out),
                "--out",
                str(imported_jsonl),
                "--catalog",
                "synthetic-catalog",
                "--id-prefix",
                "reimported-",
            ],
        )
        == 0
    )
    _assert_imported_queryset(imported_jsonl, "synthetic-catalog")

    # Overwrite protection on import
    assert (
        main(
            [
                "query",
                str(csv_out),
                "--out",
                str(imported_csv),
                "--catalog",
                "synthetic-catalog",
            ],
        )
        == 2
    )


def test_query_view_and_leak_detection(
    synthetic_skills_repo: Path,
    synthetic_query_file: Path,
    synthetic_citations_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach query view displays lexical rankings, leaks, and citations."""
    status = main(
        [
            "query",
            "view",
            str(synthetic_query_file),
            "--skills",
            str(synthetic_skills_repo),
            "--leaks",
            "--citations",
        ],
    )
    assert status == 0
    err = capsys.readouterr().err
    assert "cloud-run-basics" in err
    assert "rank" in err
    assert "leak" in err
    assert "citation" in err


# ---------------------------------------------------------------------------
# Command: reach eval (quick and formal)
# ---------------------------------------------------------------------------


def test_eval_quick_dry_run_named_skill(
    synthetic_skills_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify quick eval dry run on a target synthetic skill."""
    status = main(
        [
            "eval",
            "cloud-run-basics",
            "--skills",
            str(synthetic_skills_repo),
            "--generator-model",
            "sonnet",
            "--dry-run",
        ],
    )
    assert status == 0
    err = capsys.readouterr().err
    assert "quick" in err
    assert "cloud-run-basics" in err


def test_eval_quick_dry_run_adhoc_query(
    synthetic_skills_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify quick eval dry run with explicit --query and --expected flags."""
    status = main(
        [
            "eval",
            "--skills",
            str(synthetic_skills_repo),
            "--query",
            "Deploy container to Cloud Run",
            "--expected",
            "cloud-run-basics",
            "--dry-run",
        ],
    )
    assert status == 0
    err = capsys.readouterr().err
    assert "ground truth" in err


def test_eval_quick_fake_probe_and_save(
    synthetic_skills_repo: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify quick eval execution with fake agent and artifact preservation via --save."""
    save_dir = tmp_path / "saved_run"
    status = main(
        [
            "eval",
            "--skills",
            str(synthetic_skills_repo),
            "--query",
            "Deploy container to Cloud Run",
            "--expected",
            "cloud-run-basics",
            "--agent",
            "fake",
            "--save",
            str(save_dir),
        ],
    )
    assert status == 0
    assert (save_dir / "queries.json").exists()
    assert (save_dir / "queries.json.artifact.json").exists()
    err = capsys.readouterr().err
    assert "cloud-run-basics" in err


def test_eval_formal_dry_run(
    synthetic_skills_repo: Path,
    synthetic_query_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify formal eval dry run prints configuration plan without probing."""
    status = main(
        [
            "eval",
            "--skills",
            str(synthetic_skills_repo),
            "--queries",
            str(synthetic_query_file),
            "--workdir",
            str(tmp_path / "formal_work"),
            "--out",
            str(tmp_path / "formal.jsonl"),
            "--agent",
            "fake",
            "--mode",
            "neighborhood",
            "--catalog",
            "neighborhood:cloud-run-basics",
            "--catalog-size",
            "3",
            "--rivals",
            "2",
            "--partial",
            "--rescope",
            "--dry-run",
        ],
    )
    assert status == 0
    assert "fake/fake-model" in capsys.readouterr().err


def test_eval_formal_execution_with_flags(
    synthetic_skills_repo: Path,
    synthetic_query_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify formal eval execution with concurrency, tags, format, and fake agent."""
    out_file = tmp_path / "formal_exec.jsonl"
    status = main(
        [
            "eval",
            "--skills",
            str(synthetic_skills_repo),
            "--queries",
            str(synthetic_query_file),
            "--workdir",
            str(tmp_path / "formal_exec_work"),
            "--out",
            str(out_file),
            "--agent",
            "fake",
            "--mode",
            "neighborhood",
            "--catalog",
            "neighborhood:cloud-run-basics",
            "--catalog-size",
            "3",
            "--rivals",
            "2",
            "--partial",
            "--rescope",
            "--attempts",
            "1",
            "--workers",
            "2",
            "--tag",
            "e2e-synthetic-test",
            "--format",
            "json",
            "--quiet",
        ],
    )
    assert status == 0
    assert out_file.exists()
    assert (tmp_path / "formal_exec.jsonl.artifact.json").exists()
    assert (tmp_path / "formal_exec.jsonl.config.json").exists()

    # Verify JSON output was emitted to stdout
    payload = json.loads(capsys.readouterr().out)
    assert payload["catalog_id"] == "neighborhood:cloud-run-basics"


def test_eval_save_without_quick_eval_fails(
    synthetic_skills_repo: Path,
    synthetic_query_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify using --save in formal eval mode exits with code 2."""
    status = main(
        [
            "eval",
            "--skills",
            str(synthetic_skills_repo),
            "--queries",
            str(synthetic_query_file),
            "--workdir",
            str(tmp_path / "work"),
            "--out",
            str(tmp_path / "out.jsonl"),
            "--save",
            str(tmp_path / "saved"),
        ],
    )
    assert status == 2
    assert "--save promotes a quick run's draft" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Command: reach diff
# ---------------------------------------------------------------------------


def test_diff_e2e_between_arms(
    synthetic_two_arms: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach diff computes delta and noise floor across two synthetic arms."""
    control, treatment = synthetic_two_arms
    status = main(
        [
            "diff",
            str(control),
            str(treatment),
            "--vary",
            "scope",
            "--control-label",
            "BaseScope",
            "--treatment-label",
            "ExpandedScope",
            "--confidence",
            "0.90",
            "--noise-inflation",
            "1.2",
        ],
    )
    assert status == 0
    out = capsys.readouterr().out
    assert "diff --vary scope" in out
    assert "BaseScope" in out
    assert "ExpandedScope" in out
    assert "noise floor" in out


def test_diff_formats_json_and_csv(
    synthetic_two_arms: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach diff JSON and CSV output formats."""
    control, treatment = synthetic_two_arms

    # JSON format
    status_json = main(
        ["diff", str(control), str(treatment), "--vary", "scope", "--format", "json"],
    )
    assert status_json == 0
    parsed_json = json.loads(capsys.readouterr().out)
    assert parsed_json["factor"] == "scope"
    assert "headline" in parsed_json

    # CSV format
    status_csv = main(["diff", str(control), str(treatment), "--vary", "scope", "--format", "csv"])
    assert status_csv == 0
    csv_rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(csv_rows) > 0
    assert "top1_accuracy" in [r["figure"] for r in csv_rows]


def test_diff_missing_vary_fails(
    synthetic_two_arms: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach diff without --vary exits with code 2."""
    control, treatment = synthetic_two_arms
    status = main(["diff", str(control), str(treatment)])
    assert status == 2


# ---------------------------------------------------------------------------
# Command: reach view
# ---------------------------------------------------------------------------


def test_view_scorecard_and_html(
    synthetic_skills_repo: Path,
    synthetic_query_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify reach view renders scorecard, query records, verbose digests, and HTML."""
    out_file = tmp_path / "view_test.jsonl"
    status_eval = main(
        [
            "eval",
            "--skills",
            str(synthetic_skills_repo),
            "--queries",
            str(synthetic_query_file),
            "--workdir",
            str(tmp_path / "view_work"),
            "--out",
            str(out_file),
            "--agent",
            "fake",
            "--mode",
            "neighborhood",
            "--catalog",
            "neighborhood:cloud-run-basics",
            "--catalog-size",
            "3",
            "--rivals",
            "2",
            "--partial",
            "--rescope",
            "--attempts",
            "1",
            "--quiet",
        ],
    )
    assert status_eval == 0
    artifact_file = tmp_path / "view_test.jsonl.artifact.json"
    assert artifact_file.exists()

    # View standard text scorecard with query records and verbose digests
    status_text = main(["view", str(artifact_file), "--show-queries", "--verbose"])
    assert status_text == 0
    err_text = capsys.readouterr().err
    assert "cloud-run-basics" in err_text
    assert "fake/fake-model" in err_text

    # View HTML format
    status_html = main(["view", str(artifact_file), "--format", "html"])
    assert status_html == 0
    out_html = capsys.readouterr().out
    assert "<!DOCTYPE html>" in out_html
    assert "cloud-run-basics" in out_html


def test_view_invalid_artifact_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify reach view on non-artifact file reports validation error and exits with 2."""
    bad_file = tmp_path / "not_an_artifact.json"
    bad_file.write_text('{"random": "content"}', encoding="utf-8")
    status = main(["view", str(bad_file)])
    assert status == 2
    assert "Cannot read artifact" in capsys.readouterr().err
