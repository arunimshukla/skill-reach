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

"""Verify live agent runtime preflight, CLI execution, sandbox isolation, and skill discovery."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest

from reach.catalog import load_skills
from reach.cli.doctor import (
    _check_cli_binary,
    _check_google_adc,
    _check_python,
    _check_skills,
)
from reach.models import Catalog, CatalogMode
from reach.runtime import CliAgentRuntime, RuntimeSettings, build_runtime
from reach.runtime._subprocess import run_subprocess_probe
from reach.runtime.antigravity_sdk import _HAS_ANTIGRAVITY, AntigravitySdkRuntime
from reach.runtime.claude_code import ClaudeCodeRuntime

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

#: Registered CLI agent runtime names and their primary command-line binary names.
CLI_AGENTS: tuple[tuple[str, str], ...] = (
    ("claude-code", "claude"),
    ("goose", "goose"),
    ("pi", "pi"),
    ("antigravity-cli", "antigravity"),
)


@pytest.mark.parametrize(("agent_name", "executable"), CLI_AGENTS)
def test_agent_binary_isolated_execution(
    agent_name: str,
    executable: str,
    tmp_path: Path,
) -> None:
    """Verify agent binary executes under Reach's isolated environment without consuming tokens."""
    if not shutil.which(executable):
        pytest.skip(f"Agent binary {executable!r} not installed in PATH")

    runtime = build_runtime(RuntimeSettings(agent=agent_name))
    assert isinstance(runtime, CliAgentRuntime)

    workdir = tmp_path / f"work_{agent_name}"
    workdir.mkdir()

    env = runtime.build_env(workdir)

    # 1. Verify isolated configuration directory is created with private 0o700 permissions
    iso_dir = runtime.effective_isolation_dir(workdir)
    if iso_dir is not None:
        assert iso_dir.is_dir()
        assert (iso_dir.stat().st_mode & 0o777) == 0o700

    # 2. Execute binary --version using Reach's isolated environment
    completed, err = run_subprocess_probe(
        [executable, "--version"],
        workdir=workdir,
        env=env,
        timeout_s=5.0,
    )

    assert err is None, f"Subprocess probe failed with error: {err}"
    assert completed is not None
    assert completed.returncode == 0, f"Command exited {completed.returncode}:\n{completed.stderr}"
    assert len(completed.stdout.strip()) > 0, "Expected version string in stdout"

    # 3. Verify clean post-probe teardown
    runtime.post_probe(workdir)


@pytest.mark.parametrize(("agent_name", "executable"), CLI_AGENTS)
def test_agent_cli_help_and_flag_compatibility(
    agent_name: str,
    executable: str,
    tmp_path: Path,
) -> None:
    """Verify agent binary accepts CLI help flags under Reach's execution environment."""
    if not shutil.which(executable):
        pytest.skip(f"Agent binary {executable!r} not installed in PATH")

    runtime = build_runtime(RuntimeSettings(agent=agent_name))
    workdir = tmp_path / f"work_{agent_name}"
    workdir.mkdir()
    env = runtime.build_env(workdir)

    completed, err = run_subprocess_probe(
        [executable, "--help"],
        workdir=workdir,
        env=env,
        timeout_s=5.0,
    )

    assert err is None, f"Help probe failed with error: {err}"
    assert completed is not None
    assert completed.returncode == 0
    assert len(completed.stdout.strip()) > 0

    if isinstance(runtime, ClaudeCodeRuntime):
        cmd = runtime.build_command("preflight-query")
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--verbose" in cmd
        assert "--strict-mcp-config" in cmd
        assert "--no-session-persistence" in cmd
        settings_str = runtime.options.settings_json()
        if settings_str is not None:
            parsed = json.loads(settings_str)
            assert isinstance(parsed, dict)


@pytest.mark.parametrize(
    "agent_name",
    ["claude-code", "goose", "pi", "antigravity-cli", "keyword"],
)
def test_agent_workspace_skill_installation_and_discovery(
    agent_name: str,
    synthetic_skills_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify runtime materializes resident skills and discovers them in precedence order."""
    runtime = build_runtime(RuntimeSettings(agent=agent_name))
    workdir = tmp_path / f"work_{agent_name}"
    workdir.mkdir()

    all_skills = load_skills(synthetic_skills_repo)
    assert len(all_skills) >= 2
    subset_names = (all_skills[0].name, all_skills[1].name)
    catalog = Catalog(id="preflight-catalog", mode=CatalogMode.ALL, skills=subset_names)

    installed_workdir = runtime.install(catalog, all_skills, workdir)
    assert installed_workdir.resolve() == workdir.resolve()

    skills_dir = runtime.skills_dir(workdir)
    assert skills_dir.is_dir()
    for name in subset_names:
        skill_file = skills_dir / name / "SKILL.md"
        assert skill_file.is_file()

    roots = runtime.skill_roots(workdir)
    assert any(root.path.resolve() == skills_dir.resolve() for root in roots)

    fit = runtime.fit(catalog, all_skills[:2])
    if runtime.rations_catalog:
        assert fit.allowed > 0
        assert fit.asked > 0
    else:
        assert fit.allowed == 0
        assert fit.truncated == 0


@pytest.mark.parametrize(("agent_name", "executable"), CLI_AGENTS)
def test_agent_environment_isolation_blocks_ambient_secrets(
    agent_name: str,
    executable: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify runtime process environment strips sensitive ambient credentials."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_super_secret_token")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws_secret_key")
    monkeypatch.setenv("STRIPE_API_KEY", "sk_live_stripe_secret")

    runtime = build_runtime(RuntimeSettings(agent=agent_name))
    workdir = tmp_path / f"work_{agent_name}"
    workdir.mkdir()

    env = runtime.build_env(workdir)
    assert "GITHUB_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "STRIPE_API_KEY" not in env


def test_antigravity_sdk_preflight(
    synthetic_skills_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify Antigravity SDK runtime initializes and materializes skills when installed."""
    if not _HAS_ANTIGRAVITY:
        pytest.skip("google-antigravity SDK is not installed")

    runtime = AntigravitySdkRuntime()
    workdir = tmp_path / "work_antigravity_sdk"
    workdir.mkdir()

    all_skills = load_skills(synthetic_skills_repo)
    catalog = Catalog(
        id="sdk-catalog",
        mode=CatalogMode.ALL,
        skills=(all_skills[0].name,),
    )

    runtime.install(catalog, all_skills, workdir)
    assert runtime.skills_dir(workdir).is_dir()
    roots = runtime.skill_roots(workdir)
    assert len(roots) > 0


def test_doctor_diagnostics_live(synthetic_skills_repo: Path) -> None:
    """Verify reach doctor diagnostics report accurate binary, ADC, and skill status."""
    py_check = _check_python()
    assert py_check.status == "ok"

    for _, exe in CLI_AGENTS:
        result = _check_cli_binary(exe, exe, "runtime")
        exe_path = shutil.which(exe)
        if exe_path is not None:
            assert result.status == "ok"
            assert exe_path in result.detail
        else:
            assert result.status == "warn"

    adc_check = _check_google_adc()
    assert adc_check.status in ("ok", "warn")

    skills_check = _check_skills(synthetic_skills_repo.parent)
    assert skills_check.status in ("ok", "warn")
