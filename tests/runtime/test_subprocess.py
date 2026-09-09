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

"""Verify subprocess probing, streaming line capture, early exit, and timeout handling."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Any

from reach.runtime._subprocess import _ProcessGroupController, run_subprocess_probe

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_run_subprocess_probe_real_execution(tmp_path: Path) -> None:
    """Verify live subprocess probe captures full stdout and returncode on success."""
    cmd = [sys.executable, "-c", "print('hello'); print('world')"]
    completed, err = run_subprocess_probe(cmd, tmp_path)

    assert err is None
    assert completed is not None
    assert completed.returncode == 0
    assert completed.stdout == "hello\nworld\n"


def test_run_subprocess_probe_real_early_exit(tmp_path: Path) -> None:
    """Verify on_line predicate early-terminates process when target line appears."""
    script = (
        "import sys, time\n"
        "print('line1', flush=True)\n"
        "print('STOP_HERE', flush=True)\n"
        "time.sleep(2)\n"
        "print('line3', flush=True)\n"
    )
    cmd = [sys.executable, "-c", script]

    def _predicate(line: str) -> bool:
        return "STOP_HERE" in line

    completed, err = run_subprocess_probe(cmd, tmp_path, timeout_s=5.0, on_line=_predicate)

    assert err is None
    assert completed is not None
    assert "line1" in completed.stdout
    assert "STOP_HERE" in completed.stdout
    assert "line3" not in completed.stdout


def test_run_subprocess_probe_real_timeout(tmp_path: Path) -> None:
    """Verify long-running subprocess exceeds timeout during streaming and returns timeout."""
    cmd = [
        sys.executable,
        "-c",
        "import time; [print('ping', flush=True) or time.sleep(0.01) for _ in range(50)]",
    ]
    completed, err = run_subprocess_probe(cmd, tmp_path, timeout_s=0.05)

    assert completed is None
    assert err == "timeout"


def test_run_subprocess_probe_silent_command_timeout(tmp_path: Path) -> None:
    """Verify silent long-running command exceeding timeout returns timeout."""
    cmd = [sys.executable, "-c", "import time; time.sleep(0.08)"]
    completed, err = run_subprocess_probe(cmd, tmp_path, timeout_s=0.02)

    assert completed is None
    assert err == "timeout"


def test_run_subprocess_probe_real_spawn_failure(tmp_path: Path) -> None:
    """Verify invalid executable path gracefully returns spawn failure error."""
    cmd = ["/path/to/nonexistent/reach_executable_binary_probe"]
    completed, err = run_subprocess_probe(cmd, tmp_path)

    assert completed is None
    assert err is not None
    assert "failed to spawn process" in err


def test_run_subprocess_probe_mock_mode_happy_path(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify monkeypatched subprocess.run is handled cleanly via mock path."""
    mock_subprocess(stdout="mock line 1\nmock line 2\n")
    completed, err = run_subprocess_probe(["dummy"], tmp_path)

    assert err is None
    assert completed is not None
    assert completed.stdout == "mock line 1\nmock line 2\n"


def test_run_subprocess_probe_mock_mode_early_exit(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify monkeypatched subprocess.run applies on_line filtering and truncation."""
    mock_subprocess(stdout="line 1\nMATCH_LINE\nline 3\n")

    def _predicate(line: str) -> bool:
        return "MATCH_LINE" in line

    completed, err = run_subprocess_probe(["dummy"], tmp_path, on_line=_predicate)

    assert err is None
    assert completed is not None
    assert completed.stdout == "line 1\nMATCH_LINE\n"


def test_run_subprocess_probe_mock_mode_timeout(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify monkeypatched subprocess.run timeout returns timeout error."""
    mock_subprocess(side_effect=subprocess.TimeoutExpired(cmd="dummy", timeout=1))
    completed, err = run_subprocess_probe(["dummy"], tmp_path)

    assert completed is None
    assert err == "timeout"


def test_run_subprocess_probe_mock_mode_spawn_error(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify monkeypatched subprocess.run OSError returns spawn error."""
    mock_subprocess(side_effect=FileNotFoundError("missing executable"))
    completed, err = run_subprocess_probe(["dummy"], tmp_path)

    assert completed is None
    assert err is not None
    assert "failed to spawn process: missing executable" in err


def test_process_group_controller_lifecycle(tmp_path: Path) -> None:
    """Verify ProcessGroupController gracefully terminates and kills spawned processes."""
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    proc = subprocess.Popen(
        cmd,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        controller = _ProcessGroupController(proc)

        # Terminate gracefully
        controller.terminate_gracefully(wait_timeout=0.5)
        assert proc.poll() is not None

        # Redundant signal calls on reaped process should not raise
        controller.send_signal(15)
        controller.kill()
    finally:
        if proc.stdout and not proc.stdout.closed:
            proc.stdout.close()
        if proc.stderr and not proc.stderr.closed:
            proc.stderr.close()
