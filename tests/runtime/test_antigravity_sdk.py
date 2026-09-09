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

"""Verify antigravity-sdk agent isolation, selection schema, and execution."""

from __future__ import annotations

import asyncio
import importlib
import json
from typing import TYPE_CHECKING, Any, Never

import pytest

from reach.config import agent_default_model
from reach.runtime import AntigravityRuntime
from reach.runtime.antigravity_sdk import (
    _HAS_ANTIGRAVITY,
    AntigravitySdkGenerator,
    AntigravitySdkOptions,
    AntigravitySdkRuntime,
    _tool_name,
)

from .conftest import (
    FakeSdkAgent as _FakeAgent,
)
from .conftest import (
    FakeSdkResponse as _FakeResponse,
)
from .conftest import (
    patch_sdk_agent as _fake_agent,
)

if TYPE_CHECKING:
    from pathlib import Path

    from google.antigravity import types as ag_types
else:
    try:
        from google.antigravity import types as ag_types
    except ImportError:
        ag_types = None


@pytest.fixture(autouse=True)
def _require_antigravity(request: pytest.FixtureRequest) -> None:
    """Skip test if google.antigravity is not installed and test requires it."""
    exempt = (
        "test_model_has_default",
        "test_antigravity_sdk_options_effort",
        "test_antigravity_sdk_options_defaults",
        "test_tool_name_passes_a_custom_tool_name_through",
        "missing_dependency",
        "uninstalled",
    )
    if not _HAS_ANTIGRAVITY and not any(ex in request.node.name for ex in exempt):
        pytest.skip("google-antigravity is not installed")


@pytest.fixture
def runtime() -> AntigravitySdkRuntime:
    """Provide an AntigravitySdkRuntime instance configured with test-model."""
    return AntigravitySdkRuntime(options=AntigravitySdkOptions(model="test-model"))


@pytest.fixture
def generator() -> AntigravitySdkGenerator:
    """Provide an AntigravitySdkGenerator instance configured with test-model."""
    return AntigravitySdkGenerator(options=AntigravitySdkOptions(model="test-model"))


def test_model_has_default() -> None:
    """Verify model parameter defaults to configured agent default model."""
    default = agent_default_model("antigravity-sdk")
    assert default is not None
    assert AntigravitySdkOptions().model == default


def test_the_agent_reports_the_configured_model(runtime: AntigravitySdkRuntime) -> None:
    """Verify runtime.model returns the configured model identifier."""
    assert runtime.model == "test-model"


def test_tool_name_reads_the_plain_value_not_the_enum_repr() -> None:
    """Verify _tool_name extracts string value from BuiltinTools enum members."""
    assert _tool_name(ag_types.BuiltinTools.FINISH) == "finish"


def test_tool_name_passes_a_custom_tool_name_through() -> None:
    """Verify _tool_name returns plain string tool names unmodified."""
    assert _tool_name("my_mcp_tool") == "my_mcp_tool"


def test_select_config_builds_for_a_singleton_resident_catalog(
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify _select_config constructs valid JSON response schema for singleton catalog."""
    runtime._resident = ("gke-basics",)
    config = runtime._select_config(tmp_path / "work")
    assert isinstance(config.response_schema, str)
    schema = json.loads(config.response_schema)
    selected = schema["properties"]["selected_skill"]
    assert {"const": "gke-basics", "type": "string"} in selected["anyOf"]


def test_select_config_enables_only_finish(
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify select agent configuration restricts tools exclusively to BuiltinTools.FINISH."""
    runtime._resident = ("a", "b")
    config = runtime._select_config(tmp_path / "work")
    assert config.capabilities.enabled_tools == [ag_types.BuiltinTools.FINISH]
    assert config.capabilities.enable_subagents is False
    assert config.budget_config is not None
    assert config.budget_config.max_model_calls == 3


def test_select_config_names_the_resident_catalog_in_the_schema(
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify generated response schema contains enum of resident skill names."""
    runtime._resident = ("gke-basics", "gcs-lifecycle-rules")
    config = runtime._select_config(tmp_path / "work")
    assert isinstance(config.response_schema, str)
    schema = json.loads(config.response_schema)
    selected = schema["properties"]["selected_skill"]
    enum_values = next(branch["enum"] for branch in selected["anyOf"] if "enum" in branch)
    assert sorted(enum_values) == ["gcs-lifecycle-rules", "gke-basics"]


def test_antigravity_sdk_options_effort() -> None:
    """Verify AntigravitySdkOptions accepts model and optional effort."""
    opts_with_effort = AntigravitySdkOptions(model="gemini-3.7-flash", effort="medium")
    assert opts_with_effort.model == "gemini-3.7-flash"
    assert opts_with_effort.effort == "medium"

    opts_no_effort = AntigravitySdkOptions(model="gemini-3.8-flash")
    assert opts_no_effort.model == "gemini-3.8-flash"
    assert opts_no_effort.effort is None


def test_antigravity_sdk_effective_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify effective_api_key prioritizes options.api_key over environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    runtime = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="gemini-3.7-flash", api_key="option-key"),
    )
    assert runtime.effective_api_key == "option-key"

    runtime_env = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="gemini-3.7-flash"),
    )
    assert runtime_env.effective_api_key == "env-key"


def test_select_config_sets_thinking_config(tmp_path: Path) -> None:
    """Verify _select_config sets thinking_config when effort is specified."""
    runtime = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="gemini-3.7-flash", effort="high"),
    )
    runtime._resident = ("skill-a",)
    config = runtime._select_config(tmp_path)
    assert isinstance(config.model, ag_types.ModelTarget)
    assert config.model.name == "gemini-3.7-flash"
    assert isinstance(config.model.endpoint, ag_types.GeminiAPIEndpoint)
    assert config.model.endpoint.options is not None
    assert config.model.endpoint.options.thinking_level == ag_types.ThinkingLevel.HIGH

    runtime_default = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="gemini-3.7-flash"),
    )
    runtime_default._resident = ("skill-a",)
    config_default = runtime_default._select_config(tmp_path)
    assert isinstance(config_default.model, ag_types.ModelTarget)
    assert isinstance(config_default.model.endpoint, ag_types.GeminiAPIEndpoint)
    assert isinstance(config_default.model.endpoint.options, ag_types.GeminiModelOptions)
    assert config_default.model.endpoint.options.thinking_level == ag_types.ThinkingLevel.LOW

    runtime_no_effort = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="custom-model"),
    )
    runtime_no_effort._resident = ("skill-a",)
    config_no_effort = runtime_no_effort._select_config(tmp_path)
    assert config_no_effort.model == "custom-model"


def test_select_config_points_at_the_installed_skills_directory(
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify skills_paths in select config points to workspace skills directory."""
    workdir = tmp_path / "work"
    runtime._resident = ("gke-basics",)
    config = runtime._select_config(workdir)
    assert config.skills_paths == [str(runtime.skills_dir(workdir))]


def test_select_reports_the_structured_selection(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify select parses selected skill correctly from structured response."""
    runtime._resident = ("gke-basics",)
    _fake_agent(monkeypatch, _FakeResponse(structured={"selected_skill": "gke-basics"}))
    outcome = runtime.select("how do I set up a cluster?", tmp_path / "work")
    assert outcome.invoked_skill == "gke-basics"
    assert outcome.invoked_skills == ("gke-basics",)
    assert outcome.observed_catalog == ("gke-basics",)
    assert outcome.error is None
    assert outcome.cost_usd is None


def test_select_reports_the_structured_selection_from_pydantic_model(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify select parses selected skill from a Pydantic model instance."""
    runtime._resident = ("gke-basics",)
    schema_cls = AntigravityRuntime.selection_schema(("gke-basics",))
    model_obj = schema_cls.model_validate(
        {"selected_skill": "gke-basics", "reasoning": "Fits target"},
    )
    _fake_agent(monkeypatch, _FakeResponse(structured=model_obj))
    outcome = runtime.select("how do I set up a cluster?", tmp_path / "work")
    assert outcome.invoked_skill == "gke-basics"
    assert outcome.invoked_skills == ("gke-basics",)
    assert outcome.reasoning == ("Fits target",)
    assert outcome.observed_catalog == ("gke-basics",)
    assert outcome.error is None


def test_select_populates_reasoning_in_outcome(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify select populates reasoning in SelectionOutcome."""
    runtime._resident = ("gke-basics",)
    _fake_agent(
        monkeypatch,
        _FakeResponse(
            structured={"selected_skill": "gke-basics", "reasoning": "Target skill matches"},
        ),
    )
    outcome = runtime.select("how do I set up a cluster?", tmp_path / "work")
    assert outcome.reasoning == ("Target skill matches",)


def test_select_reports_abstention_when_nothing_was_selected(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify select reports None when structured output specifies null selected_skill."""
    runtime._resident = ("gke-basics",)
    _fake_agent(monkeypatch, _FakeResponse(structured={"selected_skill": None}))
    outcome = runtime.select("what's the weather", tmp_path / "work")
    assert outcome.invoked_skill is None
    assert outcome.invoked_skills == ()
    assert outcome.error is None


def test_select_reports_no_structured_output_as_abstention(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify missing structured output defaults to abstention outcome."""
    runtime._resident = ("gke-basics",)
    _fake_agent(monkeypatch, _FakeResponse(structured=None))
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.invoked_skill is None
    assert outcome.error is None


def test_select_flags_a_tool_surviving_denial_as_a_leak(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify unexpected non-finish tool calls in response are flagged as tool leaks."""
    runtime._resident = ("gke-basics",)
    leaked_call = ag_types.ToolCall(name=ag_types.BuiltinTools.RUN_COMMAND, args={})
    _fake_agent(
        monkeypatch,
        _FakeResponse(
            structured={"selected_skill": "gke-basics"},
            tool_calls=[leaked_call],
        ),
    )
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error is not None
    assert "run_command" in outcome.error


def test_select_accepts_max_model_calls_exceeded_as_a_clean_completion(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify MAX_MODEL_CALLS_EXCEEDED stop reason is treated as valid successful turn."""
    runtime._resident = ("gke-basics",)
    _fake_agent(
        monkeypatch,
        _FakeResponse(
            structured={"selected_skill": "gke-basics"},
            stop_reason=ag_types.StopReason.MAX_MODEL_CALLS_EXCEEDED,
        ),
    )
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error is None
    assert outcome.invoked_skill == "gke-basics"


def test_select_reports_an_unexpected_stop_reason_as_an_error(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify unexpected stop reasons like QUOTA_EXHAUSTED record errors."""
    runtime._resident = ("gke-basics",)
    _fake_agent(
        monkeypatch,
        _FakeResponse(stop_reason=ag_types.StopReason.QUOTA_EXHAUSTED),
    )
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error is not None
    assert "QUOTA_EXHAUSTED" in outcome.error


def test_select_reports_a_backend_failure_not_raises(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify backend agent exceptions are captured into outcome.error string."""
    runtime._resident = ("gke-basics",)

    class _RaisingAgent(_FakeAgent):
        async def __aenter__(self):
            msg = "no credentials found"
            raise RuntimeError(msg)

    monkeypatch.setattr("reach.runtime.antigravity_sdk.Agent", _RaisingAgent)
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error == "no credentials found"


def test_select_falls_back_to_a_thread_when_a_loop_is_already_running(
    monkeypatch: pytest.MonkeyPatch,
    runtime: AntigravitySdkRuntime,
    tmp_path: Path,
) -> None:
    """Verify select executes cleanly when called from within an existing event loop."""
    runtime._resident = ("gke-basics",)
    _fake_agent(monkeypatch, _FakeResponse(structured={"selected_skill": "gke-basics"}))

    async def call_from_inside_a_running_loop():
        return runtime.select("how do I set up a cluster?", tmp_path / "work")

    outcome = asyncio.run(call_from_inside_a_running_loop())
    assert outcome.invoked_skill == "gke-basics"
    assert outcome.error is None


def test_complete_falls_back_to_a_thread_when_a_loop_is_already_running(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravitySdkGenerator,
) -> None:
    """Verify complete executes cleanly when invoked from within a running event loop."""
    _fake_agent(monkeypatch, _FakeResponse(text="drafted query set"))

    async def call_from_inside_a_running_loop():
        return generator.complete("draft some queries")

    result = asyncio.run(call_from_inside_a_running_loop())
    assert result == "drafted query set"


def test_select_reports_a_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify select returns timeout error when asyncio.wait_for times out."""
    runtime = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model"),
    )
    runtime._resident = ("gke-basics",)

    async def timeout(fut, *_args, **_kwargs) -> Never:
        if asyncio.iscoroutine(fut):
            fut.close()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", timeout)
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error == "timeout"


def test_complete_returns_the_scripted_text(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravitySdkGenerator,
) -> None:
    """Verify complete returns plain response text from agent."""
    instances = _fake_agent(monkeypatch, _FakeResponse(text="drafted query set"))
    result = generator.complete("draft some queries")
    assert result == "drafted query set"
    assert instances[0].sent == "draft some queries"
    assert generator.completions == 1


def test_complete_uses_no_isolation(
    monkeypatch: pytest.MonkeyPatch,
    generator: AntigravitySdkGenerator,
) -> None:
    """Verify complete creates agent config without budget or skill path constraints."""
    instances = _fake_agent(monkeypatch, _FakeResponse(text=""))
    generator.complete("q")
    config = instances[0].config
    assert config.budget_config is None
    assert config.skills_paths == []


def test_complete_uses_thinking_config_when_effort_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify complete creates agent config with thinking options when effort is configured."""
    gen = AntigravitySdkGenerator(
        options=AntigravitySdkOptions(model="gemini-3.7-flash", effort="high"),
    )
    instances = _fake_agent(monkeypatch, _FakeResponse(text="response text"))
    gen.complete("prompt")
    config = instances[0].config
    assert isinstance(config.model, ag_types.ModelTarget)
    assert isinstance(config.model.endpoint, ag_types.GeminiAPIEndpoint)
    assert isinstance(config.model.endpoint.options, ag_types.GeminiModelOptions)
    assert config.model.endpoint.options.thinking_level == ag_types.ThinkingLevel.HIGH


def test_antigravity_sdk_missing_dependency_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_runtime raises actionable RuntimeError when google.antigravity is missing."""
    from reach.config import RuntimeSettings
    from reach.runtime import build_runtime

    orig_import_module = importlib.import_module

    def mock_import_module(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "reach.runtime.antigravity_sdk":
            msg = "No module named 'google.antigravity'"
            raise ImportError(msg)
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    with pytest.raises(RuntimeError, match=r"pip install 'skill-reach\[antigravity-sdk\]'"):
        build_runtime(RuntimeSettings(agent="antigravity-sdk"))


def test_antigravity_sdk_generator_missing_dependency_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_text_generator raises RuntimeError when google.antigravity is missing."""
    from reach.runtime.generator import build_text_generator

    orig_import_module = importlib.import_module

    def mock_import_module(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "reach.runtime.antigravity_sdk":
            msg = "No module named 'google.antigravity'"
            raise ImportError(msg)
        return orig_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", mock_import_module)

    with pytest.raises(RuntimeError, match=r"pip install 'skill-reach\[antigravity-sdk\]'"):
        build_text_generator(agent="antigravity-sdk")


def test_antigravity_sdk_uninstalled_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify instantiating AntigravitySdkRuntime raises RuntimeError when uninstalled."""
    monkeypatch.setattr("reach.runtime.antigravity_sdk._HAS_ANTIGRAVITY", False)
    with pytest.raises(RuntimeError, match=r"pip install 'skill-reach\[antigravity-sdk\]'"):
        AntigravitySdkRuntime()


def test_antigravity_sdk_generator_uninstalled_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify instantiating AntigravitySdkGenerator raises RuntimeError when uninstalled."""
    monkeypatch.setattr("reach.runtime.antigravity_sdk._HAS_ANTIGRAVITY", False)
    with pytest.raises(RuntimeError, match=r"pip install 'skill-reach\[antigravity-sdk\]'"):
        AntigravitySdkGenerator()


def test_antigravity_sdk_options_defaults() -> None:
    """Verify AntigravitySdkOptions default parameters for performance and isolation."""
    opts = AntigravitySdkOptions()
    assert opts.use_symlinks is True
    assert opts.isolate_config_dir is True
    assert opts.auto_clean is False
    assert opts.app_data_dir is None


def test_antigravity_sdk_build_env_synchronizes_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify build_env synchronizes GEMINI_API_KEY and GOOGLE_API_KEY bidirectionally."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Option key takes priority and sets both
    rt_opt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", api_key="my-api-key"),
    )
    env_opt = rt_opt.build_env()
    assert env_opt["GEMINI_API_KEY"] == "my-api-key"
    assert env_opt["GOOGLE_API_KEY"] == "my-api-key"

    # Ambient GEMINI_API_KEY synchronizes to GOOGLE_API_KEY
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-only")
    rt_gemini = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model"),
    )
    env_gemini = rt_gemini.build_env()
    assert env_gemini["GEMINI_API_KEY"] == "gemini-only"
    assert env_gemini["GOOGLE_API_KEY"] == "gemini-only"

    # Ambient GOOGLE_API_KEY synchronizes to GEMINI_API_KEY
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-only")
    rt_google = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model"),
    )
    env_google = rt_google.build_env()
    assert env_google["GEMINI_API_KEY"] == "google-only"
    assert env_google["GOOGLE_API_KEY"] == "google-only"


def test_select_config_sets_isolated_app_data_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _select_config sets app_data_dir and creates directory when isolation is enabled."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    workdir = tmp_path / "work"
    workdir.mkdir()

    # Default isolation sets workdir / .reach_antigravity_sdk
    rt = AntigravitySdkRuntime(options=AntigravitySdkOptions(model="test-model"))
    config = rt._select_config(workdir)
    expected_dir = (workdir / ".reach_antigravity_sdk").resolve()
    assert config.app_data_dir == str(expected_dir)
    assert expected_dir.is_dir()
    assert "GEMINI_API_KEY" in (config.env or {}) or "GOOGLE_API_KEY" in (config.env or {})

    # Custom app_data_dir is respected
    custom_dir = tmp_path / "custom_app_data"
    rt_custom = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", app_data_dir=custom_dir),
    )
    config_custom = rt_custom._select_config(workdir)
    assert config_custom.app_data_dir == str(custom_dir.resolve())
    assert custom_dir.is_dir()

    # Disabling isolation leaves app_data_dir None
    rt_no_iso = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", isolate_config_dir=False),
    )
    config_no_iso = rt_no_iso._select_config(workdir)
    assert config_no_iso.app_data_dir is None


def test_select_invokes_post_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify select cleans isolated directory when auto_clean is True."""
    workdir = tmp_path / "work"
    workdir.mkdir()

    rt = AntigravitySdkRuntime(
        options=AntigravitySdkOptions(model="test-model", auto_clean=True),
    )
    rt._resident = ("gke-basics",)
    _fake_agent(monkeypatch, _FakeResponse(structured={"selected_skill": "gke-basics"}))

    outcome = rt.select("how do I set up a cluster?", workdir)
    assert outcome.invoked_skill == "gke-basics"
    assert not (workdir / ".reach_antigravity_sdk").exists()


def test_complete_passes_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify complete passes synchronized environment to LocalAgentConfig."""
    gen = AntigravitySdkGenerator(
        options=AntigravitySdkOptions(model="test-model", api_key="complete-key"),
    )
    instances = _fake_agent(monkeypatch, _FakeResponse(text="completion text"))
    gen.complete("test prompt")
    config = instances[0].config
    assert config.env is not None
    assert config.env.get("GEMINI_API_KEY") == "complete-key"
    assert config.env.get("GOOGLE_API_KEY") == "complete-key"
