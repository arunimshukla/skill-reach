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

"""Drive the Google Antigravity Python SDK as an agent evaluation runtime."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast, override

if TYPE_CHECKING:
    from collections.abc import Coroutine, Mapping

    from google.antigravity import Agent, LocalAgentConfig
    from google.antigravity import hooks as ag_hooks
    from google.antigravity import types as ag_types
    from google.antigravity.types import AntigravityValidationError

    _HAS_ANTIGRAVITY = True
else:
    try:
        from google.antigravity import Agent, LocalAgentConfig
        from google.antigravity import hooks as ag_hooks
        from google.antigravity import types as ag_types
        from google.antigravity.types import AntigravityValidationError

        _HAS_ANTIGRAVITY = True
    except ImportError:
        Agent = None
        LocalAgentConfig = None
        ag_hooks = None
        ag_types = None

        class AntigravityValidationError(Exception):
            """Stub exception when google-antigravity is not installed."""

        _HAS_ANTIGRAVITY = False

from pydantic import BaseModel, Field

from reach.config import DEFAULT_GEMINI_MODEL, RuntimeSettings
from reach.runtime import (
    AgentOptions,
    AntigravityRuntime,
    SelectionOutcome,
    TrajectoryTracker,
    agent_default_model,
)
from reach.runtime._env import (
    raise_missing_agent_dependency,
    sync_google_and_gemini_keys,
)
from reach.runtime._fs import (
    ensure_private_directory,
    resolve_skill_from_path,
    safe_cleanup_isolated_dir,
)
from reach.runtime._subprocess import check_tool_leak
from reach.runtime.generator import BaseTextGenerator
from reach.runtime.profiles import model_profile

#: Selection tool set configured for single-turn probe evaluations.
SELECTION_TOOLS: tuple[Any, ...] = (
    (ag_types.BuiltinTools.FINISH,) if ag_types is not None else ("finish",)
)


def _build_multi_turn_selection_tools() -> tuple[Any, ...]:
    """Assemble finish tool plus available read-only workspace inspection tools."""
    if ag_types is None or not hasattr(ag_types, "BuiltinTools"):
        return ("finish", "view_file", "list_dir", "grep_search", "find_by_name")
    tools: list[Any] = [ag_types.BuiltinTools.FINISH]
    for attr in ("VIEW_FILE", "LIST_DIR", "GREP_SEARCH", "FIND_BY_NAME"):
        member = getattr(ag_types.BuiltinTools, attr, None)
        if member is not None:
            tools.append(member)
    if len(tools) == 1:
        tools.append("view_file")
    return tuple(tools)


#: Selection tool set configured for multi-turn trajectory probe evaluations.
MULTI_TURN_SELECTION_TOOLS: tuple[Any, ...] = _build_multi_turn_selection_tools()

#: Terminal stop reasons considered normal for single-turn evaluations.
EXPECTED_STOP_REASONS: frozenset[Any] = (
    frozenset({ag_types.StopReason.UNSPECIFIED, ag_types.StopReason.MAX_MODEL_CALLS_EXCEEDED})
    if ag_types is not None
    else frozenset({"UNSPECIFIED", "MAX_MODEL_CALLS_EXCEEDED"})
)


def _run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously, handling existing active event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        with asyncio.Runner() as runner:
            return runner.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        # pyrefly: ignore[redundant-cast]
        return cast("T", executor.submit(asyncio.run, coro).result())


def _tool_name(name: ag_types.BuiltinTools | str) -> str:
    """Extract the string name of a tool from an enum or raw string."""
    return (
        name.value
        if ag_types is not None and isinstance(name, ag_types.BuiltinTools)
        else str(name)
    )


def _model_supports_thinking(model: str) -> bool:
    """Return whether a model supports reasoning effort or thinking level."""
    lower = model.lower()
    return not any(p in lower for p in ("gemini-2.5", "gemini-2.0", "gemini-1.5", "gemini-1.0"))


def _build_model_spec(
    model: str,
    effort: str | None = None,
    *,
    vertex: bool = False,
    project: str | None = None,
    location: str | None = None,
    api_key: str | None = None,
) -> str | ag_types.ModelTarget:
    """Construct model target with reasoning effort endpoint options when configured."""
    valid_effort = effort if (effort and _model_supports_thinking(model)) else None
    if ag_types is not None and (valid_effort or vertex):
        options = ag_types.GeminiModelOptions(thinking_level=valid_effort) if valid_effort else None
        endpoint = (
            ag_types.VertexEndpoint(
                project=project,
                location=location,
                api_key=api_key,
                options=options,
            )
            if vertex
            else ag_types.GeminiAPIEndpoint(
                api_key=api_key,
                options=options,
            )
        )
        return ag_types.ModelTarget(name=model, endpoint=endpoint)
    return model


def _extract_selection_and_reasoning(
    data: object,
) -> tuple[str | None, tuple[str, ...]]:
    """Extract selected skill and reasoning tuple from structured output payload."""
    invoked = None
    reasoning: tuple[str, ...] = ()
    if isinstance(data, BaseModel):
        invoked = getattr(data, "selected_skill", None)
        if r := getattr(data, "reasoning", None):
            r_str = str(r).strip()
            if r_str:
                reasoning = (r_str,)
    elif isinstance(data, dict):
        invoked = data.get("selected_skill")
        if r := data.get("reasoning"):
            r_str = str(r).strip()
            if r_str:
                reasoning = (r_str,)
    return invoked, reasoning


_HTTP_TOO_MANY_REQUESTS = 429


def _format_step_error(step: object, error_status: object) -> str | None:
    """Format a single SDK conversation step error when present."""
    status = getattr(step, "status", None)
    raw_err = str(getattr(step, "error", "") or "").strip()
    if status not in (error_status, "STATE_ERROR", "ERROR") and not raw_err:
        return None
    http_code = int(getattr(step, "http_code", 0) or 0)
    err_msg = raw_err or "unknown system error"
    lower_err = err_msg.lower()
    if http_code == _HTTP_TOO_MANY_REQUESTS or any(
        k in lower_err for k in ("429", "resource exhausted", "quota")
    ):
        return f"rate limit (429): {err_msg}"
    if http_code > 0:
        return f"sdk step error (HTTP {http_code}): {err_msg}"
    return f"sdk step error: {err_msg}"


def _extract_history_error(agent: object) -> str | None:
    """Extract formatted rate-limit or system error from SDK conversation history."""
    conv = getattr(agent, "conversation", None)
    history = getattr(conv, "history", None)
    if not history:
        return None
    error_status = (
        getattr(ag_types.StepStatus, "ERROR", "STATE_ERROR")
        if ag_types is not None
        else "STATE_ERROR"
    )
    for step in reversed(history):
        if formatted := _format_step_error(step, error_status):
            return formatted
    return None


def _resolve_empty_selection_error(
    history_error: str | None,
    stop_reason: object,
    observed_tools: tuple[str, ...],
) -> str | None:
    """Resolve error string when an agent turn produces no skill selection or structured output."""
    if history_error:
        return history_error
    max_calls = (
        ag_types.StopReason.MAX_MODEL_CALLS_EXCEEDED
        if ag_types is not None
        else "MAX_MODEL_CALLS_EXCEEDED"
    )
    if stop_reason == max_calls and observed_tools:
        return None
    return "empty selection (likely rate-limited)"


class AntigravitySdkOptions(AgentOptions):
    """Specify runtime configuration options for the Antigravity SDK driver."""

    model: str = Field(
        default_factory=lambda: agent_default_model("antigravity-sdk") or DEFAULT_GEMINI_MODEL,
    )
    app_data_dir: Path | None = None
    isolation_dir_field: ClassVar[str | None] = "app_data_dir"
    vertex: bool | None = None
    project: str | None = None
    location: str | None = None
    api_max_retries: int | None = Field(default=None, ge=0)
    api_retry_jitter: float | None = Field(default=None, ge=0.0, le=1.0)


class _AntigravitySdkConfigMixin:
    """Consolidate shared Vertex AI and ADC resolution for SDK runtime and generator."""

    options: AntigravitySdkOptions

    @property
    def effective_vertex(self) -> bool:
        """Determine whether Vertex AI backend is active."""
        if self.options.vertex is not None:
            return self.options.vertex
        if getattr(self.options, "provider", None) == "vertex":
            return True
        return os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE", "").lower() in (
            "true",
            "1",
        ) or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1")

    @property
    def effective_project(self) -> str | None:
        """Resolve GCP project ID for Vertex AI execution."""
        if not self.effective_vertex:
            return None
        if self.options.api_key:
            return self.options.project
        return self.options.project or os.environ.get("GOOGLE_CLOUD_PROJECT")

    @property
    def effective_location(self) -> str | None:
        """Resolve GCP region/location for Vertex AI execution."""
        if not self.effective_vertex:
            return None
        if self.options.api_key:
            return self.options.location
        return self.options.location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"

    @property
    def effective_api_key(self) -> str | None:
        """Return configured API key or fallback to environment variables."""
        if self.options.api_key:
            return self.options.api_key
        if self.effective_vertex:
            return None
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def _sync_sdk_env(self, env: dict[str, str]) -> dict[str, str]:
        """Apply API key synchronization and Vertex credential preservation to environment."""
        if (api_key := self.effective_api_key) is not None:
            env["GEMINI_API_KEY"] = api_key
            env["GOOGLE_API_KEY"] = api_key
        elif self.effective_vertex and not self.options.api_key:
            env.pop("GEMINI_API_KEY", None)
            env.pop("GOOGLE_API_KEY", None)

        blocked = getattr(self, "blocked_env_vars", None) or ()
        if (
            self.effective_vertex
            and "GOOGLE_APPLICATION_CREDENTIALS" not in blocked
            and "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
            and "GOOGLE_APPLICATION_CREDENTIALS" not in env
        ):
            env["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

        return sync_google_and_gemini_keys(env)

    def _target_model_spec(self, model: str, effort: str | None) -> str | ag_types.ModelTarget:
        """Construct model target with reasoning effort and endpoint options when configured."""
        return _build_model_spec(
            model,
            effort,
            vertex=self.effective_vertex,
            project=self.effective_project,
            location=self.effective_location,
            api_key=self.effective_api_key,
        )

    def _build_retry_config(self) -> object | None:
        """Construct RetryConfig when api_max_retries or api_retry_jitter is configured."""
        if (
            ag_types is None
            or not hasattr(ag_types, "RetryConfig")
            or (self.options.api_max_retries is None and self.options.api_retry_jitter is None)
        ):
            return None
        api_kwargs: dict[str, Any] = {}
        if self.options.api_max_retries is not None:
            api_kwargs["max_retries"] = self.options.api_max_retries
        if self.options.api_retry_jitter is not None:
            api_kwargs["jitter_range"] = self.options.api_retry_jitter
        return ag_types.RetryConfig(api_retry=ag_types.ModelAPIRetryConfig(**api_kwargs))

    def _base_config_kwargs(
        self,
        model: str | ag_types.ModelTarget,
        env: dict[str, str],
    ) -> dict[str, Any]:
        """Assemble shared LocalAgentConfig keyword arguments across runtime and generator."""
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": self.effective_api_key,
            "vertex": self.effective_vertex,
            "project": self.effective_project,
            "location": self.effective_location,
            "env": env,
        }
        if (retry_cfg := self._build_retry_config()) is not None:
            kwargs["retry_config"] = retry_cfg
        return kwargs


class AntigravitySdkRuntime(_AntigravitySdkConfigMixin, AntigravityRuntime):
    """Execute evaluation queries using the Google Antigravity Python SDK."""

    name = "antigravity-sdk"
    options: AntigravitySdkOptions
    api_key_env_var: str | None = "GEMINI_API_KEY"
    _skills_subpath = ".agents/skills"

    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        options: AntigravitySdkOptions | None = None,
    ) -> None:
        """Initialize Antigravity SDK driver with runtime settings and options."""
        if not _HAS_ANTIGRAVITY:
            raise_missing_agent_dependency(
                "antigravity-sdk",
                ImportError("google-antigravity is not installed"),
                role="runtime",
            )
        super().__init__(settings=settings, options=options)

    @override
    def build_env(self, workdir: Path | None = None) -> dict[str, str]:
        """Assemble environment variables with API key synchronization."""
        env = super().build_env(workdir)
        return self._sync_sdk_env(env)

    def _model_spec(self) -> str | ag_types.ModelTarget:
        """Construct model target with reasoning effort endpoint options when configured."""
        return self._target_model_spec(self.options.model, self.effective_effort)

    def _select_config(
        self,
        workdir: Path,
        hooks: list[Any] | None = None,
    ) -> LocalAgentConfig:
        """Assemble schema-constrained LocalAgentConfig with turn budget and hooks."""
        app_data_dir = None
        if self.options.isolate_config_dir or self.options.app_data_dir:
            sdk_dir = ensure_private_directory(
                self.options.app_data_dir or (workdir / ".reach_antigravity_sdk")
            )
            app_data_dir = str(sdk_dir)

        enabled_tools = (
            list(MULTI_TURN_SELECTION_TOOLS)
            if not self.options.early_exit and self.options.max_turns > 1
            else list(SELECTION_TOOLS)
        )
        if self.allowed_tools:
            builtin_by_name = (
                {_tool_name(member): member for member in ag_types.BuiltinTools}
                if ag_types is not None and hasattr(ag_types, "BuiltinTools")
                else {}
            )
            enabled_tools = [builtin_by_name.get(t, t) for t in self.allowed_tools]

        kwargs = self._base_config_kwargs(self._model_spec(), self.build_env(workdir))
        kwargs.update(
            {
                "skills_paths": [str(self.skills_dir(workdir))],
                "capabilities": ag_types.CapabilitiesConfig(
                    enabled_tools=enabled_tools,
                    enable_subagents=False,
                ),
                "budget_config": ag_types.BudgetConfig(max_model_calls=self.options.max_turns),
                "response_schema": self.selection_schema(self._resident),
                "app_data_dir": app_data_dir,
                "hooks": hooks,
            }
        )
        return LocalAgentConfig(**kwargs)

    async def _select_async(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Execute chat evaluation asynchronously and return observed outcome."""
        tracker = TrajectoryTracker(
            target_skill=target_skill,
            max_turns=self.options.max_turns,
            early_exit=self.options.early_exit,
        )

        hooks_list: list[Any] = []
        if ag_hooks is not None:

            @ag_hooks.pre_tool_call_decide
            async def _on_tool_call(call: ag_types.ToolCall) -> ag_types.HookResult:
                args = getattr(call, "args", None) or getattr(call, "arguments", {}) or {}
                path = args.get("path") or args.get("AbsolutePath")
                if path:
                    skill = resolve_skill_from_path(path, self._resident)
                    if skill:
                        should_stop = tracker.observe(skill)
                        if self.options.early_exit and should_stop:
                            return ag_types.HookResult(allow=False)
                return ag_types.HookResult(allow=True)

            hooks_list.append(_on_tool_call)

        config = self._select_config(workdir, hooks=hooks_list)

        try:
            async with Agent(config) as agent:
                response = await agent.chat(query_text)
                data = await response.structured_output()
                observed_tools = tuple(
                    [_tool_name(call.name) async for call in response.tool_calls],
                )
                stop_reason = response.stop_reason
                history_error = _extract_history_error(agent)
        except (AntigravityValidationError, Exception) as err:
            if isinstance(err, RuntimeError):
                raise
            msg = f"Antigravity SDK execution error: {err}"
            raise RuntimeError(msg) from err

        active_tool_set = (
            MULTI_TURN_SELECTION_TOOLS
            if not self.options.early_exit and self.options.max_turns > 1
            else SELECTION_TOOLS
        )
        allowed_tool_names = (
            set(self.allowed_tools)
            if self.allowed_tools
            else (set(self.ANTIGRAVITY_SELECTION_TOOLS) | {_tool_name(t) for t in active_tool_set})
        )
        error = None
        if stop_reason not in EXPECTED_STOP_REASONS:
            reason_str = getattr(stop_reason, "value", str(stop_reason))
            error = f"runtime error: {reason_str}"
        elif tool_leak := check_tool_leak(observed_tools, allowed_tool_names):
            error = tool_leak

        invoked, reasoning = _extract_selection_and_reasoning(data)

        if tracker.early_exit_hit and tracker.invoked_skills:
            invoked = tracker.invoked_skills[0]

        if tracker.invoked_skills:
            invoked_skills = tuple(tracker.invoked_skills)
        elif invoked:
            invoked_skills = (invoked,)
        else:
            invoked_skills = ()

        if not error and not tracker.early_exit_hit and not invoked_skills and data is None:
            error = _resolve_empty_selection_error(history_error, stop_reason, observed_tools)

        return SelectionOutcome(
            invoked_skills=invoked_skills,
            early_exit=tracker.early_exit_hit,
            turns_taken=tracker.turns_taken,
            reasoning=reasoning,
            observed_catalog=self._resident,
            observed_tools=observed_tools,
            cost_usd=None,
            error=error,
        )

    @override
    def post_probe(self, workdir: Path) -> None:
        """Clean session and agent artifacts after probe execution if auto_clean is enabled."""
        if not self.options.auto_clean:
            return
        if self.options.isolate_config_dir:
            sdk_dir = self.options.app_data_dir or (Path(workdir) / ".reach_antigravity_sdk")
            safe_cleanup_isolated_dir(workdir, sdk_dir)

    @override
    def select(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Execute query evaluation probe and return SelectionOutcome."""
        try:
            return _run_sync(
                asyncio.wait_for(
                    self._select_async(query_text, workdir, target_skill=target_skill),
                    timeout=self.timeout_s,
                ),
            )
        except TimeoutError:
            return SelectionOutcome(error="timeout", observed_catalog=self._resident)
        except Exception as exc:  # noqa: BLE001
            return SelectionOutcome(error=str(exc), observed_catalog=self._resident)
        finally:
            self.post_probe(workdir)


class AntigravitySdkGenerator(_AntigravitySdkConfigMixin, BaseTextGenerator[AntigravitySdkOptions]):
    """Generate text completions using the Antigravity SDK."""

    name: str = "antigravity-sdk"
    options: AntigravitySdkOptions

    def __init__(
        self,
        model: str = "",
        *,
        timeout_s: int = 300,
        options: AntigravitySdkOptions | None = None,
        settings: RuntimeSettings | None = None,
    ) -> None:
        """Initialize Antigravity SDK generator with model, timeout, and options."""
        if not _HAS_ANTIGRAVITY:
            raise_missing_agent_dependency(
                "antigravity-sdk",
                ImportError("google-antigravity is not installed"),
                role="generator",
            )
        if isinstance(options, AntigravitySdkOptions):
            opts = options.model_copy(update={"model": model}) if model else options
        elif settings is not None and settings.options:
            base_opts = dict(settings.options)
            if model:
                base_opts["model"] = model
            opts = AntigravitySdkOptions.model_validate(base_opts)
        elif model:
            opts = AntigravitySdkOptions(model=model)
        else:
            opts = AntigravitySdkOptions()
        super().__init__(model=opts.model or model, timeout_s=timeout_s, options=opts)
        self.settings = settings

    @override
    def build_env(self) -> dict[str, str]:
        """Assemble environment variables with API key synchronization."""
        env = super().build_env()
        return self._sync_sdk_env(env)

    @property
    def effective_effort(self) -> str | None:
        """Return configured reasoning effort or default from model profile."""
        if self.options.effort:
            effort = self.options.effort
            return None if effort.lower() in ("none", "off") else effort
        if not _model_supports_thinking(self.model):
            return None
        try:
            return model_profile(self.model).effort
        except (KeyError, ValueError):
            return None

    def _model_spec(self) -> str | ag_types.ModelTarget:
        """Construct model target with reasoning effort endpoint options when configured."""
        return self._target_model_spec(self.options.model or self.model, self.effective_effort)

    def _resolve_schema_dict(
        self,
        schema: str | Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Parse optional JSON schema argument or options fallback into a dictionary."""
        if isinstance(schema, Mapping):
            return dict(schema)
        raw_schema = schema if isinstance(schema, str) else self.options.json_schema
        if not raw_schema:
            return None
        try:
            parsed = json.loads(raw_schema)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @override
    def complete(self, prompt: str, *, schema: str | Mapping[str, Any] | None = None) -> str:
        """Execute text completion using the Antigravity SDK."""
        schema_dict = self._resolve_schema_dict(schema)
        caps = (
            ag_types.CapabilitiesConfig(enabled_tools=[], enable_subagents=False)
            if ag_types is not None and hasattr(ag_types, "CapabilitiesConfig")
            else None
        )

        async def _complete_async() -> str:
            kwargs = self._base_config_kwargs(self._model_spec(), self.build_env())
            if schema_dict is not None:
                kwargs["response_schema"] = schema_dict
            if caps is not None:
                kwargs["capabilities"] = caps

            config = LocalAgentConfig(**kwargs)
            async with Agent(config) as agent:
                response = await agent.chat(prompt)
                if schema_dict is not None and hasattr(response, "structured_output"):
                    attr = response.structured_output
                    structured = attr() if callable(attr) else attr
                    if asyncio.iscoroutine(structured):
                        structured = await structured
                    if structured is not None:
                        if hasattr(structured, "model_dump_json"):
                            return structured.model_dump_json()
                        if hasattr(structured, "model_dump"):
                            return json.dumps(structured.model_dump())
                        return json.dumps(structured)
                    err_msg = _extract_history_error(agent) or (
                        "empty structured output (likely rate-limited)"
                    )
                    raise ValueError(err_msg)
                text_out = await response.text()
                if not text_out.strip():
                    err_msg = _extract_history_error(agent) or (
                        "empty response (likely rate-limited)"
                    )
                    raise ValueError(err_msg)
                return text_out

        try:
            text = _run_sync(asyncio.wait_for(_complete_async(), timeout=self.timeout_s))
        except TimeoutError as err:
            msg = f"generation failed: timed out after {self.timeout_s}s"
            raise RuntimeError(msg) from err
        except (AntigravityValidationError, Exception) as err:
            msg = f"generation failed: {err}"
            raise RuntimeError(msg) from err
        self.completions += 1
        return text
