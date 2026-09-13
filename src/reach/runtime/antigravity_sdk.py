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
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast, override

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from google.antigravity import Agent, LocalAgentConfig
    from google.antigravity import types as ag_types

    _HAS_ANTIGRAVITY = True
else:
    try:
        from google.antigravity import Agent, LocalAgentConfig
        from google.antigravity import types as ag_types

        _HAS_ANTIGRAVITY = True
    except ImportError:
        Agent = None
        LocalAgentConfig = None
        ag_types = None
        _HAS_ANTIGRAVITY = False

from pydantic import BaseModel, Field

from reach.config import DEFAULT_GEMINI_MODEL, RuntimeSettings
from reach.runtime import (
    AgentOptions,
    AntigravityRuntime,
    SelectionOutcome,
    agent_default_model,
    resolve_options,
)
from reach.runtime._env import (
    raise_missing_agent_dependency,
    sync_google_and_gemini_keys,
)
from reach.runtime._fs import (
    resolve_skill_from_path,
    safe_cleanup_isolated_dir,
)
from reach.runtime._subprocess import (
    check_tool_leak,
)
from reach.runtime.generator import BaseTextGenerator
from reach.runtime.profiles import model_profile

#: Selection tool set configured for single-turn probe evaluations.
SELECTION_TOOLS: tuple[Any, ...] = (
    (ag_types.BuiltinTools.FINISH,) if ag_types is not None else ("finish",)
)

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


class AntigravitySdkOptions(AgentOptions):
    """Specify runtime configuration options for the Antigravity SDK driver."""

    model: str = Field(
        default_factory=lambda: agent_default_model("antigravity-sdk") or DEFAULT_GEMINI_MODEL,
    )
    app_data_dir: Path | None = None
    isolation_dir_field: ClassVar[str | None] = "app_data_dir"


class AntigravitySdkRuntime(AntigravityRuntime):
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
        self.settings = settings if settings is not None else RuntimeSettings(agent=self.name)
        if options is None:
            resolved = resolve_options(self.settings) if self.settings.options else None
            options = (
                resolved if isinstance(resolved, AntigravitySdkOptions) else AntigravitySdkOptions()
            )
        self.options = options
        self._resident: tuple[str, ...] = ()

    @override
    def build_env(self, workdir: Path | None = None) -> dict[str, str]:
        """Assemble environment variables with API key synchronization."""
        del workdir
        env = dict(os.environ)
        if (api_key := self.effective_api_key) is not None:
            env["GEMINI_API_KEY"] = api_key
            env["GOOGLE_API_KEY"] = api_key

        return sync_google_and_gemini_keys(env)

    def _model_spec(self) -> str | ag_types.ModelTarget:
        """Construct model target with reasoning effort endpoint options when configured."""
        if effort := self.effective_effort:
            return ag_types.ModelTarget(
                name=self.options.model,
                endpoint=ag_types.GeminiAPIEndpoint(
                    options=ag_types.GeminiModelOptions(thinking_level=effort),
                ),
            )
        return self.options.model

    def _select_config(self, workdir: Path) -> LocalAgentConfig:
        """Assemble schema-constrained LocalAgentConfig with turn budget."""
        app_data_dir = None
        if self.options.isolate_config_dir or self.options.app_data_dir:
            sdk_dir = (self.options.app_data_dir or (workdir / ".reach_antigravity_sdk")).resolve()
            sdk_dir.mkdir(parents=True, exist_ok=True)
            app_data_dir = str(sdk_dir)

        return LocalAgentConfig(
            model=self._model_spec(),
            skills_paths=[str(self.skills_dir(workdir))],
            capabilities=ag_types.CapabilitiesConfig(
                enabled_tools=list(SELECTION_TOOLS),
                enable_subagents=False,
            ),
            budget_config=ag_types.BudgetConfig(max_model_calls=self.options.max_turns),
            response_schema=self.selection_schema(self._resident),
            api_key=self.effective_api_key,
            app_data_dir=app_data_dir,
            env=self.build_env(workdir),
        )

    async def _select_async(
        self,
        query_text: str,
        workdir: Path,
        target_skill: str | None = None,
    ) -> SelectionOutcome:
        """Execute chat evaluation asynchronously and return observed outcome."""
        config = self._select_config(workdir)
        detected_skills: list[str] = []
        early_exit_hit = False

        async with Agent(config) as agent:
            hooks = getattr(agent, "hooks", None)
            decide_hook: Any = getattr(hooks, "pre_tool_call_decide", None)
            if callable(decide_hook):

                async def _on_tool_call(call: ag_types.ToolCall) -> ag_types.HookResult:
                    nonlocal early_exit_hit
                    if not self.options.early_exit:
                        return ag_types.HookResult(allow=True)
                    args = getattr(call, "arguments", {}) or {}
                    path = args.get("path") or args.get("AbsolutePath")
                    if path:
                        skill = resolve_skill_from_path(path, self._resident)
                        if skill and (not detected_skills or detected_skills[-1] != skill):
                            detected_skills.append(skill)
                            if target_skill is not None and skill == target_skill:
                                early_exit_hit = True
                                return ag_types.HookResult(allow=False)
                            if len(detected_skills) >= self.options.max_turns:
                                early_exit_hit = True
                                return ag_types.HookResult(allow=False)
                    return ag_types.HookResult(allow=True)

                decide_hook(_on_tool_call)

            response = await agent.chat(query_text)
            data = await response.structured_output()
            observed_tools = tuple(
                [_tool_name(call.name) async for call in response.tool_calls],
            )
            stop_reason = response.stop_reason

        error = None
        if stop_reason not in EXPECTED_STOP_REASONS:
            error = f"runtime error: {stop_reason.value}"
        elif tool_leak := check_tool_leak(observed_tools, {_tool_name(t) for t in SELECTION_TOOLS}):
            error = tool_leak

        invoked = None
        reasoning: tuple[str, ...] = ()
        if isinstance(data, BaseModel):
            invoked = getattr(data, "selected_skill", None)
            if r := getattr(data, "reasoning", None):
                reasoning = (str(r).strip(),)
        elif isinstance(data, dict):
            invoked = data.get("selected_skill")
            if r := data.get("reasoning"):
                reasoning = (str(r).strip(),)

        if early_exit_hit and detected_skills:
            invoked = detected_skills[0]

        if detected_skills:
            invoked_skills = tuple(detected_skills)
        elif invoked:
            invoked_skills = (invoked,)
        else:
            invoked_skills = ()
        turns_taken = 1

        return SelectionOutcome(
            invoked_skills=invoked_skills,
            early_exit=early_exit_hit,
            turns_taken=turns_taken,
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
            return SelectionOutcome(error="timeout")
        except Exception as exc:  # noqa: BLE001
            return SelectionOutcome(error=str(exc))
        finally:
            self.post_probe(workdir)


class AntigravitySdkGenerator(BaseTextGenerator[AntigravitySdkOptions]):
    """Generate text completions using the Antigravity SDK."""

    name: str = "antigravity-sdk"
    options: AntigravitySdkOptions

    def __init__(
        self,
        model: str = "",
        *,
        timeout_s: int = 300,
        options: AntigravitySdkOptions | None = None,
    ) -> None:
        """Initialize Antigravity SDK generator with model, timeout, and options."""
        if not _HAS_ANTIGRAVITY:
            raise_missing_agent_dependency(
                "antigravity-sdk",
                ImportError("google-antigravity is not installed"),
                role="generator",
            )
        if isinstance(options, AntigravitySdkOptions):
            opts = options
        elif model:
            opts = AntigravitySdkOptions(model=model)
        else:
            opts = AntigravitySdkOptions()
        super().__init__(model=opts.model or model, timeout_s=timeout_s, options=opts)

    @property
    def effective_api_key(self) -> str | None:
        """Return configured API key or resolve from environment."""
        if self.options.api_key:
            return self.options.api_key
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def build_env(self) -> dict[str, str]:
        """Assemble environment variables with API key synchronization."""
        env = dict(os.environ)
        if api_key := self.effective_api_key:
            env["GEMINI_API_KEY"] = api_key
            env["GOOGLE_API_KEY"] = api_key
        return sync_google_and_gemini_keys(env)

    @property
    def effective_effort(self) -> str | None:
        """Return configured reasoning effort or default from model profile."""
        if self.options.effort:
            return self.options.effort
        profile = model_profile(self.model)
        return profile.effort

    def _model_spec(self) -> str | ag_types.ModelTarget:
        """Construct model target with reasoning effort endpoint options when configured."""
        if effort := self.effective_effort:
            return ag_types.ModelTarget(
                name=self.options.model or self.model,
                endpoint=ag_types.GeminiAPIEndpoint(
                    options=ag_types.GeminiModelOptions(thinking_level=effort),
                ),
            )
        return self.options.model or self.model

    @override
    def complete(self, prompt: str) -> str:
        """Execute text completion using the Antigravity SDK."""

        async def _complete_async() -> str:
            config = LocalAgentConfig(
                model=self._model_spec(),
                api_key=self.effective_api_key,
                env=self.build_env(),
            )
            async with Agent(config) as agent:
                response = await agent.chat(prompt)
                return await response.text()

        text = _run_sync(asyncio.wait_for(_complete_async(), timeout=self.timeout_s))
        self.completions += 1
        return text
