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

"""Provide TextGenerator protocol, base generator class, and generator factory."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from math import floor
from typing import Any, Protocol, cast, runtime_checkable

from reach.config import DEFAULT_GEMINI_MODEL
from reach.runtime._env import raise_missing_agent_dependency
from reach.runtime.profiles import model_profile

__all__ = [
    "BaseTextGenerator",
    "TextGenerator",
    "build_text_generator",
]


@runtime_checkable
class TextGenerator(Protocol):
    """Protocol for models or drivers capable of generating raw text completions."""

    @property
    def name(self) -> str:
        """Return generator name or agent identifier."""
        ...

    @property
    def model(self) -> str:
        """Return the model identifier used for generation."""
        ...

    @property
    def completions(self) -> int:
        """Return total number of completions executed."""
        ...

    @property
    def completion_cost_usd(self) -> float:
        """Return total cost accumulated across completions in USD."""
        ...

    def complete(self, prompt: str) -> str:
        """Generate a raw text completion for an arbitrary prompt."""
        ...

    def prompt_budget_chars(self) -> int | None:
        """Return maximum character length for prompts, or None if unbounded."""
        ...


class BaseTextGenerator[OptionsT](ABC):
    """Abstract base class providing character budgeting and completion accounting."""

    name: str = "generator"
    options: OptionsT

    def __init__(
        self,
        model: str,
        *,
        timeout_s: int = 300,
        options: OptionsT | None = None,
    ) -> None:
        """Initialize generator with model ID, timeout, and options."""
        self._model = model
        self.timeout_s = timeout_s
        self.options = cast("OptionsT", options)
        self.completions = 0
        self.completion_cost_usd = 0.0

    @property
    def model(self) -> str:
        """Return configured model identifier."""
        return self._model

    def prompt_budget_chars(self) -> int | None:
        """Return maximum character length for prompts, or None if unbounded."""
        try:
            profile = model_profile(self.model)
            window = profile.completion_window or profile.context_window
            return floor(window * profile.chars_per_token)
        except (KeyError, ValueError):
            return None

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Generate a raw text completion for an arbitrary prompt."""
        ...


def _build_agent_generator(
    target_agent: str,
    model: str,
    timeout_s: int,
    opts: dict[str, Any],
) -> TextGenerator | None:
    """Instantiate a concrete TextGenerator matching the requested agent name."""
    if target_agent == "fake":
        from reach.runtime.fake import FakeGenerator, FakeOptions, resolve_fake_options

        fake_opts = resolve_fake_options(opts) if opts else FakeOptions(model=model)
        budget = opts.get("prompt_budget_chars")
        return FakeGenerator(
            model=fake_opts.model or model,
            prompt_budget_chars=budget if isinstance(budget, int) else None,
            cost_usd=float(opts.get("cost_usd", 0.0) or 0.0),
            timeout_s=timeout_s,
        )
    if target_agent == "keyword":
        from reach.runtime.keyword import KeywordGenerator

        return KeywordGenerator(model=model, timeout_s=timeout_s)

    cli_map = {
        "claude-code": (
            "reach.runtime.claude_code",
            "ClaudeCodeOptions",
            "ClaudeGenerator",
        ),
        "antigravity-cli": (
            "reach.runtime.antigravity_cli",
            "AntigravityCliOptions",
            "AntigravityCliGenerator",
        ),
        "goose": ("reach.runtime.goose", "GooseOptions", "GooseGenerator"),
        "pi": ("reach.runtime.pi", "PiOptions", "PiGenerator"),
        "antigravity-sdk": (
            "reach.runtime.antigravity_sdk",
            "AntigravitySdkOptions",
            "AntigravitySdkGenerator",
        ),
    }
    spec = cli_map.get(target_agent)
    if spec is not None:
        mod_name, opt_cls_name, gen_cls_name = spec
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as err:
            raise_missing_agent_dependency(target_agent, err, role="generator")
        opt_cls = getattr(mod, opt_cls_name)
        gen_cls = getattr(mod, gen_cls_name)
        validated_opts = opt_cls.model_validate({"model": model, **opts})
        return gen_cls(model=model, timeout_s=timeout_s, options=validated_opts)
    return None


def build_text_generator(
    model: str | None = None,
    agent: str | None = None,
    timeout_s: float | None = None,
    options: Mapping[str, Any] | None = None,
) -> TextGenerator:
    """Construct a TextGenerator instance configured for query drafting or optimization."""
    from reach.config import agent_default_model, default_agent
    from reach.runtime import known_agents

    target_agent = agent or default_agent()
    opts = dict(options or {})
    target_model = (
        model
        or (opts.get("model") if isinstance(opts.get("model"), str) else None)
        or agent_default_model(target_agent)
        or DEFAULT_GEMINI_MODEL
    )
    opts["model"] = target_model

    timeout_int = int(timeout_s) if timeout_s is not None else 300
    gen = _build_agent_generator(target_agent, str(target_model), timeout_int, opts)
    if gen is not None:
        return gen

    if target_agent not in known_agents():
        agents = ", ".join(known_agents())
        msg = f"unknown runtime agent {target_agent!r}; expected one of {agents}"
        raise ValueError(msg)

    msg = f"runtime agent {target_agent!r} does not support text generation"
    raise ValueError(msg)
