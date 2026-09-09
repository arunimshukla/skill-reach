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

"""Encapsulate provider-specific API key mapping and environment synchronization."""

from __future__ import annotations

from typing import NoReturn

_PROVIDER_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def sync_google_and_gemini_keys(env: dict[str, str]) -> dict[str, str]:
    """Synchronize GEMINI_API_KEY and GOOGLE_API_KEY bidirectionally in environment dict."""
    if "GEMINI_API_KEY" in env and "GOOGLE_API_KEY" not in env:
        env["GOOGLE_API_KEY"] = env["GEMINI_API_KEY"]
    elif "GOOGLE_API_KEY" in env and "GEMINI_API_KEY" not in env:
        env["GEMINI_API_KEY"] = env["GOOGLE_API_KEY"]
    return env


def apply_provider_api_key(
    env: dict[str, str],
    provider: str | None,
    api_key: str | None,
    default_provider: str = "google",
) -> dict[str, str]:
    """Map provider-specific API key into environment variables dictionary."""
    if not api_key:
        return env

    prov = (provider or default_provider).lower()
    for name, env_vars in _PROVIDER_KEY_ENV_VARS.items():
        if name in prov:
            for var in env_vars:
                env[var] = str(api_key)
            return env

    fallback_vars = _PROVIDER_KEY_ENV_VARS.get(default_provider, ("OPENAI_API_KEY",))
    for var in fallback_vars:
        env[var] = str(api_key)
    return env


def raise_missing_agent_dependency(
    agent: str,
    err: ImportError,
    role: str = "runtime",
) -> NoReturn:
    """Raise actionable RuntimeError when an optional agent dependency is missing."""
    if agent == "antigravity-sdk":
        msg = (
            f"google-antigravity is required for the antigravity-sdk {role}. "
            "Install it with: pip install 'skill-reach[antigravity-sdk]'"
        )
        raise RuntimeError(msg) from err
    raise err
