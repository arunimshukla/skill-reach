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
    """Map provider-specific API key into environment variables dictionary.

    Supported providers:
        - google / gemini: GEMINI_API_KEY, GOOGLE_API_KEY
        - anthropic: ANTHROPIC_API_KEY
        - openai: OPENAI_API_KEY

    When provider is None, the key is mapped according to default_provider.
    When provider is explicitly specified but unrecognized, raises ValueError
    to prevent unintended key leakage to other provider endpoints.
    """
    if not api_key:
        return env

    if provider is not None:
        prov = provider.strip().lower()
        for name, env_vars in _PROVIDER_KEY_ENV_VARS.items():
            if name in prov:
                for var in env_vars:
                    env[var] = str(api_key)
                return env

        supported = ", ".join(sorted(_PROVIDER_KEY_ENV_VARS))
        msg = (
            f"Unrecognized provider {provider!r} for automatic API key mapping. "
            f"Supported providers: {supported}. "
            "For custom or self-hosted providers, set required environment variables directly."
        )
        raise ValueError(msg)

    # Provider omitted: fall back to default_provider mapping
    default_prov = default_provider.strip().lower()
    fallback_vars = _PROVIDER_KEY_ENV_VARS.get(default_prov, ("OPENAI_API_KEY",))
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
