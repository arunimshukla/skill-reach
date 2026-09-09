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

"""Model and query context window parameters and token ratios for language models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from reach.config import load_config

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "ModelProfile",
    "model_profile",
]


class ModelProfile(BaseModel):
    """Hold measured token ratios and context window parameters for a specific model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chars_per_token: float = Field(default=4.0, gt=0)

    context_window: int = Field(default=1_048_576, gt=0)

    completion_window: int | None = Field(default=None, gt=0)

    budget_fraction_places: int = Field(default=3, gt=0)
    listing_budget_fraction: float = Field(default=0.01, gt=0, le=1.0)
    effort: str | None = Field(default=None)


def _lookup_model_profile(
    models_table: dict[str, object],
    model_lower: str,
    model_hyphen: str,
) -> ModelProfile | None:
    """Look up model configuration by exact match, hyphenation, or prefix."""
    if model_lower in models_table:
        return ModelProfile.model_validate(models_table[model_lower])
    if model_hyphen in models_table:
        return ModelProfile.model_validate(models_table[model_hyphen])

    for prefix in sorted(models_table, key=len, reverse=True):
        if prefix in model_lower or prefix in model_hyphen:
            return ModelProfile.model_validate(models_table[prefix])
        if model_lower in prefix or model_hyphen in prefix:
            return ModelProfile.model_validate(models_table[prefix])
    return None


def model_profile(model: str, config_path: Path | str | None = None) -> ModelProfile:
    """Look up context window and token parameters for a model, falling back to defaults."""
    config = load_config(config_path)
    raw_models = config.get("models", {})
    if not isinstance(raw_models, dict):
        return ModelProfile()

    models_table: dict[str, object] = {
        str(k).lower(): v for k, v in raw_models.items() if isinstance(v, dict)
    }
    model_lower = model.lower()
    model_hyphen = model_lower.replace(".", "-")
    return _lookup_model_profile(models_table, model_lower, model_hyphen) or ModelProfile()
