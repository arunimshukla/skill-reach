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

"""Encapsulate skill directory installation, symlink management, and path parsing."""

from __future__ import annotations

import shutil
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from reach.catalog import resident_skills

if TYPE_CHECKING:
    from reach.models import Catalog, Skill


def probe_slot_id() -> int:
    """Return an identifier unique to the calling OS thread."""
    return getattr(threading, "get_native_id", threading.get_ident)()


def probe_slot_dir(parent: Path, prefix: str = "slot") -> Path:
    """Return the scratch directory reserved for the calling thread under parent."""
    return Path(parent) / f"{prefix}_{probe_slot_id()}"


def resolve_catalog_skills(
    catalog: Catalog,
    skills: Iterable[Skill],
) -> dict[str, Skill]:
    """Resolve and validate that all skills referenced by a catalog exist."""
    return {s.name: s for s in resident_skills(catalog, tuple(skills))}


def install_skills(
    catalog: Catalog,
    by_name: Mapping[str, Skill],
    destination: Path,
    *,
    use_symlinks: bool = False,
) -> tuple[str, ...]:
    """Copy or symlink skill directories for a catalog into destination workspace."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in catalog.skills:
        src = by_name[name].path
        dst = destination / name
        if use_symlinks:
            try:
                dst.symlink_to(src, target_is_directory=True)
                continue
            except OSError:
                pass
        shutil.copytree(src, dst)
    return tuple(catalog.skills)


def resolve_skill_from_path(
    path_str: str | Path | None,
    resident: Iterable[str],
) -> str | None:
    """Extract matching resident skill name from markdown file path if present."""
    if not path_str or not isinstance(path_str, (str, Path)):
        return None
    if isinstance(path_str, str) and not path_str.strip():
        return None

    try:
        p = Path(path_str.strip() if isinstance(path_str, str) else path_str)
    except (ValueError, TypeError, OSError):
        return None

    resident_lookup = {r.lower(): r for r in resident}

    if p.name.lower() == "skill.md":
        candidate = p.parent.name.lower()
        if candidate in resident_lookup:
            return resident_lookup[candidate]

    if p.suffix.lower() == ".md":
        candidate = p.stem.lower()
        if candidate in resident_lookup:
            return resident_lookup[candidate]

    return None
