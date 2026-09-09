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

"""Hermetic integration tests for Google Cloud Agent Registry using local HTTP server."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from reach.registry import (
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RegistryCacheManager,
    RegistryClient,
    ServiceDisabledError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import MockRegistryHandler

pytestmark = pytest.mark.integration


def test_server_authentication_and_headers(
    local_registry_server: str,
    registry_handler: type[MockRegistryHandler],
) -> None:
    """Verify RegistryClient injects Bearer token and expected user-agent and accept headers."""
    client = RegistryClient(
        token="mock-secret-bearer-token",  # noqa: S106
        base_url=local_registry_server,
    )
    skills = client.list_skills(project="test-proj", location="global")

    assert len(skills) == 1
    assert skills[0].identifier == "cloud-storage"
    headers = registry_handler.captured_headers
    assert headers.get("Authorization") == "Bearer mock-secret-bearer-token"
    assert headers.get("Accept") == "application/json"
    assert headers.get("User-Agent") == "skill-reach"


def test_server_pagination_loop(
    local_registry_server: str,
    registry_handler: type[MockRegistryHandler],
) -> None:
    """Verify RegistryClient loops across all pages until nextPageToken is exhausted."""
    base_url = f"{local_registry_server}/paginated"
    client = RegistryClient(token="valid-token", base_url=base_url)  # noqa: S106
    skills = client.list_skills(project="test-proj", location="global")

    assert len(skills) == 2
    identifiers = [s.identifier for s in skills]
    assert identifiers == ["cloud-run", "cloud-sql"]
    assert registry_handler.captured_params.get("pageToken") == ["page-2"]


def test_server_rate_limit_retry_with_backoff(
    local_registry_server: str,
    registry_handler: type[MockRegistryHandler],
) -> None:
    """Verify RegistryClient automatically retries on HTTP 429 and succeeds."""
    base_url = f"{local_registry_server}/retry-test"
    client = RegistryClient(
        token="valid-token",  # noqa: S106
        base_url=base_url,
        max_retries=2,
        backoff_factor=0.01,
    )
    skills = client.list_skills(project="test-proj", location="global")

    assert len(skills) == 1
    assert skills[0].identifier == "cloud-storage"
    retry_counts = sum(
        count for path, count in registry_handler.request_counts.items() if "/retry-test" in path
    )
    assert retry_counts == 2


@pytest.mark.parametrize(
    ("token", "path_suffix", "expected_exc"),
    [
        ("invalid-token", "", AuthenticationError),
        ("valid-token", "/forbidden", PermissionDeniedError),
        ("valid-token", "/disabled", ServiceDisabledError),
        ("valid-token", "/notfound", NotFoundError),
    ],
)
def test_server_error_mappings(
    local_registry_server: str,
    token: str,
    path_suffix: str,
    expected_exc: type[Exception],
) -> None:
    """Verify HTTP status codes map to domain-specific exceptions."""
    client = RegistryClient(
        token=token,
        base_url=f"{local_registry_server}{path_suffix}",
    )
    with pytest.raises(expected_exc):
        client.list_skills(project="test-proj", location="global")


def test_server_local_disk_cache_hydration(
    local_registry_server: str,
    tmp_path: Path,
) -> None:
    """Verify RegistryCacheManager fetches from local server, caches, and hydrates files."""
    cache_root = tmp_path / "registry_cache"
    cache_mgr = RegistryCacheManager(cache_root=cache_root)
    client = RegistryClient(token="valid-token", base_url=local_registry_server)  # noqa: S106

    # 1. First fetch: retrieves from server and hydrates local cache
    skills = cache_mgr.resolve_skills(
        project="test-proj",
        location="global",
        client=client,
    )
    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "cloud-storage"
    assert skill.path.exists()
    assert (skill.path / "SKILL.md").is_file()

    manifest_file = cache_mgr.manifest_path("test-proj", "global")
    assert manifest_file.is_file()

    # 2. Second fetch: loads directly from cache without server contact
    # Point client to invalid endpoint to ensure no HTTP requests are made
    broken_client = RegistryClient(
        token="valid-token",  # noqa: S106
        base_url="http://127.0.0.1:1",
    )
    cached_skills = cache_mgr.resolve_skills(
        project="test-proj",
        location="global",
        client=broken_client,
    )
    assert len(cached_skills) == 1
    assert cached_skills[0].name == "cloud-storage"
