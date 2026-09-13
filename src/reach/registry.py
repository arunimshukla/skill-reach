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

"""Provide client access, authentication, and two-tier caching for Google Cloud Agent Registry."""

import contextlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reach._io import atomic_write_text
from reach.config import resolve_path
from reach.models import Skill

__all__ = [
    "AuthenticationError",
    "NotFoundError",
    "PermissionDeniedError",
    "RegistryCacheManager",
    "RegistryClient",
    "RegistryError",
    "RegistryManifest",
    "RegistrySkillData",
    "ServiceDisabledError",
    "find_adc_path",
    "get_access_token",
    "is_adc_available",
]


class RegistryError(RuntimeError):
    """Base exception for Google Cloud Agent Registry operations."""


class AuthenticationError(RegistryError):
    """Raised when authentication credentials cannot be obtained or are invalid."""


class PermissionDeniedError(RegistryError):
    """Raised when caller lacks required IAM permissions (403 Forbidden)."""


class ServiceDisabledError(RegistryError):
    """Raised when the Agent Registry API is not enabled on the target project."""


class NotFoundError(RegistryError):
    """Raised when target project, location, or skill is not found (404)."""


class RegistrySkillData(BaseModel):
    """Represent raw skill metadata returned by the Agent Registry REST API."""

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    name: str
    display_name: str = Field(alias="displayName", default="")
    description: str = ""
    type: str = "SIMPLE"
    state: str = "STATE_ACTIVE"
    target_state: str = Field(alias="targetState", default="TARGET_STATE_ACTIVE")
    default_revision: str | None = Field(alias="defaultRevision", default=None)
    skill_id: str | None = Field(alias="skillId", default=None)
    publisher: str | None = None
    create_time: str | None = Field(alias="createTime", default=None)
    update_time: str | None = Field(alias="updateTime", default=None)

    @property
    def identifier(self) -> str:
        """Return the display name if available, otherwise trailing segment of resource name."""
        if self.display_name:
            return self.display_name
        return self.name.rsplit("/", 1)[-1]

    @property
    def revision_slug(self) -> str:
        """Extract the revision ID from defaultRevision or return 'default'."""
        if self.default_revision and "/" in self.default_revision:
            return self.default_revision.rsplit("/", 1)[-1]
        return "default"


class RegistryManifest(BaseModel):
    """Persisted snapshot of skills metadata in a project and location."""

    model_config = ConfigDict(frozen=True)

    project: str
    location: str
    publisher: str | None = None
    fetched_at: datetime
    skills: tuple[RegistrySkillData, ...] = ()


def find_adc_path() -> Path | None:
    """Locate local Google Cloud Application Default Credentials file if present."""
    if custom := os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        custom_path = Path(custom).expanduser()
        if custom_path.is_file():
            return custom_path
    if cloudsdk_config := os.environ.get("CLOUDSDK_CONFIG"):
        sdk_path = Path(cloudsdk_config).expanduser() / "application_default_credentials.json"
        if sdk_path.is_file():
            return sdk_path
    candidates = [
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json",
    ]
    if appdata := os.environ.get("APPDATA"):
        candidates.append(Path(appdata) / "gcloud" / "application_default_credentials.json")
    candidates.append(
        Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def is_adc_available() -> bool:
    """Return True if Application Default Credentials are configured locally without network I/O."""
    return find_adc_path() is not None


def get_access_token() -> str:
    """Resolve an OAuth2 access token for Google Cloud APIs via ADC.

    Attempts to resolve Application Default Credentials using google-auth first,
    falling back to `gcloud auth application-default print-access-token`.

    Returns:
        A valid OAuth2 bearer access token string.

    Raises:
        AuthenticationError: If credentials cannot be acquired or refreshed.
    """
    # 1. Try google-auth library if installed
    with contextlib.suppress(Exception):
        import google.auth
        import google.auth.transport.requests

        creds, _project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        request = google.auth.transport.requests.Request()
        creds.refresh(request)
        if creds.token:
            return str(creds.token)

    # 2. Fall back to gcloud auth application-default print-access-token
    gcloud_bin = shutil.which("gcloud") or "gcloud"
    try:
        proc = subprocess.run(  # noqa: S603
            [gcloud_bin, "auth", "application-default", "print-access-token"],
            check=True,
            capture_output=True,
            text=True,
        )
        token = proc.stdout.strip()
        if token:
            return token
    except (subprocess.SubprocessError, FileNotFoundError) as err:
        msg = (
            "Unable to obtain Google Cloud access token. Please run "
            "'gcloud auth application-default login' to authenticate."
        )
        raise AuthenticationError(msg) from err

    msg = (
        "Empty access token returned. Please run "
        "'gcloud auth application-default login' to refresh credentials."
    )
    raise AuthenticationError(msg)


def _matches_publisher(actual_publisher: str | None, requested_publisher: str | None) -> bool:
    """Check if actual publisher resource name or identifier matches the requested publisher."""
    if not requested_publisher:
        return True
    if not actual_publisher:
        return False
    if actual_publisher == requested_publisher:
        return True
    return actual_publisher.endswith(f"/{requested_publisher}")


class RegistryClient:
    """HTTP client for querying the Google Cloud Agent Registry REST API."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://agentregistry.googleapis.com/v1alpha",
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> None:
        """Initialize RegistryClient with optional bearer token and retry settings."""
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def _get_token(self) -> str:
        """Resolve active authentication token, fetching dynamically if omitted."""
        if self.token:
            return self.token
        return get_access_token()

    def _request(self, endpoint: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        """Execute an authenticated GET request with rate-limit retry and error mapping."""
        query_string = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"{self.base_url}/{endpoint.lstrip('/')}{query_string}"
        token = self._get_token()

        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(  # noqa: S310
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "User-Agent": "skill-reach",
                },
            )
            try:
                with urllib.request.urlopen(req) as resp:  # noqa: S310
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw)  # type: ignore[no-any-return]
            except urllib.error.HTTPError as http_err:
                status = http_err.code
                error_body = ""
                with contextlib.suppress(OSError):
                    error_body = http_err.read().decode("utf-8")
                http_err.close()

                if status == HTTPStatus.TOO_MANY_REQUESTS and attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2**attempt)
                    time.sleep(sleep_time)
                    continue

                self._handle_http_error(status, error_body, endpoint)
            except urllib.error.URLError as url_err:
                msg = f"Failed to connect to Agent Registry endpoint ({url}): {url_err.reason}"
                raise RegistryError(msg) from url_err

        msg = f"Agent Registry request exceeded maximum retries for {url}"
        raise RegistryError(msg)

    def _handle_http_error(self, status: int, body: str, endpoint: str) -> None:
        """Map HTTP error status codes to descriptive domain exceptions."""
        error_details = {}
        error_message = body
        with contextlib.suppress(json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            parsed = json.loads(body)
            if isinstance(parsed, dict) and "error" in parsed:
                err_dict = parsed["error"]
                error_message = err_dict.get("message", body)
                error_details = err_dict

        if status == HTTPStatus.UNAUTHORIZED:
            msg = (
                "Authentication failed when contacting Agent Registry. "
                "Run 'gcloud auth application-default login' to refresh credentials."
            )
            raise AuthenticationError(msg)

        if status == HTTPStatus.FORBIDDEN:
            # Check for service disabled
            details_list = error_details.get("details", [])
            for item in details_list:
                if isinstance(item, dict) and item.get("reason") == "SERVICE_DISABLED":
                    msg = (
                        "The Agent Registry API (agentregistry.googleapis.com) is disabled. "
                        "Enable it with: gcloud services enable agentregistry.googleapis.com"
                    )
                    raise ServiceDisabledError(msg)

            msg = (
                f"Permission denied on Agent Registry ({endpoint}). "
                "Ensure your account has the 'roles/agentregistry.user' role on the project."
            )
            raise PermissionDeniedError(msg)

        if status == HTTPStatus.NOT_FOUND:
            msg = f"Agent Registry resource not found ({endpoint}): {error_message}"
            raise NotFoundError(msg)

        msg = f"Agent Registry request failed with HTTP {status}: {error_message}"
        raise RegistryError(msg)

    def list_skills(
        self,
        project: str,
        location: str = "global",
        publisher: str | None = None,
    ) -> list[RegistrySkillData]:
        """Fetch all registered skills in the specified project and location."""
        endpoint = f"projects/{project}/locations/{location}/skills"
        skills: list[RegistrySkillData] = []
        page_token: str | None = None

        while True:
            params: dict[str, str] = {}
            if page_token:
                params["pageToken"] = page_token

            data = self._request(endpoint, params=params or None)
            raw_skills = data.get("skills", [])
            for s in raw_skills:
                try:
                    parsed = RegistrySkillData.model_validate(s)
                    if not _matches_publisher(parsed.publisher, publisher):
                        continue
                    skills.append(parsed)
                except (ValidationError, ValueError):
                    continue

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return skills

    def get_skill(
        self,
        project: str,
        location: str,
        skill_id: str,
    ) -> RegistrySkillData:
        """Fetch a single skill resource from Agent Registry."""
        endpoint = f"projects/{project}/locations/{location}/skills/{skill_id}"
        data = self._request(endpoint)
        return RegistrySkillData.model_validate(data)

    def fetch_manifest(
        self,
        project: str,
        location: str = "global",
        publisher: str | None = None,
    ) -> RegistryManifest:
        """Compile a complete RegistryManifest snapshot for project and location."""
        raw_skills = self.list_skills(project=project, location=location, publisher=publisher)
        return RegistryManifest(
            project=project,
            location=location,
            publisher=publisher,
            fetched_at=datetime.now(UTC),
            skills=tuple(raw_skills),
        )


class RegistryCacheManager:
    """Manage local caching and payload hydration of Agent Registry skills."""

    def __init__(self, cache_root: Path | str | None = None) -> None:
        """Initialize RegistryCacheManager with optional root directory."""
        if cache_root is not None:
            self.cache_root = resolve_path(cache_root)
        else:
            self.cache_root = resolve_path(".reach") / "cache" / "registry"

    def location_dir(self, project: str, location: str) -> Path:
        """Return the directory containing manifest and cached skills for project and location."""
        return self.cache_root / project / location

    def manifest_path(
        self,
        project: str,
        location: str,
        publisher: str | None = None,
    ) -> Path:
        """Return the path to the cached .manifest.json file, optionally scoped by publisher."""
        loc_dir = self.location_dir(project, location)
        if publisher:
            slug = re.sub(r"[^\w.-]", "_", publisher)
            return loc_dir / f".manifest.{slug}.json"
        return loc_dir / ".manifest.json"

    def skill_dir(
        self,
        project: str,
        location: str,
        skill_name: str,
        revision_slug: str = "default",
    ) -> Path:
        """Return the directory for an unpacked skill revision."""
        return self.location_dir(project, location) / skill_name / revision_slug

    def get_cached_manifest(
        self,
        project: str,
        location: str,
        publisher: str | None = None,
        max_age_seconds: int = 300,
    ) -> RegistryManifest | None:
        """Load cached manifest if present, compatible with publisher, and within TTL."""
        candidate_paths = [self.manifest_path(project, location, publisher)]
        if publisher is not None:
            candidate_paths.append(self.manifest_path(project, location, None))

        for manifest_file in candidate_paths:
            if not manifest_file.is_file():
                continue

            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
                manifest = RegistryManifest.model_validate(data)
                now = datetime.now(UTC)
                age = (now - manifest.fetched_at).total_seconds()
                if max_age_seconds >= 0 and age > max_age_seconds:
                    continue
                if manifest.publisher is not None:
                    if publisher is None:
                        continue
                    if not _matches_publisher(manifest.publisher, publisher):
                        continue
                return manifest
            except (json.JSONDecodeError, ValidationError, OSError):
                continue

        return None

    def save_manifest(self, manifest: RegistryManifest) -> None:
        """Atomically persist a RegistryManifest to disk."""
        target_path = self.manifest_path(manifest.project, manifest.location, manifest.publisher)
        atomic_write_text(target_path, manifest.model_dump_json(indent=2) + "\n")

    def hydrate_skill_file(
        self,
        project: str,
        location: str,
        skill_data: RegistrySkillData,
    ) -> Path:
        """Generate local directory and SKILL.md file for a registry skill if absent.

        Args:
            project: Google Cloud project ID.
            location: Registry location.
            skill_data: RegistrySkillData object.

        Returns:
            Path to the enclosing directory of the hydrated skill.
        """
        skill_dir = self.skill_dir(
            project=project,
            location=location,
            skill_name=skill_data.identifier,
            revision_slug=skill_data.revision_slug,
        )
        skill_file = skill_dir / "SKILL.md"

        if not skill_file.is_file():
            body_content = (
                f"---\n"
                f"name: {skill_data.identifier}\n"
                f"description: {skill_data.description}\n"
                f"---\n\n"
                f"# {skill_data.identifier}\n\n"
                f"{skill_data.description}\n\n"
                f"<!-- Registry source: {skill_data.name} -->\n"
            )
            if skill_data.skill_id:
                body_content += f"<!-- Skill ID: {skill_data.skill_id} -->\n"

            atomic_write_text(skill_file, body_content)

        return skill_dir

    def resolve_skills(
        self,
        project: str,
        location: str = "global",
        publisher: str | None = None,
        fresh: bool = False,
        no_cache: bool = False,
        cache_ttl_seconds: int = 300,
        client: RegistryClient | None = None,
    ) -> list[Skill]:
        """Fetch or load skills from cache, hydrate local files, and return Skill models.

        Args:
            project: Google Cloud project ID.
            location: Registry location (e.g. 'global').
            publisher: Optional publisher filter.
            fresh: If True, bypass metadata TTL and fetch live.
            no_cache: If True, do not use or persist cache.
            cache_ttl_seconds: TTL in seconds for metadata cache validity.
            client: Optional pre-configured RegistryClient.

        Returns:
            List of validated Skill models.
        """
        manifest: RegistryManifest | None = None
        if not fresh and not no_cache:
            manifest = self.get_cached_manifest(
                project,
                location,
                publisher=publisher,
                max_age_seconds=cache_ttl_seconds,
            )

        if manifest is None:
            active_client = client or RegistryClient()
            try:
                manifest = active_client.fetch_manifest(
                    project=project,
                    location=location,
                    publisher=publisher,
                )
                if not no_cache:
                    self.save_manifest(manifest)
            except Exception as net_err:
                # If network fails, try falling back to stale cache
                stale = self.get_cached_manifest(
                    project,
                    location,
                    publisher=publisher,
                    max_age_seconds=-1,
                )
                if stale is not None:
                    manifest = stale
                else:
                    raise net_err

        skills: list[Skill] = []
        for s in manifest.skills:
            if not _matches_publisher(s.publisher, publisher):
                continue

            skill_path = self.hydrate_skill_file(project, location, s)
            model_invocable = s.state == "STATE_ACTIVE"

            meta = {
                "urn": s.skill_id or "",
                "publisher": s.publisher or "",
                "project": project,
                "location": location,
                "default_revision": s.default_revision or "",
                "state": s.state,
            }

            skills.append(
                Skill(
                    name=s.identifier,
                    description=s.description,
                    path=skill_path,
                    metadata=meta,
                    manifest_source=f"agent-registry://{project}/{location}",
                    model_invocable=model_invocable,
                ),
            )

        return sorted(skills, key=lambda sk: sk.name)

    def clean(self, project: str | None = None, dry_run: bool = False) -> tuple[int, list[Path]]:
        """Clean cached registry files, optionally scoped to a single project.

        Args:
            project: Optional project ID to limit cleanup scope.
            dry_run: If True, compute freed bytes and paths without deleting.

        Returns:
            Tuple of (total_bytes_reclaimed, list_of_paths_removed).
        """
        target_dir = self.cache_root / project if project else self.cache_root
        if not target_dir.exists():
            return 0, []

        total_bytes = 0
        removed_paths: list[Path] = []

        for p in target_dir.rglob("*"):
            if p.is_file():
                with contextlib.suppress(OSError):
                    total_bytes += p.stat().st_size

        removed_paths.append(target_dir)

        if not dry_run:
            if project:
                shutil.rmtree(target_dir, ignore_errors=True)
            else:
                shutil.rmtree(self.cache_root, ignore_errors=True)

        return total_bytes, removed_paths
