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

"""Verify AgentRuntime interface contracts, factory registration, and model profiles."""

from __future__ import annotations

import contextlib
import subprocess
import tomllib
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast, override

import pytest
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from reach.catalog import build_catalogs, load_skills
from reach.config import RuntimeSettings, agent_profiles, default_agent, load_config
from reach.models import Catalog, CatalogMode, Skill
from reach.runtime import (
    AgentRuntime,
    AntigravityRuntime,
    CatalogFit,
    CliAgentRuntime,
    CliOptions,
    SelectionOutcome,
    SessionSummary,
    SkillRoot,
    SkillSelectionBase,
    TextGenerator,
    ToolCallInfo,
    agent_default_model,
    build_runtime,
    build_text_generator,
    cli_agents,
    find_agent_for_model,
    known_agents,
    options_model,
)
from reach.runtime._env import (
    apply_provider_api_key,
    sync_google_and_gemini_keys,
)
from reach.runtime._fs import (
    install_skills,
    resolve_skill_from_path,
)
from reach.runtime._subprocess import (
    check_tool_leak,
    extract_content_reasoning,
    format_subprocess_error,
    process_failure_reason,
)
from reach.runtime.antigravity_cli import (
    AntigravityCliOptions,
    AntigravityCliRuntime,
)
from reach.runtime.antigravity_sdk import _HAS_ANTIGRAVITY, AntigravitySdkOptions
from reach.runtime.claude_code import (
    ClaudeCodeOptions,
    ClaudeCodeRuntime,
    ClaudeGenerator,
)
from reach.runtime.fake import FakeGenerator, FakeOptions, FakeRuntime
from reach.runtime.goose import GooseOptions
from reach.runtime.keyword import KeywordOptions, KeywordRuntime
from reach.runtime.pi import PiGenerator, PiOptions
from reach.runtime.profiles import model_profile

from .conftest import MINIMAL_OPTIONS
from .conftest import build_agent as _build_agent

_EXECUTION_AGENTS = tuple(a for a in known_agents() if a not in ("keyword", "fake"))


def test_agent_runtime_cannot_be_instantiated_directly() -> None:
    """Verify AgentRuntime ABC raises TypeError when instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class AgentRuntime"):
        AgentRuntime()  # type: ignore[abstract]


def test_incomplete_runtime_subclass_cannot_be_instantiated() -> None:
    """Verify subclasses missing abstract methods cannot be instantiated."""

    class IncompleteRuntime(AgentRuntime):
        name = "incomplete"

    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteRuntime"):
        IncompleteRuntime()  # type: ignore[abstract]


def _assert_runtime_attributes(runtime: AgentRuntime, agent: str) -> None:
    """Verify runtime instance meets required attribute and counter types."""
    assert isinstance(runtime.name, str)
    assert runtime.name == agent
    assert isinstance(runtime.model, str)
    assert len(runtime.model) > 0
    assert isinstance(runtime.skills_subpath, str)
    profile = agent_profiles().get(agent)
    if profile is not None and profile.skills_dir:
        assert runtime.skills_subpath == profile.skills_dir
    assert hasattr(runtime, "options")
    assert isinstance(runtime.options, BaseModel)


def _assert_runtime_interface(runtime: AgentRuntime, tmp_path: Path) -> None:
    """Verify runtime instance satisfies core interface method contracts."""
    workdir = tmp_path / runtime.name
    sdir = runtime.skills_dir(workdir)
    assert isinstance(sdir, Path)
    assert sdir == workdir / runtime.skills_subpath

    roots = runtime.skill_roots(workdir)
    assert isinstance(roots, tuple)
    assert all(isinstance(root, SkillRoot) for root in roots)

    empty_catalog = Catalog(id="test", mode=CatalogMode.ALL, skills=())
    fit = runtime.fit(empty_catalog, ())
    assert isinstance(fit, CatalogFit)

    assert callable(runtime.skills_dir)
    assert callable(runtime.skill_roots)
    assert callable(runtime.fit)
    assert callable(runtime.install)
    assert callable(runtime.select)


def _assert_generator_interface(gen: TextGenerator, agent: str) -> None:
    """Verify text generator instance satisfies generator contract."""
    assert isinstance(gen.name, str)
    assert gen.name == agent
    assert isinstance(gen.model, str)
    assert isinstance(gen.completion_cost_usd, float)
    assert gen.completion_cost_usd >= 0.0
    assert isinstance(gen.completions, int)
    assert gen.completions >= 0
    budget = gen.prompt_budget_chars()
    assert budget is None or (isinstance(budget, int) and budget > 0)
    assert callable(gen.complete)
    assert callable(gen.prompt_budget_chars)


@pytest.mark.parametrize("agent", known_agents())
def test_every_agent_runtime_conforms_to_agent_runtime_contract(
    agent: str,
    tmp_path: Path,
) -> None:
    """Verify every registered agent implementation satisfies the AgentRuntime ABC contract."""
    if agent == "antigravity-sdk" and not _HAS_ANTIGRAVITY:
        pytest.skip("google-antigravity is not installed")
    runtime = build_runtime(
        RuntimeSettings(agent=agent, options=MINIMAL_OPTIONS.get(agent, {})),
    )
    assert isinstance(runtime, AgentRuntime)
    assert issubclass(type(runtime), AgentRuntime)
    _assert_runtime_attributes(runtime, agent)
    _assert_runtime_interface(runtime, tmp_path)


@pytest.mark.parametrize("agent", known_agents())
def test_every_advertised_agent_can_be_built(agent: str) -> None:
    """Verify build_runtime successfully instantiates every agent in known_agents."""
    if agent == "antigravity-sdk" and not _HAS_ANTIGRAVITY:
        pytest.skip("google-antigravity is not installed")
    runtime = build_runtime(
        RuntimeSettings(agent=agent, options=MINIMAL_OPTIONS.get(agent, {})),
    )
    assert isinstance(runtime, AgentRuntime)
    assert runtime.name == agent


def test_an_unknown_agent_names_the_alternatives() -> None:
    """Verify ValueError lists available agents when unknown agent is requested."""
    with pytest.raises(
        ValueError,
        match="antigravity-cli, antigravity-sdk, claude-code, fake, goose, keyword, pi",
    ):
        build_runtime(RuntimeSettings(agent="codex"))


def test_the_agent_carries_the_configured_model() -> None:
    """Verify configured model string is stored on constructed runtime instance."""
    assert build_runtime(RuntimeSettings(options={"model": "haiku"})).model == "haiku"
    keyword_runtime = build_runtime(
        RuntimeSettings(agent="keyword", options={"model": "haiku"}),
    )
    assert keyword_runtime.model == "haiku"


def test_two_agents_make_the_seam_real() -> None:
    """Verify known_agents includes both ClaudeCodeRuntime and KeywordRuntime."""
    assert {ClaudeCodeRuntime.name, KeywordRuntime.name} <= set(known_agents())


@pytest.mark.parametrize("agent", known_agents())
def test_every_agent_answers_where_it_reads_skills_from(agent: str, tmp_path: Path) -> None:
    """Verify every agent implements skill_roots returning a tuple of SkillRoot."""
    runtime = _build_agent(agent, tmp_path)
    roots = runtime.skill_roots(tmp_path / agent)
    assert isinstance(roots, tuple)
    assert all(isinstance(root, SkillRoot) for root in roots)


def test_a_root_ranks_itself_and_says_under_which_scope(tmp_path: Path) -> None:
    """Verify SkillRoot model validates path, scope, and non-negative precedence."""
    root = SkillRoot(path=tmp_path, scope="user")
    assert root.precedence == 0
    with pytest.raises(PydanticValidationError):
        SkillRoot.model_validate({"path": tmp_path, "scope": "user", "precedence": -1})


@pytest.mark.parametrize("agent", known_agents())
def test_install_rejects_a_catalog_naming_an_unloaded_skill(
    agent: str,
    catalog: Catalog,
    skills: list[Skill],
    tmp_path: Path,
) -> None:
    """Verify install raises KeyError when catalog references unindexed skill."""
    ghost = catalog.model_copy(update={"skills": ("ghost",)})
    runtime = _build_agent(agent, tmp_path)
    with pytest.raises(KeyError, match="ghost"):
        runtime.install(ghost, skills, tmp_path / agent)


def test_the_fake_reports_the_catalog_it_was_given(catalog, skills, tmp_path) -> None:
    """Verify FakeRuntime observed_catalog matches installed catalog skills."""
    runtime = FakeRuntime({"q": "a"})
    runtime.install(catalog, skills, tmp_path)
    assert runtime.select("q", tmp_path).observed_catalog == ("a", "b")


def test_the_fake_abstains_on_anything_unscripted(catalog, skills, tmp_path) -> None:
    """Verify FakeRuntime returns None for unscripted queries."""
    runtime = FakeRuntime({"scripted": "a"})
    runtime.install(catalog, skills, tmp_path)
    assert runtime.select("unscripted", tmp_path).invoked_skill is None


def test_selection_outcome_carries_no_invocations_by_default() -> None:
    """Verify default SelectionOutcome initializes with empty invoked_skills tuple."""
    assert SelectionOutcome().invoked_skills == ()


def test_the_fake_names_its_one_selection_in_invoked_skills_too(
    catalog,
    skills,
    tmp_path,
) -> None:
    """Verify FakeRuntime mirrors invoked_skill into invoked_skills tuple."""
    runtime = FakeRuntime({"q": "a"})
    runtime.install(catalog, skills, tmp_path)
    assert runtime.select("q", tmp_path).invoked_skills == ("a",)


def test_the_fakes_abstention_carries_no_invocations(catalog, skills, tmp_path) -> None:
    """Verify FakeRuntime abstention produces empty invoked_skills tuple."""
    runtime = FakeRuntime({"scripted": "a"})
    runtime.install(catalog, skills, tmp_path)
    assert runtime.select("unscripted", tmp_path).invoked_skills == ()


def test_a_scripted_outcome_can_override_residency(catalog, skills, tmp_path) -> None:
    """Verify FakeRuntime allows explicit SelectionOutcome return overrides."""
    runtime = FakeRuntime({"q": SelectionOutcome(observed_catalog=("wrong",))})
    runtime.install(catalog, skills, tmp_path)
    assert runtime.select("q", tmp_path).observed_catalog == ("wrong",)


def test_a_callable_script_sees_the_query(catalog, skills, tmp_path) -> None:
    """Verify FakeRuntime accepts callable dispatch function for query selection."""
    runtime = FakeRuntime(lambda text: "a" if "storage" in text else None)
    runtime.install(catalog, skills, tmp_path)
    assert runtime.select("about storage", tmp_path).invoked_skill == "a"
    assert runtime.select("about networks", tmp_path).invoked_skill is None


def test_the_fake_records_what_it_was_asked(catalog, skills, tmp_path) -> None:
    """Verify FakeRuntime tracks historical installs and queries."""
    runtime = FakeRuntime()
    runtime.install(catalog, skills, tmp_path)
    runtime.select("q1", tmp_path)
    assert runtime.installs == [(catalog, tmp_path)]
    assert runtime.queries == ["q1"]


@pytest.mark.parametrize("agent", known_agents())
def test_every_agent_says_how_much_of_a_catalog_it_would_show(
    agent: str,
    catalog: Catalog,
    skills: list[Skill],
    tmp_path: Path,
) -> None:
    """Verify every agent implements fit returning a CatalogFit instance."""
    runtime = _build_agent(agent, tmp_path)
    fit = runtime.fit(catalog, skills)
    assert isinstance(fit, CatalogFit)
    assert fit.whole


def test_a_runtime_that_rations_nothing_says_so_rather_than_saying_it_fits() -> None:
    """Verify CatalogFit.rations distinguishes unconstrained fits from measured limits."""
    silent = CatalogFit()
    assert silent.whole
    assert not silent.rations
    assert CatalogFit(allowed=30_000, asked=51_910, truncated=47).rations


def test_a_fit_is_whole_exactly_when_nothing_lost_its_description() -> None:
    """Verify CatalogFit.whole is True only when truncated is 0."""
    assert CatalogFit(allowed=100, asked=90).whole
    assert not CatalogFit(allowed=100, asked=900, truncated=1).whole


def test_a_fit_cannot_report_a_negative_measurement() -> None:
    """Verify CatalogFit raises ValidationError on negative count values."""
    with pytest.raises(PydanticValidationError):
        CatalogFit.model_validate({"allowed": 30_000, "asked": 51_910, "truncated": -1})


def test_the_fake_returns_the_fit_it_was_scripted_with(catalog, skills, tmp_path) -> None:
    """Verify FakeRuntime returns scripted CatalogFit and records fitting calls."""
    scripted = CatalogFit(allowed=10, asked=99, unit="columns", truncated=2, remedy="ask")
    runtime = FakeRuntime(fit=scripted)
    assert runtime.fit(catalog, skills) == scripted
    assert runtime.fittings == [catalog]


def test_model_profile_matches_by_name_prefix() -> None:
    """Verify model_profile resolves registry entries matching model prefix."""
    profile = model_profile("gemini-3.7-flash")
    assert profile is not None
    assert profile.chars_per_token > 0
    assert profile.context_window > 0


def test_model_profile_returns_defaults_for_an_unregistered_model() -> None:
    """Verify model_profile returns standard default ModelProfile for unregistered models."""
    profile = model_profile("some-future-model")
    assert profile.chars_per_token == 4.0
    assert profile.context_window == 1_048_576


def test_model_profile_is_overridden_by_project_config(tmp_path, monkeypatch) -> None:
    """Verify project reach.toml overrides existing model profile registry entries."""
    override = tmp_path / "reach.toml"
    override.write_text("[models.claude-opus-5]\nchars_per_token = 9.9\ncontext_window = 42\n")
    monkeypatch.chdir(tmp_path)
    profile = model_profile("claude-opus-5")
    assert profile is not None
    assert (profile.chars_per_token, profile.context_window) == (9.9, 42)


def test_model_profile_override_matches_regardless_of_key_case(tmp_path, monkeypatch) -> None:
    """Verify model profile TOML section names match case-insensitively."""
    override = tmp_path / "reach.toml"
    override.write_text("[models.Claude-Opus-5]\nchars_per_token = 9.9\ncontext_window = 42\n")
    monkeypatch.chdir(tmp_path)
    profile = model_profile("claude-opus-5")
    assert profile is not None
    assert (profile.chars_per_token, profile.context_window) == (9.9, 42)


def test_model_profile_defaults_are_sound() -> None:
    """Verify ModelProfile defaults chars_per_token to 4.0 and context_window to 1_048_576."""
    from reach.runtime.profiles import ModelProfile

    profile = ModelProfile()
    assert profile.chars_per_token == 4.0
    assert profile.context_window == 1_048_576


@pytest.mark.parametrize("agent", known_agents())
def test_every_agent_supports_prompt_budget_query(agent: str) -> None:
    """Verify prompt_budget_chars method returns None or positive integer across all generators."""
    if agent == "antigravity-sdk" and not _HAS_ANTIGRAVITY:
        pytest.skip("google-antigravity is not installed")
    gen = build_text_generator(agent=agent, options=MINIMAL_OPTIONS.get(agent, {}))
    budget = gen.prompt_budget_chars()
    assert budget is None or budget > 0


def test_fake_generator_custom_completion() -> None:
    """Verify FakeGenerator complete supports custom responses and tracks counts."""
    generator = FakeGenerator(completion="custom completion response")
    assert generator.complete("prompt 1") == "custom completion response"
    assert generator.complete("prompt 2") == "custom completion response"
    assert generator.completions == 2
    assert generator.prompts == ["prompt 1", "prompt 2"]


def test_full_model_ids_have_profiles_in_reach_toml() -> None:
    """Verify full model IDs for Claude and Gemini resolve directly without aliasing."""
    for model_id in (
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-haiku-4-5",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
    ):
        profile = model_profile(model_id)
        assert profile is not None, f"Full model ID {model_id!r} lacks profile in reach.toml"
        assert profile.context_window > 0
        assert profile.chars_per_token > 0


def _assert_agent_default_model(agent_name: str) -> None:
    """Verify that an agent's default model has an accompanying profile."""
    default = agent_default_model(agent_name)
    if default:
        assert isinstance(default, str)
        profile = model_profile(default)
        assert profile is not None, f"default_model {default!r} for {agent_name!r} lacks a profile"


def test_production_defaults_are_internally_consistent() -> None:
    """Verify known agents have valid default models."""
    for agent_name in known_agents():
        _assert_agent_default_model(agent_name)


def test_fake_constants_consistency() -> None:
    """Verify FAKE_AGENT and FAKE_MODEL constants match fake runtime attributes."""
    from reach.runtime import FAKE_AGENT
    from reach.runtime.fake import FAKE_MODEL, FakeOptions, FakeRuntime

    assert FAKE_AGENT == "fake"
    assert FAKE_MODEL == "fake-model"
    assert FakeRuntime.name == FAKE_AGENT
    assert FakeOptions().model == FAKE_MODEL


def test_default_agent_reads_general_section_from_reach_toml() -> None:
    """Verify default_agent reads default_agent from bundled reach.toml."""
    assert default_agent() == "antigravity-cli"
    assert RuntimeSettings().agent == "antigravity-cli"


def test_default_agent_respects_project_config_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify default_agent and RuntimeSettings dynamically track reach.toml override."""
    override = tmp_path / "reach.toml"
    override.write_text("[general]\ndefault_agent = 'fake'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert default_agent() == "fake"
    assert RuntimeSettings().agent == "fake"


@pytest.mark.usefixtures("fake_registry")
@pytest.mark.parametrize(
    ("model_name", "expected_agent"),
    [
        ("mock-model", "mock-agent"),
        ("MOCK-MODEL", "mock-agent"),
        ("mock-model-variant", "mock-agent"),
        ("mock-opus", "mock-agent"),
        ("mock-flash", "mock-antigravity"),
        ("mock-pro", "mock-antigravity"),
    ],
)
def test_find_agent_for_model_happy_paths(model_name: str, expected_agent: str) -> None:
    """Verify find_agent_for_model resolves models across case variations with fake registry."""
    assert find_agent_for_model(model_name) == expected_agent


@pytest.mark.usefixtures("fake_registry")
@pytest.mark.parametrize(
    "unknown_model",
    [
        "unregistered-model",
        "future-model",
        "",
        "   ",
    ],
)
def test_find_agent_for_model_sad_paths(unknown_model: str) -> None:
    """Verify find_agent_for_model returns None for unknown or empty model strings."""
    assert find_agent_for_model(unknown_model) is None


@pytest.mark.usefixtures("fake_registry")
def test_agent_default_model_with_fake_registry() -> None:
    """Verify agent_default_model returns correct default or None using fake registry."""
    assert agent_default_model("mock-agent") == "mock-model"
    assert agent_default_model("mock-antigravity") == "mock-flash"
    assert agent_default_model("mock-empty") is None
    assert agent_default_model("unregistered-agent") is None


def test_find_agent_for_model_edge_case_empty_or_corrupted_agents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify find_agent_for_model handles empty or missing agents configuration gracefully."""
    override = tmp_path / "reach.toml"
    override.write_text("[agents]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert find_agent_for_model("some-model") is None


def test_load_config_rejects_corrupted_config(
    tmp_path: Path,
) -> None:
    """Verify load_config fails fast with TOMLDecodeError on malformed syntax."""
    corrupted = tmp_path / "reach.toml"
    corrupted.write_text("this is not valid toml = [[[", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_config(corrupted)


def test_load_config_deep_merge_preserves_sibling_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify deep merge updates targeted field without overwriting sibling keys."""
    override = tmp_path / "reach.toml"
    override.write_text(
        "[agents.claude-code]\ndefault_model = 'custom-model'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert agent_default_model("claude-code") == "custom-model"
    assert "antigravity-cli" in known_agents()


def test_process_failure_reason_extracts_trailing_stderr_or_exit_code() -> None:
    """Verify process_failure_reason pulls last line of stderr or formats exit code."""
    with_stderr = subprocess.CompletedProcess(
        args=["fake"],
        returncode=1,
        stdout="",
        stderr="line 1\nline 2: out of memory\n",
    )
    assert process_failure_reason(with_stderr) == "line 2: out of memory"

    empty_stderr = subprocess.CompletedProcess(
        args=["fake"],
        returncode=137,
        stdout="",
        stderr="",
    )
    assert process_failure_reason(empty_stderr) == "exit 137"


def test_antigravity_runtime_resolves_skills_dir_and_roots(tmp_path: Path) -> None:
    """Verify AntigravityRuntime provides .agents/skills layout and discovers existing roots."""
    runtime = build_runtime(
        RuntimeSettings(
            agent="antigravity-cli",
            options=MINIMAL_OPTIONS["antigravity-cli"],
        ),
    )
    assert isinstance(runtime, AntigravityRuntime)
    assert runtime.skills_dir(tmp_path) == tmp_path / ".agents" / "skills"
    assert runtime.skill_roots(tmp_path) == ()

    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)
    roots = runtime.skill_roots(tmp_path)
    assert len(roots) == 1
    assert roots[0].path == skills_dir.resolve()
    assert roots[0].scope == "project"
    assert roots[0].precedence == 0


def test_agent_runtime_default_model_reads_options_or_empty() -> None:
    """Verify AgentRuntime.model extracts model name from options or falls back to empty string."""
    fake = FakeRuntime(model="custom-gemini")
    assert fake.model == "custom-gemini"

    class BareRuntime(AgentRuntime):
        name = "bare"

        def skill_roots(self, workdir: Path) -> tuple[SkillRoot, ...]:
            del workdir
            return ()

        def install(self, catalog: Catalog, skills: Any, workdir: Path) -> Path:
            del catalog, skills
            return workdir

        def select(
            self,
            query_text: str,
            workdir: Path,
            target_skill: str | None = None,
        ) -> SelectionOutcome:
            del query_text, workdir, target_skill
            return SelectionOutcome()

        def complete(self, prompt: str) -> str:
            del prompt
            return ""

    bare = BareRuntime()
    assert bare.model == ""


@pytest.mark.parametrize(
    ("agent", "expected_type"),
    [
        ("antigravity-cli", AntigravityCliOptions),
        ("antigravity-sdk", AntigravitySdkOptions),
        ("claude-code", ClaudeCodeOptions),
        ("keyword", KeywordOptions),
        ("goose", GooseOptions),
        ("pi", PiOptions),
        ("fake", FakeOptions),
    ],
)
def test_options_model_matches_registered_schema(
    agent: str,
    expected_type: type[BaseModel] | None,
) -> None:
    """Verify options_model returns designated schema model across all agents."""
    assert options_model(agent) is expected_type


@pytest.mark.parametrize("agent", known_agents())
def test_install_places_only_catalog_members(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify install copies only catalog member skill directories into workspace."""
    runtime = _build_agent(agent, tmp_path)
    skills = load_skills(skill_repo)
    catalog = Catalog(
        id="test-subset",
        mode=CatalogMode.ALL,
        skills=("gcs-lifecycle-rules", "gcs-retention-policy"),
    )
    workdir = runtime.install(catalog, skills, tmp_path / "work")
    skills_dir = runtime.skills_dir(workdir)
    installed = sorted(p.name for p in skills_dir.iterdir())
    assert installed == ["gcs-lifecycle-rules", "gcs-retention-policy"]
    assert getattr(runtime, "_resident", None) == catalog.skills


@pytest.mark.parametrize("agent", known_agents())
def test_install_is_idempotent(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify consecutive install calls replace previously installed skills."""
    runtime = _build_agent(agent, tmp_path)
    skills = load_skills(skill_repo)
    catalogs = build_catalogs(skills, CatalogMode.SINGLETON)
    workdir = tmp_path / "work"
    runtime.install(catalogs[0], skills, workdir)
    runtime.install(catalogs[1], skills, workdir)
    skills_dir = runtime.skills_dir(workdir)
    installed = sorted(p.name for p in skills_dir.iterdir())
    assert installed == [catalogs[1].skills[0]]


@pytest.mark.parametrize("agent", known_agents())
def test_skill_roots_empty_before_install_and_present_after(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify skill_roots is empty on clean workspace and discovered once catalog is installed."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "nohome"))
    runtime = _build_agent(agent, tmp_path)
    workdir = tmp_path / "work"
    assert runtime.skill_roots(workdir) == ()

    skills = load_skills(skill_repo)
    catalog = build_catalogs(skills, CatalogMode.SINGLETON)[0]
    runtime.install(catalog, skills, workdir)
    roots = runtime.skill_roots(workdir)
    assert len(roots) == 1
    assert roots[0].path == runtime.skills_dir(workdir).resolve()
    assert roots[0].scope in ("project", "user")
    assert roots[0].precedence == 0


def test_install_skills_uses_symlinks_when_requested(
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify install_skills creates fast symlinks when use_symlinks is enabled."""
    skills = load_skills(skill_repo)
    by_name = {s.name: s for s in skills}
    catalog = Catalog(id="c", mode=CatalogMode.SINGLETON, skills=(skills[0].name,))
    dest = tmp_path / "installed_skills"
    installed = install_skills(catalog, by_name, dest, use_symlinks=True)
    assert installed == (skills[0].name,)
    assert (dest / skills[0].name).is_symlink()


def test_install_skills_copies_when_use_symlinks_is_false(
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify install_skills copies directories when use_symlinks is disabled."""
    skills = load_skills(skill_repo)
    by_name = {s.name: s for s in skills}
    catalog = Catalog(id="c", mode=CatalogMode.SINGLETON, skills=(skills[0].name,))
    dest = tmp_path / "installed_skills"
    installed = install_skills(catalog, by_name, dest, use_symlinks=False)
    assert installed == (skills[0].name,)
    assert not (dest / skills[0].name).is_symlink()
    assert (dest / skills[0].name / "SKILL.md").is_file()


def test_install_skills_falls_back_to_copy_on_symlink_error(
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify install_skills falls back to copytree when symlink raises OSError."""

    def _failing_symlink(self: Path, _target: Path, target_is_directory: bool = False) -> None:
        msg = "symlink operation not permitted"
        raise OSError(msg)

    monkeypatch.setattr(Path, "symlink_to", _failing_symlink)
    skills = load_skills(skill_repo)
    by_name = {s.name: s for s in skills}
    catalog = Catalog(id="c", mode=CatalogMode.SINGLETON, skills=(skills[0].name,))
    dest = tmp_path / "installed_skills"
    installed = install_skills(catalog, by_name, dest, use_symlinks=True)
    assert installed == (skills[0].name,)
    assert not (dest / skills[0].name).is_symlink()
    assert (dest / skills[0].name / "SKILL.md").is_file()


@pytest.mark.parametrize("agent", known_agents())
def test_install_uses_symlinks_by_default(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify install creates fast directory symlinks for resident skills by default."""
    runtime = _build_agent(agent, tmp_path)
    skills = load_skills(skill_repo)
    catalog = build_catalogs(skills, CatalogMode.SINGLETON)[0]
    workdir = runtime.install(catalog, skills, tmp_path / f"work_{agent}")
    skill_entry = runtime.skills_dir(workdir) / catalog.skills[0]
    assert skill_entry.is_symlink()


@pytest.mark.parametrize("agent", known_agents())
def test_install_copies_when_use_symlinks_is_false(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
) -> None:
    """Verify install copies skill files directly when use_symlinks is disabled."""
    runtime = _build_agent(agent, tmp_path, use_symlinks=False)
    skills = load_skills(skill_repo)
    catalog = build_catalogs(skills, CatalogMode.SINGLETON)[0]
    workdir = runtime.install(catalog, skills, tmp_path / f"work_{agent}")
    skill_entry = runtime.skills_dir(workdir) / catalog.skills[0]
    assert not skill_entry.is_symlink()


@pytest.mark.parametrize("agent", known_agents())
def test_install_falls_back_to_copy_on_symlink_error(
    agent: str,
    skill_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify install falls back to shutil.copytree when symlink creation raises OSError."""

    def _failing_symlink(self: Path, _target: Path, target_is_directory: bool = False) -> None:
        msg = "symlink operation not permitted"
        raise OSError(msg)

    monkeypatch.setattr(Path, "symlink_to", _failing_symlink)
    runtime = _build_agent(agent, tmp_path)
    skills = load_skills(skill_repo)
    catalog = build_catalogs(skills, CatalogMode.SINGLETON)[0]
    workdir = runtime.install(catalog, skills, tmp_path / f"work_{agent}")
    skill_entry = runtime.skills_dir(workdir) / catalog.skills[0]
    assert not skill_entry.is_symlink()
    assert (skill_entry / "SKILL.md").is_file()


@pytest.mark.parametrize("agent", known_agents())
def test_all_agents_fit_matches_catalog_rationing_capability(
    agent: str,
    tmp_path: Path,
) -> None:
    """Verify fit reports whole and rationing matching the agent's declared capability."""
    runtime = _build_agent(agent, tmp_path)
    fit = runtime.fit(
        catalog=Catalog(id="c", mode=CatalogMode.ALL, skills=()),
        skills=[],
    )
    assert fit.rations is runtime.rations_catalog
    assert fit.whole is True


@pytest.mark.parametrize("agent", known_agents())
def test_prompt_budget_uses_default_for_unregistered_model(agent: str) -> None:
    """Verify prompt_budget_chars returns standard default budget on unregistered model."""
    if agent == "antigravity-sdk" and not _HAS_ANTIGRAVITY:
        pytest.skip("google-antigravity is not installed")
    gen = build_text_generator(agent=agent, model="unmeasured-future-model-999")
    assert gen.prompt_budget_chars() == 4_194_304


@pytest.mark.parametrize("agent", _EXECUTION_AGENTS)
def test_prompt_budget_is_measured_for_recognized_models(agent: str) -> None:
    """Verify prompt_budget_chars returns positive integer for registered model profiles."""
    if agent == "antigravity-sdk" and not _HAS_ANTIGRAVITY:
        pytest.skip("google-antigravity is not installed")
    gen = build_text_generator(agent=agent)
    budget = gen.prompt_budget_chars()
    assert budget is not None
    assert budget > 0


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_executable_is_configurable(agent: str, tmp_path: Path) -> None:
    """Verify custom executable option is placed as the first command token."""
    runtime = _build_agent(agent, tmp_path, executable="/custom/binary")
    assert isinstance(runtime, CliAgentRuntime)
    cmd = runtime.build_command("test query")
    assert cmd[0] == "/custom/binary"


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_extra_args_reach_command_line_last(agent: str, tmp_path: Path) -> None:
    """Verify extra_args tokens are placed at the end of the constructed command line."""
    runtime = _build_agent(agent, tmp_path, extra_args=("--custom-flag", "value"))
    assert isinstance(runtime, CliAgentRuntime)
    cmd = runtime.build_command("test query")
    assert cmd[-2:] == ["--custom-flag", "value"]


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_runtime_select_executes_in_workdir(
    agent: str,
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify select executes subprocess with cwd pointing to the provided workspace."""
    runtime = _build_agent(agent, tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, Any] = {}

    def mock_run(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="abort")

    mock_subprocess(handler=mock_run)
    runtime.select("test query", workdir)
    assert seen.get("cwd") == workdir


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_runtime_select_handles_timeout(
    agent: str,
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify select handles subprocess timeout by returning error outcome without raising."""
    runtime = _build_agent(agent, tmp_path)
    mock_subprocess(side_effect=subprocess.TimeoutExpired(cmd="fake", timeout=1))
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    outcome = runtime.select("query", workdir)
    assert outcome.invoked_skill is None
    assert outcome.error is not None
    assert "time" in outcome.error.lower()


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_runtime_select_handles_spawn_oserror(
    agent: str,
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify select handles subprocess spawn failure (e.g. missing binary) gracefully."""
    runtime = _build_agent(agent, tmp_path)
    mock_subprocess(
        side_effect=FileNotFoundError(2, "No such file or directory: 'fake-executable'"),
    )
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    outcome = runtime.select("query", workdir)
    assert outcome.invoked_skill is None
    assert outcome.error is not None
    assert "failed to spawn" in outcome.error or "No such file" in outcome.error


@pytest.mark.parametrize("agent", known_agents())
def test_skills_subpath_matches_expected_agent_default(
    agent: str,
    tmp_path: Path,
) -> None:
    """Verify runtime skills_subpath matches the agent's native skills directory."""
    runtime = _build_agent(agent, tmp_path)
    profile = agent_profiles().get(agent)
    expected = (
        profile.skills_dir
        if (profile is not None and profile.skills_dir)
        else getattr(runtime, "_skills_subpath", ".agents/skills")
    )
    assert runtime.skills_subpath == expected
    assert len(runtime.skills_subpath) > 0


def test_skills_subpath_fallback_and_override() -> None:
    """Verify custom runtime subclasses can override _skills_subpath or fallback to empty."""

    class CustomRuntime(AgentRuntime):
        name = "unregistered-agent"

        def select(
            self,
            query_text: str,
            workdir: Path,
            target_skill: str | None = None,
        ) -> SelectionOutcome:
            del query_text, workdir, target_skill
            return SelectionOutcome()

        def complete(self, prompt: str) -> str:
            del prompt
            return ""

    class ExplicitRuntime(AgentRuntime):
        name = "explicit-agent"
        _skills_subpath = "custom/path"

        def select(
            self,
            query_text: str,
            workdir: Path,
            target_skill: str | None = None,
        ) -> SelectionOutcome:
            del query_text, workdir, target_skill
            return SelectionOutcome()

        def complete(self, prompt: str) -> str:
            del prompt
            return ""

    assert CustomRuntime().skills_subpath == ""
    assert ExplicitRuntime().skills_subpath == "custom/path"


def test_check_tool_leak() -> None:
    """Verify check_tool_leak detects unauthorized tool attempts."""
    assert check_tool_leak(["read", "bash"], None) is None
    assert check_tool_leak(["read", "write"], ["read", "write", "glob"]) is None
    leak_err = check_tool_leak(["read", "bash", "curl"], ["read"])
    assert leak_err == "tool leak: bash, curl"


def test_session_summary_to_outcome() -> None:
    """Verify SessionSummary converts cleanly to SelectionOutcome with sync and fallback."""
    summary = SessionSummary(
        invoked_skills=("pizza-calculator",),
        reasoning=("thought 1",),
        observed_tools=("load_skill",),
        cost_usd=0.005,
        resolved_model="",
    )
    assert summary.invoked_skills == ("pizza-calculator",)
    outcome = summary.to_outcome(
        observed_catalog=("pizza-calculator", "cloud-deploy"),
        fallback_model="default-model",
    )
    assert outcome.invoked_skill == "pizza-calculator"
    assert outcome.invoked_skills == ("pizza-calculator",)
    assert outcome.reasoning == ("thought 1",)
    assert outcome.observed_catalog == ("pizza-calculator", "cloud-deploy")
    assert outcome.observed_tools == ("load_skill",)
    assert outcome.resolved_model == "default-model"
    assert outcome.cost_usd == pytest.approx(0.005)


def test_agent_runtime_timeout_s() -> None:
    """Verify AgentRuntime exposes timeout_s from settings."""
    settings_with = RuntimeSettings(agent="fake", timeout_s=42)
    rt_with = build_runtime(settings_with)
    assert rt_with.timeout_s == 42

    rt_none = FakeRuntime()
    assert rt_none.timeout_s is None


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_agent_reports_subprocess_failure(
    agent: str,
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify CLI runtimes capture non-zero subprocess returncode and report failure reason."""
    runtime = _build_agent(agent, tmp_path)
    mock_subprocess(returncode=1, stderr="Error: connection refused")
    outcome = runtime.select("q", tmp_path / "work")
    assert outcome.error is not None
    assert "connection refused" in outcome.error


def test_resolve_skill_from_path_variations() -> None:
    """Verify resolve_skill_from_path handles paths, casing, None, and edge cases."""
    residents = ("cloud-deploy", "pizza-calculator")
    assert resolve_skill_from_path("/path/to/cloud-deploy/SKILL.md", residents) == "cloud-deploy"
    assert resolve_skill_from_path("cloud-deploy/skill.md", residents) == "cloud-deploy"
    assert resolve_skill_from_path("PIZZA-CALCULATOR.MD", residents) == "pizza-calculator"
    assert (
        resolve_skill_from_path(Path("/skills/pizza-calculator.md"), residents)
        == "pizza-calculator"
    )
    assert resolve_skill_from_path("/other/README.md", residents) is None
    assert resolve_skill_from_path("", residents) is None
    assert resolve_skill_from_path(None, residents) is None
    assert resolve_skill_from_path(cast("Any", 123), residents) is None


def test_extract_content_reasoning() -> None:
    """Verify extract_content_reasoning collects thought and text entries."""
    content = [
        {"type": "thought", "thought": "Thinking step 1"},
        {"type": "thinking", "text": "Thinking step 2"},
        {"type": "text", "text": "Direct response text"},
        {"type": "toolCall", "name": "read"},
        "invalid_item",
    ]
    extracted = extract_content_reasoning(content)
    assert extracted == ["Thinking step 1", "Thinking step 2", "Direct response text"]


def test_format_subprocess_error() -> None:
    """Verify format_subprocess_error formats timeout, spawn error, and generic failures."""
    assert format_subprocess_error("pi", "timeout", 30) == "pi process timed out after 30s"
    assert format_subprocess_error("goose", "timeout", None) == "goose process timed out"
    assert (
        format_subprocess_error("pi", "executable not found", 10)
        == "failed to spawn pi: executable not found"
    )
    assert format_subprocess_error("pi", None, 10) == "pi subprocess failed"


def test_cli_options_helpers() -> None:
    """Verify CliOptions effort_args, provider_args, max_turns_args, and api_key_args helpers."""
    opts = CliOptions(
        executable="test-cli",
        model="test-model",
        effort="high",
        provider="google",
        max_turns=3,
        api_key="secret-123",
    )
    assert opts.effort == "high"
    assert opts.provider == "google"
    assert opts.max_turns == 3
    assert opts.api_key == "secret-123"
    assert opts.effort_args("--effort") == ["--effort", "high"]
    assert opts.provider_args("--provider") == ["--provider", "google"]
    assert opts.max_turns_args("--max-turns") == ["--max-turns", "3"]
    assert opts.api_key_args("--api-key") == ["--api-key", "secret-123"]

    opts_empty = CliOptions(executable="test-cli", model="test-model")
    assert opts_empty.effort_args("--effort") == []
    assert opts_empty.provider_args("--provider") == []
    assert opts_empty.max_turns_args("--max-turns") == ["--max-turns", "3"]
    assert opts_empty.api_key_args("--api-key") == []


@pytest.mark.parametrize("agent", _EXECUTION_AGENTS)
def test_cli_agent_shared_options_conformance(agent: str, tmp_path: Path) -> None:
    """Verify all CLI agents expose common options and properties uniformly."""
    rt = _build_agent(
        agent,
        tmp_path,
        model="custom-model",
        effort="medium",
        provider="custom-provider",
        max_turns=2,
        early_exit=False,
        api_key="secret-key",
        allowed_tools=("custom-tool",),
    )
    assert rt.model == "custom-model"
    assert rt.effort == "medium"
    assert rt.provider == "custom-provider"
    assert rt.max_turns == 2
    assert rt.early_exit is False
    assert rt.api_key == "secret-key"
    assert rt.allowed_tools == ("custom-tool",)


@pytest.mark.parametrize("agent", known_agents())
def test_all_agents_default_max_turns_and_early_exit(agent: str, tmp_path: Path) -> None:
    """Verify all agent runtimes default to max_turns=3 and early_exit=True."""
    rt = _build_agent(agent, tmp_path)
    assert rt.max_turns == 3
    assert rt.early_exit is True


@pytest.mark.parametrize("agent", known_agents())
def test_all_agents_configurable_max_turns_and_early_exit(agent: str, tmp_path: Path) -> None:
    """Verify max_turns and early_exit can be overridden across all agent runtimes."""
    rt = _build_agent(agent, tmp_path, max_turns=5, early_exit=False)
    assert rt.max_turns == 5
    assert rt.early_exit is False


@pytest.mark.parametrize("agent", known_agents())
def test_all_agents_default_performance_and_isolation_options(agent: str, tmp_path: Path) -> None:
    """Verify performance and isolation options default to expected values across all agents."""
    rt = _build_agent(agent, tmp_path)
    assert rt.use_symlinks is True
    assert rt.isolate_config_dir is True
    assert rt.auto_clean is False
    opts = getattr(rt, "options", None)
    assert opts is not None
    assert opts.use_symlinks is True
    assert opts.isolate_config_dir is True
    assert opts.auto_clean is False


@pytest.mark.parametrize("agent", known_agents())
def test_all_agents_configurable_performance_and_isolation_options(
    agent: str,
    tmp_path: Path,
) -> None:
    """Verify performance and isolation options can be overridden across all agents."""
    rt = _build_agent(
        agent,
        tmp_path,
        use_symlinks=False,
        isolate_config_dir=False,
        auto_clean=True,
    )
    assert rt.use_symlinks is False
    assert rt.isolate_config_dir is False
    assert rt.auto_clean is True
    opts = getattr(rt, "options", None)
    assert opts is not None
    assert opts.use_symlinks is False
    assert opts.isolate_config_dir is False
    assert opts.auto_clean is True


@pytest.mark.parametrize("agent", known_agents())
def test_all_agents_select_invokes_post_probe(
    agent: str,
    tmp_path: Path,
    mock_subprocess: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify select invokes post_probe lifecycle hook upon completion across all agents."""
    mock_subprocess(stdout="")
    if agent == "antigravity-sdk":
        if not _HAS_ANTIGRAVITY:
            pytest.skip("google-antigravity is not installed")
        from reach.runtime.antigravity_sdk import AntigravitySdkRuntime

        async def _mock_select_async(*_args: Any, **_kwargs: Any) -> SelectionOutcome:
            return SelectionOutcome()

        monkeypatch.setattr(AntigravitySdkRuntime, "_select_async", _mock_select_async)

    runtime = _build_agent(agent, tmp_path)
    called_workdirs: list[Path] = []

    def mock_post_probe(workdir: Path) -> None:
        called_workdirs.append(workdir)

    monkeypatch.setattr(runtime, "post_probe", mock_post_probe)
    workdir = tmp_path / f"work_{agent}"
    workdir.mkdir(parents=True, exist_ok=True)

    with contextlib.suppress(Exception):
        runtime.select("test query", workdir)

    assert workdir in called_workdirs


@pytest.mark.parametrize(
    "agent",
    [a for a in _EXECUTION_AGENTS if a != "claude-code"],
)
def test_agent_build_env_synchronizes_google_and_gemini_keys(
    agent: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build_env synchronizes GEMINI_API_KEY and GOOGLE_API_KEY bidirectionally."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rt = _build_agent(agent, tmp_path)
    env = rt.build_env()
    assert env["GEMINI_API_KEY"] == "gemini-secret"
    assert env["GOOGLE_API_KEY"] == "gemini-secret"

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
    rt2 = _build_agent(agent, tmp_path)
    env2 = rt2.build_env()
    assert env2["GEMINI_API_KEY"] == "google-secret"
    assert env2["GOOGLE_API_KEY"] == "google-secret"


@pytest.mark.parametrize(
    ("agent", "dir_names"),
    [
        ("claude-code", [".reach_claude_config"]),
        ("goose", [".reach_goose"]),
        ("pi", [".reach_pi_sessions", ".reach_pi_agent"]),
        ("antigravity-sdk", [".reach_antigravity_sdk"]),
    ],
)
def test_isolated_config_dir_cleanup_on_auto_clean(
    agent: str,
    dir_names: list[str],
    tmp_path: Path,
) -> None:
    """Verify post_probe removes isolated runtime directories when auto_clean is enabled."""
    if agent == "antigravity-sdk" and not _HAS_ANTIGRAVITY:
        pytest.skip("google-antigravity is not installed")
    workdir = tmp_path / f"work_{agent}"
    workdir.mkdir(parents=True, exist_ok=True)
    dirs = [workdir / name for name in dir_names]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # auto_clean=False preserves directories
    rt_no_clean = _build_agent(agent, tmp_path, auto_clean=False)
    rt_no_clean.post_probe(workdir)
    for d in dirs:
        assert d.exists()

    # auto_clean=True removes directories
    rt_clean = _build_agent(agent, tmp_path, auto_clean=True)
    rt_clean.post_probe(workdir)
    for d in dirs:
        assert not d.exists()


@pytest.mark.parametrize("agent", cli_agents())
def test_cli_generator_receives_prompt(agent: str) -> None:
    """Verify text generator receives prompt via CLI arguments or standard input."""
    gen = build_text_generator(agent=agent)
    prompt = "draft some queries"
    cmd_fn = getattr(gen, "build_completion_command", None)
    assert callable(cmd_fn)
    cmd = cmd_fn(prompt)
    assert isinstance(cmd, list)
    in_command = any(prompt in token for token in cmd)
    assert in_command or gen.name == "claude-code"


@pytest.mark.parametrize("agent", cli_agents())
def test_build_text_generator_resolves_agent_default_model(agent: str) -> None:
    """Verify build_text_generator resolves agent-specific default model when none is passed."""
    expected_model = agent_default_model(agent)
    assert expected_model is not None
    gen = build_text_generator(agent=agent)
    assert gen.model == expected_model
    opts = getattr(gen, "options", None)
    if opts is not None and hasattr(opts, "model"):
        assert opts.model == expected_model


def test_build_text_generator_explicit_model_overrides_agent_default() -> None:
    """Verify explicit model parameter overrides the agent runtime default model."""
    gen = build_text_generator(agent="claude-code", model="claude-opus-5")
    assert gen.model == "claude-opus-5"
    assert getattr(getattr(gen, "options", None), "model", None) == "claude-opus-5"


def test_build_text_generator_options_model_is_respected() -> None:
    """Verify options dictionary model is respected when model argument is omitted."""
    gen = build_text_generator(agent="claude-code", options={"model": "claude-haiku-4-5"})
    assert gen.model == "claude-haiku-4-5"
    assert getattr(getattr(gen, "options", None), "model", None) == "claude-haiku-4-5"


def test_agent_options_inheritance_hierarchy() -> None:
    """Verify AgentOptions is the root model and CliOptions subclasses it."""
    from reach.runtime import AgentOptions, CliOptions
    from reach.runtime.antigravity_sdk import AntigravitySdkOptions
    from reach.runtime.fake import FakeOptions
    from reach.runtime.keyword import KeywordOptions

    assert issubclass(CliOptions, AgentOptions)
    assert issubclass(AntigravitySdkOptions, AgentOptions)
    assert issubclass(FakeOptions, AgentOptions)
    assert issubclass(KeywordOptions, AgentOptions)

    # AgentOptions has core routing, isolation, and performance flags
    base = AgentOptions(
        model="gpt-4o",
        effort="low",
        provider="openai",
        max_turns=5,
        early_exit=False,
        allowed_tools=("search",),
        api_key="key-123",
    )
    assert base.model == "gpt-4o"
    assert base.effort == "low"
    assert base.max_turns == 5
    assert base.early_exit is False
    assert base.use_symlinks is True
    assert base.isolate_config_dir is True
    assert base.auto_clean is False

    # Subclasses inherit performance and isolation defaults
    assert FakeOptions().use_symlinks is True
    assert FakeOptions().isolate_config_dir is True
    assert FakeOptions().auto_clean is False
    assert KeywordOptions().use_symlinks is True
    assert KeywordOptions().isolate_config_dir is True
    assert KeywordOptions().auto_clean is False

    # SDK options does not have executable or extra_args
    sdk_opts = AntigravitySdkOptions(model="gemini-2.5-flash", max_turns=3, early_exit=True)
    assert not hasattr(sdk_opts, "executable")
    assert not hasattr(sdk_opts, "extra_args")
    assert sdk_opts.max_turns == 3
    assert sdk_opts.early_exit is True
    assert sdk_opts.use_symlinks is True
    assert sdk_opts.isolate_config_dir is True
    assert sdk_opts.auto_clean is False


@pytest.mark.parametrize(
    ("early_exit", "turns_taken"),
    [
        (True, 1),
        (True, 2),
        (False, 3),
    ],
)
def test_selection_outcome_and_probe_result_early_exit_propagation(
    catalog: Catalog,
    early_exit: bool,
    turns_taken: int,
) -> None:
    """Verify early_exit and turns_taken propagate faithfully to ProbeResult."""
    from reach.models import ProbeResult, Query

    outcome = SelectionOutcome(
        invoked_skills=("target-skill", "precursor")[:turns_taken],
        early_exit=early_exit,
        turns_taken=turns_taken,
        observed_catalog=catalog.skills,
        observed_tools=("Skill",),
    )
    query = Query(id="q1", text="test query", expected_skill="target-skill")
    result = ProbeResult.from_outcome(
        outcome=outcome,
        query=query,
        catalog=catalog,
        runtime_name="test-runtime",
        model="test-model",
    )
    assert result.early_exit is early_exit
    assert result.turns_taken == turns_taken
    assert result.invoked_skill == "target-skill"


@pytest.mark.parametrize(
    ("agent", "max_turns", "early_exit"),
    [
        ("antigravity-cli", 1, False),
        ("antigravity-cli", 3, True),
        ("antigravity-sdk", 1, False),
        ("antigravity-sdk", 3, True),
    ],
)
def test_antigravity_agents_enforce_schema_in_both_single_and_multi_turn(
    agent: str,
    max_turns: int,
    early_exit: bool,
    tmp_path: Path,
) -> None:
    """Verify Antigravity runtimes configure catalog response schema regardless of turn mode."""
    if agent == "antigravity-sdk" and not _HAS_ANTIGRAVITY:
        pytest.skip("google-antigravity is not installed")
    import json

    from reach.runtime.antigravity_sdk import AntigravitySdkRuntime

    rt = _build_agent(agent, tmp_path, max_turns=max_turns, early_exit=early_exit)
    rt._resident = ("skill-a", "skill-b")

    if isinstance(rt, AntigravityCliRuntime):
        cmd = rt.build_command("query")
        assert "--json-schema" in cmd
        assert "--disable-slash-commands" in cmd
        schema = json.loads(cmd[cmd.index("--json-schema") + 1])
        branches = schema["properties"]["selected_skill"]["anyOf"]
        enum = next(b["enum"] for b in branches if "enum" in b)
        assert sorted(enum) == ["skill-a", "skill-b"]
    elif isinstance(rt, AntigravitySdkRuntime):
        config = rt._select_config(tmp_path)
        assert config.response_schema is not None


# --- Trajectory tracking and session telemetry -------------------------------


def test_trajectory_tracker_direct_hit() -> None:
    """Verify TrajectoryTracker signals immediate early exit upon target detection."""
    from reach.runtime import TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="cloud-sql", max_turns=3, early_exit=True)
    stop = tracker.observe("cloud-sql")
    assert stop
    assert tracker.early_exit_hit
    assert tracker.turns_taken == 1
    assert tracker.invoked_skills == ["cloud-sql"]


def test_trajectory_tracker_multi_turn_recovery() -> None:
    """Verify TrajectoryTracker records intermediate precursor skills and stops on target."""
    from reach.runtime import TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="cloud-sql", max_turns=3, early_exit=True)
    assert not tracker.observe("gcloud")
    assert not tracker.early_exit_hit
    assert tracker.turns_taken == 1

    assert tracker.observe("cloud-sql")
    assert tracker.early_exit_hit
    assert tracker.turns_taken == 2
    assert tracker.invoked_skills == ["gcloud", "cloud-sql"]


def test_trajectory_tracker_deduplicates_consecutive_tool_calls() -> None:
    """Verify multiple consecutive calls to the same skill do not advance turn budget."""
    from reach.runtime import TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="cloud-sql", max_turns=3, early_exit=True)
    assert not tracker.observe("gcloud")
    assert not tracker.observe("gcloud")  # consecutive call to same skill
    assert tracker.turns_taken == 1
    assert tracker.invoked_skills == ["gcloud"]

    assert tracker.observe("cloud-sql")
    assert tracker.turns_taken == 2
    assert tracker.invoked_skills == ["gcloud", "cloud-sql"]


def test_trajectory_tracker_exhausts_max_turns() -> None:
    """Verify TrajectoryTracker stops when turn budget is exhausted without target hit."""
    from reach.runtime import TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="cloud-sql", max_turns=3, early_exit=True)
    assert not tracker.observe("skill-1")
    assert not tracker.observe("skill-2")
    assert tracker.observe("skill-3")  # 3rd turn reached
    assert tracker.early_exit_hit
    assert tracker.turns_taken == 3
    assert tracker.invoked_skills == ["skill-1", "skill-2", "skill-3"]


def test_trajectory_tracker_disabled_early_exit() -> None:
    """Verify TrajectoryTracker records all skills without stopping when early_exit is False."""
    from reach.runtime import TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="cloud-sql", max_turns=2, early_exit=False)
    assert not tracker.observe("cloud-sql")
    assert not tracker.observe("another-skill")
    assert not tracker.observe("third-skill")
    assert not tracker.early_exit_hit
    assert tracker.turns_taken == 3


def test_trajectory_tracker_ignores_none_or_empty() -> None:
    """Verify TrajectoryTracker safely ignores None or empty string invocations."""
    from reach.runtime import TrajectoryTracker

    tracker = TrajectoryTracker(target_skill="cloud-sql", max_turns=3, early_exit=True)
    assert not tracker.observe(None)
    assert not tracker.observe("")
    assert tracker.turns_taken == 1
    assert tracker.invoked_skills == []


def test_selection_outcome_subclasses_session_summary() -> None:
    """Verify SelectionOutcome inherits from SessionSummary with observed_catalog added."""
    from reach.runtime import SelectionOutcome, SessionSummary

    assert issubclass(SelectionOutcome, SessionSummary)
    summary = SessionSummary(
        cost_usd=0.05,
        duration_ms=1500,
        invoked_skills=("precursor", "skill-a"),
        early_exit=True,
        turns_taken=2,
    )
    outcome = summary.to_outcome(observed_catalog=("skill-a", "skill-b"))
    assert isinstance(outcome, SelectionOutcome)
    assert outcome.observed_catalog == ("skill-a", "skill-b")
    assert outcome.cost_usd == 0.05
    assert outcome.duration_ms == 1500
    assert outcome.invoked_skill == "precursor"
    assert outcome.early_exit is True
    assert outcome.turns_taken == 2


@pytest.mark.parametrize(
    ("invoked_skills", "summary_turns", "expected_turns"),
    [
        ((), 3, 3),  # Abstention: 0 skills, 3 turns taken
        (("skill-a", "skill-b"), 1, 1),  # Parallel tool calls: 2 skills, 1 turn
        (("skill-a",), 4, 4),  # Delayed execution: 1 skill, 4 turns taken
    ],
)
def test_to_outcome_preserves_telemetry_independent_of_invoked_skills(
    invoked_skills: tuple[str, ...],
    summary_turns: int,
    expected_turns: int,
) -> None:
    """Verify to_outcome does not guess or overwrite turns_taken with len(invoked_skills)."""
    from reach.runtime import SessionSummary

    summary = SessionSummary(
        invoked_skills=invoked_skills,
        turns_taken=summary_turns,
        early_exit=bool(invoked_skills),
    )
    outcome = summary.to_outcome(observed_catalog=("skill-a", "skill-b"))
    assert outcome.invoked_skills == invoked_skills
    assert outcome.turns_taken == expected_turns


def test_to_outcome_honors_explicit_turns_taken_override() -> None:
    """Verify to_outcome accepts explicit turns_taken parameter override."""
    from reach.runtime import SessionSummary

    summary = SessionSummary(
        invoked_skills=("skill-a",),
        turns_taken=1,
    )
    outcome = summary.to_outcome(turns_taken=5)
    assert outcome.turns_taken == 5


def test_selection_outcome_suppresses_cancellation_error_on_early_exit() -> None:
    """Verify SelectionOutcome automatically clears process cancellation errors on early exit."""
    from reach.runtime import SelectionOutcome

    # When early_exit=True, synthetic cancellation/timeout errors must be cleared
    outcome = SelectionOutcome(
        early_exit=True,
        invoked_skills=("skill-a",),
        error="timeout waiting for response",
    )
    assert outcome.error is None

    outcome2 = SelectionOutcome(
        early_exit=True,
        invoked_skills=("skill-a",),
        error="subprocess failed",
    )
    assert outcome2.error is None


def test_selection_outcome_preserves_security_tool_and_residency_leaks() -> None:
    """Verify SelectionOutcome strictly retains tool and residency leak errors on early exit."""
    from reach.runtime import SelectionOutcome

    # Genuine security leaks must never be masked
    outcome_tool = SelectionOutcome(
        early_exit=True,
        invoked_skills=("skill-a",),
        error="tool leak: Bash, write_file",
    )
    assert outcome_tool.error == "tool leak: Bash, write_file"

    outcome_res = SelectionOutcome(
        early_exit=True,
        invoked_skills=("skill-a",),
        error="residency leak: rogue-skill",
    )
    assert outcome_res.error == "residency leak: rogue-skill"


def test_session_summary_syncs_tool_calls_to_observed_tools() -> None:
    """Verify SessionSummary synthesizes observed_tools from tool_calls if not provided."""
    from reach.runtime import SessionSummary, ToolCallInfo

    summary = SessionSummary(
        tool_calls=(
            ToolCallInfo(name="view_file", parameters={"AbsolutePath": "/path/to/skill"}),
            ToolCallInfo(name="list_dir", parameters={}),
        ),
    )
    assert summary.observed_tools == ("list_dir", "view_file")

    # If observed_tools provided directly (e.g. Claude Code), it is preserved
    summary_direct = SessionSummary(observed_tools=("View", "Skill"))
    assert summary_direct.observed_tools == ("View", "Skill")


def test_session_summary_saw_result_property() -> None:
    """Verify saw_result returns True if status is present or early_exit is True."""
    from reach.runtime import SessionStatus, SessionSummary

    assert SessionSummary(status=SessionStatus.SUCCESS).saw_result
    assert SessionSummary(status="SUCCESS").saw_result
    assert SessionSummary(status="success").saw_result
    assert SessionSummary(early_exit=True).saw_result
    assert not SessionSummary(status=None, early_exit=False).saw_result


def test_tool_call_info_path_property_and_coercion() -> None:
    """Verify ToolCallInfo provides path property and parameter coercion."""
    from reach.runtime import ToolCallInfo

    call1 = ToolCallInfo(name="view_file", path="/work/skill/SKILL.md")
    assert call1.path == "/work/skill/SKILL.md"
    assert call1.target_path == "/work/skill/SKILL.md"
    assert call1.parameters == {"path": "/work/skill/SKILL.md"}

    call2 = ToolCallInfo(name="view_file", parameters={"AbsolutePath": "/work/abs.md"})
    assert call2.path == "/work/abs.md"
    assert call2.target_path == "/work/abs.md"


def test_agent_runtime_effective_effort_and_resident_paths(tmp_path: Path) -> None:
    """Verify AgentRuntime effective_effort and resident_skill_paths properties."""
    from reach.runtime.fake import FakeOptions, FakeRuntime

    # Default model profile effort resolution
    runtime = FakeRuntime(options=FakeOptions(model="gemini-2.5-flash"))
    assert runtime.effective_effort == "low"

    # Explicit effort override on options takes precedence
    runtime_override = FakeRuntime(options=FakeOptions(model="gemini-2.5-flash", effort="high"))
    assert runtime_override.effective_effort == "high"

    # resident_skill_paths resolves resident skills under skills_dir
    runtime._resident = ("skill-a", "skill-b")
    paths = runtime.resident_skill_paths(tmp_path)
    expected = {
        (tmp_path / ".agents/skills/skill-a").resolve(),
        (tmp_path / ".agents/skills/skill-b").resolve(),
    }
    assert paths == expected


def test_session_status_normalization_and_idiosyncrasy_handling() -> None:
    """Verify SessionStatus coerces known variants and preserves vendor states gracefully."""
    from reach.runtime import SessionStatus, SessionSummary

    # Canonical enum members
    assert SessionSummary(status=SessionStatus.SUCCESS).status == SessionStatus.SUCCESS
    assert SessionSummary(status=SessionStatus.ERROR).status == SessionStatus.ERROR
    assert SessionSummary(status=SessionStatus.TIMEOUT).status == SessionStatus.TIMEOUT
    assert SessionSummary(status=SessionStatus.CANCELLED).status == SessionStatus.CANCELLED

    # Case-insensitive string normalization to enum
    assert SessionSummary(status="SUCCESS").status == SessionStatus.SUCCESS
    assert SessionSummary(status="success").status == SessionStatus.SUCCESS
    assert SessionSummary(status="error").status == SessionStatus.ERROR
    assert SessionSummary(status="timeout").status == SessionStatus.TIMEOUT
    assert SessionSummary(status="cancelled").status == SessionStatus.CANCELLED

    # Unrecognized / vendor-specific idiosyncratic status preserved without validation crash
    vendor_summary = SessionSummary(status="rate_limited_tier_2")
    assert vendor_summary.status == "rate_limited_tier_2"
    assert vendor_summary.saw_result


# --- Antigravity domain schema and tool contract -----------------------------


def test_antigravity_runtime_selection_schema_and_json_schema() -> None:
    """Verify AntigravityRuntime generates both Pydantic model and derived JSON schema."""
    import json

    from reach.runtime import AntigravityRuntime

    schema_cls = AntigravityRuntime.selection_schema(["skill-1", "skill-2"])
    valid_instance = schema_cls(selected_skill="skill-1", reasoning="matched intent")
    assert valid_instance.selected_skill == "skill-1"
    assert valid_instance.reasoning == "matched intent"

    json_schema_str = AntigravityRuntime.selection_json_schema(["skill-1", "skill-2"])
    schema_dict = json.loads(json_schema_str)
    assert schema_dict["type"] == "object"
    assert "selected_skill" in schema_dict["properties"]
    assert "reasoning" in schema_dict["properties"]


def test_antigravity_runtime_selection_schema_empty_catalog() -> None:
    """Verify AntigravityRuntime handles empty resident catalog gracefully."""
    from reach.runtime import AntigravityRuntime

    schema_cls = AntigravityRuntime.selection_schema([])
    instance = schema_cls(selected_skill="arbitrary", reasoning="fallback")
    assert instance.selected_skill == "arbitrary"


def test_antigravity_runtime_selection_tools() -> None:
    """Verify AntigravityRuntime defines canonical selection tools set."""
    from reach.runtime import AntigravityRuntime

    expected = frozenset({"view_file", "list_dir", "grep_search", "find_by_name"})
    assert expected == AntigravityRuntime.ANTIGRAVITY_SELECTION_TOOLS


def test_antigravity_runtime_effective_model_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify AntigravityRuntime auto-detects gemini provider when API keys are set."""
    from reach.config import RuntimeSettings
    from reach.runtime.antigravity_cli import AntigravityCliOptions, AntigravityCliRuntime
    from reach.runtime.antigravity_sdk import AntigravitySdkRuntime

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    cli_rt = AntigravityCliRuntime(
        options=AntigravityCliOptions(model="gemini-3.7-flash"),
    )
    assert cli_rt.effective_model_provider == "gemini"

    if _HAS_ANTIGRAVITY:
        sdk_rt = AntigravitySdkRuntime(
            settings=RuntimeSettings(
                agent="antigravity-sdk",
                options={"model": "gemini-3.7-flash"},
            ),
        )
        assert sdk_rt.effective_model_provider == "gemini"

    # Non-gemini model does not activate provider
    non_gemini = AntigravityCliRuntime(
        options=AntigravityCliOptions(model="claude-3-opus"),
    )
    assert non_gemini.effective_model_provider is None


# --- CLI agent runtime execution template ------------------------------------


class DummyCliRuntime(CliAgentRuntime[CliOptions]):
    """Concrete dummy CLI runtime for testing template method behavior."""

    name = "dummy-cli"
    options: CliOptions

    @override
    def build_command(self, query_text: str) -> list[str]:
        """Assemble probe command."""
        return ["dummy", "-q", query_text]

    @override
    def parse_stream(
        self,
        lines: Iterable[str],
        resident: Sequence[str] = (),
        early_exit: bool = False,
    ) -> SessionSummary:
        """Parse stream lines into SessionSummary."""
        del resident
        invoked = [line.split(":", 1)[1].strip() for line in lines if line.startswith("SKILL:")]
        return SessionSummary(
            invoked_skills=tuple(invoked),
            early_exit=early_exit,
            status="SUCCESS" if invoked else None,
        )

    @override
    def extract_skill_from_line(self, line: str) -> str | None:
        """Extract skill from single line."""
        if line.startswith("SKILL:"):
            return line.split(":", 1)[1].strip()
        return None


def test_cli_template_method_clean_execution(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify CliAgentRuntime template method runs subprocess and returns outcome."""
    mock_subprocess(stdout="SKILL:target-skill\n")
    rt = DummyCliRuntime()
    rt._resident = ("target-skill", "other-skill")

    outcome = rt.select("test query", tmp_path, target_skill="target-skill")
    assert outcome.error is None
    assert outcome.invoked_skill == "target-skill"
    assert outcome.early_exit is True
    assert outcome.turns_taken == 1
    assert outcome.observed_catalog == ("target-skill", "other-skill")


def test_cli_agent_runtime_validate_outcome_status_check(tmp_path: Path) -> None:
    """Verify CliAgentRuntime base validate_outcome detects non-success status."""
    from reach.runtime import SessionStatus, SessionSummary

    rt = DummyCliRuntime()
    clean_summary = SessionSummary(status=SessionStatus.SUCCESS)
    assert rt.validate_outcome(clean_summary, tmp_path) is None

    error_summary = SessionSummary(status=SessionStatus.ERROR, error="bad state")
    assert rt.validate_outcome(error_summary, tmp_path) == "bad state"

    timeout_summary = SessionSummary(status=SessionStatus.TIMEOUT)
    assert rt.validate_outcome(timeout_summary, tmp_path) == "runtime error: TIMEOUT"


def test_cli_select_handles_spawn_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify CliAgentRuntime handles subprocess spawn failures without raising."""

    def _mock_run(*args: Any, **kwargs: Any) -> tuple[None, str]:
        return None, "executable not found: dummy"

    monkeypatch.setattr("reach.runtime.run_subprocess_probe", _mock_run)
    rt = DummyCliRuntime()

    outcome = rt.select("query", tmp_path)
    assert outcome.error == "executable not found: dummy"
    assert outcome.invoked_skill is None


def test_cli_select_handles_missing_result_event(
    mock_subprocess: Callable[..., Any],
    tmp_path: Path,
) -> None:
    """Verify CliAgentRuntime flags missing result event when process produces no status."""
    mock_subprocess(stdout="plain log without skill", returncode=1, stderr="crashed")
    rt = DummyCliRuntime()

    outcome = rt.select("query", tmp_path)
    assert outcome.error is not None
    assert "no result event" in outcome.error


def test_cli_generator_complete_success(
    mock_subprocess: Callable[..., Any],
) -> None:
    """Verify CLI generator complete executes subprocess and returns text."""
    mock_subprocess(stdout="generated answer\n")
    gen = PiGenerator()
    assert gen.complete("test prompt") == "generated answer"
    assert gen.completions == 1


def test_cli_generator_complete_failure(
    mock_subprocess: Callable[..., Any],
) -> None:
    """Verify CLI generator complete raises on subprocess failure."""
    mock_subprocess(returncode=1, stderr="fatal error")
    gen = PiGenerator()
    with pytest.raises(RuntimeError, match=r"generation failed: fatal error"):
        gen.complete("test prompt")


@pytest.mark.parametrize("agent", cli_agents())
def test_all_cli_generators_build_completion_command(agent: str, tmp_path: Path) -> None:
    """Verify all CLI text generators construct valid completion commands starting with binary."""
    gen = build_text_generator(agent=agent)
    cmd_fn = getattr(gen, "build_completion_command", None)
    assert callable(cmd_fn)
    cmd = cmd_fn("hello world")
    assert isinstance(cmd, list)
    assert len(cmd) > 0
    runtime = _build_agent(agent, tmp_path)
    assert isinstance(runtime, CliAgentRuntime)
    assert cmd[0] == runtime.options.executable


# --- Cross-client runtime conformance ----------------------------------------


@pytest.mark.parametrize("agent", known_agents())
def test_all_agents_cli_hierarchy_consistency(agent: str, tmp_path: Path) -> None:
    """Verify runtime.is_cli matches CliAgentRuntime and CliOptions hierarchy across agents."""
    runtime = _build_agent(agent, tmp_path)
    opt = options_model(agent)
    is_cli_runtime = isinstance(runtime, CliAgentRuntime)
    assert runtime.is_cli is is_cli_runtime
    assert is_cli_runtime == (opt is not None and issubclass(opt, CliOptions))


@pytest.mark.parametrize("agent", cli_agents())
def test_all_cli_agents_implement_build_command_contract(agent: str, tmp_path: Path) -> None:
    """Verify all CLI drivers implement the build_command contract."""
    runtime = _build_agent(agent, tmp_path)
    assert isinstance(runtime, CliAgentRuntime)
    cmd = runtime.build_command("test query")
    assert isinstance(cmd, list)
    assert len(cmd) > 0
    assert isinstance(cmd[0], str)


@pytest.mark.parametrize("agent", cli_agents())
def test_all_cli_agents_conform_to_timeout_handling(
    agent: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify all CLI drivers return a SelectionOutcome on timeout rather than raising."""
    runtime = _build_agent(agent, tmp_path)
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    def _mock_timeout(*_args: Any, **_kwargs: Any) -> tuple[None, str]:
        return None, "timeout"

    monkeypatch.setattr("reach.runtime.run_subprocess_probe", _mock_timeout)
    module_name = runtime.__class__.__module__
    monkeypatch.setattr(f"{module_name}.run_subprocess_probe", _mock_timeout, raising=False)
    outcome = runtime.select("query", workdir)
    assert isinstance(outcome, SelectionOutcome)
    assert outcome.error is not None
    assert "timeout" in outcome.error.lower() or "timed out" in outcome.error.lower()


@pytest.mark.parametrize(
    "agent",
    [a for a in known_agents() if a.startswith("antigravity")],
)
def test_all_antigravity_agents_share_selection_tools(agent: str, tmp_path: Path) -> None:
    """Verify that both Antigravity drivers expose identical selection inspection tools."""
    runtime = _build_agent(agent, tmp_path)
    assert isinstance(runtime, AntigravityRuntime)
    assert runtime.selection_tools == AntigravityRuntime.ANTIGRAVITY_SELECTION_TOOLS


@pytest.mark.parametrize(
    "agent",
    [a for a in known_agents() if a.startswith("antigravity")],
)
def test_all_antigravity_agents_conform_to_schema_contract(agent: str, tmp_path: Path) -> None:
    """Verify that Antigravity drivers generate consistent selection schemas."""
    runtime = _build_agent(agent, tmp_path)
    assert isinstance(runtime, AntigravityRuntime)
    schema_cls = runtime.selection_schema(("skill-a", "skill-b"))
    assert issubclass(schema_cls, SkillSelectionBase)
    json_schema = runtime.selection_json_schema(("skill-a", "skill-b"))
    assert "skill-a" in json_schema
    assert "skill-b" in json_schema


@pytest.mark.parametrize("agent", list(known_agents()))
def test_all_agents_install_catalog_and_report_skill_roots(
    agent: str,
    tmp_path: Path,
) -> None:
    """Verify all agents install catalog skills and report skill roots."""
    runtime = _build_agent(agent, tmp_path)
    skill_src = tmp_path / "src" / "alpha"
    skill_src.mkdir(parents=True)
    (skill_src / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Alpha skill.\n---\nBody",
        encoding="utf-8",
    )
    skill = Skill(name="alpha", description="Alpha skill.", path=skill_src)
    catalog = Catalog(id="test-cat", mode=CatalogMode.ALL, skills=("alpha",))

    workdir = tmp_path / "work"
    target = runtime.install(catalog, [skill], workdir)
    assert (runtime.skills_dir(target) / "alpha" / "SKILL.md").is_file()
    roots = runtime.skill_roots(workdir)
    assert len(roots) >= 1
    assert any(r.path == runtime.skills_dir(target) for r in roots)


@pytest.mark.parametrize("agent", list(known_agents()))
def test_all_agents_parse_stream_empty_input(agent: str, tmp_path: Path) -> None:
    """Verify all drivers return a valid SessionSummary without skills for empty stream."""
    runtime = _build_agent(agent, tmp_path)
    summary = runtime.parse_stream([])
    assert isinstance(summary, SessionSummary)
    assert summary.invoked_skill is None
    assert summary.invoked_skills == ()
    assert not summary.saw_result


def test_keyword_runtime_parse_stream_extracts_skill() -> None:
    """Verify KeywordRuntime extracts invoked skill from stream lines."""
    runtime = KeywordRuntime()
    summary = runtime.parse_stream(
        ["Model chose tool cloud-sql to proceed"],
        resident=("cloud-sql", "gcloud"),
    )
    assert summary.invoked_skill == "cloud-sql"
    assert summary.invoked_skills == ("cloud-sql",)
    assert summary.saw_result


def test_fake_runtime_parse_stream_uses_configured_default() -> None:
    """Verify FakeRuntime extracts default response during stream parsing."""
    runtime = FakeRuntime(default="alpha")
    summary = runtime.parse_stream(["some log line"])
    assert summary.invoked_skill == "alpha"
    assert summary.invoked_skills == ("alpha",)
    assert summary.saw_result


@pytest.mark.parametrize("agent", cli_agents())
@pytest.mark.parametrize(
    "raw_line",
    ["", "   ", "not json", "[1, 2, 3]", '"just-a-string"', "12345", "true", "null", "{}"],
    ids=[
        "empty",
        "whitespace",
        "garbage",
        "json-array",
        "json-string",
        "json-int",
        "json-bool",
        "json-null",
        "empty-dict",
    ],
)
def test_all_cli_agents_extract_skills_from_line_gracefully_handles_malformed_inputs(
    agent: str,
    raw_line: str,
    tmp_path: Path,
) -> None:
    """Verify extract_skills_from_line safely returns empty sequence for malformed inputs."""
    runtime = _build_agent(agent, tmp_path)
    assert isinstance(runtime, CliAgentRuntime)
    assert runtime.extract_skills_from_line(raw_line) == ()


@pytest.mark.parametrize("agent", list(known_agents()))
def test_all_agents_parse_stream_malformed_input(agent: str, tmp_path: Path) -> None:
    """Verify all drivers return a valid SessionSummary without raising on malformed stream."""
    runtime = _build_agent(agent, tmp_path)

    summary = runtime.parse_stream(["not json", "[1, 2, 3]", "12345", "true", "null", "{}"])
    assert isinstance(summary, SessionSummary)


def test_sync_google_and_gemini_keys_bidirectional() -> None:
    """Verify sync_google_and_gemini_keys synchronizes keys in both directions."""
    # GEMINI -> GOOGLE
    env1 = {"GEMINI_API_KEY": "secret-1"}
    sync_google_and_gemini_keys(env1)
    assert env1 == {"GEMINI_API_KEY": "secret-1", "GOOGLE_API_KEY": "secret-1"}

    # GOOGLE -> GEMINI
    env2 = {"GOOGLE_API_KEY": "secret-2"}
    sync_google_and_gemini_keys(env2)
    assert env2 == {"GEMINI_API_KEY": "secret-2", "GOOGLE_API_KEY": "secret-2"}

    # Both present -> preserve existing
    env3 = {"GEMINI_API_KEY": "gemini-orig", "GOOGLE_API_KEY": "google-orig"}
    sync_google_and_gemini_keys(env3)
    assert env3 == {"GEMINI_API_KEY": "gemini-orig", "GOOGLE_API_KEY": "google-orig"}

    # Neither present -> no changes
    env4 = {"OTHER_KEY": "other"}
    sync_google_and_gemini_keys(env4)
    assert env4 == {"OTHER_KEY": "other"}


@pytest.mark.parametrize(
    ("provider", "expected_vars"),
    [
        ("anthropic", ("ANTHROPIC_API_KEY",)),
        ("openai", ("OPENAI_API_KEY",)),
        ("google", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
        ("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    ],
)
def test_apply_provider_api_key_known_providers(
    provider: str,
    expected_vars: tuple[str, ...],
) -> None:
    """Verify apply_provider_api_key maps keys for all known providers."""
    env: dict[str, str] = {}
    apply_provider_api_key(env, provider=provider, api_key="test-key")
    for var in expected_vars:
        assert env[var] == "test-key"


def test_apply_provider_api_key_fallback_and_empty() -> None:
    """Verify provider mapping handles missing key, default provider, and unknown fallbacks."""
    # Empty / None key does not modify env
    env1 = {"EXISTING": "val"}
    apply_provider_api_key(env1, provider="google", api_key=None)
    assert env1 == {"EXISTING": "val"}
    apply_provider_api_key(env1, provider="google", api_key="")
    assert env1 == {"EXISTING": "val"}

    # None provider uses default_provider
    env2: dict[str, str] = {}
    apply_provider_api_key(env2, provider=None, api_key="k1", default_provider="openai")
    assert env2 == {"OPENAI_API_KEY": "k1"}

    env3: dict[str, str] = {}
    apply_provider_api_key(env3, provider=None, api_key="k2", default_provider="google")
    assert env3 == {"GEMINI_API_KEY": "k2", "GOOGLE_API_KEY": "k2"}

    # Unknown provider falls back to default_provider
    env4: dict[str, str] = {}
    apply_provider_api_key(env4, provider="unknown-custom", api_key="k3", default_provider="openai")
    assert env4 == {"OPENAI_API_KEY": "k3"}


@pytest.mark.parametrize(
    "agent",
    [a for a in _EXECUTION_AGENTS if a != "claude-code"],
)
def test_all_google_supporting_agents_sync_ambient_keys(
    agent: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Google-supporting agents synchronize ambient GEMINI and GOOGLE keys in build_env."""
    monkeypatch.setenv("GEMINI_API_KEY", "ambient-sync-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    runtime = _build_agent(agent, tmp_path)
    env = runtime.build_env(tmp_path / f"work_{agent}")
    assert env["GEMINI_API_KEY"] == "ambient-sync-key"
    assert env["GOOGLE_API_KEY"] == "ambient-sync-key"


def test_cli_agents_discovery() -> None:
    """Verify cli_agents returns registered CLI driver names and filters non-CLI runtimes."""
    agents = cli_agents()
    assert agents == ("antigravity-cli", "claude-code", "goose", "pi")
    for agent in agents:
        opt = options_model(agent)
        assert opt is not None
        assert issubclass(opt, CliOptions)


def test_claude_generator_build_env_forwards_api_key() -> None:
    """Verify ClaudeGenerator build_env passes ANTHROPIC_API_KEY when configured in options."""
    from reach.runtime.claude_code import ClaudeCodeOptions

    gen = ClaudeGenerator(
        options=ClaudeCodeOptions(
            api_key="claude-secret-key",
        ),
    )
    env = gen.build_env()
    assert env["ANTHROPIC_API_KEY"] == "claude-secret-key"


def test_model_validators_preserve_input_dict_immutability() -> None:
    """Verify before-validators in runtime models do not mutate caller dictionaries."""
    tool_dict = {"name": "read", "path": "/path/to/file"}
    coerced = ToolCallInfo.model_validate(tool_dict)
    assert coerced.parameters["path"] == "/path/to/file"
    assert "path" in tool_dict
    assert "parameters" not in tool_dict

    summary_dict = {"tool_calls": [{"name": "read"}]}
    summary = SessionSummary.model_validate(summary_dict)
    assert summary.observed_tools == ("read",)
    assert "observed_tools" not in summary_dict

    outcome_dict = {"early_exit": True, "error": "process killed"}
    outcome = SelectionOutcome.model_validate(outcome_dict)
    assert outcome.error is None
    assert outcome_dict["error"] == "process killed"
