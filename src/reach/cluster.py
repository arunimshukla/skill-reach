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

"""Cluster skills into cohesive subagent scopes using modularity optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from reach.overlap import rank_corpus

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reach.models import Skill

__all__ = [
    "ClusterPartition",
    "SkillCluster",
    "cluster_skills",
]


class SkillCluster(BaseModel):
    """Represent a cohesive cluster of skills for subagent scoping."""

    model_config = ConfigDict(frozen=True)

    id: str
    skills: tuple[str, ...] = ()
    cohesion: float = 0.0


class ClusterPartition(BaseModel):
    """Represent a modularity clustering partition of a skill corpus."""

    model_config = ConfigDict(frozen=True)

    clusters: tuple[SkillCluster, ...] = ()
    modularity: float = 0.0
    total_skills: int = 0

    @property
    def cluster_map(self) -> dict[str, str]:
        """Map each skill name directly to its assigned cluster identifier."""
        return {skill: c.id for c in self.clusters for skill in c.skills}


_EPSILON: float = 1e-12


def _build_adjacency_matrix(
    skills: Sequence[Skill],
) -> tuple[list[list[float]], list[float], float]:
    """Construct a symmetric BM25 adjacency matrix and degree weights for skills."""
    n = len(skills)
    overlap = rank_corpus(skills)
    skill_index: dict[str, int] = {skill.name: i for i, skill in enumerate(skills)}

    w: list[list[float]] = [[0.0] * n for _ in range(n)]
    for competition in overlap.competitions:
        u = skill_index.get(competition.skill)
        if u is None:
            continue
        for rival in competition.rivals:
            v = skill_index.get(rival.name)
            if v is not None:
                w[u][v] = rival.score

    for i in range(n):
        for j in range(i + 1, n):
            sym = 0.5 * (w[i][j] + w[j][i])
            w[i][j] = sym
            w[j][i] = sym
        w[i][i] = 0.0

    deg: list[float] = [sum(w[i]) for i in range(n)]
    total_weight: float = sum(deg)
    return w, deg, total_weight


def _find_best_merge(
    pair_w: Mapping[tuple[int, int], float],
    communities: Mapping[int, set[int]],
    deg_sum: Mapping[int, float],
    total_weight: float,
    resolution: float,
    target_size: int | None,
    enforce_size: bool,
) -> tuple[tuple[int, int] | None, float]:
    """Scan candidate community pairs and find the merge with highest modularity gain."""
    best_pair: tuple[int, int] | None = None
    best_delta_q = -float("inf")

    for (u, v), edge_wt in pair_w.items():
        if edge_wt <= 0.0:
            continue
        if (
            enforce_size
            and target_size is not None
            and len(communities[u]) + len(communities[v]) > target_size
        ):
            continue

        delta_q = (1.0 / total_weight) * (
            edge_wt - 2.0 * resolution * deg_sum[u] * deg_sum[v] / total_weight
        )
        if delta_q > best_delta_q:
            best_delta_q = delta_q
            best_pair = (u, v)

    return best_pair, best_delta_q


def _should_record_best(
    communities_count: int,
    current_q: float,
    best_q: float,
    best_communities_count: int,
    max_clusters: int | None,
) -> bool:
    """Check whether the current community state satisfies optimization bounds."""
    if max_clusters is None:
        return current_q > best_q
    if communities_count <= max_clusters:
        return current_q > best_q or best_communities_count > max_clusters
    return False


def _select_next_merge(
    pair_w: dict[tuple[int, int], float],
    communities: dict[int, set[int]],
    deg_sum: dict[int, float],
    total_weight: float,
    resolution: float,
    target_size: int | None,
    max_clusters: int | None,
) -> tuple[tuple[int, int] | None, float]:
    """Find the best community pair to merge, relaxing size constraints when required."""
    best_pair, best_delta_q = _find_best_merge(
        pair_w, communities, deg_sum, total_weight, resolution, target_size, enforce_size=True
    )
    if best_pair is None and max_clusters is not None and len(communities) > max_clusters:
        return _find_best_merge(
            pair_w,
            communities,
            deg_sum,
            total_weight,
            resolution,
            target_size,
            enforce_size=False,
        )
    return best_pair, best_delta_q


def _apply_community_merge(
    u: int,
    v: int,
    communities: dict[int, set[int]],
    deg_sum: dict[int, float],
    internal_w: dict[int, float],
    pair_w: dict[tuple[int, int], float],
) -> None:
    """Merge community v into community u and update graph weights in place."""
    communities[u].update(communities[v])
    internal_w[u] += internal_w[v] + pair_w.pop((u, v), 0.0)
    deg_sum[u] += deg_sum[v]

    other_communities = [c for c in communities if c not in (u, v)]
    for c in other_communities:
        key_uc = (min(u, c), max(u, c))
        key_vc = (min(v, c), max(v, c))
        weight_vc = pair_w.pop(key_vc, 0.0)
        if weight_vc > 0.0 or key_uc in pair_w:
            pair_w[key_uc] = pair_w.get(key_uc, 0.0) + weight_vc

    del communities[v]
    del deg_sum[v]
    del internal_w[v]


def _optimize_communities(
    communities: dict[int, set[int]],
    deg_sum: dict[int, float],
    internal_w: dict[int, float],
    pair_w: dict[tuple[int, int], float],
    total_weight: float,
    resolution: float,
    max_clusters: int | None,
    target_size: int | None,
    initial_q: float,
) -> tuple[float, list[set[int]]]:
    """Merge communities iteratively to maximize modularity until termination."""
    current_q = initial_q
    best_q = current_q
    best_communities: list[set[int]] = [set(m) for m in communities.values()]

    while len(communities) > 1:
        if _should_record_best(
            len(communities), current_q, best_q, len(best_communities), max_clusters
        ):
            best_q = current_q
            best_communities = [set(m) for m in communities.values()]

        best_pair, best_delta_q = _select_next_merge(
            pair_w, communities, deg_sum, total_weight, resolution, target_size, max_clusters
        )

        if best_pair is None or (max_clusters is None and best_delta_q <= 0.0):
            break

        _apply_community_merge(best_pair[0], best_pair[1], communities, deg_sum, internal_w, pair_w)
        current_q += best_delta_q

        if current_q > best_q:
            best_q = current_q
            best_communities = [set(m) for m in communities.values()]

    return best_q, best_communities


def _build_clusters(
    skills: Sequence[Skill],
    communities: list[set[int]],
    w: list[list[float]],
) -> tuple[SkillCluster, ...]:
    """Sort and package communities into cohesive SkillCluster instances."""
    sorted_comm = sorted(
        communities,
        key=lambda members: (
            -len(members),
            min(skills[i].name for i in members),
        ),
    )

    clusters_list: list[SkillCluster] = []
    for rank, members in enumerate(sorted_comm, start=1):
        member_names = tuple(sorted(skills[i].name for i in members))
        if len(members) <= 1:
            cohesion = 1.0
        else:
            pairwise_sum = sum(w[i][j] for i in members for j in members if i < j)
            possible_pairs = len(members) * (len(members) - 1) / 2.0
            cohesion = round(pairwise_sum / possible_pairs, 4) if possible_pairs > 0 else 1.0

        clusters_list.append(
            SkillCluster(
                id=f"cluster-{rank}",
                skills=member_names,
                cohesion=cohesion,
            )
        )
    return tuple(clusters_list)


def cluster_skills(
    skills: Sequence[Skill],
    *,
    resolution: float = 1.0,
    max_clusters: int | None = None,
    target_size: int | None = None,
) -> ClusterPartition:
    """Partition skills into cohesive communities using modularity optimization.

    Build an adjacency matrix from symmetrized BM25 overlap scores between skill descriptions,
    then iteratively merge communities that produce the greatest gain in modularity (Q).

    Args:
        skills: The corpus of skills to cluster.
        resolution: Resolution parameter (gamma). Higher values yield smaller, more compact
            clusters; lower values yield larger, merged clusters. Defaults to 1.0.
        max_clusters: Optional upper bound on the number of output clusters.
        target_size: Optional soft upper bound on the maximum number of skills per cluster.

    Returns:
        ClusterPartition with identified skill clusters and final modularity score.
    """
    unique_by_name = {s.name: s for s in skills}
    unique_skills = list(unique_by_name.values())
    total_skills = len(unique_skills)
    if total_skills == 0:
        return ClusterPartition(clusters=(), modularity=0.0, total_skills=0)

    if total_skills == 1:
        cluster = SkillCluster(id="cluster-1", skills=(unique_skills[0].name,), cohesion=1.0)
        return ClusterPartition(clusters=(cluster,), modularity=0.0, total_skills=1)

    n = total_skills
    w, deg, total_weight = _build_adjacency_matrix(unique_skills)

    if total_weight < _EPSILON:
        clusters = tuple(
            SkillCluster(id=f"cluster-{i + 1}", skills=(unique_skills[i].name,), cohesion=1.0)
            for i in range(n)
        )
        return ClusterPartition(clusters=clusters, modularity=0.0, total_skills=n)

    communities: dict[int, set[int]] = {i: {i} for i in range(n)}
    deg_sum: dict[int, float] = {i: deg[i] for i in range(n)}
    internal_w: dict[int, float] = dict.fromkeys(range(n), 0.0)

    pair_w: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if w[i][j] > 0.0:
                pair_w[(i, j)] = 2.0 * w[i][j]

    initial_q = -resolution / (total_weight**2) * sum(d**2 for d in deg)
    best_q, best_communities = _optimize_communities(
        communities=communities,
        deg_sum=deg_sum,
        internal_w=internal_w,
        pair_w=pair_w,
        total_weight=total_weight,
        resolution=resolution,
        max_clusters=max_clusters,
        target_size=target_size,
        initial_q=initial_q,
    )

    final_modularity = max(0.0, round(best_q, 4)) if best_q > -1.0 else 0.0
    clusters = _build_clusters(unique_skills, best_communities, w)

    return ClusterPartition(
        clusters=clusters,
        modularity=final_modularity,
        total_skills=total_skills,
    )
