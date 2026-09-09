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

"""Render modularity-based skill cluster partitions and cohesion tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.table import Table
from rich.text import Text

from reach.rendering import csv_document, dispatch_render

if TYPE_CHECKING:
    from rich.console import Console

    from reach.cluster import ClusterPartition

__all__ = [
    "CLUSTER_RENDERERS",
    "print_cluster",
    "render_cluster",
    "render_cluster_csv",
    "render_cluster_json",
]


def print_cluster(console: Console, partition: ClusterPartition) -> None:
    """Render a Rich table summarizing community clusters and intra-cluster cohesion."""
    console.print(
        Text.assemble(
            ("Skill Partition: ", "bold"),
            (f"{len(partition.clusters)} communities", "bold cyan"),
            f" across {partition.total_skills} skills ",
            (
                f"(Modularity Q = {partition.modularity:.4f})",
                "green" if partition.modularity > 0 else "dim",
            ),
        ),
        soft_wrap=True,
    )
    console.print()

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        title="Subagent Candidate Catalogs",
        title_justify="left",
    )

    table.add_column("Cluster ID", justify="left", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Cohesion", justify="right", style="magenta")
    table.add_column("Member Skills", justify="left")

    for cluster in partition.clusters:
        table.add_row(
            cluster.id,
            str(len(cluster.skills)),
            f"{cluster.cohesion:.2f}",
            ", ".join(cluster.skills),
        )

    console.print(table)


def render_cluster_json(partition: ClusterPartition) -> str:
    """Serialize ClusterPartition to JSON."""
    return partition.model_dump_json(indent=2)


def render_cluster_csv(partition: ClusterPartition) -> str:
    """Export ClusterPartition to CSV."""
    rows: list[list[object]] = [
        [cluster.id, f"{cluster.cohesion:.4f}", skill]
        for cluster in partition.clusters
        for skill in cluster.skills
    ]
    return csv_document(["cluster_id", "cohesion", "skill"], rows)


CLUSTER_RENDERERS = {
    "csv": render_cluster_csv,
    "json": render_cluster_json,
}


def render_cluster(partition: ClusterPartition, fmt: str) -> str:
    """Render ClusterPartition into the requested format."""
    return dispatch_render(CLUSTER_RENDERERS, fmt, partition)
