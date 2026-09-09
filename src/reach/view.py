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

"""Render a recorded run artifact as a self-contained HTML report.

For Rich-based terminal and console formatting views, see `reach.views`.
"""

from __future__ import annotations

import html
from collections import defaultdict
from typing import TYPE_CHECKING

from reach.artifact import NO_SKILL
from reach.rendering import dispatch_render

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reach.artifact import Artifact, QueryRecord, SkillScore
    from reach.leak import Leak
    from reach.uncertainty import Interval


def _esc(value: object) -> str:
    """Escape a value for safe inclusion in HTML text or attributes."""
    return html.escape(str(value), quote=True)


def _pct(value: float | None) -> str:
    """Format a floating-point ratio as a whole-percentage string, or empty if None."""
    return "" if value is None else f"{value * 100:.0f}%"


def _bounds(interval: Interval | None) -> str:
    """Format an Interval as a hyphen-separated percentage range."""
    if interval is None:
        return ""
    return f"{interval.low * 100:.0f}-{interval.high * 100:.0f}%"


def _header_html(artifact: Artifact) -> str:
    """Generate HTML header with run provenance, metrics, and digests."""
    provenance = artifact.provenance
    digests = artifact.digests
    scores = artifact.scores
    spread = artifact.spread
    error = f" ± {spread.standard_error * 100:.1f}pp" if spread.standard_error else ""
    return f"""
<h1>{_esc(artifact.catalog_id)}</h1>
<p class="subject">
  {_esc(artifact.catalog_mode.value)} &middot; {artifact.catalog_size} skills
  &middot; {_esc(provenance.runtime)}/{_esc(provenance.model)} x{provenance.attempts}
  &middot; {artifact.probes} probes
</p>
<dl class="figures">
  <dt>consistency</dt>
  <dd>{_pct(scores.consistency)}{_bounds(scores.consistency_interval)}</dd>
  <dt>top-1</dt><dd>{scores.top1_accuracy * 100:.1f}%{error}</dd>
  <dt>abstention</dt>
  <dd>{_pct(scores.abstention.rate)} ({_pct(scores.abstention.false_rate)} false)</dd>
  <dt>macro-F1</dt><dd>{scores.not_headline.macro_f1 * 100:.1f}%</dd>
</dl>
<p class="digests">
  arm {_esc(provenance.arm)} &middot; corpus {_esc(digests.corpus_digest)}
  &middot; truth {_esc(digests.queries_digest)}
</p>
"""


def _confusion_cell(
    row: str,
    col: str,
    counts: Mapping[tuple[str, str], int],
    query_samples: Mapping[tuple[str, str], str],
) -> str:
    """Format an individual table data cell in the confusion matrix."""
    n = counts.get((row, col), 0)
    css = "hit" if row == col else ("miss" if n else "")
    title = query_samples.get((row, col))
    attr = f' title="{_esc(title)}"' if title else ""
    return f'<td class="{css}"{attr}>{n or ""}</td>'


def _confusion_html(artifact: Artifact) -> str:
    """Generate the HTML table for the expected vs invoked confusion matrix."""
    pairs = artifact.confusion
    if not pairs:
        return '<p class="empty">No probes were recorded.</p>'
    counts: dict[tuple[str, str], int] = defaultdict(int)
    query_samples: dict[tuple[str, str], str] = {}
    expected_labels: set[str] = set()
    invoked_labels: set[str] = set()
    for pair in pairs:
        counts[(pair.expected, pair.invoked)] += pair.probes
        if pair.queries:
            sample = pair.queries[0]
            if sample.reasoning:
                traces = " ".join(sample.reasoning)
                query_samples[(pair.expected, pair.invoked)] = f"{sample.text}\n\nthought: {traces}"
            else:
                query_samples[(pair.expected, pair.invoked)] = sample.text
        expected_labels.add(pair.expected)
        invoked_labels.add(pair.invoked)
    rows = sorted(expected_labels)
    cols = sorted(invoked_labels)
    head = "".join(f"<th>{_esc(col)}</th>" for col in cols)
    body_rows = [
        (
            f"<tr><th>{_esc(row)}</th>"
            f"{''.join(_confusion_cell(row, col, counts, query_samples) for col in cols)}</tr>"
        )
        for row in rows
    ]
    return (
        '<table class="confusion" id="confusion-table">'
        f"<thead><tr><th>expected \\ invoked</th>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _collisions_html(artifact: Artifact) -> str:
    """Generate HTML table detailing misrouting collisions and reasoning traces."""
    collisions = [
        p for p in artifact.confusion if p.invoked not in (p.expected, NO_SKILL) and p.probes > 0
    ]
    if not collisions:
        return ""
    rows = []
    for pair in collisions:
        for q in pair.queries:
            thought_html = (
                f'<div class="thought">thought: {_esc(" ".join(q.reasoning))}</div>'
                if q.reasoning
                else ""
            )
            rows.append(
                "<tr>"
                f"<td>{_esc(pair.expected)}</td>"
                f"<td>{_esc(pair.invoked)}</td>"
                f"<td>{pair.probes}</td>"
                f"<td>{_esc(q.text)}{thought_html}</td>"
                "</tr>"
            )
    if not rows:
        return ""
    return f"""
<h2>Collisions</h2>
<table class="collisions" id="collisions-table">
  <thead>
    <tr><th>expected</th><th>invoked</th><th>n</th><th>query</th></tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>
"""


def _skills_table_html(skills: Sequence[SkillScore]) -> str:
    """Generate a filterable HTML table of resident skills with recall and precision."""
    ordered = sorted(
        skills,
        key=lambda s: (s.recall is None, s.recall or 0.0, -s.absorbed, s.skill),
    )
    body_rows = []
    for skill in ordered:
        reached = f"{skill.reached}/{skill.probes}" if skill.probes else ""
        body_rows.append(
            "<tr>"
            f"<td>{_esc(skill.skill)}</td>"
            f"<td>{_pct(skill.recall)}</td>"
            f"<td>{_esc(_bounds(skill.recall_interval))}</td>"
            f"<td>{reached}</td>"
            f"<td>{skill.absorbed or ''}</td>"
            f"<td>{_pct(skill.precision)}</td>"
            f"<td>{_pct(skill.f1)}</td>"
            "</tr>",
        )
    return f"""
<input
  type="text"
  class="filter"
  placeholder="Filter skills by name..."
  oninput="reachFilterRows(this, 'skills-table')"
  aria-controls="skills-table"
/>
<table class="skills" id="skills-table">
  <thead>
    <tr>
      <th>skill</th><th>recall</th><th>CI</th><th>reached</th>
      <th>absorbed</th><th>precision</th><th>F1</th>
    </tr>
  </thead>
  <tbody>{"".join(body_rows)}</tbody>
</table>
"""


def _leak_description(leak: Leak | None) -> str:
    """Format leak evaluation outcome or routes for query detail display."""
    if leak is None:
        return "not evaluated"
    if not leak.leaked:
        return "clean"
    return "; ".join(_esc(route) for route in leak.routes)


def _query_detail_html(record: QueryRecord, catalog_size: int) -> str:
    """Generate expanded detail HTML for an individual QueryRecord."""
    rank = (
        f"{record.difficulty_rank}/{catalog_size}"
        if record.difficulty_rank is not None
        else "unranked"
    )
    selections = ", ".join(_esc(s) for s in record.selections) or "none"
    leak = _leak_description(record.leak)
    return f"""
<dl>
  <dt>text</dt><dd>{_esc(record.text)}</dd>
  <dt>kind</dt><dd>{_esc(record.kind.value) if record.kind else "unset"}</dd>
  <dt>expected</dt><dd>{_esc(record.expected)}</dd>
  <dt>difficulty rank</dt><dd>{_esc(rank)}</dd>
  <dt>leak</dt><dd>{leak}</dd>
  <dt>selections</dt><dd>{selections}</dd>
</dl>
"""


def _queries_html(artifact: Artifact) -> str:
    """Generate collapsible HTML detail elements for each query in the artifact."""
    if not artifact.queries:
        return '<p class="empty">No queries were recorded.</p>'
    entries = []
    for record in artifact.queries:
        rate = f"{record.hits}/{record.probes}" if record.probes else "unprobed"
        css = (
            "unprobed"
            if record.probes == 0
            else ("hit" if record.hits == record.probes else "miss" if record.hits else "error")
        )
        entries.append(
            f'<details class="query {css}">'
            "<summary>"
            f'<span class="query-id">{_esc(record.query_id)}</span>'
            f'<span class="expected">{_esc(record.expected)}</span>'
            f'<span class="rate">{_esc(rate)}</span>'
            "</summary>"
            f"{_query_detail_html(record, artifact.catalog_size)}"
            "</details>",
        )
    return f"""
<input
  type="text"
  class="filter"
  placeholder="Filter queries by ID, expected skill, or text..."
  oninput="reachFilterQueries(this, 'queries-container')"
  aria-controls="queries-container"
/>
<div id="queries-container">{"".join(entries)}</div>
"""


#: Inlined CSS styling for the self-contained HTML report.
_STYLE = """
body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 2rem; color: #1a1a1a; }
h1 { margin-bottom: 0.2rem; }
.subject, .digests { color: #555; font-size: 0.9rem; }
.digests { font-family: ui-monospace, Menlo, monospace; }
dl.figures { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0; }
dl.figures dt { font-size: 0.8rem; color: #666; }
dl.figures dd { margin: 0; font-size: 1.2rem; font-weight: 600; }
h2 { margin-top: 2.5rem; border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }
table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
th, td { border: 1px solid #ddd; padding: 0.3rem 0.6rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #f5f5f5; position: sticky; top: 0; }
table.confusion td.hit { background: #e3f6e3; font-weight: 600; }
table.confusion td.miss { background: #fbe3e3; }
table.collisions td { text-align: left; }
table.collisions td:nth-child(3) { text-align: right; }
.thought { font-style: italic; color: #666; font-size: 0.85rem; margin-top: 0.3rem; }
input.filter { width: 100%; max-width: 24rem; padding: 0.4rem 0.6rem;
               margin-top: 0.5rem; box-sizing: border-box; }
details.query { border: 1px solid #ddd; border-radius: 4px;
                margin-top: 0.4rem; padding: 0.3rem 0.6rem; }
details.query summary { cursor: pointer; display: flex; gap: 1rem; }
details.query summary .query-id { flex: 2; font-weight: 600; }
details.query summary .expected { flex: 2; color: #555; }
details.query summary .rate { flex: 1; text-align: right; }
details.query.hit { border-left: 4px solid #3a3; }
details.query.miss { border-left: 4px solid #c33; }
details.query.error { border-left: 4px solid #a00; }
details.query.unprobed { border-left: 4px solid #999; opacity: 0.7; }
details.query dl { display: grid; grid-template-columns: 8rem 1fr; gap: 0.2rem 1rem;
                    margin: 0.5rem 0 0; }
details.query dt { color: #666; }
details.query dd { margin: 0; }
p.empty { color: #666; font-style: italic; }
"""

#: Inlined client-side JavaScript for dynamic table row and query filtering.
_SCRIPT = """
function reachFilterRows(input, tableId) {
  var needle = input.value.trim().toLowerCase();
  var table = document.getElementById(tableId);
  var rows = table.querySelectorAll('tbody tr');
  for (var i = 0; i < rows.length; i++) {
    var text = rows[i].textContent.toLowerCase();
    rows[i].style.display = text.indexOf(needle) === -1 ? 'none' : '';
  }
}
function reachFilterQueries(input, containerId) {
  var needle = input.value.trim().toLowerCase();
  var container = document.getElementById(containerId);
  var items = container.querySelectorAll('details.query');
  for (var i = 0; i < items.length; i++) {
    var text = items[i].textContent.toLowerCase();
    items[i].style.display = text.indexOf(needle) === -1 ? 'none' : '';
  }
}
"""


def render_view_html(artifact: Artifact) -> str:
    """Render a complete Artifact model as a standalone HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(artifact.catalog_id)} — reach view</title>
<style>{_STYLE}</style>
</head>
<body>
{_header_html(artifact)}
<h2>Confusion matrix</h2>
{_confusion_html(artifact)}
{_collisions_html(artifact)}
<h2>Skills</h2>
{_skills_table_html(artifact.skills)}
<h2>Queries</h2>
{_queries_html(artifact)}
<script>{_SCRIPT}</script>
</body>
</html>
"""


#: Supported format handlers for artifact visualization.
VIEW_RENDERERS = {
    "html": render_view_html,
}


def render_view(artifact: Artifact, fmt: str) -> str:
    """Render an artifact into the requested visualization format."""
    return dispatch_render(VIEW_RENDERERS, fmt, artifact)
