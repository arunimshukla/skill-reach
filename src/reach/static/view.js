/* Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

function filterElements(input, containerId, itemSelector) {
  const needle = input.value.trim().toLowerCase();
  const container = document.getElementById(containerId);
  if (!container) return;
  const items = container.querySelectorAll(itemSelector);
  for (let i = 0; i < items.length; i++) {
    const text = items[i].textContent.toLowerCase();
    items[i].style.display = text.includes(needle) ? '' : 'none';
  }
}

function reachFilterRows(input, tableId) {
  filterElements(input, tableId, 'tbody tr');
}

function reachFilterQueries(input, containerId) {
  filterElements(input, containerId, 'details.query');
}

function forEachQuery(fn) {
  const container = document.getElementById('queries-container');
  if (!container) return;
  const items = container.querySelectorAll('details.query');
  for (let i = 0; i < items.length; i++) {
    fn(items[i]);
  }
}

function setFilterBanner(label) {
  const banner = document.getElementById('active-filter-banner');
  const labelEl = document.getElementById('active-filter-label');
  if (!banner) return;
  if (label) {
    if (labelEl) labelEl.textContent = label;
    banner.style.display = 'flex';
    banner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } else {
    banner.style.display = 'none';
  }
}

function reachToggleAllQueries(open) {
  forEachQuery((item) => {
    item.open = open;
  });
}

function reachApplyFilter(label, filterFn) {
  setFilterBanner(label);
  forEachQuery((item) => {
    item.style.display = filterFn(item) ? '' : 'none';
  });
}

function reachClearActiveFilter() {
  setFilterBanner(null);
  forEachQuery((item) => {
    item.style.display = '';
  });
}

function getExpectedSkill(q) {
  return q.getAttribute('data-expected') || q.querySelector('.expected')?.textContent.trim() || '';
}

function getSelectedSkills(q) {
  const raw = q.getAttribute('data-selections');
  if (raw === null) return [];
  return raw ? raw.split(',').map((s) => s.trim()) : [];
}

// Cross-filter bindings via event delegation
document.addEventListener('DOMContentLoaded', () => {
  const matrixTable = document.getElementById('confusion-table');
  if (matrixTable) {
    matrixTable.addEventListener('click', (e) => {
      const td = e.target.closest('td');
      if (!td || td.textContent.trim() === '') return;
      const cellIndex = td.cellIndex;
      const tr = td.closest('tr');
      const expected = tr.querySelector('th') ? tr.querySelector('th').textContent.trim() : '';
      const headerRow = matrixTable.querySelector('thead tr');
      const invoked = headerRow ? headerRow.cells[cellIndex].textContent.trim() : '';
      if (expected && invoked) {
        const filterMsg = `Confusion cell: Expected [${expected}] → Invoked [${invoked}]`;
        reachApplyFilter(filterMsg, (q) => {
          const qExpected = getExpectedSkill(q);
          const selections = getSelectedSkills(q);
          const matched =
            invoked === '(no skill)'
              ? selections.length === 0 || selections.includes('(no skill)')
              : selections.includes(invoked);
          return qExpected === expected && matched;
        });
      }
    });
  }

  const collisionsTable = document.getElementById('collisions-table');
  if (collisionsTable) {
    collisionsTable.addEventListener('click', (e) => {
      const tr = e.target.closest('tbody tr');
      if (!tr) return;
      const cells = tr.querySelectorAll('td');
      if (cells.length >= 2) {
        const exp = cells[0].textContent.trim();
        const inv = cells[1].textContent.trim();
        const colMsg = `Collision: Expected [${exp}] → Invoked [${inv}]`;
        reachApplyFilter(colMsg, (q) => {
          return getExpectedSkill(q) === exp && getSelectedSkills(q).includes(inv);
        });
      }
    });
  }

  const skillsTable = document.getElementById('skills-table');
  if (skillsTable) {
    skillsTable.addEventListener('click', (e) => {
      const tr = e.target.closest('tbody tr');
      if (!tr) return;
      const skillCell = tr.querySelector('td:first-child');
      if (skillCell) {
        const skillName = skillCell.textContent.trim();
        reachApplyFilter(`Skill focus: [${skillName}]`, (q) => {
          return getExpectedSkill(q) === skillName || getSelectedSkills(q).includes(skillName);
        });
      }
    });
  }
});

// Explicitly bind functions invoked via HTML event attributes to window
window.reachFilterRows = reachFilterRows;
window.reachFilterQueries = reachFilterQueries;
window.reachToggleAllQueries = reachToggleAllQueries;
window.reachClearActiveFilter = reachClearActiveFilter;
