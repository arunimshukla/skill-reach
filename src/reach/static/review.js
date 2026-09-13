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

let activeCardIndex = -1;

function getTargetSkill() {
  return document.body.dataset.targetSkill || '';
}

function getOutOfScope() {
  return document.body.dataset.outOfScope || '__OUT_OF_SCOPE__';
}

function getAllCards() {
  return Array.from(document.querySelectorAll('.query-card'));
}

function setElementText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setElementDisplay(id, display) {
  const el = document.getElementById(id);
  if (el) el.style.display = display;
}

function escapeHtml(str) {
  if (typeof str !== 'string') return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderBadge(kind, text) {
  return `<span class="badge-${escapeHtml(kind)}">→ ${escapeHtml(text)}</span>`;
}

function setStatus(text, isError = false) {
  const statusBar = document.getElementById('status-bar');
  if (!statusBar) return;
  statusBar.style.color = isError ? 'var(--reach-error-text)' : 'var(--reach-success-text)';
  statusBar.textContent = text;
}

function updateSummary() {
  const triggers = document.querySelectorAll('#triggers-list .query-card').length;
  const guardrails = document.querySelectorAll('#guardrails-list .query-card').length;
  const total = triggers + guardrails;

  // Header badges
  setElementText('triggers-count', `${triggers} Triggers`);
  setElementText('guardrails-count', `${guardrails} Guardrails`);

  // Balance bar fill
  const totalSafe = total || 1;
  const triggerPct = (triggers / totalSafe) * 100;
  const guardrailPct = (guardrails / totalSafe) * 100;

  const barTriggers = document.getElementById('balance-triggers');
  if (barTriggers) barTriggers.style.width = `${triggerPct}%`;
  const barGuardrails = document.getElementById('balance-guardrails');
  if (barGuardrails) barGuardrails.style.width = `${guardrailPct}%`;

  // Summary text matching standard contract
  setElementText(
    'summary',
    `${total} queries total: ${triggers} should trigger, ${guardrails} should not trigger`,
  );

  // Column empty states
  setElementDisplay('triggers-empty', triggers === 0 ? 'flex' : 'none');
  setElementDisplay('guardrails-empty', guardrails === 0 ? 'flex' : 'none');

  // Overall empty state
  setElementDisplay('empty-state', total === 0 ? '' : 'none');
}

function getRows() {
  const cards = getAllCards();
  const targetSkill = getTargetSkill();
  const queries = [];

  cards.forEach((card) => {
    const input = card.querySelector('.query-input');
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;

    const isTrigger = card.closest('#triggers-list') !== null;
    let expected = null;
    if (isTrigger) {
      expected = targetSkill;
    } else {
      const select = card.querySelector('.rival-select');
      if (select?.value && select.value !== getOutOfScope()) {
        expected = select.value;
      } else {
        expected = null;
      }
    }

    queries.push({
      text: text,
      expected_skill: expected,
    });
  });

  return queries;
}

function swapCard(btn) {
  const card = btn.closest('.query-card');
  if (!card) return;

  const isTrigger = card.closest('#triggers-list') !== null;
  const targetList = document.getElementById(isTrigger ? 'guardrails-list' : 'triggers-list');

  // Update card UI for new column
  const badgeTarget = card.querySelector('.badge-target-wrapper');
  const rivalSelect = card.querySelector('.rival-select');
  const swapBtn = card.querySelector('.card-btn-swap');

  if (isTrigger) {
    // Moving to guardrails
    if (badgeTarget) badgeTarget.innerHTML = renderBadge('rival', 'Rival / Out-of-Scope');
    if (rivalSelect) rivalSelect.style.display = 'block';
    if (swapBtn) swapBtn.textContent = '⇄ Move to Triggers';
  } else {
    // Moving to triggers
    if (badgeTarget) badgeTarget.innerHTML = renderBadge('target', getTargetSkill());
    if (rivalSelect) rivalSelect.style.display = 'none';
    if (swapBtn) swapBtn.textContent = '⇄ Move to Guardrails';
  }

  if (targetList) targetList.appendChild(card);
  updateSummary();
}

function deleteCard(btn) {
  const card = btn.closest('.query-card');
  if (card) {
    card.remove();
    updateSummary();
  }
}

function editCard(el) {
  const card = el.closest('.query-card');
  if (!card) return;
  card.classList.add('editing');
  const input = card.querySelector('.query-input');
  if (input) {
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }
}

function closeCard(btn) {
  const card = btn.closest('.query-card');
  if (!card) return;
  const input = card.querySelector('.query-input');
  const preview = card.querySelector('.query-text-preview');
  if (input && preview) {
    preview.textContent = input.value.trim() || '(Empty query)';
  }
  card.classList.remove('editing');
  updateSummary();
}

function addRow(text = '', isTrigger = true, expected = '') {
  const targetSkill = getTargetSkill();
  const rivals = JSON.parse(document.body.dataset.rivals || '[]');

  const outOfScope = getOutOfScope();
  let rivalOptions = `<option value="${escapeHtml(outOfScope)}">Out of Scope (General Distractor)</option>`;
  for (const r of rivals) {
    const safeR = escapeHtml(r);
    const sel = expected === r ? 'selected' : '';
    rivalOptions += `<option value="${safeR}" ${sel}>Rival: ${safeR}</option>`;
  }

  const targetList = document.getElementById(isTrigger ? 'triggers-list' : 'guardrails-list');
  const card = document.createElement('div');
  card.className = 'query-card editing';

  const badgeHtml = isTrigger
    ? renderBadge('target', targetSkill)
    : renderBadge('rival', expected || 'Rival / Out-of-Scope');

  const rivalSelectDisp = isTrigger ? 'none' : 'block';
  const swapBtnText = isTrigger ? '⇄ Move to Guardrails' : '⇄ Move to Triggers';
  const safeText = escapeHtml(text);

  card.innerHTML = `
    <div class="card-top">
      <div class="card-badges">
        <span class="badge-kind">authored</span>
        <span class="badge-target-wrapper">${badgeHtml}</span>
      </div>
      <button type="button" class="card-btn card-btn-delete"
        onclick="deleteCard(this)" title="Delete query">✕</button>
    </div>
    <div class="query-text-preview" onclick="editCard(this)">${safeText ? safeText : '(Empty query)'}</div>
    <div class="query-card-editor">
      <textarea class="query-input" placeholder="Enter query prompt...">${safeText}</textarea>
      <select class="rival-select" style="display:${rivalSelectDisp};"
        onchange="updateCardRival(this)">
        ${rivalOptions}
      </select>
    </div>
    <div class="card-actions">
      <div class="card-left-actions">
        <button type="button" class="card-btn card-btn-swap"
          onclick="swapCard(this)">${swapBtnText}</button>
      </div>
      <div class="card-right-actions">
        <button type="button" class="card-btn card-btn-edit"
          onclick="editCard(this)">✎ Edit</button>
        <button type="button" class="card-btn card-btn-done"
          onclick="closeCard(this)">✓ Done</button>
      </div>
    </div>
  `;

  if (targetList) targetList.appendChild(card);
  updateSummary();
  const input = card.querySelector('.query-input');
  if (input) input.focus();
}

function updateCardRival(select) {
  const card = select.closest('.query-card');
  if (!card) return;
  const badgeTarget = card.querySelector('.badge-target-wrapper');
  if (!badgeTarget) return;

  const isOutOfScope = select.value === getOutOfScope();
  badgeTarget.innerHTML = isOutOfScope
    ? renderBadge('distractor', 'Out of Scope')
    : renderBadge('rival', select.value);
}

function exportEvalSetJson() {
  const queries = getRows();
  const targetSkill = getTargetSkill();
  const data = queries.map((q) => ({
    query: q.text,
    should_trigger: q.expected_skill === targetSkill,
    expected_skill: q.expected_skill,
  }));
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `eval_set_${targetSkill}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function approveQueries() {
  const queries = getRows();
  if (queries.length === 0) {
    alert('Please keep at least one query in the eval set.');
    return;
  }
  setStatus('Saving queries and resuming optimizer in terminal...');

  const token =
    document.querySelector('meta[name="reach-token"]')?.content ||
    new URLSearchParams(window.location.search).get('token') ||
    '';
  const headers = { 'Content-Type': 'application/json' };
  if (token) {
    headers['X-Reach-Token'] = token;
  }

  try {
    const resp = await fetch('/api/save', {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({ queries: queries }),
    });
    if (resp.ok) {
      setStatus('✓ Saved! You can close this tab and return to the terminal.');
      document.querySelectorAll('.approve-btn').forEach((btn) => {
        btn.disabled = true;
      });
    } else {
      setStatus(`Error saving queries: ${resp.statusText}`, true);
    }
  } catch (err) {
    setStatus(`Network error saving queries: ${err}`, true);
  }
}

// Keyboard Navigation & Shortcuts
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    approveQueries();
    return;
  }

  // If typing in an active input or textarea, don't trigger hotkeys
  if (['TEXTAREA', 'INPUT', 'SELECT'].includes(document.activeElement.tagName)) {
    if (e.key === 'Escape') {
      const activeCard = document.activeElement.closest('.query-card');
      if (activeCard) closeCard(activeCard);
    }
    return;
  }

  const cards = getAllCards();
  if (cards.length === 0) return;

  if (e.key === 'j' || e.key === 'ArrowDown') {
    e.preventDefault();
    activeCardIndex = (activeCardIndex + 1) % cards.length;
    highlightActiveCard(cards);
    return;
  }
  if (e.key === 'k' || e.key === 'ArrowUp') {
    e.preventDefault();
    activeCardIndex = (activeCardIndex - 1 + cards.length) % cards.length;
    highlightActiveCard(cards);
    return;
  }

  const activeCard = cards[activeCardIndex];
  if (!activeCard) return;

  if (e.key === ' ' || e.key === 't') {
    e.preventDefault();
    swapCard(activeCard.querySelector('.card-btn-swap'));
  } else if (e.key === 'Enter') {
    e.preventDefault();
    editCard(activeCard);
  } else if (e.key === 'Backspace' || e.key === 'Delete') {
    e.preventDefault();
    deleteCard(activeCard.querySelector('.card-btn-delete'));
    activeCardIndex = Math.min(activeCardIndex, getAllCards().length - 1);
  }
});

function highlightActiveCard(cards) {
  cards.forEach((c, idx) => {
    if (idx === activeCardIndex) {
      c.classList.add('active-card');
      c.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else {
      c.classList.remove('active-card');
    }
  });
}

// Click to focus active card
document.addEventListener('click', (e) => {
  const card = e.target.closest('.query-card');
  if (card) {
    const cards = getAllCards();
    activeCardIndex = cards.indexOf(card);
    highlightActiveCard(cards);
  }
});

// Initial calculation and card highlight
window.addEventListener('DOMContentLoaded', () => {
  updateSummary();
  const cards = getAllCards();
  if (cards.length > 0) highlightActiveCard(cards);
});

// Explicitly bind functions invoked via HTML event attributes to window
window.addRow = addRow;
window.updateCardRival = updateCardRival;
window.exportEvalSetJson = exportEvalSetJson;
window.approveQueries = approveQueries;
window.deleteCard = deleteCard;
window.editCard = editCard;
window.closeCard = closeCard;
window.swapCard = swapCard;
