---
name: Google Material
description: Google Material Design 3 system specification for Skill Reach HTML viewers.
colors:
  background: "#f8f9fa"
  error: "#b3261e"
  error-container: "#fce8e6"
  guardrail-accent: "#6750a4"
  on-error: "#ffffff"
  on-error-container: "#410e0b"
  on-primary: "#ffffff"
  on-primary-container: "#041e49"
  on-secondary-container: "#001d35"
  on-surface: "#1f1f1f"
  on-surface-variant: "#444746"
  outline: "#747775"
  outline-variant: "#e0e2ec"
  primary: "#0b57d0"
  primary-container: "#d3e3fd"
  secondary: "#00639b"
  secondary-container: "#c2e7ff"
  success: "#137333"
  success-container: "#e6f4ea"
  surface: "#ffffff"
  surface-container: "#f0f4f9"
  surface-container-alt: "#f4f5f8"
  surface-container-high: "#e9eef6"
  surface-container-highest: "#e1e3e1"
  surface-container-low: "#f8fafd"
components:
  badge:
    fontSize: 12px
    fontWeight: 500
    padding: "2px 8px"
    rounded: "{rounded.full}"
  button-outlined:
    backgroundColor: "{colors.surface}"
    border: "1px solid {colors.outline}"
    padding: "8px 20px"
    rounded: "{rounded.full}"
    textColor: "{colors.on-surface-variant}"
  button-primary:
    backgroundColor: "{colors.primary}"
    padding: "10px 24px"
    rounded: "{rounded.full}"
    textColor: "{colors.on-primary}"
  button-tonal:
    backgroundColor: "{colors.primary-container}"
    padding: "10px 20px"
    rounded: "{rounded.full}"
    textColor: "{colors.on-primary-container}"
  card:
    backgroundColor: "{colors.surface}"
    border: "1px solid {colors.outline-variant}"
    rounded: "{rounded.xl}"
  input-filter:
    backgroundColor: "{colors.surface}"
    border: "1px solid {colors.outline}"
    padding: "8px 12px"
    rounded: "{rounded.md}"
rounded:
  sm: 4px
  md: 8px
  lg: 12px
  xl: 16px
  full: 9999px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
typography:
  body:
    fontFamily: "Roboto, 'Google Sans Text', -apple-system, sans-serif"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
  code:
    fontFamily: "'Roboto Mono', 'Google Sans Mono', ui-monospace, Menlo, monospace"
    fontSize: 13px
  display:
    fontFamily: "Google Sans, Roboto, -apple-system, sans-serif"
    fontSize: 28px
    fontWeight: 600
  label:
    fontFamily: "Google Sans, Roboto, -apple-system, sans-serif"
    fontSize: 13px
    fontWeight: 500
  title:
    fontFamily: "Google Sans, Roboto, -apple-system, sans-serif"
    fontSize: 20px
    fontWeight: 500
---

## Overview

Skill Reach visual identity following Google Material Design 3 (M3).
The interface is clean, spacious, modern, and accessible, prioritizing content clarity, crisp information density, and distinctive Google design patterns.

## Color Palette

- **Background (`#f8f9fa`)**: Soft, clean light grey canvas providing subtle contrast for elevated cards.
- **Error & Error Container (`#b3261e` / `#fce8e6`)**: Google Red for deletion actions, misrouted queries, and confusion collisions.
- **Guardrail Accent (`#6750a4`)**: Material Purple accent for guardrail proportions in balance meters.
- **Outline & Outline Variant (`#747775` / `#e0e2ec`)**: Clean structural boundaries for cards, dividers, and text fields without harsh contrast.
- **Primary (`#0b57d0`)**: Signature Google Blue for primary actions, active card borders, and focus rings.
- **Primary Container (`#d3e3fd` / `#e8f0fe`)**: Light blue surface for tonal buttons, trigger column accents, and filter highlight banners.
- **Secondary & Secondary Container (`#00639b` / `#c2e7ff` / `#001d35`)**: Deep cyan and soft blue container (`#c2e7ff`) with dark contrast text (`#001d35`) for guardrail count badges.
- **Success & Success Container (`#137333` / `#e6f4ea`)**: Google Green for passed probes, target hits, and in-scope confirmations.
- **Surface & Containers (`#ffffff` / `#f0f4f9` / `#f4f5f8` / `#f8fafd`)**: Pure white card containers and tinted surface layers for clear visual hierarchy.

## Typography

- **Body & Inputs**: `Roboto, 'Google Sans Text', -apple-system, BlinkMacSystemFont, sans-serif` (400 weight, 14px, 1.5 line height).
- **Code & Digests**: `'Roboto Mono', 'Google Sans Mono', ui-monospace, Menlo, monospace` (13px, monospace clarity for IDs and digests).
- **Headings & Titles**: `Google Sans, Roboto, -apple-system, BlinkMacSystemFont, sans-serif` (500/600 weight, 20px–28px).
- **Labels & Badges**: `Google Sans, Roboto, -apple-system, BlinkMacSystemFont, sans-serif` (500 weight, 12px–13px, uppercase tracking).

## Component Guidelines

- **Badges & Chips**: Compact pill and rounded badges (`.badge-kind`, `.badge-rival`, `.badge-target`, `.header-badge`, `.rival-chip`, `.skill-tag`) indicating routing status and origin.
- **Balance Meter & Ribbon**: Interactive header ribbon (`.control-ribbon`, `.balance-bar`) displaying real-time trigger vs. guardrail proportions with smooth width transitions.
- **Buttons**: Pill-shaped Material 3 buttons (`rounded: 9999px`) with filled (`.btn-primary`), tonal (`.btn-tonal`), outlined (`.btn-outlined`), and compact (`.btn-sm`) variants, featuring inline keyboard badges (`.btn-kbd`).
- **Cards & Surfaces**: Rounded 12px and 16px containers (`.figure-cell`, `.query-card`, `.skill-context-card`, `.system-map-card`, `.workbench-header`) with subtle elevation shadows.
- **Filter & Search Controls**: Debounced real-time text filters (`.filter`) paired with an active filter banner (`.active-filter-banner`) for drill-down analysis.
- **Hotkey Bar**: Compact keyboard shortcut ribbon (`.hotkey-bar`) documenting hotkeys (<kbd>J</kbd>/<kbd>K</kbd>, <kbd>Space</kbd>/<kbd>T</kbd>, <kbd>Enter</kbd>, <kbd>⌫</kbd>, <kbd>Ctrl/⌘↵</kbd>).
- **Tables & Heatmaps**: Responsive tabular layouts (`table.confusion`, `table.collisions`, `table.skills`) with interactive cross-filtering hover states.
- **Territory Board**: Two-column split board (`.territory-board`, `.territory-column`, `.card-list`) cleanly partitioning in-scope triggers from neighbor guardrails.

## CSS Class Reference

| Class                      | Stylesheet                 | Purpose                                                                |
| :------------------------- | :------------------------- | :--------------------------------------------------------------------- |
| `.action-buttons`          | `review.css`               | Flex container for export and approval action buttons.                 |
| `.active-card`             | `review.css`               | Currently selected card in keyboard navigation with focus outline.     |
| `.active-filter-banner`    | `view.css`                 | Banner displaying active confusion matrix or collision query filter.   |
| `.approve-btn`             | `review.html`, `review.js` | Action button triggering review approval.                              |
| `.badge-distractor`        | `review.css`               | Neutral badge for distractor queries.                                  |
| `.badge-kind`              | `review.css`               | Metadata badge showing query origin kind (`implicit`, etc.).           |
| `.badge-rival`             | `base.css`, `review.css`   | Badge showing targeted rival skill on guardrail queries.               |
| `.badge-target`            | `base.css`, `review.css`   | Badge indicating target skill routing expectation.                     |
| `.badge-target-wrapper`    | `review.css`               | Flex wrapper for target routing status badge.                          |
| `.balance-bar`             | `review.css`               | Visual meter showing percentage split between triggers and guardrails. |
| `.balance-fill-guardrails` | `review.css`               | Purple fill segment in balance bar representing guardrails.            |
| `.balance-fill-triggers`   | `review.css`               | Blue fill segment in balance bar representing triggers.                |
| `.balance-labels`          | `review.css`               | Text row above balance bar with title and summary count.               |
| `.balance-section`         | `review.css`               | Wrapper for balance labels and balance bar.                            |
| `.btn`                     | `review.css`               | Base Material 3 pill button with transition effects.                   |
| `.btn-kbd`                 | `review.css`               | Translucent badge within buttons indicating shortcut keys.             |
| `.btn-link`                | `view.css`                 | Text button styled as a link for expand/collapse actions.              |
| `.btn-outlined`            | `review.css`               | Outlined secondary button style with hover fill.                       |
| `.btn-primary`             | `review.css`               | Filled primary action button (Google Blue).                            |
| `.btn-sm`                  | `review.css`               | Compact button modifier for card actions and column headers.           |
| `.btn-tonal`               | `review.css`               | Tonal button with container background for secondary actions.          |
| `.card-actions`            | `review.css`               | Footer row in query card containing swap and edit buttons.             |
| `.card-badges`             | `review.css`               | Header row inside query card displaying status badges.                 |
| `.card-btn`                | `review.css`               | Base style for query card action buttons.                              |
| `.card-btn-delete`         | `review.css`               | Top-right query card button for card deletion.                         |
| `.card-btn-done`           | `review.css`               | Confirmation button to save inline query edits.                        |
| `.card-btn-edit`           | `review.css`               | Action button triggering inline edit mode on a card.                   |
| `.card-btn-swap`           | `review.css`               | Action button moving query between triggers and guardrails.            |
| `.card-left-actions`       | `review.css`               | Left action group within query card footer.                            |
| `.card-list`               | `review.css`               | Vertical container holding query cards in a column.                    |
| `.card-list.empty`         | `review.css`               | Contextual placeholder displayed when a column has zero queries.       |
| `.card-right-actions`      | `review.css`               | Right action group within query card footer.                           |
| `.card-top`                | `review.css`               | Header row inside query card with badges and delete control.           |
| `.collisions`              | `view.css`                 | Collisions table detailing misrouting pairs and query traces.          |
| `.column-header`           | `review.css`               | Header row for territory board columns.                                |
| `.column-subtitle`         | `review.css`               | Explanatory routing rule subtitle for territory columns.               |
| `.column-title`            | `review.css`               | Column title text in territory board.                                  |
| `.column-title-group`      | `review.css`               | Flex container for column title and count badge.                       |
| `.confusion`               | `base.css`, `view.css`     | Cross-tabulation confusion matrix table.                               |
| `.control-ribbon`          | `base.css`, `review.css`   | Surface card containing balance meter and primary actions.             |
| `.count-badge`             | `review.css`               | Pill badge displaying query counts in column headers.                  |
| `.description`             | `review.css`               | Target skill description text within context card.                     |
| `.digests`                 | `view.css`                 | Monospace footer section for commit and artifact hashes.               |
| `.dim`                     | `base.css`                 | Dimmed text modifier for secondary explanatory labels.                 |
| `.editing`                 | `review.css`               | State modifier on query cards when textarea editor is open.            |
| `.empty`                   | `view.css`, `review.css`   | Italicized muted text for empty lists.                                 |
| `.empty-state-card`        | `review.css`               | Dashed empty state card shown when zero queries remain.                |
| `.error`                   | `base.css`, `view.css`     | Error styling for failed or errored query executions.                  |
| `.expand-controls`         | `view.css`                 | Controls container for expanding or collapsing all queries.            |
| `.expected`                | `view.css`                 | Column or label for expected skill in query lists.                     |
| `.figure-cell`             | `view.css`                 | Summary metric tile card with title and figure.                        |
| `.figures`                 | `view.css`                 | Grid container for summary figure cells.                               |
| `.filter`                  | `base.css`, `view.css`     | Text input for live filtering tables and query lists.                  |
| `.filter-row`              | `view.css`                 | Flex row containing filter input and helper controls.                  |
| `.guardrails-count`        | `review.css`               | Counter badge for neighbor guardrails column.                          |
| `.guardrails-header`       | `review.css`               | Column header styling for neighbor guardrails.                         |
| `.header-badge`            | `base.css`                 | Uppercase tag in report headers identifying the view.                  |
| `.header-main`             | `view.css`                 | Container for catalog header title and subject metadata.               |
| `.hit`                     | `base.css`, `view.css`     | Green success styling for correct routes and passed probes.            |
| `.hotkey-bar`              | `review.css`               | Container displaying keyboard navigation shortcut hints.               |
| `.matrix-container`        | `view.css`                 | Horizontally scrollable container for wide confusion matrices.         |
| `.miss`                    | `base.css`, `view.css`     | Red error styling for misrouted queries and collision cells.           |
| `.query`                   | `base.css`, `view.css`     | Collapsible accordion item for individual query records.               |
| `.query-card`              | `review.css`               | Interactive prompt query card in boundary curation.                    |
| `.query-card-editor`       | `review.css`               | Inline editing container with textarea and save controls.              |
| `.query-id`                | `view.css`                 | Monospace query ID badge in diagnostic summary rows.                   |
| `.query-input`             | `base.css`, `review.css`   | Textarea input field for query editing.                                |
| `.query-preview`           | `review.css`               | Text preview displayed on query cards.                                 |
| `.query-text`              | `view.css`                 | Query prompt string in diagnostic summary row.                         |
| `.query-text-preview`      | `review.css`               | Text preview displayed on query cards.                                 |
| `.rate`                    | `base.css`, `view.css`     | Pill badge displaying pass rate ratio (`3/3`, `1/3`).                  |
| `.rival-chip`              | `review.css`               | Monospace chip displaying competing rival skill names.                 |
| `.rival-select`            | `review.css`               | Dropdown selector to bind negative queries to a rival skill.           |
| `.rivals-bar`              | `review.css`               | Container row displaying rival skill chips.                            |
| `.skill-context-card`      | `base.css`, `review.css`   | Hero context card with target skill details and rivals.                |
| `.skill-tag`               | `base.css`                 | Uppercase pill badge identifying boundary curation mode.               |
| `.skills`                  | `view.css`                 | Skills evaluation breakdown table.                                     |
| `.status-bar`              | `review.css`               | Notification bar displaying feedback and auto-dismiss alerts.          |
| `.subject`                 | `view.css`                 | Run description and model configuration string.                        |
| `.summary`                 | `review.css`               | Text summary of query counts and trigger balance.                      |
| `.system-map-card`         | `base.css`, `view.css`     | Card container housing matrix, skills, or queries.                     |
| `.territory-board`         | `review.css`               | Split-column grid layout for boundary territories.                     |
| `.territory-column`        | `review.css`               | Territory column container (in-scope or guardrail).                    |
| `.thought`                 | `view.css`                 | Italicized callout displaying agent reasoning traces.                  |
| `.title-row`               | `base.css`                 | Top row layout for card titles and badge tags.                         |
| `.triggers-count`          | `review.css`               | Counter badge for in-scope trigger queries.                            |
| `.triggers-header`         | `review.css`               | Column header styling for in-scope territory.                          |
| `.unprobed`                | `view.css`                 | Gray neutral styling for queries without probe results.                |
| `.workbench-header`        | `base.css`, `view.css`     | Header card displaying evaluation metadata and metric tiles.           |
