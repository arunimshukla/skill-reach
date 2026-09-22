# `reach overlap`

Find which of your installed skills compete to answer the same user requests using asymmetric Lucene BM25 lexical ranking.

---

## Why Overlap is a Targeting Device

/// warning | High overlap does not prove collision
High description overlap does **not** prove that two skills collide. Two skills with nearly identical templates can route with 100% precision if their distinguishing keywords are clear to the model.
Conversely, zero lexical overlap does not guarantee safety: a single vague phrase can pull an LLM towards the wrong skill.

`reach overlap` is a **targeting device**: it ranks which pairs of skills are most likely to compete, telling you where to spend your evaluation probe budget.
///

---

## Synopsis

```bash
reach overlap [SKILLS] [OPTIONS]
```

---

## Key Scenarios

/// tab | Corpus-wide overlap ranking
Display the top competitive pairs across all installed skills:

```bash
reach overlap ./skills
```

Output:

```text
58 skills, ranked by how hard each is competed for

 skill                           overlap   nearest rival
 ────────────────────────────────────────────────────────────────────────
 google-agents-cli-deploy           0.69   google-agents-cli-scaffold
 google-agents-cli-scaffold         0.63   google-agents-cli-deploy
 google-agents-cli-adk-code         0.60   google-agents-cli-deploy
 google-agents-cli-observability    0.59   google-agents-cli-deploy
 google-agents-cli-eval             0.58   google-agents-cli-deploy
 google-agents-cli-workflow         0.51   google-agents-cli-deploy
 google-agents-cli-publish          0.49   google-agents-cli-deploy
```

///

/// tab | Analyze a specific skill's competitors
Identify the top rivals competing with a specific skill:

```bash
reach overlap ./skills --skill google-agents-cli-deploy
```

Output:

```text
google-agents-cli-deploy: outranked by 0 of 57 rivals for its own vocabulary

 rank   skill                              score
 ────────────────────────────────────────────────────────────
    1   google-agents-cli-deploy          68.168   this skill
    2   google-agents-cli-scaffold        46.923
    3   google-agents-cli-observability   45.944
    4   google-agents-cli-eval            45.201
    5   google-agents-cli-adk-code        43.225
    6   google-agents-cli-publish         39.664
    7   google-agents-cli-workflow        34.416
```

///

/// tab | Suggest wording rewrites
Analyze lexical overlap and inspect disclaimed rival terms:

```bash
reach overlap ./skills --skill google-agents-cli-deploy --suggest
```

Output:

```text
google-agents-cli-deploy and google-agents-cli-scaffold: no wording change is indicated

 already disclaimed   project, scaffold
 also within reach    3 more rivals score within a tenth of it
```

///

/// tab | Analyze Agent Registry skills
Rank lexical overlap directly across remote skills in Google Cloud Agent Registry:

```bash
reach overlap --project your-project-id --location global
```

///

/// tab | Dense semantic similarity matrix
Include dense embedding similarity alongside lexical BM25 scores:

```bash
reach overlap ./skills --semantic
```

///

/// tab | Diagnose query misrouting drivers
Explain which query terms biased routing toward a rival skill:

```bash
reach overlap explain "Set retention policy to archive old files" --skill gcs-lifecycle-rules
```

Output:

```text
Set retention policy to archive old files
target: gcs-lifecycle-rules (0.000)  rival: gcs-retention-policy (5.412)  net bias: +5.412

 token       target (gcs-lifecycle-rules)   rival (gcs-retention-policy)    delta   role
 retention                          0.000                          2.845   +2.845   target gap
 policy                             0.000                          2.567   +2.567   target gap
```

///

---

## Subcommands

### `reach overlap explain`

Itemize BM25 score contributions for a specific query to identify terms pulling routing toward a competitor.

```bash
reach overlap explain <QUERY> --skill <SKILL> [--rival <RIVAL>] [OPTIONS]
```

## Options

### General & Analysis Options

| Option                 | Type   | Default         | Description                                                                                                 |
| :--------------------- | :----- | :-------------- | :---------------------------------------------------------------------------------------------------------- |
| `[SKILLS]`, `--skills` | Path    | Auto-discovered | Path to the skill directory, `SKILL.md` file, or corpus to analyze (discovered from precedence if omitted). |
| `--skill`              | String  | -               | Analyze overlap specifically for this skill against all competitors (repeatable).                           |
| `--suggest`            | Flag    | `false`         | Generate suggested description rewrites to reduce lexical overlap (single-skill or corpus-wide).            |
| `--semantic`           | Flag    | `false`         | Include dense semantic similarity and dual-axis diagnostic quadrant matrix.                                 |
| `--top`                | Integer | -               | Show only the top N ranked skills.                                                                          |
| `--all`                | Flag    | `false`         | Show all skills without the default 30-row cap or actionable-only filter.                                   |
| `--quadrant`           | String  | -               | Filter by diagnostic quadrant: `near-duplicate`, `latent-collision`, `boilerplate`, `distinct`.             |
| `--no-truncate`        | Flag    | `false`         | Render full skill names without middle truncation.                                                          |
| `--agent`              | Choice  | -               | Agent runtime to query for installed skill locations (`claude-code`, `antigravity-cli`, etc.).              |
| `--global`, `-g`       | Flag    | `false`         | Discover and inspect skills from user global configuration (`~/`).                                          |
| `--format`             | Choice  | `text`          | Output format: `text`, `csv`, `json`, `jsonl`.                                                              |
| `--config`, `-c`       | Path    | -               | Path to reach.toml configuration file.                                                                      |

### Agent Registry Options

| Option            | Type   | Default    | Description                                                               |
| :---------------- | :----- | :--------- | :------------------------------------------------------------------------ |
| `--project`, `-p` | String | None       | Google Cloud project ID hosting the Agent Registry.                       |
| `--location`      | String | `"global"` | Agent Registry location endpoint (`global`, `us`, `eu`).                  |
| `--publisher`     | String | None       | Filter registry skills by publisher identifier (e.g. `cloud.google.com`). |
| `--registry`      | Flag   | `false`    | Target Google Cloud Agent Registry instead of local workspace.            |
| `--fresh`         | Flag   | `false`    | Bypass cached metadata and re-fetch latest skill definitions.             |
| `--no-cache`      | Flag   | `false`    | Run without reading or writing local disk cache.                          |
