# `reach diff`

Change one factor, hold the evaluation queries fixed, and report the empirical performance delta between two experimental arms.

---

## Controlled Single-Factor A/B Testing

In prompt and skill engineering, changing multiple variables at once masks the root cause of regressions. `reach diff` compares two recorded runs (`CONTROL` vs `TREATMENT`), holding the evaluation queries constant while varying exactly one factor:

- **`description`**: Same rivals and query set, but one or more skill descriptions were altered.
- **`rival`**: Same target skill and queries, but new competing skills were installed or removed from the catalog.
- **`scope`**: Same skills, but catalog size or listing budget parameters were adjusted.

---

## Synopsis

```bash
reach diff --vary {description,rival,scope} CONTROL TREATMENT [OPTIONS]
```

---

## Key Scenarios

/// tab | Compare description changes
Measure the impact of rewording descriptions between baseline and treatment runs:

```bash
reach diff --vary description runs/v1.jsonl runs/v2.jsonl
```

///

/// tab | Compare evaluation artifacts directly
`reach diff` accepts both raw probe `.jsonl` files and `.artifact.json` evaluation files directly:

```bash
reach diff --vary description runs/v1.artifact.json runs/v2.artifact.json
```

///

/// tab | Statistical noise estimation
`reach diff` computes an estimated noise floor with inflation to guard against stochastic LLM variance:

```bash
reach diff --vary description runs/v1.jsonl runs/v2.jsonl --confidence 0.95 --noise-inflation 1.265
```

///

---

## Options

| Option               | Type   | Default                   | Description                                                                          |
| :------------------- | :----- | :------------------------ | :----------------------------------------------------------------------------------- |
| `CONTROL`            | Path   | -                         | Recorded results for the baseline/control arm (JSONL or `.artifact.json`, required). |
| `TREATMENT`          | Path   | -                         | Recorded results for the treatment arm (JSONL or `.artifact.json`, required).        |
| `--vary`             | Choice | -                         | The single factor varied: `description`, `rival`, or `scope` (required).             |
| `--queries-root`     | Path   | -                         | Root directory containing query sets if their paths moved since evaluation.          |
| `--control-corpus`   | Path   | -                         | Corpus the control arm was probed against, if it moved.                              |
| `--treatment-corpus` | Path   | -                         | Corpus the treatment arm was probed against, if it moved.                            |
| `--control-label`    | String | Filename                  | Custom display label for the control arm.                                            |
| `--treatment-label`  | String | Filename                  | Custom display label for the treatment arm.                                          |
| `--queries`, `-q`    | Path   | -                         | Subset query set file used to slice both arms before comparing.                      |
| `--filter-skill`     | String | `()`                      | Glob pattern(s) matching target skill names to slice both arms before comparing.     |
| `--filter-id`        | String | `()`                      | Glob pattern(s) matching query IDs to slice both arms before comparing.              |
| `--confidence`       | Float  | `from reach.toml` (0.95)  | Confidence level used to estimate the noise floor.                                   |
| `--noise-inflation`  | Float  | `from reach.toml` (1.265) | Multiplier to inflate estimated noise floor for over-dispersion.                     |
| `--format`           | Choice | `text`                    | How to render comparison: `text`, `csv`, `json`, `jsonl`.                            |
| `--config`, `-c`     | Path   | -                         | Path to custom `reach.toml` configuration file.                                      |
