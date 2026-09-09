# `reach cluster`

Partition skill catalogs into cohesive subagent scopes to prevent routing decay.

---

## Synopsis

```bash
reach cluster [SKILLS] [OPTIONS]
```

---

## Key Scenarios

/// tab | Partition skills into subagent candidate catalogs
Partition a skill corpus into high-modularity clusters:

```bash
reach cluster ./skills
```

///

/// tab | Target subagent catalog size
Specify soft upper bounds on cluster size to create modular subagent workspaces:

```bash
reach cluster ./skills --target-size 10
```

///

/// tab | Export clusters to JSON or CSV
Export cluster assignments for automated subagent scaffolding:

```bash
reach cluster ./skills --format json > clusters.json
```

///

---

## Options

| Option           | Type    | Default | Description                                                                 |
| :--------------- | :------ | :------ | :-------------------------------------------------------------------------- |
| `SKILLS`         | Path    | `.`     | Path to the skill directory or corpus to partition (discovered if omitted). |
| `--resolution`   | Float   | `1.0`   | Resolution parameter: higher values yield smaller clusters.                 |
| `--target-size`  | Integer | -       | Target maximum skills per cluster.                                          |
| `--max-clusters` | Integer | -       | Maximum number of clusters.                                                 |
| `--agent`        | Choice  | -       | Agent runtime to query for installed skill locations.                       |
| `--format`       | Choice  | `text`  | Output format: `text`, `json`, `csv`.                                       |
| `--config`, `-c` | Path    | -       | Path to reach.toml configuration file.                                      |
