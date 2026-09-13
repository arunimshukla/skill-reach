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

### Clustering Options

| Option                 | Type    | Default | Description                                                                 |
| :--------------------- | :------ | :------ | :-------------------------------------------------------------------------- |
| `[SKILLS]`, `--skills` | Path    | `.`     | Path to the skill directory or corpus to partition (discovered if omitted). |
| `--resolution`         | Float   | `1.0`   | Resolution parameter: higher values yield smaller clusters.                 |
| `--target-size`        | Integer | -       | Target maximum skills per cluster.                                          |
| `--max-clusters`       | Integer | -       | Maximum number of clusters.                                                 |
| `--agent`              | Choice  | -       | Agent runtime to query for installed skill locations.                       |
| `--global`, `-g`       | Flag    | `false` | Discover and inspect skills from user global configuration (`~/`).          |
| `--format`             | Choice  | `text`  | Output format: `text`, `json`, `csv`.                                       |
| `--config`, `-c`       | Path    | -       | Path to reach.toml configuration file.                                      |

### Agent Registry Options

| Option            | Type   | Default    | Description                                                               |
| :---------------- | :----- | :--------- | :------------------------------------------------------------------------ |
| `--project`, `-p` | String | None       | Google Cloud project ID hosting the Agent Registry.                       |
| `--location`      | String | `"global"` | Agent Registry location endpoint (`global`, `us`, `eu`).                  |
| `--publisher`     | String | None       | Filter registry skills by publisher identifier (e.g. `cloud.google.com`). |
| `--registry`      | Flag   | `false`    | Target Google Cloud Agent Registry instead of local workspace.            |
| `--fresh`         | Flag   | `false`    | Bypass cached metadata and re-fetch latest skill definitions.             |
| `--no-cache`      | Flag   | `false`    | Run without reading or writing local disk cache.                          |
