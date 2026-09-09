# `reach clean`

Clean cached Agent Registry payloads, sandboxes, and evaluation artifacts.

---

## Synopsis

```bash
reach clean [OPTIONS]
```

---

## Behavior

1. **Safe Default**: By default, `reach clean` safely purges the `.reach/cache/` directory containing downloaded Google Cloud Agent Registry skill bundles and metadata cache files without deleting user-created benchmark queries or evaluation records.
2. **Project Scoping**: Passing `--project <PROJECT_ID>` restricts the cache purge to bundles from that specific Google Cloud project.
3. **Dry-Run Preview**: Passing `--dry-run` (`-n`) calculates and reports the list of cached directories and total reclaimed space without modifying disk files.
4. **Full Artifact & Query Purge**: Passing `--all` removes local evaluation run files (`.reach/eval.json`, `.reach/report.html`) as well as local benchmark query files (`.reach/queries.*`).

---

## Key Scenarios

/// tab | Safe cache cleaning
Purge cached Agent Registry payloads and manifests:

```bash
reach clean
```

///

/// tab | Dry-run space calculation
Preview files and disk space that would be reclaimed:

```bash
reach clean --dry-run
reach clean -n
```

///

/// tab | Project-scoped cache purge
Purge cache for a specific Google Cloud project:

```bash
reach clean --project your-project-id
```

///

/// tab | Purge all caches, runs, and queries
Delete cache along with local `.reach/eval.json`, `.reach/report.html`, and benchmark query sets (`.reach/queries.*`):

```bash
reach clean --all
```

///

---

## Options

| Option            | Type   | Default | Description                                                                                                     |
| :---------------- | :----- | :------ | :-------------------------------------------------------------------------------------------------------------- |
| `--project`, `-p` | `TEXT` | None    | Purge cache only for the specified Google Cloud project ID.                                                     |
| `--dry-run`, `-n` | `flag` | `false` | Display paths and space that would be reclaimed without deleting files.                                         |
| `--all`           | `flag` | `false` | Purge all caches, temporary sandboxes, run artifacts (`.reach/eval.json`), and query sets (`.reach/queries.*`). |
| `--quiet`, `-q`   | `flag` | `false` | Suppress summary output.                                                                                        |
