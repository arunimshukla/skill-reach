# `reach completion`

Generate or install shell tab-completion scripts for `bash`, `zsh`, and `fish`.

---

## Synopsis

```bash
reach completion [SHELL] [OPTIONS]
```

---

## Key Scenarios

/// tab | Generate completion script
Print shell completion script to stdout:

```bash
reach completion zsh
reach completion bash
reach completion fish
```

///

/// tab | Install completion script automatically
Install completion script directly into your user shell configuration:

```bash
reach completion --install
reach completion zsh --install
```

///

/// tab | Load in current shell
Source the completion script directly into your active shell session:

```bash
# Zsh
source <(reach completion zsh)

# Bash
source <(reach completion bash)
```

///

---

## Options

| Option            | Type   | Default          | Description                                                             |
| :---------------- | :----- | :--------------- | :---------------------------------------------------------------------- |
| `SHELL`           | `TEXT` | Current `$SHELL` | Shell dialect: `bash`, `zsh`, or `fish`.                                |
| `--install`, `-i` | `flag` | `false`          | Install completion script directly to user shell startup configuration. |
