# Genkit Conformance Runner

A parallel model conformance test runner for Genkit plugins across
multiple runtimes (Python, JS, Go).

---

## What it does

The `conform` tool runs model conformance tests against Genkit plugins.
It uses a **native runner** by default:

- **Python runtime** — imports the entry point in-process (no subprocess,
  no HTTP).  **No genkit CLI dependency.**
- **JS/Go runtimes** — communicates with reflection servers via async HTTP.
- **CLI fallback** — `--runner cli` delegates to `genkit dev:test-model`.

## Quick start

```bash
# From anywhere in the repo
py/bin/conform check-model                 # Run all plugins (all runtimes)
py/bin/conform check-model anthropic xai   # Run specific plugins
py/bin/conform check-model --runner cli     # Genkit CLI runner
py/bin/conform --runtime python check-model  # Single runtime only
py/bin/conform check-plugin                # Verify plugin files exist
py/bin/conform list                        # Show runtimes and env-var readiness
py/bin/conform                             # Help + plugin table
```

## Subcommands

| Command | Purpose |
|---------|---------|
| `check-model [PLUGIN...]` | Run model conformance tests (native runner by default) |
| `check-plugin` | Lint-time check that plugins have conformance files |
| `list` | Show plugins, runtimes, and environment variable readiness |

## Key features

- **Native test runner** — in-process for Python, async HTTP for JS/Go (no genkit CLI needed)
- **Parallel execution** — `asyncio.Semaphore` bounds concurrency (default: 8), configurable via `-j N`
- **Live progress** — Rich table pinned at bottom, log lines scroll above
- **Inline progress bars** — per-row colored bars (green/red/dim) with pre-calculated totals
- **Log redaction** — data URIs auto-truncated in debug logs for readability
- **Error log** — last 15 lines per failure, full output with `-v`
- **10 validators** — ported 1:1 from the canonical JS implementation
- **Rust-style messages** — color-coded errors and warnings
- **Multi-runtime** — runs across Python, JS, Go runtimes by default
- **TOML-configured** — all settings in `pyproject.toml`, CLI overrides

!!! info "Installation"

    The tool is installed automatically as part of the workspace via
    `uv sync`. No separate installation needed.
