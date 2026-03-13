# Usage

## Running conformance tests

The `check-model` subcommand runs tests using the native runner by
default — no genkit CLI dependency required.

### All plugins (default)

```bash
conform check-model
```

### Specific plugins

```bash
conform check-model anthropic deepseek google-genai
```

For the Python runtime, this imports the entry point **in-process**
(no subprocess, no HTTP).  For JS/Go runtimes, it starts a subprocess
and communicates via async HTTP with the reflection server.

### Specific runtime only

```bash
conform --runtime python check-model google-genai
```

### Multi-runtime matrix

```bash
conform --runtime python go check-model
```

### With increased concurrency

```bash
conform check-model -j 8
```

### With verbose output

Plain-text log lines (no live table), full output on failure:

```bash
conform check-model -v
```

### Fallback to genkit CLI

```bash
conform check-model --runner cli
```

This delegates to `genkit dev:test-model` via subprocess.

## Multi-runtime behavior

When `--runtime` is omitted, **all configured runtimes** are used.
Results from all runtimes are displayed in a **unified table** with a
Runtime column.

```bash
# Runs across python, js, and go runtimes (all configured)
conform check-model

# Restrict to a single runtime
conform --runtime python check-model

# Matrix: python + go only
conform --runtime python go check-model
```

## Checking plugin files

Verify every model plugin has the required conformance files
(`model-conformance.yaml` + `conformance_entry.py`):

```bash
conform check-plugin
```

This is also called automatically by `bin/check_consistency` during
`bin/lint`.

## Listing plugins

```bash
conform list
```

Shows a table with:

- **Runtimes** — which runtimes have entry points (green = available,
  dim = not available)
- **Green ●** — all environment variables set
- **Red ○** — some environment variables missing
- **Blue VAR** — variable is set
- **Red VAR** — variable is not set

## Using the wrapper script

The `py/bin/conform` wrapper avoids needing to be in the right directory:

```bash
# From anywhere
py/bin/conform check-model
py/bin/conform check-model google-genai
py/bin/conform list
```
