# Architecture

## Module structure

The tool follows a clean dependency graph with no circular imports:

```
types.py          ← Shared types (Status, PluginResult, Runtime Protocol)
   ↑
config.py         ← TOML config loader, RuntimeConfig (satisfies Runtime)
   ↑
paths.py          ← Path constants (PACKAGE_DIR, TOOL_DIR)
   ↑
plugins.py        ← Plugin discovery, env-var checking
   ↑
display.py        ← Rich tables, inline progress bars, Rust-style messages
   ↑
log_redact.py     ← Structlog processor to truncate data URIs in logs
   ↑
┌──────────────┬──────────────────┬──────────────────────┐
│ checker.py   │ runner.py        │ util_test_model.py   │
│ (check-plugin)│ (--runner cli)  │ (check-model default)│
│              │ genkit CLI runner│ ActionRunner Protocol │
└──────────────┴──────────────────┴──────────────────────┘
   ↑                                      ↑
   │              ┌────────────┬──────────┘
   │              │            │
   │     reflection.py    test_cases.py
   │     (httpx client)   (12 built-in)
   │              │
   │     validators/
   │     ├── __init__.py   ← Protocol + @register decorator
   │     ├── helpers.py    ← Response parsing utilities
   │     ├── json.py       ← valid-json
   │     ├── media.py      ← valid-media
   │     ├── reasoning.py  ← reasoning
   │     ├── streaming.py  ← stream-text-includes, stream-has-tool-request, stream-valid-json
   │     ├── text.py       ← text-includes, text-starts-with, text-not-empty
   │     └── tool.py       ← has-tool-request
   ↑
cli.py            ← CLI entry point, argparse, dispatch
   ↑
__main__.py       ← python -m conform support
```

## Key design decisions

### Types in a separate module

`Status`, `PluginResult`, and the `Runtime` Protocol live in `types.py`
which has **zero internal dependencies**. This breaks what would otherwise
be a circular import between `display.py` (needs `PluginResult` for type
signatures) and `runner.py` (needs display functions for output).

### ActionRunner Protocol

```python
class ActionRunner(Protocol):
    async def run_action(
        self, key: str, input_data: dict, *, stream: bool = False,
    ) -> tuple[dict, list[dict]]: ...
    async def close(self) -> None: ...
```

Three implementations:

| Runner | When | How |
|--------|------|-----|
| `InProcessRunner` | Python runtime (default) | Imports entry point, calls `action.arun_raw()` via Genkit SDK |
| `NativeExecRunner` | `--runner native` | JSONL-over-stdio subprocess (Python/JS/Go) |
| `ReflectionRunner` | JS/Go/other runtimes | Subprocess + async HTTP to reflection server |
| genkit CLI | `--runner cli` | Delegates to `genkit dev:test-model` via `runner.py` |

### Runtime Protocol

```python
@runtime_checkable
class Runtime(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def specs_dir(self) -> Path: ...

    @property
    def plugins_dir(self) -> Path: ...

    @property
    def entry_command(self) -> list[str]: ...
```

The `RuntimeConfig` dataclass in `config.py` satisfies this protocol.
New runtimes can be added by:

1. Adding a `[tool.conform.runtimes.<name>]` section to `pyproject.toml`
2. Or creating a new class that satisfies the protocol

### Concurrency model (check-model)

```text
┌─────────────┐
│   cli.py    │  asyncio.run()
└──────┬──────┘
       ▼
┌─────────────┐
│  runner.py  │  asyncio.Queue + N workers
└──────┬──────┘
       ├──► Worker 1: pulls from queue, runs test
       ├──► Worker 2: pulls from queue, runs test
       └──► Worker N: (bounded by config.concurrency)
```

The runner creates an `asyncio.Queue` of plugins to test and a pool of
`N` worker coroutines. Each worker pulls a plugin from the queue and runs
it as a subprocess using `asyncio.create_subprocess_exec`. The number
of workers limits concurrency (default: 8, configurable via `-j` or TOML).

### Retry strategy

Conformance tests hit external LLM APIs that are subject to transient
errors (rate limits, timeouts, server errors).  A single failure should
not mark a plugin as broken if the cause is a momentary API hiccup.

**Algorithm:** Exponential backoff with **full jitter** (AWS-recommended).

```text
delay = random() × min(base × 2^k, 60)
```

where `k` is the zero-indexed attempt number, `base` is configurable
(default 1.0s), and 60s is the hard cap.

**Why full jitter over other strategies:**

| Strategy | Formula | Trade-off |
|---|---|---|
| Full Jitter ✅ | `random() × min(base × 2^k, cap)` | Maximum spread; best for reducing contention |
| Equal Jitter | `max/2 + random() × max/2` | Guarantees ≥50% backoff; less spread |
| Decorrelated | `min(cap, random() × 3 × prev)` | Good for correlated failures; harder to reason about |
| No Jitter | `min(base × 2^k, cap)` | Deterministic; risks thundering herd |

Full jitter is optimal here because conformance tests run against
shared API endpoints where contention is the primary concern.  The
uniform `[0, max]` distribution maximizes the spread of retry times
across concurrent workers.

**Serial fallback:** When tests run in parallel (`test-concurrency > 1`)
and any fail, the failures are automatically re-run **serially** before
being reported.  This prevents concurrent retry storms from compounding
rate-limit pressure.  The user sees a note:

```
  ⚠ 2 test(s) failed — re-running serially with retries to rule out flakes.
```

**Configuration:** `max-retries` (default 2) and `retry-base-delay`
(default 1.0s) are set in `conform.toml` and overridable via CLI
(`--max-retries`, `--retry-base-delay`).  Set `--max-retries 0` to
disable retries entirely.

### Timeout configuration

Timeouts are configurable at three levels in `conform.toml`.  The most
specific value wins (model → plugin → global):

```text
Resolution order (most specific wins):

  1. [conform.model-overrides."provider/model-name"]
     action-timeout = 60.0

  2. [conform.plugin-overrides.<plugin>]
     action-timeout = 180.0

  3. [conform]
     action-timeout  = 120.0   # per LLM action call
     health-timeout  = 5.0     # reflection server health check
     startup-timeout = 30.0    # wait for reflection server to start
```

| Timeout | Default | Scope | Description |
|---------|---------|-------|-------------|
| `action-timeout` | 120s | global, plugin, model | Single LLM generate request |
| `health-timeout` | 5s | global | Reflection server health-check poll |
| `startup-timeout` | 30s | global | Wait for reflection server + runtime file |

**Example `conform.toml`:**

```toml
[conform]
action-timeout  = 120.0
health-timeout  = 5.0
startup-timeout = 30.0

[conform.plugin-overrides.cloudflare-workers-ai]
action-timeout = 180.0   # slow provider

[conform.model-overrides."googleai/gemini-2.5-flash"]
action-timeout = 60.0    # fast model, fail quickly
```

The `action_timeout_for(plugin, model)` method on `ConformConfig`
implements the resolution chain.  `health-timeout` and
`startup-timeout` are global-only (no per-plugin/model override) since
they apply to the reflection server, not individual LLM calls.

### CLI flag architecture

All shared flags (`--config`, `--runtime`, `--specs-dir`, `--plugins-dir`)
are added directly to each subcommand parser via `_add_common_args()`
rather than using argparse's `parents=` mechanism.  This avoids a known
argparse bug where parent-parser defaults silently overwrite values
parsed by the main parser, and prevents `uv run` from intercepting
flags intended for the `conform` command.

Common arg extraction is handled by `_extract_common_args()` in
`main()`, keeping the dispatch logic DRY.

### Progress bar width invariant

The inline progress bar in `display.py` uses `round()` to convert
pass/fail counts to block characters.  Because `round()` can
independently round both segments up, the sum `p + f` may exceed
`bar_width`.  The fix clamps each segment:

```python
p = min(round(passed / total * bar_width), bar_width)
f = min(round(failed / total * bar_width), bar_width - p)
r = bar_width - p - f  # always >= 0
```

This guarantees `p + f + r == bar_width` for all inputs.  A brute-force
regression test in `display_test.py` verifies every combination of
`passed`, `failed`, `total` across multiple bar widths.

### Validator registry

Validators use a Protocol + `@register` decorator pattern:

```python
@register('has-tool-request')
def has_tool_request(
    response: dict, arg: str | None = None, chunks: list[dict] | None = None,
) -> None:
    ...  # raise ValidationError on failure
```

10 validators are ported 1:1 from the canonical JS source
(`genkit-tools/cli/src/commands/dev-test-model.ts`).
