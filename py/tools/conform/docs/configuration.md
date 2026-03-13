# Configuration

All repo-specific settings live in a `conform.toml` file alongside the
conformance specs (e.g. `py/tests/conform/conform.toml`).

**Auto-discovery:** When `--config` is not passed, the tool walks up from
the current working directory looking for `conform.toml`.  The wrapper
script `py/bin/conform` passes `--config` explicitly.

> **Note:** All shared flags (`--config`, `--runtime`, `--specs-dir`,
> `--plugins-dir`) are defined on each subcommand, not on the top-level
> parser.  This avoids argparse parent-parser bugs and prevents `uv run`
> from intercepting flags meant for `conform`.

Both `conform.toml` (top-level `[conform]`) and `pyproject.toml`
(`[tool.conform]`) layouts are supported.

## Concurrency

```toml
[conform]
concurrency = 8
```

Override with `-j N`:

```bash
conform check-model -j 8
```

## Test concurrency

Controls how many tests run in parallel within a single model spec.
Defaults to 3.  Set to 1 for strictly sequential execution if you hit rate limits.

```toml
[conform]
test-concurrency = 3
```

Override with `-t N`:

```bash
conform check-model -t 3
```

### Provider rate limits reference

When choosing `-j` (plugin concurrency) and `-t` (test concurrency), keep
provider rate limits in mind.  Worst-case concurrent requests =
`concurrency × test-concurrency` (default 8 × 3 = 24).

| Provider | Lowest paid tier RPM | Free tier RPM | Source |
|----------|---------------------|---------------|--------|
| Google Gemini (API key) | 1,000 | 15 | [ai.google.dev](https://ai.google.dev/gemini-api/docs/rate-limits) |
| Vertex AI (Google Cloud) | TPM-based (no RPM cap) | — | [cloud.google.com](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/quotas) |
| Anthropic Claude | 50 (Tier 1) | — | [platform.claude.com](https://platform.claude.com/docs/en/api/rate-limits) |
| OpenAI (GPT-4o) | 500 (Tier 1) | 3 | [platform.openai.com](https://platform.openai.com/docs/guides/rate-limits) |
| Mistral AI | ~60–300 (RPS-based) | Restrictive | [docs.mistral.ai](https://docs.mistral.ai/deployment/ai-studio/tier) |
| Cloudflare Workers AI | 300 (text gen) | Same | [developers.cloudflare.com](https://developers.cloudflare.com/workers-ai/platform/limits/) |
| Amazon Bedrock | ~100–1,000 (varies) | — | [docs.aws.amazon.com](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html) |
| Ollama (local) | ∞ (hardware-bound) | ∞ | — |

> **Tip:** The defaults (`-j 8 -t 3`) are safe for all paid tiers.
> If you are on a free tier, use `-t 1` to avoid 429 errors.

### Per-plugin overrides

Override `test-concurrency` for specific plugins that are more
rate-sensitive:

```toml
[conform.plugin-overrides.cloudflare-workers-ai]
test-concurrency = 1
```

Priority: CLI `-t N` > per-plugin override > global `test-concurrency`.

## Retry on failure

Failed tests are automatically retried with exponential backoff and full
jitter to handle transient API errors and rate limiting.

```toml
[conform]
max-retries      = 2
retry-base-delay = 1.0
```

Override with CLI flags:

```bash
conform check-model --max-retries 3 --retry-base-delay 2.0
```

Set `--max-retries 0` to disable retries entirely.

When tests run in parallel (`-t` > 1) and any fail, the failures are
automatically **re-run serially** with retries.  This avoids hammering
rate-limited APIs with concurrent retry storms.  A note is printed:

```text
  ☢ 2 test(s) failed — re-running serially with retries to rule out flakes.
```

See [Architecture § Retry strategy](architecture.md#retry-strategy) for
the engineering rationale.

## Environment variables

Required env vars per plugin:

```toml
[conform.env]
google-genai          = ["GEMINI_API_KEY"]
anthropic             = ["ANTHROPIC_API_KEY"]
compat-oai            = ["OPENAI_API_KEY"]
mistral               = ["MISTRAL_API_KEY"]
deepseek              = ["DEEPSEEK_API_KEY"]
xai                   = ["XAI_API_KEY"]
cohere                = ["COHERE_API_KEY"]
amazon-bedrock        = ["AWS_REGION"]
huggingface           = ["HF_TOKEN"]
microsoft-foundry     = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
cloudflare-workers-ai = ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"]
vertex-ai             = ["GOOGLE_CLOUD_PROJECT"]
ollama                = []                        # No credentials needed
```

Plugins with missing env vars are **skipped** (not failed) during
`check-model`.

## Additional model plugins

Plugins that use dynamic model registration (no `model_info.py`):

```toml
[conform]
additional-model-plugins = ["google-genai", "vertex-ai", "ollama"]
```

These are included in `check-plugin` verification even though they
lack a `model_info.py` file.

## Runtime configuration

Each runtime defines how to locate specs, plugins, and what command
to use for running entry points:

Paths are relative to the `conform.toml` file.

```toml
[conform.runtimes.python]
cwd           = "../../.."
specs-dir     = "."
plugins-dir   = "../../plugins"
entry-command = ["uv", "run", "--project", "py", "--active"]

[conform.runtimes.js]
cwd           = "../../../js"
specs-dir     = "."
plugins-dir   = "../../../js/plugins"
entry-command = ["npx", "tsx"]

[conform.runtimes.go]
cwd           = "../../../go"
specs-dir     = "."
plugins-dir   = "../../../go/plugins"
entry-command = ["go", "run"]
```

Runtime-specific entry filenames and model markers default to
sensible values per runtime but can be overridden:

| Runtime | `entry-filename` | `model-marker` |
|---------|-------------------|----------------|
| python  | `conformance_entry.py` | `model_info.py` |
| js      | `conformance_entry.ts` | `models.ts` |
| go      | `conformance_entry.go` | `model_info.go` |

## CLI flags vs TOML

CLI flags override TOML values.  All flags below are per-subcommand
(e.g. `conform check-model --config FILE`), not global:

| Flag | TOML Key | Description |
|------|----------|-------------|
| `--config FILE` | — | Path to `conform.toml` (auto-discovered if omitted) |
| `--runtime NAME` | — | Filter to a single runtime |
| `--runner TYPE` | — | Runner type: `auto`, `native`, `reflection`, `in-process`, or `cli` |
| `--specs-dir DIR` | `runtimes.<name>.specs-dir` | Override specs directory |
| `--plugins-dir DIR` | `runtimes.<name>.plugins-dir` | Override plugins directory |
| `-j N` | `concurrency` | Max concurrent plugins |
| `-t N` | `test-concurrency` | Max concurrent tests per model spec |
| `--max-retries N` | `max-retries` | Max retries per failed test (default: 2) |
| `--retry-base-delay SECS` | `retry-base-delay` | Base delay for backoff (default: 1.0) |
| `--verbose` | — | Print full output for failures |

## Adding a new plugin

1. Add the plugin's env vars to `[conform.env]` in `conform.toml`
2. Create `py/tests/conform/<plugin>/model-conformance.yaml`
3. Create entry points for each runtime:
    - `conformance_entry.py` (Python)
    - `conformance_entry.ts` (JS)
    - `conformance_entry.go` (Go)
4. Run `conform check-plugin` to verify
5. Run `conform check-model <plugin>` to test
