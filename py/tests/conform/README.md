# Model Conformance Testing

Cross-runtime model conformance tests for Genkit Python plugins. Each plugin
has a test spec (`model-conformance.yaml`) and a minimal entry point
(`conformance_entry.py`) that the `genkit dev:test-model` CLI drives.

## Quick start

```bash
# First-time setup: install tools + configure API keys interactively
./setup.sh

# Check what's installed without making changes
./setup.sh --check

# Skip tools, just configure API keys
./setup.sh --keys-only
```

Once set up:

```bash
# Test a single plugin (native runner, all runtimes)
py/bin/conform check-model anthropic

# Test all plugins (requires ALL env vars below)
py/bin/conform check-model

# Test with a specific runtime matrix
py/bin/conform --runtime python go check-model

# Check env-var readiness
py/bin/conform list

# Lint-time check (are conformance files present?)
py/bin/conform check-plugin
```

## Directory layout

```
tests/conform/
├── README.md                          ← you are here
├── amazon-bedrock/
│   ├── conformance_entry.py           ← minimal Genkit + plugin init
│   └── model-conformance.yaml         ← models + supported capabilities
├── anthropic/
│   ├── conformance_entry.py
│   └── model-conformance.yaml
├── cloudflare-workers-ai/
│   ├── conformance_entry.py
│   └── model-conformance.yaml
├── cohere/
│   ├── conformance_entry.py
│   └── model-conformance.yaml
├── compat-oai/
│   ├── conformance_entry.py
│   └── model-conformance.yaml
├── deepseek/
│   ├── conformance_entry.py
│   └── model-conformance.yaml
├── google-genai/
│   ├── conformance_entry.py
│   └── model-conformance.yaml         ← symlink to JS SDK spec
├── huggingface/
│   ├── conformance_entry.py
│   └── model-conformance.yaml
├── microsoft-foundry/
│   ├── conformance_entry.py
│   └── model-conformance.yaml
├── mistral/
│   ├── conformance_entry.py
│   └── model-conformance.yaml
├── ollama/
│   ├── conformance_entry.py
│   └── model-conformance.yaml
├── vertex-ai/
│   ├── conformance_entry.py
│   └── model-conformance.yaml
└── xai/
    ├── conformance_entry.py
    └── model-conformance.yaml
```

## Required environment variables

| Plugin | Environment variable(s) |
|--------|------------------------|
| `google-genai` | `GEMINI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `compat-oai` | `OPENAI_API_KEY` |
| `mistral` | `MISTRAL_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `xai` | `XAI_API_KEY` |
| `cohere` | `COHERE_API_KEY` |
| `amazon-bedrock` | `AWS_REGION` + AWS credentials |
| `huggingface` | `HF_TOKEN` |
| `microsoft-foundry` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| `cloudflare-workers-ai` | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` |
| `vertex-ai` | `GOOGLE_CLOUD_PROJECT` + GCP credentials |
| `ollama` | *(none — uses local server)* |

## Model coverage (26 models across 13 plugins)

| Plugin | Models | Key capabilities tested |
|--------|--------|------------------------|
| **google-genai** | gemini-2.5-pro, gemini-2.5-flash, gemini-3-pro-preview, imagen-4.0, gemini-2.5-flash-tts | tool-request, structured-output, vision, image-gen, TTS |
| **anthropic** | claude-sonnet-4, claude-sonnet-4-5, claude-haiku-4-5, claude-opus-4-6 | tool-request, structured-output, vision, streaming |
| **compat-oai** | gpt-4o, gpt-4o-mini, dall-e-3, tts-1 | tool-request, structured-output, image-gen, TTS |
| **mistral** | mistral-small-latest, mistral-large-latest | tool-request, structured-output, vision, streaming |
| **deepseek** | deepseek-chat, deepseek-reasoner | tool-request (chat only), structured-output, streaming |
| **xai** | grok-4-fast-non-reasoning, grok-2-vision-1212 | tool-request, structured-output, vision, streaming |
| **cohere** | command-a-03-2025 | tool-request, structured-output, multiturn |
| **amazon-bedrock** | us.anthropic.claude-sonnet-4-5-...v1:0 | tool-request, structured-output, vision, streaming |
| **microsoft-foundry** | gpt-4o | tool-request, structured-output, vision, streaming |
| **huggingface** | meta-llama/Llama-3.1-8B-Instruct | multiturn, system-role |
| **cloudflare-workers-ai** | @cf/meta/llama-3.1-8b-instruct | tool-request, multiturn, streaming |
| **vertex-ai** | anthropic/claude-sonnet-4 (via Model Garden) | tool-request, vision, multiturn, streaming |
| **ollama** | gemma3 | tool-request, vision, multiturn |

## Supported test capabilities

The YAML spec files use these capability tags:

| Tag | Description |
|-----|-------------|
| `tool-request` | Model can call tools (function calling) |
| `structured-output` | Model can produce JSON output conforming to a schema |
| `multiturn` | Model supports multi-turn conversations |
| `system-role` | Model accepts a system message |
| `input-image-base64` | Model accepts base64-encoded image input |
| `input-image-url` | Model accepts image URLs |
| `input-video-youtube` | Model accepts YouTube video URLs |
| `output-image` | Model generates images |
| `output-audio` | Model generates audio |
| `streaming-multiturn` | Streaming works with multi-turn conversations |
| `streaming-tool-request` | Streaming works with tool calls |
| `streaming-structured-output` | Streaming works with structured output |

## Lint integration

`py/bin/check_consistency` (Check 21) delegates to `conform check-plugin`
which verifies that every model plugin (any plugin with a `model_info.py`)
has a conformance spec and entry point. This check runs as part of
`bin/lint` and does **not** require API keys — it only checks that the
files exist.

The actual conformance tests (`conform check-model`) require live API keys
and are run manually, not as part of CI lint.

## Adding a new plugin

1. Create `tests/conform/<plugin>/model-conformance.yaml` with the models
   and capabilities to test.
2. Create `tests/conform/<plugin>/conformance_entry.py` that initializes
   Genkit with the plugin and keeps the process alive.
3. Add the plugin's required env vars to
   `py/tools/conform/pyproject.toml` under `[tool.conform.env]`.
4. Run `py/bin/conform check-plugin` to verify the files are detected.

Use any existing plugin directory as a template — the pattern is identical
across all plugins.

## Cross-runtime parity

The `google-genai` spec is a symlink to the JS SDK's test file:

```
tests/conform/google-genai/model-conformance.yaml
  → ../../../../js/plugins/google-genai/tests/model-tests-tts.yaml
```

This ensures the Python plugin is tested against the exact same models and
capabilities as the JavaScript SDK.
