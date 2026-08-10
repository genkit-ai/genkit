# Model Tester

Automated conformance test suite that runs Genkit's built-in model tests
against multiple providers — Google AI, Vertex AI, Vertex AI Model Garden,
and Ollama. Reports pass/fail/skip for each model.

## Models Tested

| Provider | Models |
|----------|--------|
| Google AI | `gemini-2.5-pro`, `gemini-2.5-flash` |
| Vertex AI | `gemini-2.5-pro`, `gemini-2.5-flash` |
| Vertex AI Model Garden | `claude-sonnet-4@20250514`, `meta/llama-4-maverick-17b-128e-instruct-maas` |
| Ollama | `gemma2` |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager

### API Keys

For Google AI models:

```bash
export GEMINI_API_KEY='<your-api-key>'
```

For Vertex AI models:

```bash
gcloud auth application-default login
```

### Ollama (optional)

For Ollama models, install and start Ollama:

```bash
ollama serve
ollama pull gemma2
```

### Build and Install

From the repo root:

```bash
pnpm install
pnpm run setup
```

## Run the Tests

```bash
pnpm run genkit:dev
```

Or run directly:

```bash
pnpm build && pnpm start
```

## Expected Behavior

- Each model is tested against the standard conformance tests
- Results are printed with colored PASSED/FAILED/SKIPPED indicators
- Failed tests include error message and stack trace
- Process exits with code 1 if any test fails
