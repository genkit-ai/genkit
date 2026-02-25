# Format Tester

Automated test harness for Genkit's output format support — text, JSON, array,
JSONL, and enum — across multiple models. Runs all format × model combinations
and reports pass/fail results.

## Formats Tested

| Format | Prompt | Output Schema |
|--------|--------|---------------|
| `text` | Short pirate story | Plain text |
| `json` | RPG creature | `{name, backstory, hitPoints, attacks[]}` |
| `array` | Futurama characters | `[{name, description, friends[], enemies[]}]` |
| `jsonl` | Pet store products | `[{name, description, price, stock, color, tags[]}]` |
| `enum` | Skydiving risk level | `VERY_LOW \| LOW \| MEDIUM \| HIGH \| VERY_HIGH` |

## Models Tested (default)

- `vertexai/gemini-2.5-pro`
- `vertexai/gemini-2.5-flash`
- `googleai/gemini-2.5-pro`
- `googleai/gemini-2.5-flash`
- Vertex AI Model Garden: Claude 3.5 Sonnet (v1 and v2)

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager

### API Keys

```bash
export GEMINI_API_KEY='<your-api-key>'
```

For Vertex AI models, also configure GCP credentials:

```bash
gcloud auth application-default login
```

### Build and Install

From the repo root:

```bash
pnpm install
pnpm run setup
```

## Run the Test

```bash
pnpm build && pnpm start
```

### Custom models

Pass model names as command-line arguments:

```bash
pnpm build && node lib/index.js googleai/gemini-2.5-flash vertexai/gemini-2.5-pro
```

## Expected Behavior

- Each format × model combination is tested
- Streaming output is printed chunk-by-chunk
- Final structured output is printed after each test
- A summary of failures is printed at the end
