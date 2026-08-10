# Vertex AI Model Garden

Demonstrates using third-party models via the Vertex AI Model Garden —
Anthropic Claude, Meta Llama, Mistral, and Codestral — alongside native
Gemini models, with tool calling, structured output, and streaming.

## Features Demonstrated

| Feature | Flow | Model | Description |
|---------|------|-------|-------------|
| Tool Calling | `claude-sonnet-4 - toolCallingFlow` | Claude Sonnet 4 | Weather + unit conversion tools |
| Basic Generation | `llama4 - basicFlow` | Llama 4 Maverick | Simple greeting |
| Structured Output | `mistral-medium - explainConcept` | Mistral Medium 3 | Concept explanation with examples |
| Code Analysis | `mistral-small - analyzeCode` | Mistral Small | Code review and suggestions |
| Code Generation | `codestral - generateFunction` | Codestral 2 | TypeScript function generation |
| Gemini Tools | `gemini-2.5-flash - tool flow` | Gemini 2.5 Flash | Weather tools with native Gemini |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager
- **Google Cloud project** with Vertex AI API enabled

### GCP Authentication

```bash
gcloud auth application-default login
gcloud config set project <your-project-id>
```

### Build and Install

From the repo root:

```bash
pnpm install
pnpm run setup
```

## Run the Sample

```bash
pnpm run genkit:dev
```

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test Claude** (Anthropic via Model Garden):
   - [ ] `claude-sonnet-4 - toolCallingFlow` — Input: `"Paris, France"` (calls weather + unit converter tools)

3. **Test Llama** (Meta via Model Garden):
   - [ ] `llama4 - basicFlow` — Simple greeting flow

4. **Test Mistral**:
   - [ ] `mistral-medium - explainConcept` — Input: `{"concept": "concurrency"}`
   - [ ] `mistral-small - analyzeCode` — Input: `{"code": "console.log('hello world');"}`

5. **Test Codestral**:
   - [ ] `codestral - generateFunction` — Input: `{"description": "greets me and asks my favourite colour"}`

6. **Test Gemini** (native, not via Model Garden):
   - [ ] `gemini-2.5-flash - tool flow` — Input: `"Paris, France"`

7. **Expected behavior**:
   - All models respond through the Model Garden API
   - Tool calling works with Claude and Gemini
   - Structured output works with Mistral
   - Streaming shows incremental output for Claude, Llama, and Gemini flows
