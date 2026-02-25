# Dev UI Gallery

Loads plugins and flows to demonstrate and test Genkit Dev UI features.

This test app registers a wide variety of flows, prompts, and tools to
exercise the Dev UI — including Firebase Functions-compatible flows,
Dotprompt templates, custom tools, evaluators, and multiple vector store
backends.

## Features Demonstrated

| Feature | Module | Description |
|---------|--------|-------------|
| Flows | `flows.ts` | Various generation flows (streaming, structured output, etc.) |
| Firebase Functions | `flows-firebase-functions.ts` | Flows wrapped for Firebase Functions compatibility |
| Dotprompt | `prompts.ts` | Prompt file templates with schemas |
| Tools | `tools.ts` | Custom tools for model use |
| Multiple Plugins | `genkit.ts` | Google AI, Vertex AI, Ollama, ChromaDB, Pinecone, Firebase |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager

### API Keys

```bash
export GEMINI_API_KEY='<your-api-key>'
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

2. **Verify UI features**:
   - [ ] Flows panel lists all defined flows
   - [ ] Prompts panel lists all Dotprompt templates
   - [ ] Tools panel lists all defined tools
   - [ ] Actions panel shows registered actions

3. **Test flows**:
   - [ ] Run various flows from the Dev UI
   - [ ] Test streaming flows and verify incremental output
   - [ ] Test structured output flows

4. **Expected behavior**:
   - All flows, prompts, and tools appear in the Dev UI
   - Flows execute and return results
   - Streaming shows incremental output
