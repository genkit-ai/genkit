# ESM Compatibility Test

Verifies that all Genkit packages and plugins can be imported and used in
an ESM (ECMAScript Modules) project with `"type": "module"` in `package.json`.

This is a build-time smoke test, not a runtime sample — it imports every
plugin and core module to ensure there are no CommonJS/ESM interop issues.

## Plugins Tested

| Plugin | Import |
|--------|--------|
| `genkit` | Core library |
| `@genkit-ai/google-genai` | Google AI + Vertex AI |
| `@genkit-ai/googleai` | Legacy Google AI |
| `@genkit-ai/vertexai` | Legacy Vertex AI (evaluation, model garden, rerankers) |
| `@genkit-ai/firebase` | Firebase telemetry + context |
| `@genkit-ai/google-cloud` | Google Cloud telemetry |
| `@genkit-ai/express` | Express handler |
| `@genkit-ai/next` | Next.js integration |
| `@genkit-ai/mcp` | MCP client/server/host |
| `@genkit-ai/evaluator` | Evaluation framework |
| `@genkit-ai/dev-local-vectorstore` | Local vector store |
| `@genkit-ai/checks` | Checks plugin |
| `@genkit-ai/compat-oai` | OpenAI-compatible (OpenAI, DeepSeek, xAI) |
| `genkitx-ollama` | Ollama |
| `genkitx-chromadb` | ChromaDB |
| `genkitx-pinecone` | Pinecone |
| `genkitx-cloud-sql-pg` | Cloud SQL Postgres |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager

### Build and Install

From the repo root:

```bash
pnpm install
pnpm run setup
```

## Run the Test

```bash
pnpm test
```

This builds the TypeScript and then runs the compiled output. If any
ESM import fails, the process will exit with an error.

## Expected Behavior

- All imports resolve without errors
- A simple `hello` flow is defined and can be invoked
- `expressHandler` and `appRoute` wrappers work with the flow
