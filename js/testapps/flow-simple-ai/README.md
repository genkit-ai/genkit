# Flow + AI Kitchen Sink

Comprehensive test app that exercises nearly every Genkit feature — generation,
streaming, structured output, tool calling, Dotprompt, RAG, multimodal,
evaluators, TTS, and custom model middleware. Uses Google AI (Gemini) with
Google Cloud telemetry.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Joke Generation | `jokeFlow` | Generate jokes with configurable model |
| Streaming (Vertex) | `streamFlowVertex` | Streaming generation with Vertex AI |
| Streaming JSON | `streamJsonFlow` | Stream structured JSON (RPG characters) |
| Structured Output | `jokeWithOutputFlow` | JSON output with Zod schema |
| Multimodal | `multimodalFlow` | Describe an image from URL |
| RAG | `ragFlow` | Retrieval-augmented generation |
| Dotprompt | `dotpromptContext` | Prompt templates with context |
| Tool Calling | `toolTester` | Generate joke using a tool |
| Math Tool | `gablorkenFlow` | Custom gablorken calculator tool |
| TTS | `tts-flow` | Text-to-speech generation |
| LLM-as-Judge | `llmJudgeFlow` | Evaluate output quality with a model |
| Custom Middleware | `constrainedFlow` | Simulated constrained generation |

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

Or:

```bash
pnpm start
```

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test basic flows**:
   - [ ] `jokeFlow` — Joke generation (input: `{"modelName": "vertexai/gemini-2.5-pro", "subject": "cats"}`)
   - [ ] `streamFlowVertex` — Streaming response
   - [ ] `streamJsonFlow` — Streaming JSON (RPG characters)

3. **Test structured output**:
   - [ ] `jokeWithOutputFlow` — JSON output with schema

4. **Test tools**:
   - [ ] `toolTester` — Tool-assisted joke generation
   - [ ] `gablorkenFlow` — Gablorken calculator

5. **Test multimodal + RAG**:
   - [ ] `multimodalFlow` — Image description
   - [ ] `ragFlow` — Retrieval-augmented generation

6. **Expected behavior**:
   - All flows complete without errors
   - Streaming shows incremental output
   - Structured output matches Zod schemas
   - Tools are called and responses integrated
