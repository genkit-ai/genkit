# Basic Gemini

Comprehensive demo of the Google GenAI plugin — from basic generation to
thinking modes, tool calling, structured output, multimodal, image generation,
TTS, and advanced features like search grounding and URL context.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Simple Generation | `basic-hi` | Basic text generation with Gemini |
| Retry Middleware | `basic-hi-with-retry` | Generation with automatic retry on failure |
| Fallback Middleware | `basic-hi-with-fallback` | Model fallback chain on failure |
| Thinking (Argon) | `thinking-level` | Chain-of-thought reasoning (Gemini 4 Argon) |
| Thinking (Flash) | `thinking-level-flash` | Chain-of-thought reasoning (Gemini 3 Flash) |
| Vision (Image) | `multimodal-input` | Describe a base64-encoded image |
| YouTube Videos | `youtube-videos` | Transcribe YouTube video content |
| Search Grounding | `search-grounding` | Web search for real-time info |
| URL Context | `url-context` | Analyze content from web pages |
| File Search (RAG) | `file-search` | Search uploaded documents |
| Tool Calling | `toolCalling` | Streaming tool use (weather + unit conversion) |
| Screenshot Tool | `multipart-tool-calling` | Screenshot-based multipart tool use |
| Structured Tool Calling | `structured-tool-calling` | Tool calling with structured output |
| Structured Output | `structured-output` | JSON output with Zod schema (RPG character) |
| Media Resolution | `media-resolution` | Media resolution config (Gemini 3.1 Pro Preview) |
| Image Editing | `image-editing` | Multi-image editing (Gemini Nano Banana 2.1) |
| Image Config | `image-config` | Image generation with Gemini 3 Pro Image |
| Image Generation | `image-generation` | Image generation with Gemini Nano Banana 2.1 |
| Text-to-Speech | `tts-default-voice` | Text-to-speech with Gemini 3.8 Flash TTS |
| Vertex AI | `basic-hi` (Vertex) | Same features via Vertex AI backend |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager

### How to Get Your Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/).
2. Click on "Get API key" and create a key in a new or existing Google Cloud project.

For more details, check out the [official documentation](https://ai.google.dev/gemini-api/docs/api-key).

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

### Google AI backend

```bash
pnpm run genkit:dev:googleai
```

### Vertex AI backend

```bash
pnpm run genkit:dev:vertexai
```

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test basic flows**:
   - [ ] `basic-hi` — Simple greeting generation
   - [ ] `basic-hi-with-retry` — Generation with retry middleware
   - [ ] `basic-hi-with-fallback` — Fallback chain on model error

3. **Test thinking modes**:
   - [ ] `thinking-level` — Input: `HIGH` (or `LOW`, `MEDIUM`)
   - [ ] `thinking-level-flash` — Input: `MINIMAL`, `LOW`, `MEDIUM`, or `HIGH`

4. **Test multimodal**:
   - [ ] `multimodal-input` — Image description from base64
   - [ ] `youtube-videos` — YouTube video transcription

5. **Test tools**:
   - [ ] `toolCalling` — Weather tool with streaming
   - [ ] `structured-tool-calling` — Tool calling with structured output

6. **Test advanced features**:
   - [ ] `search-grounding` — Web search grounding
   - [ ] `url-context` — URL content analysis
   - [ ] `file-search` — File search RAG
   - [ ] `structured-output` — RPG character generation
   - [ ] `image-generation` — Image generation

7. **Expected behavior**:
   - All flows complete without errors
   - Streaming shows incremental output
   - Structured output matches Zod schemas
   - Tools are called and responses processed
