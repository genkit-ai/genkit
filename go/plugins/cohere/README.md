# Cohere Plugin

This plugin provides a first-class interface to Cohere's [Chat v2](https://docs.cohere.com/v2/docs/chat-api)
and [Embed](https://docs.cohere.com/docs/embeddings) APIs for Genkit, built on the official
[`github.com/cohere-ai/cohere-go/v2`](https://github.com/cohere-ai/cohere-go) SDK. It supports
multi-turn chat, streaming, tool calling, JSON-mode structured output, and text embeddings.

## Prerequisites

- Go installed on your system
- A Cohere API key

## Setup

```go
import (
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/cohere"
)

g := genkit.Init(ctx, genkit.WithPlugins(&cohere.Cohere{}))
```

The plugin reads the API key from the `COHERE_API_KEY` environment variable. You may instead set it
explicitly, and optionally override the base URL (e.g. to target the OpenAI-Compatibility API or a
Bedrock-hosted deployment):

```go
&cohere.Cohere{
    APIKey:  "...",                       // defaults to COHERE_API_KEY
    BaseURL: "https://api.cohere.com",     // defaults to COHERE_BASE_URL
}
```

## Why Chat v2

The plugin targets Cohere's **Chat v2** (`/v2/chat`) endpoint rather than the legacy `Chat`/`Generate`
APIs. Chat v2 is the actively developed, messages-style API: it has native tool calls, JSON-mode
`response_format`, RAG `documents`, citations, and `safety_mode`. The legacy v1 endpoints are
deprecated and lack these capabilities.

## Language Models

```go
resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("cohere/command-r-plus"),
    ai.WithPrompt("What is the capital of France?"),
)
```

Curated models: `command-a-03-2025`, `command-r-plus`, `command-r`, `command-r7b-12-2024`.
Any other valid Cohere chat model name also resolves (with default capabilities), so newly released
models can be used immediately.

### Configuration

Pass a `*cohere.V2ChatRequest` (the SDK request type) via `ai.WithConfig` to set any chat parameter.
Because the whole request struct is accepted, advanced features pass through without extra wiring:

```go
import sdk "github.com/cohere-ai/cohere-go/v2"

safety := sdk.V2ChatRequestSafetyModeContextual
maxTokens := 1024
genkit.Generate(ctx, g,
    ai.WithModelName("cohere/command-r-plus"),
    ai.WithConfig(&sdk.V2ChatRequest{
        MaxTokens:  &maxTokens,
        SafetyMode: &safety,
        Documents:  []*sdk.V2ChatRequestDocumentsItem{ /* RAG documents */ },
    }),
    ai.WithPrompt("..."),
)
```

`documents`, `citation_options`, `safety_mode`, `response_format`, `temperature`, `p`, `k`, `seed`,
and `stop_sequences` are all settable this way.

### Streaming and tool calling

Streaming uses `ai.WithStreaming`; tool calls and JSON-mode structured output work as in any other
Genkit model provider. Tool definitions map to Cohere `tools`, and `tool_call_id` is round-tripped on
tool-result messages.

### Reasoning

For models and configurations that emit thinking (enable it via the request's `Thinking` field in
`ai.WithConfig`), the reasoning is surfaced as Genkit reasoning parts — `resp.Reasoning()` and
`ai.NewReasoningPart` content — both for non-streaming and streamed responses. Cohere thinking does
not carry a signature.

### Citations

When Cohere returns citation spans (typically with `documents`), they are preserved on the response
under `resp.Custom["citations"]` (a `[]*cohere.Citation`) so downstream code can surface them. This
applies to both non-streaming and streamed responses.

## Embedding Models

```go
resp, err := genkit.Embed(ctx, g,
    ai.WithEmbedderName("cohere/embed-v4.0"),
    ai.WithTextDocs("the quick brown fox"),
)
```

Curated embedders: `embed-v4.0`, `embed-english-v3.0`, `embed-multilingual-v3.0`.

Tune the embedding for the downstream task with `cohere.EmbedOptions` via `ai.WithConfig`:

```go
genkit.Embed(ctx, g,
    ai.WithEmbedderName("cohere/embed-v4.0"),
    ai.WithConfig(&cohere.EmbedOptions{
        InputType:       "search_query", // or search_document (default), classification, clustering
        OutputDimension: 1024,           // embed-v4 only: 256 / 512 / 1024 / 1536
        Truncate:        "END",          // NONE / START / END
    }),
    ai.WithTextDocs("what is the capital of France"),
)
```

## Running Tests

Unit tests for the request/response mapping run without network access:

```bash
go test ./plugins/cohere/
```

Live tests exercise the real API and are skipped unless `COHERE_API_KEY` is set:

```bash
export COHERE_API_KEY=<your-api-key>
go test -v ./plugins/cohere/ -run TestCohereLive
```

A runnable sample lives at [`go/samples/cohere`](../../samples/cohere):

```bash
COHERE_API_KEY=<your-api-key> go run ./samples/cohere
```
