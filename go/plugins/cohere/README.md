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

The plugin reads the API key from `COHERE_API_KEY`, falling back to the Cohere SDK's standard
`CO_API_KEY`. You may instead set it explicitly and optionally override the native Cohere API base
URL:

```go
&cohere.Cohere{
    APIKey:  "...",                       // COHERE_API_KEY, then CO_API_KEY
    BaseURL: "https://api.cohere.com",     // defaults to COHERE_BASE_URL
}
```

`BaseURL` must expose Cohere's native `/v2/chat` and `/v2/embed` API. It does not translate native
Cohere requests to the OpenAI-compatible API and it does not configure AWS request signing. Use
Genkit's `compat_oai` plugin for Cohere's OpenAI-compatible endpoint. Bedrock routing requires a
separately configured AWS client and is not supported by this plugin.

## Why Chat v2

The plugin targets Cohere's **Chat v2** (`/v2/chat`) endpoint rather than the legacy `Chat`/`Generate`
APIs. Chat v2 is the actively developed, messages-style API: it has native tool calls, JSON-mode
`response_format`, RAG `documents`, citations, and `safety_mode`. The legacy v1 endpoints are
deprecated and lack these capabilities.

## Language Models

```go
resp, err := genkit.Generate(ctx, g,
    ai.WithModel(cohere.ModelRef("command-a-03-2025", nil)),
    ai.WithPrompt("What is the capital of France?"),
)
```

Curated models: `command-a-plus-05-2026`, `command-a-03-2025`,
`command-a-reasoning-08-2025`, `command-r-plus-08-2024`, `command-r-08-2024`, and
`command-r7b-12-2024`.
Any other valid Cohere chat model name also resolves (with default capabilities), so newly released
models can be used immediately.

### Configuration

Pass `*cohere.ChatOptions` to `cohere.ModelRef` to carry typed Chat v2 options with the model.
Genkit owns the model, messages, tools, and streaming fields and constructs the SDK request
internally:

```go
import (
    sdk "github.com/cohere-ai/cohere-go/v2"
    genkitcohere "github.com/firebase/genkit/go/plugins/cohere"
)

safety := sdk.V2ChatRequestSafetyModeContextual
maxTokens := 1024
genkit.Generate(ctx, g,
    ai.WithModel(genkitcohere.ModelRef("command-a-03-2025", &genkitcohere.ChatOptions{
        MaxTokens:  &maxTokens,
        SafetyMode: &safety,
        Documents:  []*sdk.V2ChatRequestDocumentsItem{ /* RAG documents */ },
    })),
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

For models and configurations that emit thinking (enable it through `ChatOptions.Thinking` on
`ModelRef`), the reasoning is surfaced as Genkit reasoning parts — `resp.Reasoning()` and
`ai.NewReasoningPart` content — both for non-streaming and streamed responses. Cohere thinking does
not carry a signature.

### Citations

When Cohere returns citation spans (typically with `documents`), they are preserved on the response
under `resp.Custom["citations"]` (a `[]*cohere.Citation`) so downstream code can surface them. This
applies to both non-streaming and streamed responses.

## Embedding Models

```go
resp, err := genkit.Embed(ctx, g,
    ai.WithEmbedder(cohere.NewEmbedderRef("embed-v4.0", nil)),
    ai.WithTextDocs("the quick brown fox"),
)
```

Curated embedders: `embed-v4.0`, `embed-english-v3.0`, `embed-multilingual-v3.0`,
`embed-english-light-v3.0`, and `embed-multilingual-light-v3.0`.

Tune the embedding for the downstream task with typed `cohere.EmbedOptions` on the embedder ref:

```go
genkit.Embed(ctx, g,
    ai.WithEmbedder(cohere.NewEmbedderRef("embed-v4.0", &cohere.EmbedOptions{
        InputType:       "search_query",          // search_document (default), classification, clustering
        OutputDimension: 1024,                    // embed-v4 only: 256 / 512 / 1024 / 1536
        Truncate:        "END",                   // NONE / START / END
        EmbeddingType:   sdk.EmbeddingTypeInt8,   // float (default), int8, uint8, binary, ubinary
    })),
    ai.WithTextDocs("what is the capital of France"),
)
```

Genkit represents embedding vectors as `float32`, so integer and packed binary representations are
returned as their numeric values converted to `float32`. Binary and ubinary vectors remain packed.

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
