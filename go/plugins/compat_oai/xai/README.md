# xAI Plugin

This plugin provides Genkit support for xAI's OpenAI-compatible Grok models,
including Grok 4.5, Grok 4.3, the Grok 4.20 line, and the Grok Build coding
model.

## Usage

Set an xAI API key:

```bash
export XAI_API_KEY=<your-api-key>
```

The plugin uses `https://api.x.ai/v1` by default. Set `XAI_BASE_URL` to use
another compatible endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/xai"
)

ctx := context.Background()
plugin := &xai.XAI{}
g := genkit.Init(
    ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("xai/grok-4.5"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain reinforcement learning."))
```

Grok's `reasoning_content` output is returned as Genkit reasoning parts and is
available through `response.Reasoning()`.

Models take a typed `xai.ChatConfig`: the generation fields xAI accepts plus
its own controls (`reasoningEffort`, `searchParameters`). `xai.ModelRef`
carries the config with the model ID:

```go
response, err := genkit.Generate(
    ctx,
    g,
    ai.WithModel(xai.ModelRef("grok-4.3", &xai.ChatConfig{
        ReasoningEffort: "high",
    })),
    ai.WithPrompt("Work through this step by step."),
)
```

`searchParameters` turns on xAI's live search, and its `sources` entries pass
through as xAI documents them:

```go
response, err := genkit.Generate(
    ctx,
    g,
    ai.WithModel(xai.ModelRef("grok-4.5", &xai.ChatConfig{
        SearchParameters: &xai.SearchParameters{
            Mode:            "on",
            ReturnCitations: openai.Ptr(true),
            Sources:         []map[string]any{{"type": "x", "x_handles": []any{"xai"}}},
        },
    })),
    ai.WithPrompt("What did xAI announce this week?"),
)
```

`maxOutputTokens` reaches xAI as `max_completion_tokens`, since xAI deprecated
`max_tokens`, and `topLogProbs` accepts 0 through 8 rather than OpenAI's 20.

xAI's image, video, and voice models are not part of this plugin: it serves the
chat completions endpoint only. Deferred completions (`deferred: true`) are not
exposed either, since they answer with a request ID to poll rather than a
completion.

## Live tests

Live tests are skipped unless `XAI_API_KEY` is set:

```bash
go test -race ./plugins/compat_oai/xai -run '^TestPluginLive$' -v -count=1
```
