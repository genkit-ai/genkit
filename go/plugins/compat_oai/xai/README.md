# xAI Plugin

This plugin provides Genkit support for xAI's OpenAI-compatible Grok language
and vision models.

## Supported models

- `grok-3`
- `grok-3-fast`
- `grok-3-mini`
- `grok-3-mini-fast`
- `grok-2-vision-1212`

Image generation models are not included because the Go OpenAI-compatible
adapter does not yet support image generation.

## Usage

Set an xAI API key:

```bash
export XAI_API_KEY=<your-api-key>
```

The plugin uses `https://api.x.ai/v1` by default. Set `XAI_BASE_URL` to use
another xAI-compatible endpoint.

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
    genkit.WithDefaultModel("xai/"+xai.ModelGrok3),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain mixture-of-experts models."))
```

xAI-specific options use Genkit's camel-case config names and are translated
to the provider's API fields:

```go
response, err := genkit.Generate(
    ctx,
    g,
    ai.WithPrompt("Summarize recent developments in fusion energy."),
    ai.WithConfig(map[string]any{
        "deferred": true,
        "reasoningEffort": "high",
        "webSearchOptions": map[string]any{
            "search_context_size": "high",
        },
    }),
)
```

## Tests

The package tests use a local HTTP server and do not require an API key:

```bash
go test -v ./plugins/compat_oai/xai
```
