# DeepSeek Plugin

This plugin provides Genkit support for the OpenAI-compatible DeepSeek API,
including the `deepseek-chat` and `deepseek-reasoner` models.

## Usage

Set a DeepSeek API key:

```bash
export DEEPSEEK_API_KEY=<your-api-key>
```

The plugin uses `https://api.deepseek.com` by default. Set
`DEEPSEEK_BASE_URL` to use another DeepSeek-compatible endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
)

ctx := context.Background()
plugin := &deepseek.DeepSeek{}
g := genkit.Init(
    ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("deepseek/"+deepseek.ModelDeepSeekChat),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain mixture-of-experts models."))
```

Genkit's standard `maxOutputTokens` option is sent to the DeepSeek API as
`max_tokens`:

```go
response, err := genkit.Generate(
    ctx,
    g,
    ai.WithPrompt("Answer concisely."),
    ai.WithConfig(map[string]any{"maxOutputTokens": 1024}),
)
```

The `deepseek-reasoner` model's `reasoning_content` output is returned as
Genkit reasoning parts and is available through `response.Reasoning()`.

## Live tests

Live tests are skipped unless `DEEPSEEK_API_KEY` is set:

```bash
go test -v ./plugins/compat_oai/deepseek
```
