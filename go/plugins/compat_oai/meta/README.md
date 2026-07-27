# Meta Model API Plugin

This plugin provides Genkit support for Meta's OpenAI-compatible Model API and
the Muse Spark 1.1 multimodal reasoning model.

## Usage

Set a Meta Model API key:

```bash
export MODEL_API_KEY=<your-api-key>
```

`META_API_KEY` is also accepted. The plugin uses
`https://api.meta.ai/v1` by default; set `META_BASE_URL` to use another
compatible endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/meta"
)

ctx := context.Background()
plugin := &meta.Meta{}
g := genkit.Init(
    ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("meta/"+meta.ModelMuseSpark11),
)

response, err := genkit.Generate(
    ctx,
    g,
    ai.WithPrompt("Explain mixture-of-experts models."),
)
```

Meta Model API is currently in public preview. This plugin uses its
OpenAI-compatible Chat Completions endpoint.

## Live tests

Live tests are skipped unless `MODEL_API_KEY` or `META_API_KEY` is set:

```bash
go test -v ./plugins/compat_oai/meta
```
