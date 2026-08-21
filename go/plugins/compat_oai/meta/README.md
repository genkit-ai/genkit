# Meta Model API Plugin

This plugin provides Genkit support for Meta's OpenAI-compatible Model API and
the Muse Spark multimodal reasoning models.

## Usage

Set a Meta Model API key:

```bash
export MODEL_API_KEY=<your-api-key>
```

The plugin uses `https://api.meta.ai/v1` by default. Set `META_BASE_URL` or
provide `option.WithBaseURL` through `meta.Meta.Opts` to use another compatible
endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/meta"
)

ctx := context.Background()
g := genkit.Init(ctx,
    genkit.WithPlugins(&meta.Meta{}),
    genkit.WithDefaultModel("meta/muse-spark-1.2"),
)

response, err := genkit.Generate(ctx, g,
    ai.WithPrompt("Explain mixture-of-experts models."),
)
```

The plugin uses Meta Model API's OpenAI-compatible Chat Completions endpoint.

## Live tests

Live tests are skipped unless `MODEL_API_KEY` is set:

```bash
go test -v ./plugins/compat_oai/meta
```
