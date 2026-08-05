# Z.ai Plugin

This plugin provides Genkit support for Z.ai's OpenAI-compatible GLM text and
vision models.

## Usage

Set a Z.ai API key:

```bash
export ZAI_API_KEY=<your-api-key>
```

The plugin uses `https://api.z.ai/api/paas/v4` by default. Set
`ZAI_BASE_URL` to use another compatible endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/zai"
)

ctx := context.Background()
plugin := &zai.ZAI{}
g := genkit.Init(
    ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("zai/"+zai.ModelGLM51),
)

response, err := genkit.Generate(
    ctx,
    g,
    ai.WithPrompt("Explain mixture-of-experts models."),
)
```

GLM's `reasoning_content` output is returned as Genkit reasoning parts and is
available through `response.Reasoning()`. Provider-specific fields such as
`thinking` are passed through when configuration is supplied as a map:

```go
response, err := genkit.Generate(
    ctx,
    g,
    ai.WithPrompt("Answer concisely."),
    ai.WithConfig(map[string]any{
        "thinking": map[string]any{
            "type": "disabled",
        },
    }),
)
```

Z.ai currently documents only automatic tool choice. Tool calling is supported,
but forced `required` and `none` modes are not advertised by this plugin.

## Live tests

Live tests are skipped unless `ZAI_API_KEY` is set:

```bash
go test -race ./plugins/compat_oai/zai -run '^TestPluginLive$' -v -count=1
```
