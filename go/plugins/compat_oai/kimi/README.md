# Kimi Plugin

This plugin provides Genkit support for Moonshot AI's OpenAI-compatible Kimi
models, including Kimi K3, Kimi K2.6, and Kimi K2.7 Code. Kimi K2.5 remains
registered as deprecated for existing users during its platform sunset period.

## Usage

Set a Moonshot API key:

```bash
export KIMI_API_KEY=<your-api-key>
```

`MOONSHOT_API_KEY` is also accepted. The plugin uses
`https://api.moonshot.ai/v1` by default; set `KIMI_BASE_URL` or
`MOONSHOT_BASE_URL` to use another Moonshot-compatible endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/kimi"
)

ctx := context.Background()
plugin := &kimi.Kimi{}
g := genkit.Init(
    ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("kimi/"+kimi.ModelKimiK3),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain mixture-of-experts models."))
```

Kimi's `reasoning_content` output is returned as Genkit reasoning parts and is
available through `response.Reasoning()`. Reasoning parts are also preserved as
`reasoning_content` during multi-turn and tool-call requests.

Kimi-specific request fields are passed through when configuration is supplied
as a map. For example, Kimi K2.6 thinking can be disabled per request:

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

## Live tests

Live tests are skipped unless `KIMI_API_KEY` or `MOONSHOT_API_KEY` is set:

```bash
go test -v ./plugins/compat_oai/kimi
```
