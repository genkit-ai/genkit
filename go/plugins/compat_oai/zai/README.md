# Z.ai Plugin

This plugin provides Genkit support for Z.ai's OpenAI-compatible GLM text and
vision models.

## Setup

Set a Z.ai API key:

```bash
export ZAI_API_KEY=<your-api-key>
```

The plugin uses `https://api.z.ai/api/paas/v4` by default. Set
`ZAI_BASE_URL`, or pass `option.WithBaseURL` through the plugin's `Opts`,
to use another compatible endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/zai"
)

ctx := context.Background()
plugin := &zai.ZAI{}
g := genkit.Init(ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("zai/glm-5.1"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain mixture-of-experts models."))
```

GLM's `reasoning_content` output is returned as Genkit reasoning parts and is
available through `response.Reasoning()`.

## Models

The registered catalog spans the GLM text line (`glm-5.1`, `glm-5-turbo`,
`glm-5`, the `glm-4.7` and `glm-4.5` families) and the GLM vision line
(`glm-5v-turbo`, `glm-4.6v`, `glm-4.5v` and variants). The catalog is not a
ceiling: any model ID Z.ai serves resolves on demand, and the `Models` field
describes or corrects any model, curated or not:

```go
plugin := &zai.ZAI{Models: map[string]ai.ModelOptions{
    "glm-6": {Label: "GLM 6", Supports: &compat_oai.Multimodal},
}}
```

Z.ai's API documentation is at https://docs.z.ai, and the current model list
is at https://docs.z.ai/guides/overview/pricing.

## Config

Models take a typed `zai.ChatConfig`: the generation fields Z.ai accepts plus
its own controls (`thinking`, `doSample`). `zai.ModelRef` carries the config
with the model ID:

```go
response, err := genkit.Generate(ctx, g,
    ai.WithModel(zai.ModelRef("glm-5.1", &zai.ChatConfig{
        Thinking: &zai.ThinkingConfig{Type: zai.ThinkingTypeDisabled},
    })),
    ai.WithPrompt("Answer concisely."),
)
```

Every config also carries the settings Genkit owns: `version` pins the exact
model version a request is served by, `apiKey` (settable only from Go code)
serves one request with a different credential, and `extra` forwards request
body fields the config does not declare, keyed by Z.ai's wire names (for
example `user_id`).

## Tool choice

Z.ai currently documents only automatic tool choice. Tool calling is
supported, but forced `required` and `none` modes are not advertised by this
plugin.

## Live tests

Live tests are skipped unless `ZAI_API_KEY` is set:

```bash
go test -race ./plugins/compat_oai/zai -run '^TestPluginLive$' -v -count=1
```
