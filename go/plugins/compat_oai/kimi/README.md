# Kimi Plugin

This plugin provides Genkit support for Moonshot AI's OpenAI-compatible Kimi
models.

## Setup

Set a Moonshot API key:

```bash
export KIMI_API_KEY=<your-api-key>
```

`MOONSHOT_API_KEY` is also accepted. The plugin uses
`https://api.moonshot.ai/v1` by default; set `KIMI_BASE_URL` or
`MOONSHOT_BASE_URL`, or pass `option.WithBaseURL` through the plugin's
`Opts`, to use another Moonshot-compatible endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/kimi"
)

ctx := context.Background()
plugin := &kimi.Kimi{}
g := genkit.Init(ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("kimi/kimi-k3"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain mixture-of-experts models."))
```

Kimi's `reasoning_content` output is returned as Genkit reasoning parts and is
available through `response.Reasoning()`. Reasoning parts are also preserved
as `reasoning_content` during multi-turn and tool-call requests.

## Models

`kimi-k3`, `kimi-k2.6`, `kimi-k2.7-code`, and `kimi-k2.7-code-highspeed` are
registered, with `kimi-k2.5` kept as deprecated for existing users during its
platform sunset period. Tool-choice steering (`ai.WithToolChoice`) is
advertised for `kimi-k3` only: the K2 generation rejects a forced tool call
as incompatible with thinking, which is on by default. The catalog is not a
ceiling: any model ID Moonshot serves resolves on demand, and the `Models`
field describes or corrects any model, curated or not:

```go
plugin := &kimi.Kimi{Models: map[string]ai.ModelOptions{
    "kimi-k4": {Label: "Kimi K4", Supports: &compat_oai.Multimodal},
}}
```

Moonshot's API documentation, including the current model list, is at
https://platform.kimi.ai/docs.

## Config

Models take a typed `kimi.ChatConfig`: the generation fields the K-series
accepts plus the Kimi-specific controls (`thinking`, `reasoningEffort`).
`kimi.ModelRef` carries the config with the model ID. For example, Kimi K2.6
thinking can be disabled per request:

```go
response, err := genkit.Generate(ctx, g,
    ai.WithModel(kimi.ModelRef("kimi-k2.6", &kimi.ChatConfig{
        Thinking: &kimi.ThinkingConfig{Type: kimi.ThinkingTypeDisabled},
    })),
    ai.WithPrompt("Answer concisely."),
)
```

`reasoningEffort` controls how hard a Kimi K3 generation thinks (`low`,
`high`, or `max`, the default), and `thinking.keep` controls how much
reasoning is preserved across turns. Moonshot documents `temperature`,
`topP`, and the frequency and presence penalties for the legacy `moonshot-v1`
family only, so the config does not offer them, and `maxOutputTokens` reaches
Moonshot as `max_completion_tokens`.

Every config also carries the settings Genkit owns: `version` pins the exact
model version a request is served by, `apiKey` (settable only from Go code)
serves one request with a different credential, and `extra` forwards request
body fields the config does not declare, keyed by Moonshot's wire names.

## Live tests

Live tests are skipped unless `KIMI_API_KEY` or `MOONSHOT_API_KEY` is set:

```bash
go test -v ./plugins/compat_oai/kimi
```
