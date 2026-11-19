# NVIDIA Plugin

This plugin provides Genkit support for models served by NVIDIA NIM through
its OpenAI-compatible API, on NVIDIA's hosted endpoint or a self-hosted NIM.

> **Building with a coding agent? Install the Genkit Go skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-go
> ```
>
> It teaches your agent the current Genkit Go APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

## Setup

Set an NVIDIA API key, created at https://build.nvidia.com:

```bash
export NVIDIA_API_KEY=<your-nvidia-api-key>
```

The plugin uses `https://integrate.api.nvidia.com/v1` by default; set
`NVIDIA_BASE_URL`, or pass `option.WithBaseURL` through the plugin's `Opts`,
to use a self-hosted NIM.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/nvidia"
)

ctx := context.Background()
plugin := &nvidia.NVIDIA{}
g := genkit.Init(ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain mixture-of-experts models."))
```

NIM model IDs are `publisher/model`, so a Genkit model name is the plugin's
`nvidia/` prefix followed by the full NIM ID: NVIDIA's own models read
`nvidia/nvidia/...`, and another publisher's read, for example,
`nvidia/google/...`. `nvidia.ModelRef` accepts the NIM ID with or without the
plugin prefix.

NIM's `reasoning_content` output is returned as Genkit reasoning parts and is
available through `response.Reasoning()`. Reasoning parts are also preserved
as `reasoning_content` during multi-turn and tool-call requests.

## Models

`nvidia/nemotron-3.5-super-vl-120b-a12b`, `nvidia/nemotron-3.5-lightning-30b-a3b`,
and `nvidia/nemotron-3.5-bolt-e4b` are registered. All three support tools,
tool-choice steering (`ai.WithToolChoice`), and natively constrained structured
output; image input is advertised for Super VL and Bolt only, since Lightning
is text-only. The catalog is not a ceiling: any model ID the endpoint serves
resolves on demand as a text chat model with tools, and the `Models` field
describes or corrects any model, curated or not:

```go
plugin := &nvidia.NVIDIA{Models: map[string]ai.ModelOptions{
    "google/gemma-3-12b-it": {Label: "Gemma 3 12B", Supports: &compat_oai.Multimodal},
}}
```

The current model catalog is at https://build.nvidia.com/models, and the API
reference at https://docs.api.nvidia.com/nim/reference/llm-apis.

## Config

Models take a typed `nvidia.ChatConfig`: the generation fields NIM's chat
completions API accepts (`temperature`, `topP`, `maxOutputTokens`,
`stopSequences`, `frequencyPenalty`, `presencePenalty`, and `seed`).
`nvidia.ModelRef` carries the config with the model ID:

```go
response, err := genkit.Generate(ctx, g,
    ai.WithModel(nvidia.ModelRef("nvidia/nemotron-3.5-lightning-30b-a3b", &nvidia.ChatConfig{
        MaxOutputTokens: 1024,
    })),
    ai.WithPrompt("Answer concisely."),
)
```

Accepted ranges for `temperature` and `maxOutputTokens` vary by model, so the
config enforces only the limits common to every NIM model.
`maxOutputTokens` reaches NIM as `max_tokens`.

Every config also carries the settings Genkit owns: `version` pins the exact
model version a request is served by, `apiKey` (settable only from Go code)
serves one request with a different credential, and `extra` forwards request
body fields the config does not declare, keyed by NIM's wire names. For
example, Nemotron reasoning can be turned off for one request:

```go
ai.WithConfig(map[string]any{
    "extra": map[string]any{
        "chat_template_kwargs": map[string]any{"enable_thinking": false},
    },
})
```

## Live tests

Live tests are skipped unless `GENKIT_LIVE=1` and `NVIDIA_API_KEY` are set:

```bash
GENKIT_LIVE=1 go test -v ./plugins/compat_oai/nvidia
```
