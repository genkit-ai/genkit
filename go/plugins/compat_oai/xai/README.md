# xAI Plugin

This plugin provides Genkit support for xAI's OpenAI-compatible Grok models.

## Setup

Set an xAI API key:

```bash
export XAI_API_KEY=<your-api-key>
```

The plugin uses `https://api.x.ai/v1` by default. Set `XAI_BASE_URL`, or pass
`option.WithBaseURL` through the plugin's `Opts`, to use another compatible
endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/xai"
)

ctx := context.Background()
plugin := &xai.XAI{}
g := genkit.Init(ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("xai/grok-4.5"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain reinforcement learning."))
```

Grok's `reasoning_content` output is returned as Genkit reasoning parts and is
available through `response.Reasoning()`.

## Models

Grok 4.6, Grok 4.5, Grok 4.3, the Grok 4.20 line, and the Grok Build coding
model are registered. The catalog is not a ceiling: any model ID xAI serves
resolves on demand, and the `Models` field describes or corrects any model,
curated or not:

```go
plugin := &xai.XAI{Models: map[string]ai.ModelOptions{
    "grok-5": {Label: "Grok 5", Supports: &compat_oai.Multimodal},
}}
```

The current model list is at https://docs.x.ai/docs/models, and the API
reference is at https://docs.x.ai.

## Config

Models take a typed `xai.ChatConfig`: the generation fields xAI accepts plus
its own controls (`reasoningEffort`, `serviceTier`, `promptCacheKey`).
`xai.ModelRef` carries the config with the model ID:

```go
response, err := genkit.Generate(ctx, g,
    ai.WithModel(xai.ModelRef("grok-4.6", &xai.ChatConfig{
        ReasoningEffort: xai.ReasoningEffortHigh,
    })),
    ai.WithPrompt("Work through this step by step."),
)
```

`reasoningEffort` runs from `none` through `xhigh`. Which levels a model takes
is xAI's to decide: `none` is documented for the chat completions endpoint and
`xhigh` for Grok 4.6, so a level a model does not accept comes back as an error
from xAI.

`maxOutputTokens` reaches xAI as `max_completion_tokens`, since xAI deprecated
`max_tokens`, and `topLogProbs` accepts 0 through 8 rather than OpenAI's 20.
Streamed generations report token usage, including reasoning tokens, the same
way non-streamed ones do.

Every config also carries the settings Genkit owns: `version` pins the exact
model version a request is served by, `apiKey` (settable only from Go code)
serves one request with a different credential, and `extra` forwards request
body fields the config does not declare, keyed by xAI's wire names.

## Not supported

xAI's image, video, and voice models are not part of this plugin: it serves the
chat completions endpoint only.

Search is not reachable through this plugin. xAI retired the chat completion
endpoint's `search_parameters` (the API answers any request carrying it with
410 Gone), and its successors, the `web_search` and `x_search` server-side
tools, are Responses API features, while chat completions accepts only
function tools.

Two request fields are left out on purpose. `n` asks for several completion
choices and bills for all of them while Genkit reads only the first, and
`deferred` answers with a request ID to poll rather than a completion.

## Live tests

Live tests are skipped unless `XAI_API_KEY` is set:

```bash
go test -race ./plugins/compat_oai/xai -run '^TestPluginLive$' -v -count=1
```
