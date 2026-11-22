# OrcaRouter Plugin

This plugin provides Genkit support for [OrcaRouter](https://www.orcarouter.ai),
a gateway that serves models from many vendors behind one OpenAI-compatible
endpoint.

## Setup

Set an OrcaRouter API key:

```bash
export ORCAROUTER_API_KEY=<your-api-key>
```

The plugin uses `https://api.orcarouter.ai/v1` by default. Set
`ORCAROUTER_BASE_URL`, or pass `option.WithBaseURL` through the plugin's
`Opts`, to use another compatible endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/orcarouter"
)

ctx := context.Background()
g := genkit.Init(ctx,
    genkit.WithPlugins(&orcarouter.OrcaRouter{}),
    genkit.WithDefaultModel("orcarouter/anthropic/claude-sonnet-4.5"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain reinforcement learning."))
```

Reasoning output is returned as Genkit reasoning parts and is available
through `response.Reasoning()` when the model emits it.

## Models

The plugin registers no models and carries no catalog. OrcaRouter serves
hundreds of models and adds more regularly, so every model resolves on demand
instead:

```go
ai.WithModelName("orcarouter/openai/gpt-4o")
ai.WithModelName("orcarouter/anthropic/claude-sonnet-4.5")
ai.WithModelName("orcarouter/deepseek/deepseek-v4-flash")
```

A model ID keeps its upstream vendor's prefix, so a Genkit action name carries
two slashes: the first is this plugin's provider prefix and the rest is the
ID OrcaRouter serves.

For the same reason, the plugin advertises nothing to the Dev UI's model list.
Models stay usable by name; only the browsable catalog is absent.

Every resolved model is described permissively: multi-turn, tools, tool
choice, system role, and media are all claimed. This is deliberate. A
capability declared too narrow is refused by Genkit before the request is
sent, which blocks a model that would have worked, while one declared too wide
reaches OrcaRouter and comes back with the real reason. Native constrained
output is the exception, left unclaimed so that structured output falls back
to schema instructions in the prompt, which every model handles and which
returns the same typed result.

Correct a model whose real capabilities are narrower through `Models`:

```go
plugin := &orcarouter.OrcaRouter{Models: map[string]ai.ModelOptions{
    // A text-only model, so Genkit refuses media locally rather than paying
    // for the upstream rejection.
    "mistralai/mistral-7b-instruct": {Supports: &ai.ModelSupports{
        Multiturn: true, Tools: true, SystemRole: true,
    }},
}}
```

Fields left at their zero value keep what the plugin resolves, so an entry can
pin one capability without restating the rest.

The current model list is at https://www.orcarouter.ai/models, and the API
reference is at https://docs.orcarouter.ai.

## Config

Models take a typed `orcarouter.ChatConfig` covering the sampling fields the
OpenAI-compatible surface OrcaRouter exposes. `orcarouter.ModelRef` carries
the config with the model ID:

```go
response, err := genkit.Generate(ctx, g,
    ai.WithModel(orcarouter.ModelRef("deepseek/deepseek-v4-flash", &orcarouter.ChatConfig{
        MaxOutputTokens: 512,
        ReasoningEffort: orcarouter.ReasoningEffortLow,
    })),
    ai.WithPrompt("Work through this step by step."),
)
```

- `temperature`, `topP`, `maxOutputTokens`, `stopSequences`,
  `frequencyPenalty`, `presencePenalty`, `seed`, `logProbs`, `topLogProbs`,
  and `parallelToolCalls` are the standard OpenAI-compatible sampling knobs.
- `reasoningEffort` sets how hard a reasoning-capable model thinks: `low`,
  `medium`, or `high`, sent as `reasoning_effort`.
- `user` identifies the end user a request is made for.

`maxOutputTokens` reaches OrcaRouter as `max_tokens`, which reasoning models
spend on thinking before they emit anything visible. Leave it unset, or budget
for the thinking as well, on any model that reasons.

Every config also carries the settings Genkit owns: `version` pins the exact
model version a request is served by, `apiKey` (settable only from Go code)
serves one request with a different credential, and `extra` forwards request
body fields the config does not declare, keyed by OrcaRouter's wire names.

## Not supported

This plugin serves the chat completions endpoint only.

One request field is left out on purpose. `n` asks for several completion
choices and bills for all of them while Genkit reads only the first. Anything
else undeclared still reaches the wire through `extra`.

## Live tests

Live tests are skipped unless `ORCAROUTER_API_KEY` is set:

```bash
go test -race ./plugins/compat_oai/orcarouter -run '^TestPluginLive$' -v -count=1
```

They spend on ordinary catalog models named at the top of
`orcarouter_live_test.go`. Swap in whatever the key has credit for.
