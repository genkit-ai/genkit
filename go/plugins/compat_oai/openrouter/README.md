# OpenRouter Plugin

This plugin provides Genkit support for [OpenRouter](https://openrouter.ai), a
gateway that serves models from many vendors behind one OpenAI-compatible
endpoint.

## Setup

Set an OpenRouter API key:

```bash
export OPENROUTER_API_KEY=<your-api-key>
```

The plugin uses `https://openrouter.ai/api/v1` by default. Set
`OPENROUTER_BASE_URL`, or pass `option.WithBaseURL` through the plugin's
`Opts`, to use another compatible endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/openrouter"
)

ctx := context.Background()
g := genkit.Init(ctx,
    genkit.WithPlugins(&openrouter.OpenRouter{}),
    genkit.WithDefaultModel("openrouter/anthropic/claude-sonnet-4.5"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain reinforcement learning."))
```

`SiteURL` and `AppName` are optional. They name the application on
OpenRouter's public rankings, sent as the `HTTP-Referer` and `X-Title`
headers, and change nothing else about a request.

Reasoning output is returned as Genkit reasoning parts and is available
through `response.Reasoning()`.

## Models

The plugin registers no models and carries no catalog. OpenRouter serves
hundreds of models and adds more weekly, so every model resolves on demand
instead:

```go
ai.WithModelName("openrouter/openai/gpt-5")
ai.WithModelName("openrouter/anthropic/claude-sonnet-4.5")
ai.WithModelName("openrouter/meta-llama/llama-4-70b-instruct:free")
```

A model ID keeps its upstream vendor's prefix, so a Genkit action name carries
two slashes: the first is this plugin's provider prefix and the rest is the
ID OpenRouter serves. Variant suffixes work the same way, including `:free`,
`:nitro` for the fastest provider, and `:floor` for the cheapest.

For the same reason, the plugin advertises nothing to the Dev UI's model list.
Models stay usable by name; only the browsable catalog is absent.

Every resolved model is described permissively: multi-turn, tools, tool
choice, system role, and media are all claimed. This is deliberate. A
capability declared too narrow is refused by Genkit before the request is
sent, which blocks a model that would have worked, while one declared too wide
reaches OpenRouter and comes back with the real reason. Native constrained
output is the exception, left unclaimed so that structured output falls back
to schema instructions in the prompt, which every model handles and which
returns the same typed result.

Correct a model whose real capabilities are narrower through `Models`:

```go
plugin := &openrouter.OpenRouter{Models: map[string]ai.ModelOptions{
    // A text-only model, so Genkit refuses media locally rather than paying
    // for the upstream rejection.
    "mistralai/mistral-7b-instruct": {Supports: &ai.ModelSupports{
        Multiturn: true, Tools: true, SystemRole: true,
    }},
}}
```

Fields left at their zero value keep what the plugin resolves, so an entry can
pin one capability without restating the rest.

The current model list is at https://openrouter.ai/models, and the API
reference is at https://openrouter.ai/docs.

## Config

Models take a typed `openrouter.ChatConfig`: the generation fields OpenRouter
normalizes across vendors, plus the gateway controls that are the reason to
route through it. `openrouter.ModelRef` carries the config with the model ID:

```go
response, err := genkit.Generate(ctx, g,
    ai.WithModel(openrouter.ModelRef("openai/gpt-5", &openrouter.ChatConfig{
        Provider: &openrouter.ProviderRouting{
            Sort:           openrouter.ProviderSortPrice,
            DataCollection: openrouter.DataCollectionDeny,
        },
        Models:    []string{"anthropic/claude-sonnet-4.5"},
        Reasoning: &openrouter.ReasoningConfig{Effort: openrouter.ReasoningEffortHigh},
    })),
    ai.WithPrompt("Work through this step by step."),
)
```

- `provider` chooses which upstream providers may serve the request: `order`,
  `only`, `ignore`, `sort` by price, throughput, or latency, `maxPrice`,
  `dataCollection`, `zdr`, `requireParameters`, and `quantizations`.
- `models` is a fallback chain, tried in order when the requested model is
  unavailable, rate-limited, or refuses.
- `reasoning` sets `effort` or an exact `maxTokens` budget, with `exclude` to
  reason without returning the reasoning.
- `plugins` enables OpenRouter's request plugins, such as web search. They are
  sent verbatim rather than typed, since the roster and each plugin's options
  change on OpenRouter's schedule:
  `Plugins: []map[string]any{{"id": "web", "max_results": 3}}`.
- `transforms`, `sessionId`, `serviceTier`, and `metadata` cover context
  compression, sticky routing, scheduling, and request tagging.
- `topK`, `minP`, `topA`, and `repetitionPenalty` are the sampling knobs the
  OpenAI schema has no home for.

`maxOutputTokens` reaches OpenRouter as `max_tokens`, which reasoning models
spend on thinking before they emit anything visible. A budget that is
comfortable for a plain model can be consumed entirely by reasoning, leaving
an empty answer and a `length` finish reason. Leave it unset, or budget for
the thinking as well, on any model that reasons.

Every config also carries the settings Genkit owns: `version` pins the exact
model version a request is served by, `apiKey` (settable only from Go code)
serves one request with a different credential, and `extra` forwards request
body fields the config does not declare, keyed by OpenRouter's wire names.

`extra` is also the way to reach the request shapes this config does not
declare, such as the object form of `provider.sort` or the per-percentile form
of the throughput and latency preferences. An extra wins over the field it
collides with, so it replaces the whole declared object:

```go
&openrouter.ChatConfig{RequestConfig: compat_oai.RequestConfig{
    Extra: map[string]any{
        "provider": map[string]any{"sort": map[string]any{"by": "price", "partition": "model"}},
    },
}}
```

## Usage and failures

OpenRouter prices every request it routes and reports what it charged. The
amount arrives as `cost` on the response's usage, in credits:

```go
response.Usage.Custom["cost"]
```

It needs no request field. OpenRouter used to gate this behind
`usage: {include: true}`, which is now deprecated and does nothing; the full
accounting is returned either way, on streaming and non-streaming responses
alike. A `:free` model reports an explicit cost of `0`, which is kept: the
key's presence says the request was priced.

A provider that fails part-way through a generation is a case worth handling.
OpenRouter has already sent a 200 by then, so it cannot answer with an error
status. It returns the text produced before the failure, a finish reason of
`other`, and the reason in `response.FinishMessage`. Check the finish reason
before trusting a short answer:

```go
if response.FinishReason == ai.FinishReasonOther && response.FinishMessage != "" {
    // Upstream failed mid-generation. response.Text() is partial.
}
```

The error object also rides whole on the response metadata, as
`response.Raw.(map[string]any)["error"]`. It carries what the message alone
does not: the failure's code, and metadata naming the upstream provider that
failed, which is what `provider.ignore` takes to route the retry around it.

A request that fails before any output returns an ordinary error instead, as
does a stream that breaks part-way through.

## Not supported

This plugin serves the chat completions endpoint only. OpenRouter's image and
audio generation endpoints are not part of it.

Three request fields are left out on purpose. `n` asks for several completion
choices and bills for all of them while Genkit reads only the first, and
`route` and `usage` are deprecated by OpenRouter and have no effect. Anything
else undeclared still reaches the wire through `extra`.

## Live tests

Live tests are skipped unless `OPENROUTER_API_KEY` is set:

```bash
go test -race ./plugins/compat_oai/openrouter -run '^TestPluginLive$' -v -count=1
```

They spend on ordinary catalog models named at the top of
`openrouter_live_test.go`. Swap in whatever the key has credit for.
