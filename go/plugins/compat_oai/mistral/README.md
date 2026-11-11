# Mistral Plugin

This plugin provides Genkit support for Mistral AI's models through Mistral's
OpenAI-compatible chat completions and embeddings endpoints.

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

Set a Mistral API key:

```bash
export MISTRAL_API_KEY=<your-api-key>
```

The plugin uses `https://api.mistral.ai/v1` by default. Set `MISTRAL_BASE_URL`,
or pass `option.WithBaseURL` through the plugin's `Opts`, to use another
compatible endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/mistral"
)

ctx := context.Background()
g := genkit.Init(ctx,
    genkit.WithPlugins(&mistral.Mistral{}),
    genkit.WithDefaultModel("mistral/mistral-small-latest"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain mixture-of-experts models."))
```

## Models

These chat models are registered, by the aliases Mistral moves to each new
snapshot:

| Model | Input |
| --- | --- |
| `mistral-large-latest` | text, images |
| `mistral-large-4` (public preview) | text, images |
| `mistral-medium-latest` | text, images |
| `mistral-small-latest` | text, images |
| `ministral-8b-latest` | text, images |
| `ministral-3b-latest` | text, images |
| `codestral-latest` | text |

All of them call tools and constrain JSON output to a schema. Whether Mistral
honors the two together is not confirmed, so a structured-output request that
carries tools sends its schema as prompt instructions rather than as a
`json_schema` response format. Magistral, Pixtral and Devstral are left out because
Mistral has deprecated or retired them.

The catalog is not a ceiling: any chat model Mistral serves resolves on
demand, a config `version` pins a dated snapshot such as
`mistral-small-2603`, and the `Models` field describes or corrects any model,
curated or not. A model the plugin does not curate is described
conservatively, as text only and with no constrained output (a schema goes
out as prompt instructions with JSON mode on), since it may predate either
feature; a `Models` entry turns on what it supports. The model list is at
https://docs.mistral.ai/getting-started/models and the API reference at
https://docs.mistral.ai/api.

## Config

Models take a typed `mistral.ChatConfig`, carried with the model ID by
`mistral.ModelRef`:

```go
response, err := genkit.Generate(ctx, g,
    ai.WithModel(mistral.ModelRef("mistral-medium-latest", &mistral.ChatConfig{
        ReasoningEffort: mistral.ReasoningEffortHigh,
        MaxOutputTokens: 1024,
    })),
    ai.WithPrompt("How many r's are in strawberry?"),
)
```

Mistral's request schema is closed: a field it does not declare fails the
request. The config follows it, so three fields differ from OpenAI's.
`maxOutputTokens` is sent as `max_tokens`, `seed` as `random_seed`, and there
is no `user`. The Mistral-specific fields are `safePrompt` (`safe_prompt`),
`promptCacheKey` (`prompt_cache_key`) and `serviceTier` (`auto` or
`standard_only`).

Every config also carries the settings Genkit owns: `version` pins the exact
model version a request is served by, `apiKey` (settable only from Go code)
serves one request with a different credential, and `extra` forwards request
body fields the config does not declare, keyed by Mistral's wire names (for
example `min_tokens`).

## Reasoning

`reasoningEffort` turns reasoning on for the models that take it. Each model
takes a subset of the levels: Mistral Small and Medium take `none` and `high`
and reason only when asked; Mistral Large 4 takes `low`, `medium` and `high`
and always reasons. Mistral's API reference also lists `minimal`, `xhigh` and
`max`, which no current model takes; the config leaves them out, and `extra`
can still send one. Mistral returns the reasoning as a `thinking` chunk in the
message content, which the plugin turns into Genkit reasoning parts, available
through `response.Reasoning()`. When the conversation goes back to Mistral, as
in a tool loop, the reasoning is sent back as the same `thinking` chunk.

## Streaming

Mistral rejects the `stream_options` field the plugin sends other providers,
and reports usage on the final chunk of every stream without it, so streaming
requests leave it out.

## Embedders

`mistral-embed` (1024 dimensions) and `codestral-embed` (1536 by default, up
to 3072) are registered; other embedders do not resolve on demand. They take a
`mistral.EmbedConfig`, carried by `mistral.EmbedderRef`. `dimensions` is sent
as Mistral's `output_dimension`, for models that let you choose it:

```go
res, err := genkit.Embed(ctx, g,
    ai.WithEmbedder(mistral.EmbedderRef("codestral-embed", &mistral.EmbedConfig{
        Dimensions: 256,
    })),
    ai.WithTextDocs("func main() {}"),
)
```

## Not supported

PDF input is not sent: Mistral reads documents as `document_url` chunks
rather than OpenAI's `file` parts, so the models accept images only. Mistral's
FIM, OCR, moderation, and audio endpoints are not exposed by this plugin, and
`ListActions` lists only the models whose card reports chat completion.

## Live tests

Live tests are skipped unless `GENKIT_LIVE` and `MISTRAL_API_KEY` are both
set (`GENKIT_LIVE=all` adds the expensive cases):

```bash
GENKIT_LIVE=1 go test -race ./plugins/compat_oai/mistral -run '^TestPluginLive$' -v -count=1
```
