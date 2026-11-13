# OpenAI Plugin

This plugin provides Genkit support for OpenAI chat models and embedders
through the Chat Completions API.

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

Set an OpenAI API key:

```bash
export OPENAI_API_KEY=<your-api-key>
```

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    oai "github.com/firebase/genkit/go/plugins/compat_oai/openai"
)

ctx := context.Background()
plugin := &oai.OpenAI{}
g := genkit.Init(ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("openai/gpt-5.4"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Write a haiku about Go."))
```

Client options such as a different base URL or an organization ride the
plugin's `Opts` (`option.WithBaseURL`, `option.WithOrganization`, and the rest
of the SDK's request options).

## Models

The plugin registers a curated catalog spanning the GPT-6 line (`gpt-6-astra`,
`gpt-6-sol`, `gpt-6-luna`, `gpt-6.1-sol`, `gpt-6.1-terra`), the GPT-5 line
(`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`, and
earlier, plus `gpt-5-search-api`), `gpt-4.1`, `gpt-4o`, and `o3`. Models OpenAI has shut down are not
registered. Every catalog model with an announced shutdown date is marked
deprecated, its dated snapshots and fine-tunes included, so generating with
one logs a warning: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, and `o3`
(2026-12-11); `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, and `gpt-4o-mini`
(2027-01-14); `gpt-5.1` and `gpt-5.4-nano` (2027-04-01). Models outside the
catalog, such as `gpt-audio-mini` (shutdown 2027-01-20), resolve with the
generic defaults and are not marked deprecated, as do models OpenAI has
already shut down, such as `o4-mini`.

`gpt-6-astra`, `gpt-6.1-sol`, and `gpt-6.1-terra` call tools only through the
Responses API, so they advertise no tool support here. `gpt-5-search-api`
takes no tools either. The other `gpt-5.4` and later chat models (`gpt-5.4`,
`gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.5`, `gpt-5.6-sol`, `gpt-5.6-terra`,
`gpt-5.6-luna`, `gpt-6-sol`, and `gpt-6-luna`) advertise tool support, but they
call tools in Chat Completions only with `ReasoningEffort` set to `none`. The
`gpt-5.4` models and `gpt-5.5` default to `none`; the others default to a higher
effort, so a request with tools must set `none` or the API rejects it. Earlier
models (`gpt-5`, `gpt-5.1`, `gpt-5.2`, `o3`) take tools at any effort.

The catalog is not a ceiling: any model ID the API serves resolves on demand,
and the chat models the endpoint reports are listed dynamically. The listing
leaves out the image, speech (TTS), transcription, realtime, audio, embedding,
moderation, legacy completions, and Responses-only models OpenAI's endpoint
also reports. IDs that don't follow OpenAI's naming, as a proxy or self-hosted
server behind a custom base URL may report, are always listed. If the endpoint
cannot be listed, the plugin logs a warning. Use the `Models` field to describe or correct any
model, most often one released after this plugin:

```go
plugin := &oai.OpenAI{Models: map[string]ai.ModelOptions{
    "gpt-7": {Label: "GPT-7", Supports: &compat_oai.Multimodal},
}}
```

The current model list and capabilities are at
https://developers.openai.com/api/docs/models.

Models served only by the Responses API (the `-pro` variants) are not part of
this plugin, which speaks the Chat Completions API; naming one still resolves
it and fails at request time.

## Config

Models take the OpenAI SDK's own request type,
`openai.ChatCompletionNewParams`, with the SDK's wire names. `oai.ModelRef`
carries a config with the model ID:

```go
import "github.com/openai/openai-go"

response, err := genkit.Generate(ctx, g,
    ai.WithModel(oai.ModelRef("gpt-5.4", &openai.ChatCompletionNewParams{
        Temperature:         openai.Float(0.2),
        MaxCompletionTokens: openai.Int(1024),
    })),
    ai.WithPrompt("Answer concisely."),
)
```

The GPT-5 generation rejects the legacy `max_tokens`, so cap output with
`MaxCompletionTokens`; the older field remains for the models that still
read it.

Set the config's `Model` to a dated snapshot to pin the exact version a
request is served by. The fields Genkit builds from the request (`messages`,
`tools`, and their variants) are not config: a config naming one is rejected.

The advertised schema is reflected from the openai-go version your build
links, so a field OpenAI ships tomorrow becomes usable, and validated, by
bumping `github.com/openai/openai-go` in your own go.mod.

## Embedders

`text-embedding-3-large`, `text-embedding-3-small`, and
`text-embedding-ada-002` are registered, and the `Embedders` field overrides
what the plugin knows the way `Models` does. Embedders take a typed
`oai.TextEmbeddingConfig`:

```go
res, err := genkit.Embed(ctx, g,
    ai.WithEmbedder(oai.NewEmbedderRef("text-embedding-3-small", &oai.TextEmbeddingConfig{
        Dimensions: 256,
    })),
    ai.WithTextDocs("Genkit is an AI framework."),
)
```

The embedding config also carries the settings Genkit owns: `apiKey`
(settable only from Go code) serves one request with a different credential,
and `extra` forwards request body fields the config does not declare, keyed
by OpenAI's wire names.

## Live tests

Live tests are skipped unless `OPENAI_API_KEY` is set:

```bash
go test -v ./plugins/compat_oai/openai
```
