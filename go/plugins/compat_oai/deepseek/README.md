# DeepSeek Plugin

This plugin provides Genkit support for DeepSeek's OpenAI-compatible models,
DeepSeek V4 Flash and DeepSeek V4 Pro.

## Setup

Set a DeepSeek API key:

```bash
export DEEPSEEK_API_KEY=<your-api-key>
```

The plugin uses `https://api.deepseek.com` by default. Set `DEEPSEEK_BASE_URL`,
or pass `option.WithBaseURL` through the plugin's `Opts`, to use another
compatible endpoint, such as DeepSeek's beta endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
)

ctx := context.Background()
plugin := &deepseek.DeepSeek{}
g := genkit.Init(ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("deepseek/deepseek-v4-flash"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain mixture-of-experts models."))
```

DeepSeek's `reasoning_content` output is returned as Genkit reasoning parts and
is available through `response.Reasoning()`.

## Models

`deepseek-v4-flash` and `deepseek-v4-pro` are registered. The catalog is not
a ceiling: any model ID DeepSeek serves resolves on demand, and the `Models`
field describes or corrects any model, curated or not. The current model list
and pricing, including the dated snapshots a config `version` can pin, are at
https://api-docs.deepseek.com/quick_start/pricing, and the API reference is
at https://api-docs.deepseek.com.

## Config

Models take a typed `deepseek.ChatConfig`: the generation fields DeepSeek
accepts plus its thinking controls, `thinking` for the mode and
`reasoningEffort` (`low`, `high`, or `max`) for the depth.
`deepseek.ModelRef` carries the config with the model ID. Thinking is on by
default, so turning it off is the common case:

```go
response, err := genkit.Generate(ctx, g,
    ai.WithModel(deepseek.ModelRef("deepseek-v4-flash", &deepseek.ChatConfig{
        Thinking: &deepseek.ThinkingConfig{Type: deepseek.ThinkingTypeDisabled},
    })),
    ai.WithPrompt("Answer concisely."),
)
```

`maxOutputTokens` reaches DeepSeek as the `max_tokens` it reads. DeepSeek no
longer supports the frequency and presence penalties, so the config does not
offer them.

Every config also carries the settings Genkit owns: `version` pins the exact
model version a request is served by, `apiKey` (settable only from Go code)
serves one request with a different credential, and `extra` forwards request
body fields the config does not declare, keyed by DeepSeek's wire names (for
example `logprobs`).

## Prompt caching

DeepSeek caches prompt prefixes automatically and bills a cache hit at a lower
rate; the hits come back as `response.Usage.CachedContentTokens`. Set `userId`
to partition that cache per end user, so users neither read nor evict each
other's cached prefixes:

```go
response, err := genkit.Generate(ctx, g,
    ai.WithModel(deepseek.ModelRef("deepseek-v4-flash", &deepseek.ChatConfig{
        UserID: "tenant-42",
    })),
    ai.WithPrompt("Answer concisely."),
)
```

## Not supported

The V4 models are text-only, so image inputs are not advertised. DeepSeek's
beta features (chat prefix completion and FIM) are not exposed by this plugin.

## Live tests

Live tests are skipped unless `DEEPSEEK_API_KEY` is set:

```bash
go test -race ./plugins/compat_oai/deepseek -run '^TestPluginLive$' -v -count=1
```
