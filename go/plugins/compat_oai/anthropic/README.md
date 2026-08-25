# Anthropic Plugin

This plugin provides Genkit support for Claude models through Anthropic's
OpenAI-compatible Chat Completions endpoint.

Anthropic positions that endpoint as a compatibility layer for testing and
comparing Claude rather than a long-term integration surface: it does not
return thinking content, ignores `response_format`, and hoists system
messages to the start of the conversation. The full support matrix is at
https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk.
For the native Messages API surface (thinking output, prompt caching,
citations, structured outputs), use the `go/plugins/anthropic` plugin
instead; this one is for code that must speak the OpenAI shape.

## Setup

Set an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=<your-api-key>
```

The plugin's `APIKey` field overrides the environment. The plugin uses
`https://api.anthropic.com/v1` by default; set `ANTHROPIC_BASE_URL` or pass
`option.WithBaseURL` through the plugin's `Opts` to use another compatible
endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/anthropic"
)

ctx := context.Background()
plugin := &anthropic.Anthropic{}
g := genkit.Init(ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("anthropic/claude-haiku-4-5-20251001"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain constitutional AI."))
```

## Models

A catalog of dated Claude releases is registered at Init, and the catalog is
not a ceiling: any Claude model ID resolves on demand, and the plugin lists
what Anthropic's native models API reports (paged, authenticated with
`x-api-key`, since listing is not part of the compatible surface). Use the
`Models` field to describe or correct any model, most often one released
after this plugin:

```go
plugin := &anthropic.Anthropic{Models: map[string]ai.ModelOptions{
    "claude-opus-5": {Label: "Claude Opus 5", Supports: &compat_oai.Multimodal},
}}
```

The current model list is at
https://platform.claude.com/docs/en/about-claude/models/overview.

## Config

Models take a typed `anthropic.ChatConfig`: the fields the compatible
endpoint honors, and none of the OpenAI fields it documents as ignored (the
penalties, `logprobs`, `seed`). `temperature` runs 0 to 1, where the endpoint
caps it. `anthropic.ModelRef` carries the config with the model ID:

```go
response, err := genkit.Generate(ctx, g,
    ai.WithModel(anthropic.ModelRef("claude-sonnet-4-5-20250929", &anthropic.ChatConfig{
        MaxOutputTokens: 2048,
        Thinking:        &anthropic.ThinkingConfig{Type: "enabled", BudgetTokens: 2000},
    })),
    ai.WithPrompt("Work through this carefully."),
)
```

`thinking` spends a reasoning budget, but the compatible endpoint does not
return the thinking content itself; only the native API does. On Claude 5
models thinking is adaptive and on by default, which makes the manual control
a legacy mode.

Every config also carries the settings Genkit owns: `version` pins the exact
model version a request is served by, `apiKey` (settable only from Go code)
serves one request with a different credential, and `extra` forwards request
body fields the config does not declare, keyed by Anthropic's wire names.

## Live tests

Live tests are skipped unless `ANTHROPIC_API_KEY` is set:

```bash
go test -v ./plugins/compat_oai/anthropic
```
