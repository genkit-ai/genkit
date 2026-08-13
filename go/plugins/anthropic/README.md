# Anthropic Plugin

This plugin provides Genkit support for Claude models through Anthropic's
native Messages API, using the official Go SDK,
`github.com/anthropics/anthropic-sdk-go`.

Unlike the OpenAI-compatible `compat_oai/anthropic` plugin, this one speaks
the Messages API itself: thinking content comes back as Genkit reasoning
parts with their signatures preserved across turns, structured output uses
Anthropic's native support, and Anthropic's server-side tools are reachable.

## Setup

Set an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=<your-api-key>
```

`ANTHROPIC_BASE_URL` overrides the endpoint; both settings can also be set as
plugin struct fields.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/anthropic"
)

ctx := context.Background()
g := genkit.Init(
    ctx,
    genkit.WithPlugins(&anthropic.Anthropic{}),
    genkit.WithDefaultModel("anthropic/claude-sonnet-4-5"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain constitutional AI."))
```

Claude's thinking output is returned as Genkit reasoning parts, available
through `response.Reasoning()`, and each block's signature rides along so
multi-turn conversations replay their thinking intact.

## Models

Init registers nothing: every Claude model resolves on first use, and the
plugin lists Anthropic's current catalog through the models API, cached for an
hour. Curated entries describe the current families (`claude-fable-5`,
`claude-opus-5`, `claude-sonnet-5`, the Claude 4 line, `claude-haiku-4-5`),
dated snapshots such as `claude-sonnet-4-5-20250929` resolve to the same
descriptions, and the `Models` field describes or corrects any model, most
often one released after this plugin:

```go
plugin := &anthropic.Anthropic{Models: map[string]ai.ModelOptions{
    "claude-opus-6": {Label: "Claude Opus 6"},
}}
```

The current model list is at
https://platform.claude.com/docs/en/about-claude/models/overview.

## Config

Models take the SDK's own request type, `anthropic.MessageNewParams`, with
Anthropic's wire names. This package and the SDK are both named `anthropic`,
so alias one of them:

```go
import (
    sdk "github.com/anthropics/anthropic-sdk-go"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/anthropic"
)

response, err := genkit.Generate(ctx, g,
    ai.WithModel(anthropic.ModelRef("claude-opus-4-5", &sdk.MessageNewParams{
        MaxTokens: 1024,
    })),
    ai.WithPrompt("Answer concisely."),
)
```

The fields Genkit builds from the request are not config: a config carrying
`messages`, `system`, `model`, an output format, or a custom function tool is
rejected with an error naming the Genkit option to use instead
(`ai.WithMessages`, `ai.WithSystem`, `ai.WithModel`, `ai.WithOutputType`,
`ai.WithTools`). The config-level `tools` field stays available for
Anthropic's server-side tools, such as web search and code execution.

The advertised schema is reflected from the anthropic-sdk-go version your
build links, so a field Anthropic ships tomorrow becomes usable, and
validated, by bumping the SDK in your own go.mod.

The API reference is at https://platform.claude.com/docs/en/api/overview.

## Live tests

Live tests are skipped unless `ANTHROPIC_API_KEY` is set:

```bash
go test -v ./plugins/anthropic
```
