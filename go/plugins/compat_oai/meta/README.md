# Meta Plugin

This plugin provides Genkit support for Meta's Muse models, including Muse
Wonder, through the OpenAI-compatible chat completions endpoint of the Meta
Model API.

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

Set a Meta Model API key:

```bash
export META_API_KEY=<your-api-key>
```

Meta's own examples use `MODEL_API_KEY`. The plugin does not read it, because
a variable with so generic a name can hold another provider's key, which would
then go to Meta. The plugin uses `https://api.meta.ai/v1` by default; set
`META_BASE_URL`, or pass `option.WithBaseURL` through the plugin's `Opts`, to
use another endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/meta"
)

ctx := context.Background()
g := genkit.Init(ctx,
    genkit.WithPlugins(&meta.Meta{}),
    genkit.WithDefaultModel("meta/muse-wonder-1.0"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Plan a three-step refactor."))
```

## Models

`muse-wonder-1.0` (in preview, registered as unstable) and `muse-spark-1.3`,
`muse-spark-1.2`, and `muse-spark-1.1` are registered, along with
`muse-spark-1.3-contributor` and `muse-spark-1.2-contributor`. The
contributor variants cost less because Meta trains on the prompts, files and
responses sent to them. Muse Wonder Full is served in Muse Code only, not by
the Model API.

Every model reads text, images, WAV and MP3 audio, and PDFs, and calls tools.
Meta also reads MP4 video, which the plugin does not send yet. Tool-choice
steering (`ai.WithToolChoice`) is not advertised: Meta accepts only `auto` and
answers `none`, `required`, or a named tool with a 400.

The catalog is not a ceiling: any model ID the Model API serves resolves on
demand, and the `Models` field describes or corrects any model, curated or
not:

```go
plugin := &meta.Meta{Models: map[string]ai.ModelOptions{
    "muse-wonder-1.0": {Stage: ai.ModelStageStable},
}}
```

Meta's API documentation, including the current model list, is at
https://dev.meta.ai/docs/models.

## Config

Models take a typed `meta.ChatConfig`, and `meta.ModelRef` carries it with the
model ID:

```go
response, err := genkit.Generate(ctx, g,
    ai.WithModel(meta.ModelRef("muse-wonder-1.0", &meta.ChatConfig{
        ReasoningEffort: meta.ReasoningEffortHigh,
    })),
    ai.WithPrompt("Find the bug in this function."),
)
```

Muse models always reason. `reasoningEffort` sets the depth from `minimal` to
`max`; unset, the model picks it. `maxOutputTokens` reaches Meta as
`max_completion_tokens` and counts reasoning tokens too, so a small value can
end a response before any visible text. The config also offers `temperature`,
`topP`, the frequency and presence penalties, `seed`, `parallelToolCalls`, the
prompt cache controls (`promptCacheKey`, `promptCacheRetention`), and
`safetyIdentifier`. It does not offer `stopSequences` or `logProbs`, which
Meta rejects.

Every config also carries the settings Genkit owns: `version` pins the exact
model version a request is served by, `apiKey` (settable only from Go code)
serves one request with a different credential, and `extra` forwards request
body fields the config does not declare, keyed by Meta's wire names.

## Reasoning across turns

Meta's chat completions do not return the model's reasoning: the
`reasoning_content` field comes back empty, so responses carry no reasoning
parts. Each turn also reasons from scratch, because chat completions cannot
carry reasoning from one turn to the next. Meta recommends its Responses API
for long agent loops for that reason; this plugin speaks chat completions
only.

## Live tests

Live tests are skipped unless `META_API_KEY` is set:

```bash
go test -v ./plugins/compat_oai/meta
```
