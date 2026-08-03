# Groq Plugin

This plugin provides Genkit support for [Groq](https://groq.com)'s
OpenAI-compatible chat models (Llama, GPT-OSS, Compound, Qwen, and more).

## Usage

Set a Groq API key:

```bash
export GROQ_API_KEY=<your-api-key>
```

The plugin uses `https://api.groq.com/openai/v1` by default. Set `GROQ_BASE_URL`
to point at another Groq-compatible endpoint.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/groq"
)

ctx := context.Background()
plugin := &groq.Groq{}
g := genkit.Init(
    ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("groq/"+groq.ModelLlama3370bVersatile),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain mixture-of-experts models."))
```

Groq-specific request fields are passed through when configuration is supplied
as a map (snake_case OpenAI-compatible extras):

```go
response, err := genkit.Generate(
    ctx,
    g,
    ai.WithModelName("groq/"+groq.ModelGPTOss120b),
    ai.WithPrompt("Reason carefully."),
    ai.WithConfig(map[string]any{
        "reasoning_effort":  "high",
        "reasoning_format":  "parsed",
        "include_reasoning": true,
        "service_tier":      "on_demand",
    }),
)
```

Built-in models mirror the JS `@genkit-ai/compat-oai/groq` table, including
`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `openai/gpt-oss-120b`,
`openai/gpt-oss-20b`, `groq/compound`, `groq/compound-mini`, and
`qwen/qwen3.6-27b` (media-capable). Additional chat models from the Groq
endpoint can be resolved dynamically.

## Live tests

Live tests are skipped unless `GROQ_API_KEY` is set:

```bash
go test -v ./plugins/compat_oai/groq
```
