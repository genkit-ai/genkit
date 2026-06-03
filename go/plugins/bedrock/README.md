# Genkit Amazon Bedrock plugin (Go)

First-party Genkit plugin for Amazon Bedrock, covering:

- **Text generation** via the [Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html) — Claude 3/3.5/3.7/4, Nova micro/lite/pro/premier, Llama 3.x + 4, Mistral, AI21 Jamba, Cohere Command R/R+, DeepSeek r1, Writer Palmyra.
- **Streaming** via `ConverseStream` with full tool-use re-assembly.
- **Tool calling** with `auto` / `any` / specific-tool choice.
- **Embedders** via `InvokeModel` — Titan text + image (`amazon.titan-embed-text-*`, `amazon.titan-embed-image-v1`), Cohere text + image (`cohere.embed-english-v3`, `cohere.embed-multilingual-v3`, `cohere.embed-v4:0`), and Amazon Nova multimodal (`amazon.nova-2-multimodal-embeddings-v1:0`).
- **Reranking** via `bedrock.Rerank` — Cohere Rerank (`cohere.rerank-v3-5:0`).
- **Image generation** via `InvokeModel` — Titan Image, Nova Canvas, Stability Stable Diffusion.

```go
import "github.com/firebase/genkit/go/plugins/bedrock"
```

## Quick start

```go
g := genkit.Init(ctx,
    genkit.WithPlugins(&bedrock.Bedrock{Region: "us-east-1"}),
)

claude, _ := bedrock.DefineModel(g, "us.anthropic.claude-haiku-4-5-20251001-v1:0", nil)

resp, err := genkit.Generate(ctx, g,
    ai.WithModel(claude),
    ai.WithPrompt("Write a haiku about AWS Bedrock."),
    ai.WithConfig(&bedrock.Config{MaxTokens: 256}),
)
```

## AWS setup

The plugin uses the standard AWS credential chain via `config.LoadDefaultConfig`. Credentials resolve from (in order):

1. `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` env vars
2. `~/.aws/credentials` (named profile via `AWS_PROFILE`)
3. EC2 / ECS / EKS instance role
4. SSO via `aws sso login`
5. Web-identity / OIDC role assumption (e.g. GitHub Actions)

Region resolves from `Bedrock.Region` → `AWS_REGION` env → `~/.aws/config`. There is no hardcoded default — `Init` panics if no region is resolvable, so you don't accidentally talk to `us-east-1` from a config-less environment.

For advanced wiring (custom credential providers, fakes, shared config across plugins), pass a pre-built `aws.Config`:

```go
cfg, _ := config.LoadDefaultConfig(ctx, config.WithRegion("eu-central-1"))
g := genkit.Init(ctx,
    genkit.WithPlugins(&bedrock.Bedrock{AWSConfig: &cfg}),
)
```

### Model access

Every Claude, Nova, Llama, Mistral, and Cohere foundation model on Bedrock requires a one-time **"Request model access"** approval per region in the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess). Until access is granted, calls return `AccessDeniedException`. Some newer models also require an inference profile ID (for example `us.anthropic...`) instead of the base foundation-model ID. The plugin surfaces the underlying AWS error so account access, model lifecycle, and inference-profile issues are visible.

### Cross-region inference profiles

Pass cross-region inference profile IDs verbatim to `DefineModel`:

```go
m, _ := bedrock.DefineModel(g, "us.anthropic.claude-haiku-4-5-20251001-v1:0", nil)
```

Supported prefixes: `us.`, `eu.`, `apac.`, `jp.`, `au.`, `global.`, `us-gov.`. The plugin strips the prefix only when looking up capability metadata; the full prefixed ID is sent to Bedrock so AWS handles the region routing.

For newer Claude models, prefer the inference profile ID shown in the Bedrock console or AWS model docs. The live tests currently use `us.anthropic.claude-haiku-4-5-20251001-v1:0` in `us-east-1`.

## Configuration

Per-call config via `ai.WithConfig(&bedrock.Config{...})`:

| Field | Default | Notes |
|---|---|---|
| `MaxTokens` | 4096 | Bedrock requires a max-tokens value for most models. |
| `Temperature` | model default | `[0.0, 1.0]` for most models. |
| `TopP` | model default | Nucleus-sampling cutoff. |
| `StopSequences` | nil | |
| `ToolChoice` | "auto" | `"auto"`, `"any"`, or a specific tool name. |
| `AdditionalModelRequestFields` | nil | Model-specific knobs forwarded verbatim as the Converse `additionalModelRequestFields` document. Use for Claude `thinking`, Nova reasoning levels, etc. |

There is no `RequestTimeout` field — use `context.WithTimeout(ctx, ...)` at the call site instead.

## Tool calling

```go
type weatherIn struct{ Location string `json:"location"` }
type weatherOut struct{ TempF float32 `json:"temp_f"` }

tool := genkit.DefineTool(g, "get_weather", "Look up the temperature in a city.",
    func(ctx *ai.ToolContext, in weatherIn) (weatherOut, error) {
        return weatherOut{TempF: 72}, nil
    })

resp, _ := genkit.Generate(ctx, g,
    ai.WithModel(claude),
    ai.WithPrompt("What's the temperature in SF?"),
    ai.WithTools(tool),
    ai.WithConfig(&bedrock.Config{ToolChoice: "auto"}),
)
```

`stopReason == "tool_use"` is mapped to `ai.FinishReasonStop`. The runtime invokes the tool and feeds the result back via standard Genkit machinery.

## Embedders

```go
emb, _ := bedrock.DefineEmbedder(g, "amazon.titan-embed-text-v2:0", nil)
resp, _ := genkit.Embed(ctx, g, ai.WithEmbedder(emb), ai.WithTextDocs("hello"))
```

Each family has a different JSON wire shape; the plugin routes on model-ID prefix:

- `amazon.titan-embed-text-*` — one call per document, single text input.
- `amazon.titan-embed-image-v1` — one call per document, text and/or image input.
- `cohere.embed-*` — text-only requests are batched into a single `texts: []` call; when any document carries image media, the plugin issues one `images: []` call per document (Cohere accepts a single input type per call). Image requests include `embedding_types: ["int8", "float"]`, which Bedrock requires for the tested Cohere Embed v3 image path. Image documents take precedence over accompanying text.
- `amazon.nova-*-multimodal-embeddings-*` — one `SINGLE_EMBEDDING` call per document; image input takes precedence over text. Uses `embeddingPurpose: GENERIC_INDEX` and the model's default dimension.

Image documents are matched by `image/*` media parts (`png`/`jpeg`/`gif`/`webp`).

## Reranking

```go
import "github.com/firebase/genkit/go/ai"

resp, _ := bedrock.Rerank(ctx, g, "cohere.rerank-v3-5:0", &ai.RerankerRequest{
    Query:     ai.DocumentFromText("Which city is the capital of the US?", nil),
    Documents: []*ai.Document{
        ai.DocumentFromText("Carson City is the capital of Nevada.", nil),
        ai.DocumentFromText("Washington, D.C. is the capital of the United States.", nil),
    },
    Options: &bedrock.RerankOptions{TopN: 2},
})
// resp.Documents are ordered by descending relevance; each carries
// Metadata.Score (0..1) from the reranker.
```

`Rerank` is a standalone call rather than a registered Genkit reranker action — the Go framework does not yet expose a first-class reranker primitive. It reuses the already-initialised plugin on `g` for credentials. `Options` accepts a `*bedrock.RerankOptions`; `TopN <= 0` returns all documents ranked.

## Image generation

```go
img, _ := bedrock.DefineImager(g, "amazon.titan-image-generator-v1", nil)
resp, _ := genkit.Generate(ctx, g,
    ai.WithModel(img),
    ai.WithPrompt("A futuristic Seattle skyline at dusk, photorealistic."),
)
// resp.Message.Content[0] is a *ai.Part with ContentType="image/png" and a base64 data URL.
```

Routes on model-ID prefix:

- `amazon.titan-image-*`, `amazon.nova-canvas-*` — Titan-style `taskType: TEXT_IMAGE` payload.
- `stability.sd3-*`, `stability.stable-image-*` — current Stability `prompt` / `images` payload.
- `stability.stable-diffusion-xl-*`, `stable-*` — legacy Stable Diffusion `text_prompts` / `artifacts` payload.

Defaults are 1024×1024 PNG; tune via `AdditionalModelRequestFields` on Converse-backed models or wait for v2 of this plugin for an image-specific config struct.

## Prompt caching

Insert a cache point in your messages to opt into Bedrock's prompt caching:

```go
ai.WithMessages(
    ai.NewSystemMessage(ai.NewTextPart(longSystemPrompt), bedrock.NewCachePointPart()),
)
```

Bedrock requires a minimum of ~1024 cacheable tokens; small inputs silently bypass the cache.

## Known limitations (v1)

- One plugin instance = one region. Cross-region apps register two instances under separate aliases.
- Tools' input schema is forwarded verbatim — no client-side strict-mode validation.
- No streaming for embedders or image generation.
- No model auto-discovery; foundation-model IDs come from a hand-curated list. Caller can register any model ID via `DefineModel`.
- Image gen returns one 1024×1024 PNG. Larger fan-out / size tuning is a v2 feature.
- `Rerank` is a standalone function, not a registered `ai.Reranker` action (the Go framework has no reranker primitive yet). Only Cohere Rerank's text path is wired.
- Nova multimodal embedders cover synchronous text + image only — audio, video, segmented (async) embeddings, and non-default `embeddingPurpose`/`embeddingDimension` are not yet exposed.
- Embedders embed a single modality per document (image takes precedence over text); no combined text+image vectors.

## Live testing

Live tests are opt-in: each test skips unless its model flag is provided. Use credentials from the standard AWS chain, or source a local `.env` first:

```
cd go
set -a; source ../.env; set +a
go test ./plugins/bedrock/... -v -run 'TestBedrockLive_' \
    -test-bedrock-region=us-east-1 \
    -test-bedrock-model-claude=us.anthropic.claude-haiku-4-5-20251001-v1:0 \
    -test-bedrock-model-nova=amazon.nova-pro-v1:0 \
    -test-bedrock-titan-embedder=amazon.titan-embed-text-v2:0 \
    -test-bedrock-titan-image-embedder=amazon.titan-embed-image-v1 \
    -test-bedrock-cohere-embedder=cohere.embed-english-v3 \
    -test-bedrock-nova-mm-embedder=amazon.nova-2-multimodal-embeddings-v1:0 \
    -test-bedrock-rerank-model=cohere.rerank-v3-5:0
```

This matrix covers:

| Live test | Bedrock path covered |
|---|---|
| `TestBedrockLive_ClaudeSync` | Claude Converse sync |
| `TestBedrockLive_ClaudeStream` | Claude ConverseStream chunks and final response |
| `TestBedrockLive_ClaudeTool` | Claude tool request/response round trip |
| `TestBedrockLive_NovaSync` | Nova Converse sync |
| `TestBedrockLive_TitanEmbedder` | Titan text embedding |
| `TestBedrockLive_TitanImageEmbedder` | Titan image embedding |
| `TestBedrockLive_CohereEmbedderTextAndImage` | Cohere batched text embedding and per-image embedding |
| `TestBedrockLive_NovaMultimodalEmbedder` | Nova text and image multimodal embeddings |
| `TestBedrockLive_Rerank` | Cohere rerank request and score ordering |

Failures with `AccessDeniedException`, legacy/EOL model messages, or "inference profile required" messages usually mean the account, region, or model ID needs adjustment rather than a plugin code change.

Unit tests need no AWS access and run by default with plain `go test ./plugins/bedrock/...`.
