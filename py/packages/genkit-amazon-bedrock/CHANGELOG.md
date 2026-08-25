# Changelog

All notable changes to the `genkit-amazon-bedrock` package are documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First release of the plugin.

### Added

- `Bedrock` plugin registering Bedrock-hosted models as Genkit model actions,
  with AWS client knobs for `region`, `max_retries`, `read_timeout`,
  `connect_timeout`, `max_pool_connections`, `total_timeout` and a
  pre-configured `session`.
- Text generation over the Converse and ConverseStream APIs, covering
  multi-turn chat, system prompts, tool calling, reasoning content, and
  streamed deltas.
- `BedrockConfig` with Bedrock-specific knobs on top of the core `ModelConfig`
  fields, plus `additional_model_request_fields` for anything Converse does not
  model.
- Prompt caching via `cache_point_part()`.
- Embedders for the Titan text, Titan multimodal, Cohere v3 and Nova-2
  families, over InvokeModel.
- Image generation for the Nova Canvas, Titan Image and Stability families,
  configured with `BedrockImageConfig`.
- `Bedrock.rerank()` helper for the Cohere and Amazon rerank families; Genkit
  Python has no reranker action kind to register against.
- Inference-profile and ARN model IDs, sent to Bedrock verbatim while
  capability lookup falls back to the base model.
- AWS error codes mapped onto Genkit statuses, with `Retry-After` surfaced as
  `retry_after_ms` on throttling errors.
- Metadata-only debug logging through `structlog`.
- Runnable sample under `py/samples/amazon-bedrock-sample/` covering chat,
  streaming, tool calling, structured output, reasoning, prompt caching,
  vision, PDF input, embeddings, image generation, and reranking.
