# Changelog

All notable changes to the `genkit-ollama` package are documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- First-party support for the plugin (graduated from community status).
- `OllamaConfig` with Ollama-specific knobs: `think`, `keep_alive`,
  `num_ctx`, `min_p`, `seed`, `num_predict`.
- `OllamaSupports.media` opt-in flag for vision models (`llava`,
  `llama3.2-vision`, etc.).
- `request_headers` accepts a sync or async callable in addition to a
  static dict.
- `timeout` constructor argument propagated to the underlying httpx client.
- Friendly `OllamaConnectionError` when the Ollama server is unreachable.
- `EmbeddingDefinition` is now exported from the package root
  `genkit_ollama` (previously importable only via the
  `genkit_ollama.embedders` submodule).
- Runnable sample under `py/samples/ollama-sample/` covering chat, streaming,
  tool calling, and embeddings.

### Changed

- Plugin metadata now reflects per-API-type capabilities (e.g. the `generate`
  API no longer advertises `multiturn`/`tools`).
- `request_headers` are now propagated through to `ollama.AsyncClient`
  (previously stored but never sent).
- Tool input schemas that omit an explicit `type` but declare `properties`
  are inferred as object schemas instead of being dropped.

### Fixed

- `top_p` from `ModelConfig` is now mapped correctly into
  `ollama.Options` (previously sent as `topP` and ignored).

[Unreleased]: https://github.com/genkit-ai/genkit/compare/py/genkit-plugin-ollama-v0.6.0...HEAD
