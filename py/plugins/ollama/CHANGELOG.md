# Changelog

All notable changes to the `genkit-plugin-ollama` package are documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0]

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
- Runnable sample under `py/samples/ollama/` covering chat, streaming,
  tool calling, and embeddings.

### Changed

- Plugin metadata now reflects per-API-type capabilities (e.g. `generate`
  API no longer advertises `multiturn`/`tools`).
- `request_headers` are now propagated through to `ollama.AsyncClient`
  (previously stored but never sent).
- Tool input schemas must declare `type: 'object'`; non-object schemas now
  raise `ValueError` instead of being silently dropped.

### Fixed

- `top_p` from `ModelConfig` is now mapped correctly into
  `ollama.Options` (previously sent as `topP` and ignored).
