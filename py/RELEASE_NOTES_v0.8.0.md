# Genkit Python 0.8.0

> **Scope of this release.** This `py/v0.8.0` publishes only three artifacts: `genkit`
> (core) 0.8.0, `genkit-google-genai` 0.8.0, and a deprecation tombstone for
> `genkit-plugin-google-genai` 0.8.0. **All other plugins remain at 0.7.0** and are
> verified to run against core 0.8.0 — no action needed for them.

## genkit (core) 0.8.0

### Breaking changes

- The `wrap_generate` hook receives different params. `GenerateHookParams` no longer
  carries `request`; it now exposes `options` (`GenerateActionOptions`), `iteration`,
  and `message_index`. The raw `ModelRequest` is available on the new `wrap_model` hook
  via `ModelHookParams.request`, and per-tool execution is covered by the new
  `ToolHookParams`. Middleware that read `params.request` should switch to
  `params.options`, or move to `wrap_model` if they need the raw request (#5694).
- Core no longer ships the `genkit.plugins` namespace. Old `genkit.plugins.*` import
  paths are served by the installed plugin packages (or their deprecation tombstones)
  via namespace packaging, not by core (#5703).

### Added

- `ChunkAccumulator` for aggregating streamed chunks (#5694).
- `output_instructions` on `ai.generate()` now accepts `bool` (#5681).

### Fixed

- Type-checking, streaming syntax, and a missing-default-value runtime crash (#5340).

## genkit-google-genai 0.8.0

### Breaking changes

- Package renamed from `genkit-plugin-google-genai` to `genkit-google-genai`, and the
  import path from `genkit.plugins.google_genai` to `genkit_google_genai`. The old
  distribution installs as a tombstone that re-exports from the new module with a
  `DeprecationWarning`, so existing apps keep working — update to the new name when you
  can (#5703).
- Some deprecated Gemini models were removed. Referencing removed model ids will fail;
  move to current ids (#5126).

### Added

- Vertex AI multi-region support and per-request location override (#5763).
- Vertex multimodal embeddings via `:predict` (#5649).
- Registered Gemini embedding-2 (#5596).
- Registered Gemini 3.1 text models (#5559, #5588) and Gemini 3.x image models (#5579).
- Veo 3.x support on Google AI (#5174).
- Tuned Gemini endpoint model support (#5182).
- Friendly onboarding message when `GEMINI_API_KEY` is missing (#5665).

### Fixed

- Map tool role to `"user"` for Gemini compatibility. Gemini 3.6 / `gemini-flash-latest`
  enforce strict turn-role validation and reject role `"tool"` with a 400; `Role.TOOL`
  is now mapped to `"user"` when building request contents (#5780).
- Report finish reason and cumulative usage on Gemini `generate_stream` turns (#5736).
- List only callable Vertex embedders (#5695).
- Discover Vertex AI models by name (#5575).

### Docs

- Promote generic "latest" Gemini model aliases (#5540).

## Migration

```bash
# google-genai users: swap the dependency
uv remove genkit-plugin-google-genai
uv add genkit-google-genai
```

```python
- from genkit.plugins.google_genai import GoogleAI, VertexAI
+ from genkit_google_genai import GoogleAI, VertexAI
```

- **Custom middleware authors:** update `wrap_generate` hooks per the core breaking
  change above (`params.request` → `params.options`, or move to `wrap_model`).
- **Everyone else:** no changes required. Other plugins stay at 0.7.0 and remain
  compatible with core 0.8.0.

## Compatibility

`genkit-plugin-*` 0.7.0 (anthropic, compat-oai, google-cloud, middleware, ollama,
vertex-ai, django, fastapi, flask, evaluators) were each smoke-tested against core
0.8.0 and import cleanly.
