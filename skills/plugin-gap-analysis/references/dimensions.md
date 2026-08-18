# Gap analysis dimensions

The frozen checklist. Walk it in order, once per plugin. Every dimension produces either a
row in the gap list or nothing.

Extracted from the anthropic JS -> Go run and corrected by the google-genai replay
(2026-08-18). Append to it when a plugin forces a question these do not cover; never silently
skip one.

## Group 0. Coverage - do this first

Coverage decides scope. Running it late means auditing the shared path in depth while a whole
model family is missing. It also decides direction: a plugin where the target is ahead on half
these dimensions is a two-way convergence, not a catch-up.

| # | Dimension | What to compare |
|---|-----------|-----------------|
| 0.1 | Action types served | model, background-model, check-operation, embedder, and anything else. Diff the sets. |
| 0.2 | Model families | Text/multimodal, TTS, image generation, video, music, research/agent, and each provider-specific family. A family absent on one side is one row, not thirty. |
| 0.3 | Family-specific config and capabilities | Whether each family gets its own schema and capability set, or is folded into the generic one. Folding a restricted family (e.g. Gemma) into the general one makes the plugin advertise capabilities the model does not have - that is an **X**. |
| 0.4 | Backend variants | Whether one plugin serves several backends (dev API vs enterprise), and how. Shared implementation with per-backend model lists vs duplicated per-backend trees. Duplicated trees drift from *each other* - a same-language parity audit worth its own run. |
| 0.5 | Non-model plugin features | Context caching, code execution, batching, operation polling, file upload - features that are not a config field on a model action. |
| 0.6 | Direction check | Tally which side leads per group. State the verdict at the top of the report. Do not assume the reference language leads. |
| 0.7 | Pinned SDK versions | Record the provider SDK version each side pins (`go/go.mod`, the installed JS package). Skew is common and it reattributes gaps: one row blamed the target for a stale enum when the reference was rejecting a value its own newer SDK accepts. Any row citing an SDK capability must name the version. |
| 0.8 | Pending dependency bumps | Check for an open dependency-bump PR against either SDK. A large version jump can invalidate a row's conclusion or trigger a fragility the audit only flagged as theoretical. |

## A. Plugin surface and entry points

| # | Dimension | What to compare |
|---|-----------|-----------------|
| A1 | Entry points | Every place the provider is served from in each language. More than one is common. |
| A2 | Plugin options | Field-by-field: what can be configured at construction time. |
| A3 | Auth resolution | Order of precedence across request config, plugin option, env vars. Whether request-level keys exist. Whether auth can be deferred to request time. Whether a missing key panics, throws, or defers. |
| A4 | Client escape hatch | Base URL override, custom transport/fetch, request options, middleware, timeouts, retries, extra headers, credentials objects. This is also how alternative routing (Bedrock, Vertex, Express Mode, a proxy) is reached - absence blocks whole deployment modes. Note whether supplying a custom client silently drops default instrumentation. |
| A5 | Model reference helper | Name, signature, and config type of the "name a model and carry config" helper. |
| A6 | Per-model capability override | Whether a caller can correct or extend what the plugin believes about a model, e.g. one released after the plugin version. |
| A7 | Deprecated surface | Legacy define/lookup APIs present on one side only. Cleanup candidates, not parity gaps - tag **C** and say so. |
| A8 | Debug and compatibility valves | Raw-request tracing toggles, legacy-format switches, and anything else that exists to unblock a user on an older deployment. Easy to miss because they are not features. |

## B. Model catalog

| # | Dimension | What to compare |
|---|-----------|-----------------|
| B1 | Curated model list | Diff the ID sets **both ways**. Neither side is reliably a superset. |
| B2 | Declared capabilities | Per model: multiturn, tools, toolChoice, media, systemRole, output formats, constrained generation. Diff the flags, not just their presence. |
| B3 | Stage, versions, label | Dev UI surfaces all three; one side often sets none. |
| B4 | Dynamic listing | Whether the plugin lists models from the API, whether results are cached and for how long, and how a discovery failure is reported. |
| B5 | Dynamic fallback | What an unknown/undeclared model resolves to, and what it claims to support. |
| B6 | Name normalisation | Provider-prefix trimming, date-suffix stripping, alias handling. A mismatch here means the curated list silently never matches. |

## C. Config schema

| # | Dimension | What to compare |
|---|-----------|-----------------|
| C1 | Schema provenance | Curated (hand-written, Genkit-flavoured names) vs reflected from the provider SDK struct (raw provider names). This decides the whole user-facing config vocabulary and is usually the largest single row in the report. |
| C2 | Genkit-common coverage | maxOutputTokens, temperature, topK, topP, stopSequences, version - present, absent, or renamed on each side. |
| C3 | Provider-specific coverage | Field by field, for every provider config field the reference exposes. |
| C4 | Validation | Refinements, ranges, mutual exclusions, integer-vs-float constraints, and where they are enforced (plugin vs provider API). |
| C5 | Managed fields | Fields a Genkit primitive owns (messages, system, model, output format, function tools). Whether setting one is rejected with a pointer to the right option, silently overwritten, or silently dropped. |
| C6 | Dev UI presentation | Field descriptions, hidden fields, and reflection artefacts leaking into the schema. |
| C7 | Schema type mapping | For reflected schemas: how SDK wrapper types are mapped to JSON Schema primitives, and what happens to a wrapper type the mapping does not name. Matching on `reflect.Type.Name()` string literals degrades silently on an SDK rename; comparing against `reflect.TypeOf(...)` fails at compile time instead. Check whether sibling plugins share the problem before generalising it - one did not, because its SDK uses plain pointer scalars with nothing to remap. |
| C8 | Passthrough behaviour | Whether the schema rejects unknown keys or spreads them into the wire request, and *where* it spreads them. This decides whether a field absent from the schema is genuinely unreachable: a nested-config passthrough will not reach a field the provider places at the request root. The most common cause of a false "unreachable" row. |
| C9 | Backend-conditional fields | Fields the SDK accepts on one backend and hard-errors on for another. Where one description map serves both backends, check the descriptions carry the caveat - otherwise a user reads documentation for a field that cannot work. |

## D. Request conversion

| # | Dimension | What to compare |
|---|-----------|-----------------|
| D1 | Role mapping | Including tool roles, and how a tool-result message is re-roled. |
| D2 | System messages | Where they are extracted from, whether non-text content is rejected, whether an empty system block can be sent. |
| D3 | Part types accepted | text, media, data, toolRequest, toolResponse, reasoning, custom. Diff the accepted set. |
| D4 | Media sources | base64, remote URL, provider file ID. Accepted content types per source. A base64-only implementation silently inlines bytes the other side would send as a URL. |
| D5 | Tool definitions | Name validation, empty-schema defaulting, strict mode, schema rewriting, per-provider carve-outs. |
| D6 | Tool choice | Mapping of each Genkit tool-choice value, and whether an unset value clobbers a config-provided one. |
| D7 | Structured output | Native constrained path vs prompt-instruction fallback, which models it is enabled for, and schema rewriting applied. |
| D8 | Provider extras | Caching directives, citations/documents, server-side tools, beta feature flags, multipart tool results. |
| D9 | Concurrency safety | Whether a shared/hoisted config can be mutated by a request (slice aliasing, in-place appends). |

## E. Response conversion

| # | Dimension | What to compare |
|---|-----------|-----------------|
| E1 | Content block coverage | Enumerate every block type the provider can return and diff which side handles each. A block landing in a `default:` error branch is an **X**. |
| E2 | Stop reason mapping | Every provider stop reason, and what an unmapped one becomes. |
| E3 | Usage fields | Input, output, cached-read, cache-creation, reasoning/thoughts tokens. |
| E4 | Raw passthrough | Whether the raw provider response and/or a `custom` field is populated. |
| E5 | Streaming coverage | Which events produce chunks, chunk shape, whether non-text blocks stream at all, and whether signatures/metadata that only arrive on a separate event are captured. |
| E6 | Stream termination | What happens when a stream ends without a terminal event. |

## E'. Long-running actions

Only applies when group 0.1 found a background-model or check-operation action.

| # | Dimension | What to compare |
|---|-----------|-----------------|
| E'1 | Start/check split | Whether both actions exist and whether one registration serves both keys. |
| E'2 | Operation polling | How an operation handle is represented, and how a not-yet-done operation is reported. |
| E'3 | Result retrieval | Output shape per backend - a URI to fetch vs inline bytes is a real user-visible divergence. |
| E'4 | Failure and filtering | How a provider-side rejection mid-operation surfaces. |
| E'5 | Batching | Request-splitting limits per backend and per model, for embedders and any other batched action. Absence means a large request fails at the API instead of being chunked. |

## F. Errors

| # | Dimension | What to compare |
|---|-----------|-----------------|
| F1 | Status classification | Per HTTP code. Check provider-specific codes (e.g. 529) individually - a generic `>= 500 -> Internal` fallback misclassifies them. |
| F2 | Retry metadata | Whether `retry-after` (both delay-seconds and HTTP-date forms) reaches the caller. |
| F3 | Caller-fault vs server-fault | Whether a bad request is classified so retry middleware does not reissue it. |
| F4 | Message quality | Whether the error names the option to use instead. |

## G. Docs and DX

| # | Dimension | What to compare |
|---|-----------|-----------------|
| G1 | README sections | Diff the heading lists, not the word counts. |
| G2 | Samples and testapps | Which features each sample actually exercises. |
| G3 | Exported helpers | Public helper functions and types available to users on each side. |
| G4 | Doc-site page | **Not in this repo.** Emit a row noting it must be tracked in the docs repo. |

## H. Tests

| # | Dimension | What to compare |
|---|-----------|-----------------|
| H1 | Unit coverage | Which of the dimensions above have a test on each side. |
| H2 | Live tests | Presence, and what they cover. |
| H3 | Conformance tests | Whether the plugin participates in the shared conformance suite. |
