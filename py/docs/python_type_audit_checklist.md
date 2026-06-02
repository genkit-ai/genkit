# Hand-Written Type Audit — Checklist

121 classes total (119 audited + 2 private: `_LatencyTrackable`, `_ModelCopyable`).

Detailed write-ups: [python_beta_type_design.md](./python_beta_type_design.md),
[python_class_audits.md](./python_class_audits.md),
[GENKIT_CLASS_DESIGN.md](../GENKIT_CLASS_DESIGN.md).

---

## Must fix (5) — significant design rework

- [ ] `Genkit` — god object, 38 methods, positional args, `generate_stream`
      returns raw tuple, `define_prompt` has 23 params. Audited in
      GENKIT_CLASS_DESIGN.md.

- [ ] `ExecutablePrompt` — `opts: TypedDict` kills IDE autocomplete on
      `__call__`, `stream`, `render`. 220-line `render()`. Fragile
      `_ensure_resolved()` copies 20 fields. Audited in python_class_audits.md §2.

- [ ] `GenerateStreamResponse` — not used by `Genkit.generate_stream()` (returns
      raw tuple instead), not directly iterable (no `__aiter__`), lives in
      wrong module (`blocks/prompt.py`). Audited in python_class_audits.md §5.

- [ ] `GenerateResponseWrapper` — `assert_valid()`/`assert_valid_schema()` are
      empty placeholders, missing `reasoning`/`media`/`data`/`model` properties
      that JS has. Audited in python_class_audits.md §3.

- [ ] `ToolInterruptError` — extends `Exception` not `GenkitError` (blocked on
      #4346), `str(err)` returns empty string, `metadata` not keyword-only.
      Audited in python_class_audits.md §6.

---

## Should fix (28) — non-trivial changes needed

- [ ] `UserFacingError` — positional args, should be keyword-only.
- [ ] `GenkitError` — two serialization methods + standalone function, consolidate.
- [ ] `Document` — `.text()` is a method, not property. Inconsistent with every
      other `.text` in the SDK. Breaking change. Audited in python_class_audits.md §4.
- [ ] `FlowWrapper` — `stream()` returns tuple, should return `GenerateStreamResponse`.
- [ ] `GenerationResponseError` — positional args, should be keyword-only.
- [ ] `Plugin` — has `model()`/`embedder()` convenience but not `retriever()` etc.
      Causes layering violation (circular import).
- [ ] `TelemetryServerSpanExporter` — creates new `httpx.Client()` per `export()`
      call (no connection pooling), ignores HTTP errors.
- [ ] `ServerSpec` — confusingly similar name to `ServerConfig` (being deleted).
      Rename to `ReflectionServerConfig` or similar.
- [ ] `ModelReference` / `EmbedderRef` / `RetrieverRef` / `IndexerRef` /
      `RerankerRef` / `EvaluatorRef` — wildly inconsistent shapes. `ModelReference`
      allows extras, `EvaluatorRef` uses different fields, `EmbedderRef` missing
      `info`. See python_beta_type_design.md §20.
- [ ] `GenerateResponseChunkWrapper` / `MessageWrapper` — missing `reasoning`,
      `media`, `data` properties that JS has. See python_beta_type_design.md §21.
- [ ] `Action` — mutable `input_schema`/`output_schema` (should be immutable),
      `on_chunk`/`on_trace_start` callbacks on public API (Python uses `async for`),
      `run()` should be deleted, `arun()`/`arun_raw()` confusing, no `__call__`.
      Audited in python_class_audits.md §1.
- [ ] `ActionRunContext` / `ToolRunContext` — missing trace_id/span_id (JS
      provides), `ToolRunContext` accesses parent private fields.
- [ ] `FormatDef` — uses `@abc.abstractmethod` but doesn't extend `abc.ABC`.
      One-line fix.
- [ ] `Logger` — 20-method Protocol. `warn`/`warning` redundant alias,
      `fatal`/`critical` redundant alias. JS Logger has 7 methods.
- [ ] `AdjustingTraceExporter` — belongs in telemetry plugin, not core SDK.
      JS equivalent lives in `js/plugins/google-cloud/`.
- [ ] `RealtimeSpanProcessor` — belongs in telemetry plugin, not core SDK.
- [ ] `RedactedSpan` — used exclusively by `AdjustingTraceExporter`, moves
      with it to telemetry plugin.
- [ ] `GablorkenInput` — test fixture exported publicly in `__all__`. Should
      be private or inlined into `test_models()`.
- [ ] `PromptCache` — plain class with 3 optional fields, not even a dataclass.
      Fold into `ExecutablePrompt` as private attributes.
- [ ] `RerankerParams` — misnamed. Has `reranker`, `query`, `documents` — this
      is action input, should be `RerankerRequest` for consistency with
      `RetrieverRequest`/`IndexerRequest`.
- [ ] `ResumeOptions` — TypedDict (same autocomplete-killer as
      `PromptGenerateOptions`). Convert to dataclass or flatten when
      `PromptGenerateOptions` is replaced with kwargs.

---

## Delete (34) — remove entirely

**Replaced by kwargs on `define_*` methods:**
- [ ] `EmbedderOptions` — flatten to kwargs on `define_embedder()`
- [ ] `RetrieverOptions` — flatten to kwargs on `define_retriever()`
- [ ] `IndexerOptions` — flatten to kwargs on `define_indexer()`
- [ ] `RerankerOptions` — flatten to kwargs on `define_reranker()`
- [ ] `ResourceOptions` — `define_resource()` already has the same kwargs
- [ ] `DapConfig` — flatten to kwargs on `define_dynamic_action_provider()`
- [ ] `DapCacheConfig` — one-field dataclass (`ttl_millis`), fold into parent
- [ ] `DefineBackgroundModelOptions` — flatten to kwargs on `define_background_model()`
- [ ] `SimpleRetrieverOptions` — flatten to kwargs on `define_simple_retriever()`

**Replaced by flat kwargs on prompt methods:**
- [ ] `PromptGenerateOptions` — 17-field TypedDict, THE autocomplete-killer
- [ ] `OutputOptions` — dies when `PromptGenerateOptions` is replaced
- [ ] `OutputConfigDict` — dies when `Output[T]` is replaced

**Inlined into helpers (class unnecessary):**
- [ ] `GenkitSpan` — `__getattr__` proxy kills type checking. Replace with free
      functions in `_tracing.py` (`_set_genkit_attr`, `_set_span_input`,
      `_set_span_output`). `is_root` becomes `span.parent is None` at call site.
      `_trace/_types.py` deleted.

**Dead code / unused:**
- [ ] `Input` / `Output` — replace with `output_schema` kwarg + `@overload`
- [ ] `Retriever` — dead code, never instantiated
- [ ] `ToolRequestLike` — used in 1 place as cast target, delete
- [ ] `ResourceFn` — dead weight, only used in union with `Callable[..., ...]`
- [ ] `MatchableAction` — code smell, `Action` already has `.matches` field
- [ ] `ASGIApp` — defined but never used as type annotation
- [ ] `ServerManagerProtocol` — lives in `web/manager/` being deleted

**Error wire formats (consolidate into one `ErrorResponse`):**
- [ ] `HttpErrorWireFormat` — dies with `core/flows.py`
- [ ] `GenkitReflectionApiDetailsWireFormat` — collapse into `ErrorResponse`
- [ ] `GenkitReflectionApiErrorWireFormat` — collapse into `ErrorResponse`

**Server over-engineering (`web/manager/` deleted — 15-line problem):**
- [ ] `ServerManager`
- [ ] `ASGIServerAdapter`
- [ ] `UvicornAdapter`
- [ ] `GranianAdapter`
- [ ] `Server`
- [ ] `ServerConfig`
- [ ] `ServerLifecycle`
- [ ] `AbstractBaseServer`
- [ ] `SignalHandler`
- [ ] `ServerType`

---

## Clean (52) — no changes needed

**User-facing types:**
- [x] `RankedDocument` — `Document` subclass with `.score`. All 3 SDKs.
- [x] `EmbedderSupports` — value type for embedder capabilities. All 3 SDKs.
- [x] `Formatter` / `FormatterConfig` — format system base types. All 3 SDKs.
- [x] `ActionMetadata` — 9-field data bag for action registration. All 3 SDKs.
- [x] `GenerationCommonConfig` — extends schema type with `api_key`. ~36 files.
- [x] `ContextMetadata` / `RequestData` — request-level context for web frameworks.
- [x] `BackgroundAction` — wraps start/check/cancel for long-running ops. All 3 SDKs.
- [x] `DynamicActionProvider` — runtime action discovery (MCP). All 3 SDKs.
- [x] `Channel` — async iteration channel for streaming. JS has same.
- [x] `Registry` — central action/plugin/schema registry. All 3 SDKs.
- [x] `Embedder` — wraps embedder `Action` with `embed()`. Go has same.

**Enums:**
- [x] `ActionKind` — StrEnum, 17 action types.
- [x] `ActionMetadataKey` — StrEnum, 3 keys.
- [x] `StatusCodes` — IntEnum, gRPC-style status codes.
- [x] `EnvVar` — StrEnum, `GENKIT_ENV`.
- [x] `GenkitEnvironment` — StrEnum, `DEV`/`PROD`.
- [x] `DeprecationStatus` — Enum, 3 values. Python-only.

**Internal plumbing:**
- [x] `ActionResponse` — action result wrapper. All 3 SDKs.
- [x] `Status` — status with code + message.
- [x] `ResourceInput` / `ResourceOutput` — action I/O for resources. All 3 SDKs.
- [x] `RetrieverRequest` / `RetrieverSupports` / `RetrieverInfo` — retriever wire types.
- [x] `IndexerRequest` / `IndexerInfo` — indexer wire types.
- [x] `RerankerSupports` / `RerankerInfo` — reranker wire types.
- [x] `PartCounts` — token counting helper.
- [x] `PromptConfig` — BaseModel stored with prompt action. Internal.
- [x] `ExtractItemsResult` — JSON extraction helper. Python-only.
- [x] `DeprecationInfo` — deprecation metadata. Python-only.

**Format implementations (all subclass FormatDef):**
- [x] `TextFormat` / `JsonFormat` / `JsonlFormat` / `EnumFormat` / `ArrayFormat`

**Runtime/testing:**
- [x] `RuntimeManager` — writes `.genkit/runtimes/` for Dev UI discovery.
- [x] `SimpleCache` — thread-safe TTL cache for DAP. Internal to `DynamicActionProvider`.
- [x] `ProgrammableModel` / `EchoModel` / `StaticResponseModel` — test doubles.
- [x] `SkipTestError` / `ModelTestError` / `ModelTestResult` / `TestCaseReport` — test infra.

**Other:**
- [x] `UnstableApiError` — `GenkitError` subclass for beta API gating. Matches JS.
- [x] `DeprecatedEnumMeta` — metaclass for enum deprecation warnings. Python-only.
- [x] `GenkitBase` / `GenkitRegistry` — `Genkit` class hierarchy. Audited in
      GENKIT_CLASS_DESIGN.md.
