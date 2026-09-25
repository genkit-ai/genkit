# Genkit Python — Beta API Design Review

This doc covers the full public API surface being locked at beta: what's importable,
how the client is constructed, the high-traffic method signatures, and the return types
users interact with. Section 5 lists open design questions requiring explicit sign-off.

---

## 1. Import Surface

Every symbol exported at beta. This is exhaustive by design — the list itself is what's
being approved.

### `from genkit import ...` — app developers

```python
from genkit import (
    # Core
    Genkit,
    ActionRunContext,
    ModelResponse, # renamed from GenerateResponse, wire format + veneer unified
    ModelResponseChunk, # renamed from GenerateResponse, wire format + veneer unified

    ExecutablePrompt,
    GenkitError,
    PublicError,         # renamed from UserFacingError

    # Content types
    Part, TextPart, MediaPart, Media,
    DataPart, ToolRequestPart, ToolResponsePart, CustomPart,
    ReasoningPart,

    # Messages
    Message, Role,

    # Documents
    Document, DocumentPart,

    # Tool context
    ToolRunContext,
    ToolInterruptError, 
    ToolChoice,

    # Generation config
    ModelConfig # Renamed from GenerationCommonConfig

    # Evaluation
    BaseEvalDataPoint,

    Flow,                     # Useful for annotation? 50/50 on this one

    StreamResponse # Flow.stream(), Action.stream() return generic StreamResponse, ExecutablePrompt.stream() returns StreamResponse[GenerateStreamResponse]

)
```

### `genkit.model`

```python
from genkit.model import (
    ModelRequest, # Renamed from GenerateRequest
    ModelResponse, # Renamed from GenerateResponse, wire format + veneer unified
    ModelResponseChunk, # Renamed from GenerateResponseChunk, wire format + veneer unified
    GenerationUsage,
    Candidate,
    OutputConfig,
    FinishReason,
    GenerateActionOptions,
    Error,
    Operation,
    ToolRequest,
    ToolDefinition,
    ToolResponse,
    ModelInfo,
    Supports,
    Constrained,
    Stage,
    model_action_metadata,
    model_ref,
    ModelRef, # Renamed from ModelReference
    BackgroundAction,
    lookup_background_action,
    compute_usage_stats,
    resolve_api_key,
    ModelConfig # Renamed from GenerationCommonConfig
)
```

Note: DAP and Model Middleware exports will be included in `genkit.model` namespace. Still working on the re-design of these features. Will update this API surface when done.

### `genkit.retriever`

```python
from genkit.retriever import (
    RetrieverRequest,
    RetrieverResponse,
    retriever_action_metadata,
    retriever_ref,
    RetrieverRef,
    IndexerRequest,
    indexer_action_metadata,
    indexer_ref,
)
```

### `genkit.embedder`

```python
from genkit.embedder import (
    EmbedRequest,
    EmbedResponse,
    Embedding,
    embedder_action_metadata,
    embedder_ref,
    EmbedderRef,
    EmbedderSupports,
)
```

### `genkit.reranker`

```python
from genkit.reranker import (
    reranker_action_metadata,
    reranker_ref,
    RerankerRef,
    RankedDocument,
    RerankerRequest,
    RerankerResponse,
    RankedDocumentData,
    RankedDocumentMetadata,
)
```

### `genkit.evaluator`

```python
from genkit.evaluator import (
    EvalRequest,
    EvalResponse,
    EvalFnResponse,
    Score,
    Details,
    BaseEvalDataPoint,
    EvalStatusEnum,
    evaluator_action_metadata,
    EvaluatorRef,
    evaluator_ref,
)
```

### `genkit.plugin_api` — all plugin authors

```python
from genkit.plugin_api import (
    # Base class and framework primitives
    Plugin,
    Action,
    ActionMetadata,
    ActionKind,
    StatusCodes,

    # HTTP / version stamping (for setting x-goog-api-client and user-agent headers)
    GENKIT_CLIENT_HEADER,
    GENKIT_VERSION,

    # Convenience re-exports from domain modules
    # (identical to importing from genkit.model, genkit.retriever, etc.)
    model_action_metadata, model_ref, ModelReference,
    embedder_action_metadata, embedder_ref,
    retriever_action_metadata, retriever_ref,
    indexer_action_metadata, indexer_ref,
    reranker_action_metadata, reranker_ref,
    evaluator_action_metadata, evaluator_ref,
)
```

Note: The domain sub-modules (`genkit.model`, `genkit.retriever`, etc.) are still the canonical
paths for domain-specific types. `genkit.plugin_api` re-exports the cross-cutting framework primitives
and provides a single entry point for plugin authors who don't want to hunt across multiple paths.

**Canonical import policy (beta):**
- App developers use `from genkit import ...` for the application-facing API.
- Plugin authors use `from genkit.plugin_api import ...` for framework primitives (`Plugin`, `Action`, etc.).
- Domain modules (`genkit.model`, `genkit.retriever`, `genkit.embedder`, `genkit.reranker`, `genkit.evaluator`) are canonical for domain-specific types.
- Prefer domain-specific imports over importing from `genkit.plugin_api` in all app-developer facing docs and samples. `genkit.plugin_api` convenience exports should be reserved for plugin author-facing documentation.
- Telemetry/tracing helpers remain core/internal for beta (`genkit.core.tracing`) and align to OpenTelemetry semantics rather than a separate public tracing namespace. (WIP, need to flesh out the primary user journeys more clearly here)

---

## 2. Client Construction

```python
ai = Genkit(
    plugins: list[Plugin] | None = None,
    model: str | None = None,
    prompt_dir: str | Path | None = None,
)
```

- `plugins` — list of initialized plugin instances
- `model` — default model name used when `model=` is omitted from `generate()`
- `prompt_dir` — directory to load `.prompt` files from; defaults to `./prompts` if it exists

---

## 3. Method Signatures

High-traffic paths only — not exhaustive.

### `Genkit`

```python
C = TypeVar('C', bound=GenerationCommonConfig)
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

# generate(): exact 4-overload matrix
# Shared params omitted below:
# prompt, system, messages, tools, return_tool_requests, tool_choice, tool_responses,
# max_turns, context, output_format, output_content_type, output_instructions,
# output_constrained, use, docs
#
# 1) typed model + typed output
@overload
async def generate(
    self,
    *,
    model: ModelReference[C],
    config: C | None = None,
    output_schema: type[OutputT],
    ...,
) -> ModelResponse[OutputT]: ...

# 2) typed model + untyped output
@overload
async def generate(
    self,
    *,
    model: ModelReference[C],
    config: C | None = None,
    output_schema: dict[str, object] | None = None,
    ...,
) -> ModelResponse[Any]: ...

# 3) string model + typed output
@overload
async def generate(
    self,
    *,
    model: str | None = None,
    config: GenerationCommonConfig | None = None,
    output_schema: type[OutputT],
    ...,
) -> ModelResponse[OutputT]: ...

# 4) string model + untyped output
@overload
async def generate(
    self,
    *,
    model: str | None = None,
    config: GenerationCommonConfig | None = None,
    output_schema: dict[str, object] | None = None,
    ...,
) -> ModelResponse[Any]: ...

# generate_stream(): same 4-overload matrix as generate()
# Shared params omitted below:
# prompt, system, messages, tools, return_tool_requests, tool_choice,
# max_turns, context, output_format, output_content_type, output_instructions,
# output_constrained, use, docs, timeout
#
# 1) typed model + typed output
@overload
def generate_stream(
    self,
    *,
    model: ModelReference[C],
    config: C | None = None,
    output_schema: type[OutputT],
    ...,
) -> ModelStreamResponse[OutputT]: ...

# 2) typed model + untyped output
@overload
def generate_stream(
    self,
    *,
    model: ModelReference[C],
    config: C | None = None,
    output_schema: dict[str, object] | None = None,
    ...,
) -> ModelStreamResponse[Any]: ...

# 3) string model + typed output
@overload
def generate_stream(
    self,
    *,
    model: str | None = None,
    config: GenerationCommonConfig | None = None,
    output_schema: type[OutputT],
    ...,
) -> ModelStreamResponse[OutputT]: ...

# 4) string model + untyped output
@overload
def generate_stream(
    self,
    *,
    model: str | None = None,
    config: GenerationCommonConfig | None = None,
    output_schema: dict[str, object] | None = None,
    ...,
) -> ModelStreamResponse[Any]: ...

# Retrieval
async def retrieve(
    self,
    retriever: str | RetrieverRef,
    query: str | Document,
    *,
    options: dict[str, object] | None = None,  # plugin-defined schema; shape varies per retriever
) -> list[Document]: ...

# Embedding
async def embed(
    self,
    embedder: str | EmbedderRef,
    content: str | Document,
    *,
    options: dict[str, object] | None = None,  # plugin-defined schema; shape varies per embedder
) -> list[Embedding]: ...

# Prompt lookup: same 4-overload input/output matrix as define_prompt()
# Shared params omitted below:
# variant
#
# 1) typed input + typed output
@overload
def prompt(
    self,
    name: str,
    *,
    input_schema: type[InputT],
    output_schema: type[OutputT],
    ...,
) -> ExecutablePrompt[InputT, OutputT]: ...

# 2) typed input + untyped output
@overload
def prompt(
    self,
    name: str,
    *,
    input_schema: type[InputT],
    output_schema: dict[str, object] | None = None,
    ...,
) -> ExecutablePrompt[InputT, Any]: ...

# 3) untyped input + typed output
@overload
def prompt(
    self,
    name: str,
    *,
    input_schema: dict[str, object] | None = None,
    output_schema: type[OutputT],
    ...,
) -> ExecutablePrompt[Any, OutputT]: ...

# 4) untyped input + untyped output
@overload
def prompt(
    self,
    name: str,
    *,
    input_schema: dict[str, object] | None = None,
    output_schema: dict[str, object] | None = None,
    ...,
) -> ExecutablePrompt[Any, Any]: ...

# Decorators
@ai.flow(name: str | None = None)
async def my_flow(input: InputT) -> OutputT: ...
# Returns: Flow

@ai.tool(name: str | None = None, description: str | None = None)
def my_tool(input: InputT, ctx: ToolRunContext) -> OutputT: ...
```

### `ExecutablePrompt` — returned by `ai.prompt()` / `@ai.define_prompt`

```python
# Call like a function
await prompt(input: InputT | None = None) -> ModelResponse[OutputT]

# Stream
def stream(
    self,
    input: InputT | None = None,
    *,
    timeout: float | None = None,
) -> ModelStreamResponse[OutputT]

# Render without executing
async def render(
    self,
    input: InputT | dict[str, Any] | None = None,
) -> GenerateActionOptions
```

### `Flow` — returned by `@ai.flow`

```python
# Call like a function — same signature as the wrapped flow
flow(*args, **kwargs) -> Awaitable[OutputT]

# Stream
def stream(
    self,
    input: InputT = None,
    *,
    context: dict[str, object] | None = None,
    telemetry_labels: dict[str, object] | None = None,
    timeout: float | None = None,
) -> FlowStreamResponse[ChunkT, OutputT]
```

### Plugin authoring surface

```python
# define_prompt(): 4-overload input/output matrix only
# Shared params omitted below:
# name, variant, model, config, description, system, prompt, messages,
# docs, output_format, output_content_type, output_instructions,
# output_constrained, tools, tool_choice, return_tool_requests, max_turns, use
#
# 1) typed input + typed output
@overload
def define_prompt(
    self,
    *,
    input: Input[InputT],
    output: Output[OutputT],
    ...,
) -> ExecutablePrompt[InputT, OutputT]: ...

# 2) typed input + untyped output
@overload
def define_prompt(
    self,
    *,
    input: Input[InputT],
    output: Output[Any] | None = None,
    ...,
) -> ExecutablePrompt[InputT, Any]: ...

# 3) untyped input + typed output
@overload
def define_prompt(
    self,
    *,
    input: Input[Any] | None = None,
    output: Output[OutputT],
    ...,
) -> ExecutablePrompt[Any, OutputT]: ...

# 4) untyped input + untyped output
@overload
def define_prompt(
    self,
    *,
    input: Input[Any] | None = None,
    output: Output[Any] | None = None,
    ...,
) -> ExecutablePrompt[Any, Any]: ...

def define_model(
    self,
    name: str,
    fn: ModelFn,
    *,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
    label: str | None = None,
    supports: Supports | None = None,
    versions: list[str] | None = None,
    stage: Stage | None = None,
) -> Action: ...

def define_embedder(
    self,
    name: str,
    fn: EmbedderFn,
    *,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
    label: str | None = None,
    supports: EmbedderSupports | None = None,
    dimensions: int | None = None,
) -> Action: ...

def define_retriever(
    self,
    name: str,
    fn: RetrieverFn,
    *,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
    label: str | None = None,
    supports: RetrieverSupports | None = None,
) -> Action: ...

# InputT binds through input_schema — all Callables and the return type are typed accordingly

def define_prompt(
    self,
    name: str | None = None,
    *,
    variant: str | None = None,
    model: str | None = None,
    config: ModelConfig | None = None,  # or GeminiConfig, OpenAIConfig, etc. for model-specific fields
    description: str | None = None,
    input_schema: type[InputT] | None = None,      # binds InputT for callables below
    system: str | Part | list[Part] | Callable[[InputT, dict | None], str | Part | list[Part]] | None = None,
    prompt: str | Part | list[Part] | Callable[[InputT, dict | None], str | Part | list[Part]] | None = None,
    messages: str | list[Message] | Callable[[InputT, dict | None], list[Message]] | None = None,
    docs: list[Document] | Callable[[InputT, dict | None], list[Document]] | None = None,
    output_schema: type | dict[str, object] | None = None,
    output_format: str | None = None,
    output_content_type: str | None = None,
    output_instructions: bool | str | None = None,
    output_constrained: bool | None = None,
    tools: list[str | Action | ExecutablePrompt] | None = None,  # str = registered name, Action = inline tool, ExecutablePrompt = sub-agent
    tool_choice: ToolChoice | None = None,
    return_tool_requests: bool | None = None,
    max_turns: int | None = None,
    use: list[ModelMiddleware] | None = None,
) -> ExecutablePrompt[InputT]: ...

# Streaming - WIP
# Action — returned by define_model, define_tool, etc.
# Calling streams and returns the base wrapper; Flow/generate_stream build on top
action.stream(
    input: InputT | None = None,
    *,
    context: dict[str, object] | None = None,
    telemetry_labels: dict[str, object] | None = None,
    timeout: float | None = None,
) -> ActionStreamResponse[ChunkT, OutputT]

# ActionRunContext[ChunkT] — producer interface inside action/flow/tool functions
# Go: StreamCallback[Stream] param (nil = not streaming)
# JS: ActionFnArg<S> / FlowSideChannel<S> — two types; Python unifies into one
ctx.is_streaming              # bool — whether caller requested a stream
ctx.send_chunk(chunk: ChunkT) # type-safe push; no-op if not streaming
ctx.context                   # dict[str, object] — request context
```

---

## 4. Return Type Surfaces

What users get back from calls and interact with.

### `ModelResponse` — from `generate()`, `await prompt(input)`

```python
response.text          # str — full text of the response
response.output        # OutputT — typed output if output schema was provided
response.message       # Message — the final message
response.messages      # list[Message] — full conversation history
response.tool_requests # list[ToolRequestPart] — pending tool calls
```

### `Message` — used for both inputs and returned responses

```python
message.text           # str — text content of the message
message.tool_requests  # list[ToolRequestPart]
message.interrupts     # list[ToolRequestPart] — tool calls requiring user input
```

### `ModelResponseChunk` — stream chunks from `generate_stream()`

```python
chunk.text             # str — text in this chunk
chunk.output           # object — partial typed output
chunk.accumulated_text # str — all text so far
```

### Streaming wrappers — WIP

Three wrapper types, one hierarchy (`ActionStreamResponse` → `FlowStreamResponse` → `ModelStreamResponse`). All expose the same two properties:

```python
result.stream    # AsyncIterable[ChunkT]
result.response  # Awaitable[OutputT]
```

| Type | Returned by | ChunkT | OutputT |
|---|---|---|---|
| `ActionStreamResponse[C, O]` | `action.stream()` | action-defined | action-defined |
| `FlowStreamResponse[C, O]` | `flow.stream()` | flow-defined | flow-defined |
| `ModelStreamResponse[O]` | `generate_stream()`, `prompt.stream()` | `ModelResponseChunk` (fixed) | `ModelResponse[O]` |

### `retrieve()` return value

```python
documents              # list[Document]
```

---

## 5. Design Flags

### Single public type per concept

For beta, Python uses one public type per concept (no split between "wire type" and
"veneer type" in the public API):

- `ModelResponse` is the single public response type used by app code and plugin contracts.
- `ModelResponseChunk` is the single public streaming chunk type.
- `Message` and `Document` are the single public message/document types for both construction and returned values.

This is an explicit beta design decision:

- Originally, JSON-schema-exported wire types were intended to be the plugin contract.
- JS then added veneer/helper layers for frequently used types.
- Python copied that split initially, and the resulting surface was too confusing.
- We adopt Go's approach for common response/message-result types:
  - Omit the most common response wire types from default autogen output.
  - Handwrite canonical runtime types (`ModelResponse`, `ModelResponseChunk`).
  - Use those same types for both plugin contracts and app-developer annotations/usages.
- Rule: if a wire type is common enough that we'd add a veneer helper layer, do not expose two public types; use one handwritten canonical type instead.

### Plugin namespace role and boundaries

- We considered `genkit.plugin`, but it collides semantically with `genkit.plugins.*` (actual provider/plugin packages) and repeatedly confused app developers.
- We therefore standardize on `genkit.plugin_api` for framework/plugin-author primitives.
- It exists to gather framework primitives plus convenience domain re-exports in one place. Otherwise, it's unclear what common stuff a plugin developer might need and the surface of concepts to grasp suddenly looks huge.
- Canonical domain contracts should still be documented/imported from domain modules (`genkit.model`, `genkit.retriever`, etc.) to avoid import-path drift.

### Tradeoff: overload-heavy typing for `generate()` and `prompt()`

**Decision**

- `generate()` and `generate_stream()` each use a 4-overload matrix across two axes:
  - model path (`ModelReference[C]` vs `str`)
  - output typing (`output_schema: type[OutputT]` vs untyped schema)
- Prompt APIs (`prompt()` and `define_prompt()`) also use 4 overloads, but only across:
  - input typing
  - output typing
- We do **not** add model-config as a prompt overload axis.

**Why this split**

- For `generate*`, `config` is where plugin-specific correctness matters most. `ModelReference[C]`
  lets type checking enforce that model config matches the selected model family.
- For prompt APIs, the highest-value contracts are prompt input/output shapes. Those are what
  prompt authors and prompt callers interact with most directly.
- Adding model-config to prompt overload axes would increase prompt overloads from 4 to 8 for
  relatively low additional value.

**What this buys us**

- Strong config safety on the typed model path (`ModelReference[C]`).
- Strongly typed `response.output` for schema-typed output paths.
- Bounded overload growth (4 overloads per high-traffic API instead of 8+ for prompt APIs).
- Practical parity with JS ergonomics while keeping one public response type per concept.

**Cross-language note**

- JS has the same dynamic lookup limitation: `prompt(name)` cannot infer types from runtime registry
  names unless types/schemas are provided at the call site.
- Go does not provide equivalent generic config typing on model refs.

---

## Appendix: Pre-review action items

Smaller decisions we made to clean up the API surface as part of auditing the existing codebase. Referenced here to help with implementation later and remember why we made some of these decisions.

- Rename `UserFacingError` → `PublicError` (matches Go's `NewPublicError`; intent is "safe to return in HTTP response")
- Remove `reflection_server_spec` from `Genkit.__init__` — server starts automatically via `GENKIT_ENV=dev`, port is auto-selected; expose port override as env var `GENKIT_REFLECTION_PORT` if needed (PR #4812 does the right thing but left the param in)
- Make `ai.registry` private (`ai._registry`); remove direct access from all samples
- Fix `part.root.text` / `part.root.media` ergonomics — Pydantic `RootModel` internals should not surface to users
- Flatten `ExecutablePrompt` `opts: PromptGenerateOptions` TypedDict → flat kwargs (consistent with `generate()`)
- Remove `on_chunk` callback from `generate()` — use `generate_stream()` instead
- Change `generate_stream()` return type from `tuple[AsyncIterator, Future]` to `ModelStreamResponse` — unifies with `prompt.stream()` which already returns `ModelStreamResponse`
- Introduce streaming type hierarchy (see `streaming.md`): `ActionStreamResponse[ChunkT, OutputT]` as base, `FlowStreamResponse[ChunkT, OutputT]` subclasses it, `ModelStreamResponse[OutputT]` subclasses `FlowStreamResponse` with `ChunkT` pinned to `ModelResponseChunk`
- Fix `Action.stream()` to return `ActionStreamResponse[ChunkT, OutputT]` instead of raw tuple
- Make `ActionRunContext` generic: `ActionRunContext[ChunkT]` so `send_chunk(chunk: ChunkT)` is type-safe — matches Go's `StreamCallback[Stream]` and JS's `ActionFnArg<S>` which are both typed on the chunk; currently Python uses `send_chunk(chunk: object)` which accepts anything
- Fix `Flow.stream()` to return `FlowStreamResponse[ChunkT, OutputT]` instead of raw tuple; fix `input: object` → `input: InputT`
- Fix `Channel` internals: (1) simplify to `Generic[T]` — the `R` close-result type parameter is unnecessary coupling; (2) fix `_pop()` falsy check `if not r` → `if r is None` — current code incorrectly stops iteration on any falsy chunk value (empty string, `0`, `False`)
- Tighten `Callable[..., Any]` on `define_prompt()` resolver params — current code uses `Callable[..., Any]` everywhere; correct parametrized forms are `Callable[[InputT, dict | None], str | Part | list[Part]]` for `system`/`prompt`, `Callable[[InputT, dict | None], list[Message]]` for `messages`, `Callable[[InputT, dict | None], list[Document]]` for `docs`
- `ai.retrieve()` should return `list[Document]` not `RetrieverResponse` — JS converts wire `DocumentData` to `Document` veneers before returning (`response.documents.map(d => new Document(d))`); Python currently leaks the raw wire type, breaking the retrieve → generate pipeline ergonomics
