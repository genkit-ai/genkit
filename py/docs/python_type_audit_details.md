# Class Audits — Action, ExecutablePrompt, GenerateResponseWrapper, Document

Method-by-method audit of the four most important classes users interact with
(after `Genkit` itself, which is covered in [GENKIT_CLASS_DESIGN.md](../GENKIT_CLASS_DESIGN.md)).

---

## 1. `Action`

`core/action/_action.py` — the foundational type. Everything in Genkit is an
Action. ~65 files reference it.

### Class shape

```
Action(Generic[InputT, OutputT, ChunkT])
  ├── Properties (read-only): kind, name, description, metadata, input_type, is_async
  ├── Properties (read/write!): input_schema, output_schema
  ├── run(input, on_chunk, context, _telemetry_labels)         → ActionResponse
  ├── arun(input, on_chunk, context, on_trace_start, ...)      → ActionResponse
  ├── arun_raw(raw_input, on_chunk, context, on_trace_start, ...) → ActionResponse
  └── stream(input, context, telemetry_labels, timeout)        → tuple[AsyncIterator, Future]
```

### JS comparison

JS `Action` is a **type alias**, not a class — it's a callable function with
attached properties:

```typescript
type Action<I, O, S> = ((input?, options?) => Promise<O>) & {
  __action: ActionMetadata<I, O, S>;
  run(input?, options?): Promise<ActionResult<O>>;
  stream(input?, opts?): StreamingResponse<O, S>;
};
```

Key differences:
- **JS actions are callable** — `await action(input)` works. Python requires
  `await action.arun(input)`.
- **JS has one `run()` method** that returns `ActionResult`. Python has three:
  `run()` (sync), `arun()` (async), `arun_raw()` (async + validation).
- **JS `stream()` returns `StreamingResponse`** with `.stream` and `.response`
  properties. Python returns a raw tuple.

### `run()`

```python
# Today
def run(
    self,
    input: InputT | None = None,
    on_chunk: StreamingCallback | None = None,
    context: dict[str, object] | None = None,
    _telemetry_labels: dict[str, object] | None = None,
) -> ActionResponse[OutputT]
```

**Verdict: delete entirely.** Only exists to support sync flow/tool wrappers
(2 callsites in `_registry.py`). The framework is async-first — sync user
functions should be auto-wrapped with `ensure_async()` at registration time
instead of maintaining a parallel sync execution path. JS and Go don't have
this because they don't have separate sync/async function types.

### `arun()`

```python
# Today
async def arun(
    self,
    input: InputT | None = None,
    on_chunk: StreamingCallback | None = None,
    context: dict[str, object] | None = None,
    on_trace_start: Callable[[str, str], None] | None = None,
    _telemetry_labels: dict[str, object] | None = None,
) -> ActionResponse[OutputT]
```

**Issues:**
1. **`on_chunk` is a JS callback pattern leaking into the public API.** Python's
   streaming convention is `async for` (async iterators), not callbacks. `on_chunk`
   is internal plumbing — `Action.stream()` already wraps it into a `Channel`
   (async iterator) for users. The only external caller passing `on_chunk` directly
   is `core/reflection.py` (Dev UI server). Regular users should never see this
   parameter; they should use `stream()` instead.
2. **`on_trace_start` is internal Dev UI plumbing** — only called by
   `core/reflection.py` to grab trace/span IDs for the Dev UI response. No user
   ever passes this. Shouldn't be on the public method.
3. `_telemetry_labels` — same underscore issue.
4. `input` is optional but many actions require it — fails at runtime.

### `arun_raw()`

```python
# Today
async def arun_raw(
    self,
    raw_input: InputT | None = None,
    on_chunk: StreamingCallback | None = None,
    context: dict[str, object] | None = None,
    on_trace_start: Callable[[str, str], None] | None = None,
    telemetry_labels: dict[str, object] | None = None,
) -> ActionResponse[OutputT]
```

**Issues:**
1. **Confusing name** — "raw" means "I'll validate for you via Pydantic." But
   `arun()` does NOT validate. So `arun_raw` does more work, not less. The name
   implies the opposite.
2. The only difference from `arun()` is Pydantic validation on input. This
   should be a flag, not a separate method.
3. Same `on_chunk`/`on_trace_start` callback leakage as `arun()`.
4. `telemetry_labels` has no underscore here but has underscore in `arun()`.
   Inconsistent.

### `stream()`

```python
# Today
def stream(
    self,
    input: InputT | None = None,
    context: dict[str, object] | None = None,
    telemetry_labels: dict[str, object] | None = None,
    timeout: float | None = None,
) -> tuple[AsyncIterator[ChunkT], asyncio.Future[ActionResponse[OutputT]]]
```

**Issues:**
1. **Returns a tuple** — not directly iterable. Must destructure:
   `chunks, future = action.stream(input)`. JS returns a `StreamingResponse`
   with `.stream` and `.response`.
2. **Not async** — synchronously creates a Channel and kicks off a task. This
   is fine mechanically but unexpected (an async operation that doesn't use `await`).
3. Creates a redundant `result_future` that wraps `stream.closed` — why not
   just expose `stream.closed` directly?

### `input_schema` / `output_schema` (property setters)

```python
@input_schema.setter
def input_schema(self, value: dict[str, object]) -> None:
    self._input_schema = value
    self._metadata[ActionMetadataKey.INPUT_KEY] = value

@output_schema.setter
def output_schema(self, value: dict[str, object]) -> None:
    self._output_schema = value
    self._metadata[ActionMetadataKey.OUTPUT_KEY] = value
```

**Issues:**
1. **Actions should be immutable after construction.** Mutable schemas invite
   subtle bugs — if someone stores a reference to `action.input_schema` and
   the schema later changes, they have stale data.
2. The setters exist for lazy-loaded prompts that set schema after registration.
   This is a hack around the construction order — the prompt system should pass
   schemas at construction time, or the action should accept a schema-factory.

### Proposed `Action` changes

```python
class Action(Generic[InputT, OutputT, ChunkT]):
    # All properties read-only (remove setters)
    kind: ActionKind       # read-only
    name: str              # read-only
    input_schema: dict     # read-only
    output_schema: dict    # read-only

    async def __call__(
        self,
        input: InputT | None = None,
        *,
        context: dict[str, object] | None = None,
    ) -> ActionResponse[OutputT]:
        """Primary execution method. Validates input, runs async."""

    def stream(
        self,
        input: InputT | None = None,
        *,
        context: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> StreamResponse[OutputT, ChunkT]:
        """Returns a StreamResponse (iterable + awaitable response)."""

    def __repr__(self) -> str:
        return f"Action(kind={self.kind}, name={self.name!r})"
```

**Removed:** `run()` (delete — only 2 internal callsites in `_registry.py`,
rewrite to use `__call__` with sync-async bridging), `arun()` (replaced by
`__call__`), `arun_raw()` (merge validation into `__call__`), schema setters,
`on_chunk`/`on_trace_start` from public signatures.

---

## 2. `ExecutablePrompt`

`blocks/prompt.py` — returned by `define_prompt()` and `prompt()`. The primary
way users work with prompts.

### Class shape

```
ExecutablePrompt(Generic[InputT, OutputT])
  ├── ref (property)                                → dict
  ├── __call__(input, opts: TypedDict | None)       → GenerateResponseWrapper
  ├── stream(input, opts, *, timeout)               → GenerateStreamResponse
  ├── render(input, opts)                           → GenerateActionOptions
  ├── as_tool()                                     → Action
  └── _ensure_resolved()                            → None (lazy loading)
```

25 constructor params. The constructor stores every prompt option as an instance
field — model, config, system, prompt, messages, output_format,
output_content_type, output_instructions, output_schema, output_constrained,
max_turns, return_tool_requests, metadata, tools, tool_choice, use, docs,
resources, plus internal fields (_name, _ns, _prompt_action, _cache_prompt).

### JS comparison

JS `ExecutablePrompt` is an **interface**:

```typescript
interface ExecutablePrompt<I, O, CustomOptions> {
  ref: { name: string; metadata?: Record<string, any> };
  (input?, opts?): Promise<GenerateResponse<O>>;
  stream(input?, opts?): GenerateStreamResponse<O>;
  render(input?, opts?): Promise<GenerateOptions<O>>;
  asTool(): Promise<ToolAction>;
}
```

Same methods, same shape. The difference is in how `opts` works.

### `__call__()`

```python
# Today
async def __call__(
    self,
    input: InputT | None = None,
    opts: PromptGenerateOptions | None = None,
) -> GenerateResponseWrapper[OutputT]
```

**Issues:**
1. **`opts` is a TypedDict** — `PromptGenerateOptions` is a dict-like type.
   This kills IDE autocomplete. Users must know the TypedDict keys by heart.
   Compare with kwargs:
   ```python
   # TypedDict (today) — no autocomplete on keys
   await prompt(input, opts={'model': 'gemini-2.0-flash', 'config': {...}})

   # Kwargs (proposed) — full autocomplete
   await prompt(input, model='gemini-2.0-flash', config={...})
   ```
2. **JS does the same thing** — `opts` is an object parameter there too. But
   TypeScript has much better autocomplete for object literals. Python
   TypedDicts don't get the same treatment from IDEs.

### `stream()`

```python
# Today
def stream(
    self,
    input: InputT | None = None,
    opts: PromptGenerateOptions | None = None,
    *,
    timeout: float | None = None,
) -> GenerateStreamResponse[OutputT]
```

**Clean.** Returns `GenerateStreamResponse` (not a tuple). This is correct —
it's the one place streaming is done right. The irony is that
`Genkit.generate_stream()` doesn't use this type but `ExecutablePrompt.stream()`
does.

### `render()`

```python
# Today
async def render(
    self,
    input: InputT | dict[str, Any] | None = None,
    opts: PromptGenerateOptions | None = None,
) -> GenerateActionOptions
```

**Issues:**
1. **220 lines of merging logic** — the method body is enormous. It merges
   config, model, tools, output options, tool_choice, return_tool_requests,
   max_turns, metadata, docs, resources, and messages from three sources
   (prompt defaults, opts overrides, and input rendering). This is the
   complexity center of the entire prompt system.
2. `input` accepts `InputT | dict[str, Any]` — mixed typing. Should be one
   or the other. The method body has 4 branches to handle different input types
   (None, dict, Pydantic v2, Pydantic v1, fallback cast).

### `as_tool()`

```python
# Today
async def as_tool(self) -> Action
```

**Clean.** Simple lookup. Minor naming difference from JS (`asTool`).

### `_ensure_resolved()`

```python
async def _ensure_resolved(self) -> None
```

**Issues:**
1. **Lazy loading that can fail** — if the prompt was created via `ai.prompt(name)`,
   it's unresolved until first use. The first `__call__`, `stream`, `render`, or
   `as_tool` triggers resolution. If the prompt file doesn't exist, the error
   appears at call time, not at construction time.
2. **Copies all fields from resolved prompt** — 20 field assignments. If a new
   field is added to `ExecutablePrompt`, someone must remember to add it here too.
   This is fragile.

### Proposed `ExecutablePrompt` changes

```python
class Prompt(Generic[InputT, OutputT]):
    """Renamed from ExecutablePrompt (shorter, clearer)."""

    @property
    def ref(self) -> PromptRef: ...

    async def __call__(
        self,
        input: InputT | None = None,
        *,
        model: str | None = None,
        config: dict | GenerationCommonConfig | None = None,
        tools: list[str] | None = None,
        tool_choice: ToolChoice | None = None,
        return_tool_requests: bool | None = None,
        max_turns: int | None = None,
        context: dict[str, object] | None = None,
        output_schema: type | None = None,
        output_format: str | None = None,
        docs: list[DocumentData] | None = None,
    ) -> GenerateResponse[OutputT]:
        """Execute the prompt. Flat kwargs instead of opts TypedDict."""

    def stream(
        self,
        input: InputT | None = None,
        *,
        # same kwargs as __call__
        timeout: float | None = None,
    ) -> GenerateStreamResponse[OutputT]: ...

    async def render(
        self,
        input: InputT | None = None,
        *,
        # same kwargs as __call__
    ) -> GenerateOptions: ...

    async def as_tool(self) -> Action: ...
```

**Key changes:**
- Rename `ExecutablePrompt` → `Prompt` (shorter).
- Replace `opts: TypedDict` with flat kwargs for IDE autocomplete.
- Simplify `render()` — extract merging logic into a shared helper.

---

## 3. `GenerateResponseWrapper`

`blocks/model.py` — the response users get from `generate()`. The thing they
interact with most after calling the model.

### Class shape

```
GenerateResponseWrapper(GenerateResponse, Generic[OutputT])
  ├── Private: _message_parser, _schema_type
  ├── message: MessageWrapper | None
  ├── text (cached_property)                → str
  ├── output (cached_property)              → OutputT
  ├── messages (cached_property)            → list[Message]
  ├── tool_requests (cached_property)       → list[ToolRequestPart]
  ├── interrupts (cached_property)          → list[ToolRequestPart]
  ├── assert_valid()                        → None (PLACEHOLDER)
  └── assert_valid_schema()                 → None (PLACEHOLDER)
```

### JS comparison

JS `GenerateResponse<O>` has everything Python has, plus:

| Property/Method | JS | Python |
|---|---|---|
| `text` | getter | cached_property |
| `output` | getter | cached_property |
| `reasoning` | getter | **missing** |
| `media` | getter | **missing** |
| `data` | getter | **missing** |
| `toolRequests` | getter | cached_property |
| `interrupts` | getter | cached_property |
| `messages` | getter | cached_property |
| `model` | field | **missing** |
| `raw` | field | **missing** |
| `assertValid()` | **implemented** | **placeholder (TODO)** |
| `assertValidSchema()` | **implemented** | **placeholder (TODO)** |
| `isValid()` | method | **missing** |
| `toJSON()` | method | Pydantic handles it |

### `__init__()`

```python
# Today
def __init__(
    self,
    response: GenerateResponse,
    request: GenerateRequest,
    message_parser: Callable[[MessageWrapper], object] | None = None,
    schema_type: type[BaseModel] | None = None,
) -> None
```

**Issues:**
1. Wraps a `GenerateResponse` but copies all fields into `super().__init__()`.
   Could just store the response and delegate. The copy-and-reconstruct
   pattern is fragile — if `GenerateResponse` adds a field, this breaks.
2. `message_parser` and `schema_type` are internal — users never pass these.
   They should be keyword-only or prefixed.

### `assert_valid()` / `assert_valid_schema()`

```python
def assert_valid(self) -> None:
    # TODO(#4343): implement
    pass

def assert_valid_schema(self) -> None:
    # TODO(#4343): implement
    pass
```

**Issue:** Empty placeholders since initial implementation. JS has these
fully implemented — they check for empty responses, missing messages,
malformed content, and schema violations. These are important for production
use — without them, users can't validate responses programmatically.

### `messages`

```python
@cached_property
def messages(self) -> list[Message]:
    if self.message is None:
        return list(self.request.messages) if self.request else []
    return [
        *(self.request.messages if self.request else []),
        self.message._original_message,  # private field access!
    ]
```

**Issue:** Accesses `self.message._original_message` (private field on
`MessageWrapper`). Should expose a public method on `MessageWrapper` for this,
like `.unwrap()` or `.to_message()`.

### `output`

```python
@cached_property
def output(self) -> OutputT:
    if self._message_parser and self.message is not None:
        parsed = self._message_parser(self.message)
    else:
        parsed = extract_json(self.text)

    if self._schema_type is not None and parsed is not None and isinstance(parsed, dict):
        return cast(OutputT, self._schema_type.model_validate(parsed))

    return cast(OutputT, parsed)
```

**Issue:** Falls back to `extract_json(self.text)` when no parser is set. This
regex-based JSON extraction is fragile — it scans the text for `{...}` or
`[...]`. If the model returns markdown with JSON in a code fence, this might
extract it or might not. JS has the same pattern, so this is cross-language
consistent at least.

### Proposed `GenerateResponseWrapper` changes

```python
class GenerateResponse(Generic[OutputT]):
    """Rename from GenerateResponseWrapper (drop 'Wrapper' suffix)."""

    # Existing
    text: str                              # property
    output: OutputT                        # property
    messages: list[Message]                # property
    tool_requests: list[ToolRequestPart]   # property
    interrupts: list[ToolRequestPart]      # property

    # Add (parity with JS)
    reasoning: str                         # for chain-of-thought models
    media: MediaPart | None                # first media part
    data: OutputT | None                   # first data part
    model: str | None                      # which model generated this

    # Implement
    def assert_valid(self) -> None: ...            # actually check response
    def assert_valid_schema(self) -> None: ...     # actually check schema
    def is_valid(self) -> bool: ...                # non-throwing version
```

---

## 4. `Document`

`blocks/document.py` — used by every retrieval, embedding, and reranking
operation. ~25 files reference it.

### Class shape

```
Document(DocumentData)
  ├── text()                                → str  (METHOD, not property!)
  ├── media()                               → list[Media]
  ├── data()                                → str
  ├── data_type()                           → str | None
  ├── get_embedding_documents(embeddings)   → list[Document]
  ├── from_document_data(data)              → Document  (static)
  ├── from_text(text, metadata)             → Document  (static)
  ├── from_media(url, content_type, meta)   → Document  (static)
  └── from_data(data, data_type, metadata)  → Document  (static)
```

### JS comparison

| Member | JS | Python |
|---|---|---|
| `text` | **getter** (property) | **method** `text()` |
| `media` | **getter** (property) | **method** `media()` |
| `data` | **getter** (property) | **method** `data()` |
| `dataType` | **getter** (property) | **method** `data_type()` |
| `toJSON()` | method | (Pydantic handles) |
| `getEmbeddingDocuments()` | method | method |
| `fromText()` | static | static |
| `fromMedia()` | static | static |
| `fromData()` | static | static |

### `text()` — method vs property

```python
# Today (Python)
def text(self) -> str:
    texts = []
    for p in self.content:
        part = p.root if hasattr(p, 'root') else p
        text_val = getattr(part, 'text', None)
        if isinstance(text_val, str):
            texts.append(text_val)
    return ''.join(texts)
```

```typescript
// JS — property
get text(): string {
    return this.content.map((part) => part.text || '').join('');
}
```

**Issues:**
1. **The single most confusing inconsistency in the SDK.** `MessageWrapper.text`
   is a property. `GenerateResponseWrapper.text` is a property.
   `GenerateResponseChunkWrapper.text` is a property. `Document.text()` is a
   method. Users will write `doc.text` (no parens) and get a bound method
   reference instead of a string. No error, no warning, just silent bugs.
2. **Breaking change to fix** — this is a public API. Changing from method to
   property will break every call site that uses `doc.text()`. But the
   inconsistency is worse than the break.
3. Same issue applies to `media()`, `data()`, `data_type()`.

### `data()` — calls `text()` twice

```python
def data(self) -> str:
    if self.text():      # first call — scans all content
        return self.text()  # second call — scans all content again
    if self.media():
        return self.media()[0].url
    return ''
```

**Issue:** Scans content twice. Should cache or store the result. Not a
correctness bug but wasteful. Same issue with `data_type()` calling `text()`
and `media()` again.

### Constructor — deep copies

```python
def __init__(
    self,
    content: list[DocumentPart],
    metadata: dict[str, Any] | None = None,
) -> None:
    doc_content = deepcopy(content)
    doc_metadata = deepcopy(metadata)
    super().__init__(content=doc_content, metadata=doc_metadata)
```

**Issue:** Always deep-copies content and metadata. JS does the same, so
this is cross-language consistent. But in Python, `deepcopy` on Pydantic
models is expensive. For large documents (e.g., embedding pipelines with
thousands of documents), this could be a performance bottleneck.

### Proposed `Document` changes

```python
class Document(DocumentData):
    @property
    def text(self) -> str: ...          # Change to property

    @property
    def media(self) -> list[Media]: ... # Change to property

    @property
    def data(self) -> str: ...          # Change to property

    @property
    def data_type(self) -> str | None: ...  # Change to property

    # Static factories stay the same
    @staticmethod
    def from_text(text: str, metadata: dict | None = None) -> Document: ...
    @staticmethod
    def from_media(url: str, content_type: str | None = None, ...) -> Document: ...
    @staticmethod
    def from_data(data: str, data_type: str | None = None, ...) -> Document: ...
```

**Key changes:**
- All accessors become `@property` (or `@cached_property` for perf).
- Breaking change for `text()`, `media()`, `data()`, `data_type()` call sites.
- Consider lazy `@cached_property` to avoid scanning content multiple times.

---

## 5. `GenerateStreamResponse`

**File:** `blocks/prompt.py` (lines 414–539)
**Base class:** `Generic[OutputT]`

### Class shape

```python
class GenerateStreamResponse(Generic[OutputT]):
    _channel: Channel[GenerateResponseChunkWrapper, GenerateResponseWrapper[OutputT]]
    _response_future: asyncio.Future[GenerateResponseWrapper[OutputT]]

    @property stream -> AsyncIterable[GenerateResponseChunkWrapper]
    @property response -> Awaitable[GenerateResponseWrapper[OutputT]]
```

Two properties, two private fields. That's the entire class.

### JS comparison

```typescript
// js/ai/src/generate.ts
export interface GenerateStreamResponse<O extends z.ZodTypeAny = z.ZodTypeAny> {
  get stream(): AsyncIterable<GenerateResponseChunk>;
  get response(): Promise<GenerateResponse<O>>;
}
```

JS has the identical interface — `stream` + `response`. But critically, JS uses
this type everywhere: both `generateStream()` and `prompt.stream()` return it.

### Go comparison

Go has no wrapper class. `GenerateStream()` returns `iter.Seq2[*ModelStreamValue, error]`
— a native Go iterator. Each yielded `ModelStreamValue` has either `.Chunk` (streaming)
or `.Done == true` with `.Response` (final). Go-idiomatic, no need for a wrapper.

### Issue 1: Not used by `Genkit.generate_stream()`

This is the biggest problem. The main streaming entry point returns a raw tuple:

```python
# ai/_aio.py
def generate_stream(self, ...) -> tuple[
    AsyncIterator[GenerateResponseChunkWrapper],
    asyncio.Future[GenerateResponseWrapper[Any]],
]:
```

But `ExecutablePrompt.stream()` returns `GenerateStreamResponse`. So there are
two inconsistent streaming APIs in the same SDK:

```python
# Prompt streaming — nice wrapper
result = prompt.stream({"topic": "AI"})
async for chunk in result.stream:
    print(chunk.text)
final = await result.response

# Genkit.generate_stream() — raw tuple
stream, future = ai.generate_stream(prompt="hello")
async for chunk in stream:
    print(chunk.text)
final = await future
```

JS doesn't have this split — both paths return `GenerateStreamResponse`.

### Issue 2: Not directly iterable

You can't do `async for chunk in result:` — you must access `.stream` first.
Python convention for iterable wrappers is to implement `__aiter__`:

```python
# Current — requires .stream access
async for chunk in result.stream:
    print(chunk.text)

# Expected Pythonic pattern
async for chunk in result:
    print(chunk.text)
```

JS has the same `.stream` access pattern, but Python's `async for` protocol
makes direct iteration a stronger convention.

### Issue 3: Lives in wrong module

Defined in `blocks/prompt.py` even though it's a general streaming response type.
It's not prompt-specific — `Genkit.generate_stream()` should use it too.
Should live in `blocks/generate.py` or `blocks/model.py`.

### Issue 4: No `__await__`

You can't `await` the response directly on the object:

```python
# Current — must access .response
final = await result.response

# Could support direct await
final = await result
```

This is a minor convenience but makes the object more Pythonic.

### Issue 5: No `__repr__`

`repr(result)` gives `<GenerateStreamResponse object at 0x...>`. Should show
useful state (e.g., whether stream is consumed, whether response is resolved).

### Proposed `GenerateStreamResponse` changes

1. **Wire into `Genkit.generate_stream()`** — return `GenerateStreamResponse`
   instead of raw tuple. This is the highest-priority fix. One streaming API,
   not two.

2. **Add `__aiter__`** — delegate to `self._channel` so `async for chunk in result:`
   works directly.

3. **Add `__await__`** — delegate to `self._response_future` so `final = await result`
   works as a shortcut for `await result.response`.

4. **Move to `blocks/generate.py`** or a shared module — it's not prompt-specific.

5. **Rename to `StreamResponse`** — shorter, matches the pattern of removing
   redundant prefixes (`GenerateResponseWrapper` → `GenerateResponse`).

6. **Add `__repr__`** — show stream/response state.

After these changes:

```python
# Unified streaming API
result = ai.generate_stream(prompt="hello")
# OR
result = prompt.stream({"topic": "AI"})

# Direct iteration (no .stream needed)
async for chunk in result:
    print(chunk.text)

# Direct await (no .response needed)
final = await result

# .stream and .response still work for explicit access
async for chunk in result.stream:
    print(chunk.text)
final = await result.response
```

---

## 6. `ToolInterruptError`

**File:** `blocks/tools.py` (lines 172–188)
**Base class:** `Exception`

### Class shape

```python
class ToolInterruptError(Exception):
    metadata: dict[str, Any]

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.metadata = metadata or {}
```

One field, one constructor. No methods, no `__str__`, no `__repr__`.

### JS comparison

```typescript
// js/ai/src/tool.ts
export class ToolInterruptError extends Error {
  constructor(readonly metadata?: Record<string, any>) {
    super();
    this.name = 'ToolInterruptError';
  }
}
```

Same shape but sets `this.name`. Both extend base error (not framework error).
JS comment: "It's meant to be caught by the framework, not public API."

### Go comparison

```go
// go/ai/tools.go (unexported)
type toolInterruptError struct {
    Metadata map[string]any
}

func (e *toolInterruptError) Error() string {
    if e.Metadata != nil {
        data, _ := json.MarshalIndent(e.Metadata, "", "  ")
        return fmt.Sprintf("tool execution interrupted: \n\n%s", string(data))
    }
    return "tool execution interrupted"
}

func IsToolInterruptError(err error) (bool, map[string]any) { ... }
```

Go is the best here: unexported type (can't be constructed by users),
public `IsToolInterruptError()` helper for checking, and a useful `Error()` string.

### Issue 1: Extends `Exception` not `GenkitError`

The TODO at line 171 says it all:

```python
# TODO(#4346): make this extend GenkitError once it has INTERRUPTED status
```

This means `except GenkitError` won't catch tool interrupts. Users who write
broad Genkit error handlers will miss these. Blocked on adding an `INTERRUPTED`
status code to `StatusCodes`.

### Issue 2: No error message

```python
err = ToolInterruptError(metadata={"step": "confirm"})
str(err)   # => ''
repr(err)  # => 'ToolInterruptError()'
```

Compare Go: `"tool execution interrupted: \n\n{\"step\": \"confirm\"}"` — actually
useful in logs. Python's version is silent, which makes debugging painful.

### Issue 3: `metadata` should be keyword-only

```python
# Currently allows positional
ToolInterruptError({"key": "val"})

# Should require keyword
ToolInterruptError(metadata={"key": "val"})
```

All other error constructors in the SDK are being moved to keyword-only.
This should follow.

### Issue 4: No `__repr__`

As noted above, `repr()` is useless. Should show metadata.

### Issue 5: Mutable default via `or {}`

```python
self.metadata = metadata or {}
```

This creates a new dict each time (which is fine), but the pattern is inconsistent
with the rest of the codebase which uses `field(default_factory=dict)` for dataclasses
or explicit `None` checks. Minor.

### Proposed `ToolInterruptError` changes

1. **Extend `GenkitError`** once `StatusCodes.INTERRUPTED` exists (unblock #4346).
   This gives: status code, serialization, cause chaining for free.

2. **Add `__str__`** — `"tool execution interrupted"` + metadata dump (match Go).

3. **Add `__repr__`** — `ToolInterruptError(metadata={'step': 'confirm'})`.

4. **Make `metadata` keyword-only**:
   ```python
   def __init__(self, *, metadata: dict[str, Any] | None = None) -> None:
   ```

5. **Consider Go pattern** — make the class private (`_ToolInterruptError`) with
   a public `is_tool_interrupt(err)` helper, since the JS comment says "not public
   API." Python can't fully hide it (users need `except ToolInterruptError`), but
   the Go pattern is worth noting.

After these changes:

```python
class ToolInterruptError(GenkitError):
    def __init__(self, *, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(status=StatusCodes.INTERRUPTED, message="tool execution interrupted")
        self.metadata: dict[str, Any] = metadata or {}

    def __str__(self) -> str:
        if self.metadata:
            return f"tool execution interrupted: {json.dumps(self.metadata, indent=2)}"
        return "tool execution interrupted"

    def __repr__(self) -> str:
        return f"ToolInterruptError(metadata={self.metadata!r})"
```

---

## Summary of all issues

### High priority (user-facing, correctness, or API consistency)

| Class | Issue | Effort |
|---|---|---|
| `Action` | `stream()` returns tuple instead of iterable object | medium |
| `Action` | No `__call__` — can't do `await action(input)` | low |
| `Action` | `on_chunk` callback on public API — Python uses `async for` not callbacks | medium |
| `Action` | `arun()` vs `arun_raw()` confusing, inconsistent naming | medium |
| `Action` | Mutable `input_schema`/`output_schema` setters | low |
| `ExecutablePrompt` | `opts: TypedDict` kills autocomplete | medium |
| `ExecutablePrompt` | `render()` is 220 lines of merging | refactor |
| `GenerateResponseWrapper` | `assert_valid()`/`assert_valid_schema()` empty | medium |
| `GenerateResponseWrapper` | Missing `reasoning`, `media`, `data` | low |
| `GenerateResponseWrapper` | Missing `model` field | low |
| `GenerateStreamResponse` | Not used by `Genkit.generate_stream()` — two streaming APIs | medium |
| `GenerateStreamResponse` | Not directly iterable (no `__aiter__`) | low |
| `ToolInterruptError` | Extends `Exception` not `GenkitError` — blocked on #4346 | medium |
| `Document` | `text()` is method, not property — inconsistent | **breaking** |

### Medium priority (engineering quality)

| Class | Issue | Effort |
|---|---|---|
| `Action` | No `__repr__` | low |
| `Action` | `_telemetry_labels` inconsistent underscore | low |
| `Action` | `on_trace_start` Dev UI plumbing leaked into public API | low |
| `ExecutablePrompt` | 25 constructor params | refactor |
| `ExecutablePrompt` | `_ensure_resolved()` copies 20 fields — fragile | refactor |
| `GenerateResponseWrapper` | Accesses `message._original_message` | low |
| `GenerateResponseWrapper` | Constructor copies fields from response — fragile | refactor |
| `GenerateStreamResponse` | Lives in `blocks/prompt.py` — not prompt-specific | low |
| `ToolInterruptError` | No `__str__` — empty string in logs | low |
| `ToolInterruptError` | `metadata` should be keyword-only | low |
| `Document` | `data()` calls `text()` twice | low |
| `Document` | `deepcopy` on every construction — perf risk | low |

### Low priority (nice to have)

| Class | Issue | Effort |
|---|---|---|
| `Action` | `run()` sync method — remove entirely (2 internal callsites) | low |
| `ExecutablePrompt` | Rename to `Prompt` | **breaking** |
| `GenerateResponseWrapper` | Rename to `GenerateResponse` | **breaking** |
| `GenerateResponseWrapper` | Add `is_valid()` non-throwing check | low |
| `GenerateStreamResponse` | No `__await__` for direct `await result` | low |
| `GenerateStreamResponse` | Rename to `StreamResponse` | **breaking** |
| `ToolInterruptError` | No `__repr__` | low |
