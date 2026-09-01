# Python Streaming Design

> **Status**: design proposal — see pre-review action items at the bottom for gaps between this design and the current implementation.

---

## Model

Go and JS expose streaming as a single iterator that interleaves chunks and the final response. Python diverges deliberately: every streaming call returns a **two-channel wrapper object** with separate properties for chunks and the final response.

```python
result = flow.stream(input)

async for chunk in result.stream:    # AsyncIterable[ChunkT]
    print(chunk)

response = await result.response     # Awaitable[OutputT]
```

This avoids the awkward "last item is the response" sentinel pattern that Go uses (`iter.Seq2[*StreamingFlowValue[S, O], error]`) and lets callers consume the stream and the response independently — e.g. start displaying chunks while also `await`-ing the final value in a separate task.

---

## Type hierarchy

Three concrete wrapper classes, one inheritance chain:

```
ActionStreamResponse[ChunkT, OutputT]          ← base (action.stream())
    └── FlowStreamResponse[ChunkT, OutputT]    ← flow.stream()
            └── GenerateStreamResponse[OutputT] ← generate_stream(), prompt.stream()
                                                   ChunkT pinned to GenerateResponseChunk
                                                   OutputT wrapped in GenerateResponse[OutputT]
```

```python
from typing import Generic, AsyncIterable, Awaitable
ChunkT = TypeVar('ChunkT')
OutputT = TypeVar('OutputT')

class ActionStreamResponse(Generic[ChunkT, OutputT]):
    @property
    def stream(self) -> AsyncIterable[ChunkT]: ...
    @property
    def response(self) -> Awaitable[OutputT]: ...

class FlowStreamResponse(ActionStreamResponse[ChunkT, OutputT]):
    pass  # same interface, narrows the source

class GenerateStreamResponse(FlowStreamResponse[GenerateResponseChunk, GenerateResponse[OutputT]]):
    # ChunkT is pinned — generate always emits GenerateResponseChunk
    # OutputT is the user's schema type (e.g. MyModel), wrapped in GenerateResponse
    pass
```

`GenerateStreamResponse[OutputT]` is effectively `FlowStreamResponse[GenerateResponseChunk, GenerateResponse[OutputT]]` with the chunk type fixed. This lets callers write `async for chunk in result.stream` and get `GenerateResponseChunk` objects with `.text`, `.index`, etc. without needing to annotate the type themselves.

---

## Surfaces

### `action.stream()`

```python
action.stream(
    input: InputT | None = None,
    *,
    context: dict[str, object] | None = None,
    telemetry_labels: dict[str, object] | None = None,
    timeout: float | None = None,
) -> ActionStreamResponse[ChunkT, OutputT]
```

```python
result = my_action.stream(input_data)
async for chunk in result.stream:
    print(chunk)
output = await result.response
```

### `flow.stream()`

```python
flow.stream(
    input: InputT | None = None,
    *,
    context: dict[str, object] | None = None,
    timeout: float | None = None,
) -> FlowStreamResponse[ChunkT, OutputT]
```

```python
result = my_flow.stream({"query": "hello"})
async for chunk in result.stream:
    print(chunk)
final = await result.response
```

### `generate_stream()`

```python
# 4 overloads — see python_beta_api_proposal.md §2 for full signatures
def generate_stream(
    self,
    *,
    model: ModelReference[C] | str | None = None,
    output_schema: type[OutputT] | dict[str, object] | None = None,
    ...
) -> GenerateStreamResponse[OutputT]
```

```python
result = ai.generate_stream(
    model=gemini_flash,
    prompt="Tell me a story",
    output_schema=StorySchema,
)
async for chunk in result.stream:
    print(chunk.text, end="", flush=True)
story: GenerateResponse[StorySchema] = await result.response
print(story.output.title)
```

### `prompt.stream()`

```python
# On ExecutablePrompt[InputT, OutputT]
def stream(
    self,
    input: InputT | None = None,
    *,
    timeout: float | None = None,
) -> GenerateStreamResponse[OutputT]
```

```python
result = my_prompt.stream({"topic": "space"})
async for chunk in result.stream:
    print(chunk.text, end="")
response = await result.response
```

---

## Internal: `Channel[T]`

All streaming wrappers are backed by a `Channel[T]` — a thin async queue that bridges the producer (action implementation) and consumer (caller).

```python
class Channel(Generic[T]):
    async def send(self, chunk: T) -> None: ...      # producer pushes a chunk
    def close(self) -> None: ...                     # producer signals completion
    def set_response(self, value: Any) -> None: ...  # producer delivers final result
    def __aiter__(self) -> AsyncIterator[T]: ...     # consumer iterates chunks
```

**Key invariants**:
- `None` is the sentinel that signals the iterator to stop — chunk types must not be `None` (use `Optional`-wrapped types if needed).
- The response future is separate from the chunk channel — `await result.response` never needs to drain the stream first.
- `_pop()` must use `if r is None` (not `if not r`) — otherwise falsy chunks (empty string `""`, `0`, `False`) incorrectly terminate iteration. *(Pre-review action item — current code uses `if not r`.)*

**Current implementation** (`genkit.aio.channel`): `Channel` is typed as `Generic[T, R]` with a second type parameter `R` for the close-result type. The design simplifies this to `Generic[T]` — the close-result type adds coupling without benefit. The response is a separate `asyncio.Future` on the wrapper object, not baked into the channel.

---

## Producer interface

Action, flow, and model implementations emit chunks through `ActionRunContext[ChunkT]`, passed as the second argument to the action function:

```python
@ai.flow()
async def my_flow(input: str, ctx: ActionRunContext[str]) -> str:
    for word in input.split():
        await ctx.send_chunk(word)   # type-safe: ChunkT is str
    return input

ctx.is_streaming              # bool — False means caller didn't request a stream; send_chunk is a no-op
ctx.send_chunk(chunk: ChunkT) # pushes chunk to consumer; no-op if not streaming
ctx.context                   # dict[str, object] — request-scoped metadata
```

**Cross-language comparison**:

| | Producer interface | Notes |
|---|---|---|
| **Go** | `StreamCallback[Stream]` callback param (nil if not streaming) | Caller checks nil before calling |
| **JS** | `ActionFnArg<S>` + `FlowSideChannel<S>` — two separate types | Flows and actions have different producer objects |
| **Python** | `ActionRunContext[ChunkT]` — unified | Single class for actions, flows, and models; `is_streaming` replaces nil check |

**`ToolRunContext`**: Tools do not define their own chunk schema — they borrow the parent `generate()` call's callback. Therefore `ToolRunContext` is `ActionRunContext[object]` (ChunkT = `object`, explicitly untyped), matching JS's `ToolAction` which hardcodes the streaming type as `z.ZodTypeAny`.

---

## Transport layer

The reflection server (Dev UI ↔ Python runtime) uses **Server-Sent Events (SSE)** to forward chunks over HTTP. This is an implementation detail — it does not affect the consumer API. The `Channel` is the in-process abstraction; SSE is how it crosses the wire to the Dev UI during local development.

---

## Cross-language comparison

| Surface | Go | JS | Python |
|---|---|---|---|
| **action.stream()** | `action.Stream(ctx, input, cb)` — `cb StreamCallback[S]` | `action.stream(input)` → `ActionStreamResponse<S, O>` | `action.stream(input)` → `ActionStreamResponse[ChunkT, OutputT]` |
| **flow.stream()** | `flow.Stream(ctx, input)` → `iter.Seq2[*StreamingFlowValue[S,O], error]` | `flow.stream(input)` → `FlowStreamResponse<S, O>` | `flow.stream(input)` → `FlowStreamResponse[ChunkT, OutputT]` |
| **generate_stream()** | `genkit.GenerateStream(ctx, req)` → `iter.Seq2[*GenerateResponseChunk, error]` | `ai.generateStream(opts)` → `GenerateStreamResponse<O>` | `ai.generate_stream(...)` → `GenerateStreamResponse[OutputT]` |
| **prompt.stream()** | `prompt.Stream(ctx, input)` | `prompt.stream(input)` → `GenerateStreamResponse<O>` | `prompt.stream(input)` → `GenerateStreamResponse[OutputT]` |
| **chat.stream()** | n/a | `chat.sendStream(input)` → `GenerateStreamResponse<O>` | not yet implemented |
| **Chunk/response split** | Single iterator, last value is response | Two-channel wrapper object | Two-channel wrapper object |
| **Producer** | `StreamCallback[S]` func param | `ActionFnArg<S>` / `FlowSideChannel<S>` | `ActionRunContext[ChunkT]` |

---

## What's not implemented yet

- **`Chat.send_stream()`** — no streaming equivalent for `chat.send()`.
- **`Action.stream()`** — currently returns a raw tuple `(AsyncIterator, Awaitable)`, not `ActionStreamResponse`. Needs to be updated to return the wrapper.
- **`FlowWrapper.stream()`** — same: currently returns raw tuple. Needs to return `FlowStreamResponse[ChunkT, OutputT]`.
- **`Channel` cleanup** — needs two fixes: simplify to `Generic[T]` (drop `R`), and fix `_pop()` falsy sentinel check.
- **`ActionRunContext` generics** — currently `send_chunk(chunk: object)`. Needs to become `ActionRunContext[ChunkT]` with `send_chunk(chunk: ChunkT)` for type safety.
