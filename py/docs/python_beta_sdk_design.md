# Genkit Python SDK — Design Review

Related docs:
- [python_beta_api_proposal.md](./python_beta_api_proposal.md) — public API surface (what's importable)
- [python_package_reorg.md](./python_package_reorg.md) — internal package structure
- [python_type_audit_checklist.md](./python_type_audit_checklist.md) — type deletions/fixes
- [python_beta_sdk_audit.md](./python_beta_sdk_audit.md) — initial friction audit

## 1. Background

The Python SDK launched to match JS and Go feature timelines. It achieved feature parity, but the API surface was never designed independently. Patterns were ported from JS rather than designed natively for Python. 

The Python SDK is public but hasn't cut a stable release. The JS SDK went through a similar cleanup between v0.5 and v1.0, and the migration cost grew with each release. Python is earlier in that curve and changes are still cheap.

This doc covers **design decisions** — the "how" and "why" behind internal architecture choices. For the public import surface (the "what"), see [python_beta_api_proposal.md](./python_beta_api_proposal.md).

## 2. Principles

### Pythonic API conventions

The JS SDK uses options objects, camelCase, and callback patterns. "Pythonic" means a set of concrete conventions:

**Zero-to-one positional arguments.** Every public method allows at most one positional arg — the "obvious" one (e.g., `input` for a prompt call). Everything else is keyword-only via the `*` marker. This prevents positional abuse and makes call sites self-documenting:

```python
# Bad: what are these arguments?
ai.generate("gemini", "Hi", None, None, ["search"])

# Good: every argument is named
ai.generate(model="gemini", prompt="Hi", tools=["search"])
```

**Kwargs over options dicts.** JS groups parameters into an options object. Python has first-class keyword arguments. Dict-based configuration loses autocomplete, type checking, and discoverability. This applies to `generate()`, `prompt()`, and every public method.

**Flat imports, intentional boundaries.** Python has no access modifiers — any module is importable, and there's no way to enforce "private." This makes API boundary design a deliberate choice, not a language feature. Public entry points are `from genkit import ...` (app developers) and domain sub-modules like `genkit.model`, `genkit.retriever`, `genkit.tracing` (plugin authors). Internal modules use `_internal/` directories following the Pydantic v2 convention. There is no public `genkit.core` or `genkit.ai` import path — those are internal structure only. Full symbol lists are in [python_beta_api_proposal.md](./python_beta_api_proposal.md). The internal package structure is in [python_package_reorg.md](./python_package_reorg.md).

## 3. Initial Audit

While working on updated docs, we identified several friction points in the developer experience. 

For many of these friction points, there was a clear Pythonic standard to follow — keyword-only arguments on all methods, sequence protocol on `RetrieverResponse`, convenience properties like `response.media`, veneer aliasing (`GenerateResponseWrapper` → `GenerateResponse`), and cleanup of internal utilities from the public surface. More details here: [python_beta_sdk_audit.md](./python_beta_sdk_audit.md)

The remaining sections in this doc are open questions that need some discussion to resolve.

## 4. Public API surface & type architecture

Today there is no formal public/internal boundary. The documentation audit found samples importing from `genkit.core.action`, `genkit.blocks.model`, and `genkit.ai` — all internal paths that happen to work. This means any internal module rename or refactor is a breaking change for external developers, even if the public API hasn't changed.

**Resolved decisions:**

- **Single entry point.** `from genkit import ...` covers both app developers (~25 symbols) and plugin authors (~9 additional). No separate `genkit.types` or `genkit.plugin`. Domain sub-modules (`genkit.model`, `genkit.retriever`, `genkit.tracing`, etc.) provide wire-format types for plugin authors who need them.

- **No public `genkit.core`.** Internal packages (`core/`, `ai/`) use `_internal/` directories following Pydantic v2's convention. `genkit/__init__.py` re-exports everything users need. See [python_package_reorg.md](./python_package_reorg.md) for the full structure.

- **Veneer aliasing.** `GenerateResponseWrapper` → `GenerateResponse` via inheritance (so `isinstance` works). `MessageWrapper` stays as-is because it uses composition — aliasing would break `Message(role="user", content=[...])`. App developers get `MessageWrapper` via `response.messages` but never construct it directly.

- **`__all__` on every public `__init__.py`.** Enforced by `import-linter` in CI.

- **Internal code organization.** `blocks/` is deleted (merged into `ai/`). `aio/`, `lang/`, `types/` are deleted (absorbed into `core/_internal/`). `web/` renamed to `_web/`. See [python_package_reorg.md](./python_package_reorg.md).

Full symbol lists and rationale for each inclusion/exclusion: [python_beta_api_proposal.md](./python_beta_api_proposal.md).

## 5. Output configuration

The `generate()` method currently accepts output configuration multiple ways:

```python
# Way 1: Inline kwargs
await ai.generate(prompt="...", output_format="json", output_content_type="application/json",
                  output_instructions="Return valid JSON", output_constrained=True)

# Way 2: Config helper with generics
await ai.generate(prompt="...", output=Output(schema=MyModel))
```

**Flat kwargs vs. wrapper object.** We considered both approaches:

A **wrapper object** (`output=OutputConfig(Recipe)` or `output=Recipe`) bundles the schema with secondary options (format, constrained, instructions) into one param, reducing `generate()`'s parameter count. But it introduces a new type developers have to learn and import.

**Flat kwargs** (`output_schema=Recipe`) is the more Pythonic approach. Python functions embrace explicit parameters with defaults — `requests.get()` has 15+ kwargs, `json.dumps()` has 8, `subprocess.run()` has 12. No config objects. The secondary output params (`output_format`, `output_constrained`, `output_content_type`, `output_instructions`) stay as kwargs with sensible defaults — `output_format` auto-defaults to `'json'` when a schema is set, the rest default to `None` and are rarely used. The common case is just:

```python
response = await ai.generate(prompt="...", output_schema=Recipe)
response.output.name  # typed as str — IDE autocomplete works
```

No new types, no imports beyond the Pydantic model, and the 95% case is one kwarg.

**Recommendation.** Flat kwargs. Remove the `output` param. Keep `output_schema`, `output_format`, `output_constrained`, `output_content_type`, and `output_instructions` as individual keyword-only params. Use `@overload` so `output_schema: type[T]` parameterizes the return type. The wire-format `OutputConfig` remains an internal/plugin type — plugin authors use it when implementing model plugins to read output configuration from the `GenerateRequest`, but app developers never see it.

`output_schema` accepts three forms: a Pydantic model class (`type[T]` — the common case, gives typed returns), a raw JSON schema dict (`dict` — for dynamic schemas, returns `Any`), or a registered schema name (`str`, looked up from registry at runtime, returns `Any`). Only the Pydantic class form carries the generic type. 

**The same applies to `input_schema`.** With flat kwargs and overloads, `input_schema: type[T]` carries the generic directly — `Input[T]` can be removed:

```python
prompt = ai.define_prompt(
    name='recipe',
    input_schema=RecipeInput,
    output_schema=Recipe,
    prompt='Make a recipe for {dish}',
)

response = await prompt(RecipeInput(dish='pizza'))
response.output.name  # typed as str — IDE knows this is Recipe
```

Dotprompt should work in a similar way but with one additional nuance. When a schema is defined in a `.prompt` file's YAML frontmatter (`output: { schema: Recipe }`), the SDK uses it to constrain the model's JSON output at runtime. But the type checker doesn't know this — `.prompt` files can't carry Python type references — so `response.output` is `Any`. To get typed output, pass the schema at the call site:

```python
# Without output_schema — runtime parsing works, but typing is Any
recipe = ai.prompt('recipe')
response = await recipe({'food': 'pizza'})
response.output  # Any — no autocomplete

# With output_schema — typed
recipe = ai.prompt('recipe', output_schema=Recipe)
response = await recipe({'food': 'pizza'})
response.output.name  # str — IDE knows this is Recipe
```

This is inherent to Python's static type system. The redundancy (schema in both `.prompt` and Python) is the cost of typed output, and it's a cost every framework with external schema files pays. The flat kwarg `output_schema=Recipe` keeps this as lightweight as possible — no wrapper type needed, just name the class.

## 6. Streaming API

The SDK currently has two streaming patterns:

```python
# generate_stream() — returns a tuple
stream, future = ai.generate_stream(prompt="Tell me a story")
async for chunk in stream:
    print(chunk.text, end="")
response = await future

# prompt.stream() — returns an object with .stream accessor
result = prompt.stream({"topic": "AI"})
async for chunk in result.stream:
    print(chunk.text, end="")
response = await result.response
```

The Python standard for streaming is iterators — OpenAI, Anthropic, and every major Python SDK use them.

Genkit can't use a plain async generator (the OpenAI pattern) because Genkit responses carry more than text — structured output parsing, usage statistics, tool request handling, and the assembled `Message` for multi-turn conversations. A plain generator can't expose a final response object after iteration.

**Proposed streaming syntax:**

```python
# Simple case — looks identical to a plain generator
async for chunk in ai.generate_stream(prompt="Tell me a story"):
    print(chunk.text, end="")

# When you need the final response — assign, iterate, then access
result = ai.generate_stream(prompt="Tell me a story")
async for chunk in result:
    print(chunk.text, end="")
response = await result.response  # structured output, usage stats, tool requests
```

**What changes:**
- `generate_stream()` returns a directly iterable object (implements `__aiter__`) with a `.response` property for the final assembled response
- `prompt.stream()` uses the same pattern — one streaming convention across the SDK
- The tuple return is removed — no more destructuring

## 7. Sync and async support

Every Genkit Python method is `async def`. There is no sync API. This is a Python-specific problem — JS is inherently async, Go handles concurrency transparently with goroutines. Python is the only language where the developer has to explicitly choose.

The practical consequences: a Flask route handler can't call `ai.generate()` without managing an event loop. A Jupyter notebook cell needs `await` or `nest_asyncio` workarounds. A CLI script requires wrapping everything in `async def main()` and `ai.run_main()`. These are the most common entry points for developers trying Genkit for the first time.

For context, every major Python LLM SDK offers both sync and async: OpenAI and Anthropic ship dual clients (`OpenAI` / `AsyncOpenAI`), LangChain has dual methods (`.invoke()` / `.ainvoke()`), Google Cloud AI uses a separate async transport. Even Hugging Face, which is sync-only, made a deliberate choice. Genkit is the only async-only SDK in the ecosystem. A developer coming from OpenAI's `client.chat.completions.create()` — no `await`, no `async def` — hits immediate friction.

**Proposal.** Dual clients — `Genkit` (sync) and `AsyncGenkit` (async). This is the industry standard: OpenAI, Anthropic, and Cohere all ship it. The async client holds the real implementation; the sync client delegates to it. We prefer the dual-client pattern (separate classes) over dual methods (`generate()` / `agenerate()` on the same class) because it keeps each class's type signatures clean — every method on `Genkit` returns `T`, every method on `AsyncGenkit` returns `Awaitable[T]` — and avoids polluting autocomplete with `a`-prefixed duplicates of every method.

```python
from genkit import Genkit, AsyncGenkit

# Sync (scripts, Flask, notebooks)
ai = Genkit(plugins=[GoogleAI()])
response = ai.generate(model="googleai/gemini-2.0-flash", prompt="Hi")

# Async (FastAPI, high-concurrency)
ai = AsyncGenkit(plugins=[GoogleAI()])
response = await ai.generate(model="googleai/gemini-2.0-flash", prompt="Hi")
```

The maintenance cost is manageable: the sync client is auto-generated from the async client's method signatures, as OpenAI and Anthropic do. No duplicate implementation, no diverging logic.

## 8. Method signatures

The current `generate()` signature has 20 parameters:

```python
async def generate(
    self,
    model=None, prompt=None, system=None, messages=None,
    tools=None, return_tool_requests=None, tool_choice=None,
    tool_responses=None, config=None, max_turns=None,
    on_chunk=None, context=None,
    output_format=None, output_content_type=None,
    output_instructions=None, output_constrained=None, *,
    output=None, use=None, docs=None,
) -> GenerateResponseWrapper[Any]:
```

Pretty much none are keyword-only. The original decision of where to put the * in the first place seems arbitrary. Several params don't belong.

**What changes:**

- **Add `*`** (section 3) — all params keyword-only.
- **Keep output as flat kwargs** (section 5) — `output_schema`, `output_format`, `output_constrained` stay as individual kwargs with defaults. Remove the `output` param that accepted `OutputConfig | OutputConfigDict | Output[T]`. Net: same param count for output, but one way to configure instead of five.
- **Remove `on_chunk`** — `generate()` has a streaming callback parameter, but streaming belongs on `generate_stream()`.
- **Move `tool_responses` to `resume`** — only used when resuming from a tool interrupt. JS already groups this under a `resume` options object.

**After cleanup:**

```python
async def generate(
    self,
    *,
    model: str | None = None,
    prompt: str | Part | list[Part] | None = None,
    system: str | Part | list[Part] | None = None,
    messages: list[Message] | None = None,
    tools: list[str] | None = None,
    tool_choice: ToolChoice | None = None,
    return_tool_requests: bool | None = None,
    config: GenerationCommonConfig | dict | None = None,
    max_turns: int | None = None,
    context: dict[str, object] | None = None,
    output_schema: type[OutputT] | None = None,
    output_format: str | None = None,
    output_constrained: bool | None = None,
    use: list[ModelMiddleware] | None = None,
    docs: list[DocumentData] | None = None,
) -> GenerateResponse[OutputT]:
```

**`prompt.__call__()` also changes.** Today it takes a JS-style opts dict:

```python
# Before — opts dict, no autocomplete
response = await my_prompt({"name": "Ted"}, opts={"config": {"temperature": 0.4}})

# After — kwargs, full IDE support
response = await my_prompt({"name": "Ted"}, config={"temperature": 0.4})
```

The `opts` dict (a TypedDict with 16 fields) is replaced with individual kwargs:

```python
async def __call__(
    self,
    input: InputT | None = None,
    *,
    model: str | None = None,
    config: GenerationCommonConfig | dict[str, Any] | None = None,
    messages: list[Message] | None = None,
    docs: list[DocumentData] | None = None,
    tools: list[str] | None = None,
    tool_choice: ToolChoice | None = None,
    output_schema: type | dict[str, Any] | None = None,
    output_format: str | None = None,
    output_constrained: bool | None = None,
    return_tool_requests: bool | None = None,
    max_turns: int | None = None,
    use: list[ModelMiddleware] | None = None,
    context: dict[str, Any] | None = None,
) -> GenerateResponse[OutputT]:
```

`input` stays as the one positional arg (the template variables). Everything else is keyword-only. The `resume` options move to a separate `resume()` method or a `resume` kwarg (matching section 8's `generate()` cleanup). `on_chunk` is removed — streaming belongs on `prompt.stream()`.

**`generate_stream()` — after cleanup:**

```python
async def generate_stream(
    self,
    *,
    model: str | None = None,
    prompt: str | Part | list[Part] | None = None,
    system: str | Part | list[Part] | None = None,
    messages: list[Message] | None = None,
    tools: list[str] | None = None,
    tool_choice: ToolChoice | None = None,
    return_tool_requests: bool | None = None,
    config: GenerationCommonConfig | dict | None = None,
    max_turns: int | None = None,
    context: dict[str, object] | None = None,
    output_schema: type[OutputT] | None = None,
    output_format: str | None = None,
    output_constrained: bool | None = None,
    use: list[ModelMiddleware] | None = None,
    docs: list[DocumentData] | None = None,
) -> GenerateStreamResponse[OutputT]:
```

Same params as `generate()`. The return type changes from `tuple[AsyncIterator, Future]` to a single `GenerateStreamResponse` object that is directly async-iterable and exposes `.response` (see section 7).

**`retrieve()` — after cleanup:**

```python
async def retrieve(
    self,
    *,
    retriever: str,
    query: str | DocumentData,
    options: dict[str, object] | None = None,
) -> RetrieverResponse:
```

Already clean — just needs the `*` marker to enforce keyword-only. `retriever` and `query` become required (they were Optional before, but calling without them always fails).

**`embed()` — after cleanup:**

```python
async def embed(
    self,
    *,
    embedder: str,
    content: str | Document | DocumentData,
    metadata: dict[str, object] | None = None,
    options: dict[str, object] | None = None,
) -> list[Embedding]:
```

Same treatment — `*` marker, `embedder` and `content` become required.

## 9. Serialization cleanup — `GenkitBaseModel`

### The problem

Every Genkit type extends raw `pydantic.BaseModel`. Serialization to the wire
(camelCase JSON, no null fields) requires passing two flags every time:

```python
obj.model_dump(exclude_none=True, by_alias=True)
```

Nobody remembers both flags. So `codec.py` provides `dump_dict()` and `dump_json()`
wrappers. But call sites are split three ways:

| Pattern | Correct? | Count |
|---|---|---|
| `dump_dict(obj)` / `dump_json(obj)` | Yes (both flags) | ~20 calls across 13 files |
| `.model_dump(exclude_none=True, by_alias=True)` | Yes (both flags) | 5 calls |
| `.model_dump()` with partial or no flags | **No** | **11 calls** |

### The fix: `GenkitBaseModel`

Pydantic's `model_config` doesn't support `exclude_none` as a config key — it's
a parameter to `model_dump()`. So we override the methods to change the defaults:

```python
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

class GenkitBaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    def model_dump(self, *, exclude_none=True, by_alias=True, **kwargs):
        return super().model_dump(exclude_none=exclude_none, by_alias=by_alias, **kwargs)

    def model_dump_json(self, *, exclude_none=True, by_alias=True, **kwargs):
        return super().model_dump_json(exclude_none=exclude_none, by_alias=by_alias, **kwargs)
```

Now `obj.model_dump()` does the right thing. You can still override:
`obj.model_dump(exclude_none=False)` when you actually want nulls.

**Where it lives:** `genkit/core/_internal/_base.py` — Level 0 in the import DAG
(see §12). Zero genkit imports, no circular import risk.

**Not re-exported.** `GenkitBaseModel` is strictly internal. App developers
construct `Message(...)`, `Document(...)`, etc. and never see the base class.
Plugin authors extend exported types like `GenerationCommonConfig` or use plain
`BaseModel` for plugin-internal types.

### Migration plan

| Step | Scope | Risk |
|---|---|---|
| Create `GenkitBaseModel` in `genkit.core._internal._base` | 1 file | None |
| Change core schema types to inherit from it | ~10 files in `genkit/core/`, `genkit/blocks/` | Low — behavioral change only on direct `.model_dump()` calls |
| Audit the 11 inconsistent calls — some may intentionally want no aliases | Case-by-case | Medium — need to check if any internal-only paths rely on snake_case keys |
| Simplify `dump_dict`/`dump_json` | `codec.py` | Low |
| Remove `dump_dict`/`dump_json` from public API | `__init__.py` | None — already proposed for removal |

### Open questions

1. **Do any internal paths intentionally use snake_case keys?** The `prompt.py`
   calls that skip `by_alias` might be feeding data back into `model_validate()`,
   where snake_case is fine. Need to audit each of the 11 sites.

2. **`document.py` dedup hash** — uses `model_dump_json()` with no flags for
   equality comparison. If we change defaults, the hash changes for any model
   that has aliases. This could break dedup for in-flight data. Probably fine
   (dedup is ephemeral), but worth noting.

3. **Third-party model types** — e.g. Google AI SDK types that Genkit wraps.
   These won't inherit `GenkitBaseModel`, so `dump_dict()` still needs to handle
   the `isinstance(obj, BaseModel)` case with explicit flags. Or we only use
   `dump_dict` for third-party types and `.model_dump()` for our own.

## 10. `define_*` should accept raw Python types

### The problem

17 plugins and 8 core files call `to_json_schema()` manually before passing
schemas to `define_model`, `define_retriever`, etc.:

```python
# Current — every plugin does this:
from genkit.core.schema import to_json_schema

ai.define_model(
    name='my-model',
    metadata={'model': {'customOptions': to_json_schema(MyConfig)}},
    config_schema=to_json_schema(MyConfig),
    ...
)
```

This is unnecessary boilerplate. The framework should handle the conversion.

### Cross-language comparison

- **JS** — `toJsonSchema` is public at `genkit/schema` (alongside `parseSchema`,
  `validateSchema`, `JSONSchema`). But JS `defineModel` also accepts Zod schemas
  directly — plugins don't *have* to call `toJsonSchema` manually.
- **Go** — `jsonschema.Reflect()` is internal. `defineModel` in `ai/gen.go`
  accepts Go types and converts internally.
- **Python** — `to_json_schema` is public but lives at a deep path
  (`genkit.core.schema`). And `define_*` functions *require* pre-converted dicts.

Python is the only SDK where plugins are *forced* to call the schema conversion
themselves. JS has it public but optional; Go internalizes it entirely.

### The fix

`define_*` functions accept `type | dict | None` directly:

```python
# After — plugins just pass the type:
ai.define_model(
    name='my-model',
    config_schema=MyConfig,   # Python type, not JSON Schema dict
    ...
)
```

The framework calls `to_json_schema()` internally when building action metadata.
Same for `define_retriever`, `define_embedder`, `define_reranker`, `define_evaluator`.

`to_json_schema` moves to `core/_internal/_schema.py`. No plugin needs it.
`extract_json` moves to `core/_internal/_extract.py`. Zero plugin consumers —
only used by `formats/` internally.

### Migration

| Step | Scope | Risk |
|---|---|---|
| Update `define_*` signatures to accept `type \| dict \| None` | ~6 functions in ai/ | Low — dict passthrough preserves backward compat |
| Move `to_json_schema` calls inside `define_*` functions | Same 6 functions | Low |
| Move `schema.py` to `core/_internal/_schema.py` | 1 file | None |
| Move `extract.py` to `core/_internal/_extract.py` | 1 file | None |
| Update 16 plugins to drop `to_json_schema` import + calls | 16 plugin files | Medium — mechanical but wide |

## 11. `ErrorResponse` — internal type consolidation

Replaces 3 error wire format types (`HttpErrorWireFormat`,
`GenkitReflectionApiDetailsWireFormat`, `GenkitReflectionApiErrorWireFormat`).
Single Pydantic model with `message`, `status`, `details: dict | None`.
Internal only — used by the reflection server (`_web/_reflection.py`).

## 12. Import DAG

The internal import graph of the `genkit` package, simplified. Every new module
or dependency should be evaluated against this to prevent circular imports.

```
Level 0  (no genkit imports — leaf modules):
  core/_internal/_base.py      GenkitBaseModel
  core/_internal/_compat.py    StrEnum, override, wait_for backfills
  core/_internal/_schema.py    to_json_schema
  core/_internal/_extract.py   extract_json, extract_items
  core/_internal/_constants.py GENKIT_VERSION, GENKIT_CLIENT_HEADER
  core/_internal/_logging.py   get_logger

Level 1  (imports Level 0 only):
  core/_internal/_typing.py    60+ BaseModel classes (imports _compat + _base)
  core/error.py                GenkitError, UserFacingError, StatusCodes, Status
                               (absorbs status_types.py — imports _base only)

Level 2  (imports Level 0–1):
  core/action.py               Action, ActionRunContext, ActionMetadata, ActionKind,
                               ActionResponse (absorbs action_types.py)
  core/_internal/_registry.py  Registry
  core/_internal/_context.py   RequestData, ContextMetadata
  core/_internal/_environment.py  EnvVar, is_dev_environment
  core/_internal/_aio.py       Channel, run_async, ensure_async
  core/_internal/_http_client.py  per-event-loop httpx.AsyncClient cache
  core/plugin.py               Plugin ABC
  core/_internal/_flow.py      FlowWrapper (~50 lines)
  core/_internal/_background.py  BackgroundAction (imports action, error)
  core/_internal/_dap.py       DynamicActionProvider (imports action)

Level 3  (imports Level 0–2):
  ai/model.py                  define_model, GenerateResponseWrapper, etc.
  ai/retriever.py              define_retriever, RetrieverRef, etc.
  ai/embedding.py              define_embedder, EmbedderRef, etc.
  ai/evaluator.py              define_evaluator, EvaluatorRef
  ai/tools.py                  define_tool, ToolRunContext
  ai/prompt.py                 ExecutablePrompt, define_prompt
  ai/_internal/_generate.py    generate() orchestration, tool loop
  ai/_internal/_dotprompt.py   dotprompt template engine

Level 4  (imports Level 0–3):
  ai/_internal/_genkit.py      Genkit class body
  ai/_internal/_genkit_base.py Genkit __init__, server startup
  _web/_reflection.py          Dev UI ASGI app
  _web/_runtime.py             RuntimeManager
```

**Rules:**
- Each level may only import from levels below it.
- `core/` has zero imports from `ai/` or `_web/`.
- `ai/` has zero imports from `_web/`.
- All `_internal/` modules are plumbing — can change between versions. Parent packages re-export what's needed. `import-linter` blocks plugins from importing `_internal/` paths.
- `core/` has only 3 package-level files (`action.py`, `error.py`, `plugin.py`) — everything else is `_internal/`. These are stable abstractions listed in `core/__init__.py`'s `__all__`. They can still have `_`-prefixed private helpers inside — normal Python.
- Since there's no public `genkit.core` or `genkit.ai` import path, the split is for SDK developer clarity, not external API.
- Enforced by `import-linter` in CI (see [python_package_reorg.md](./python_package_reorg.md)).
