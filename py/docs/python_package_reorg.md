# Python SDK — Package Reorganization

Proposal to align the Python SDK's internal package structure with Go and JS,
enforce public/internal boundaries, and split oversized files.

Related docs:
- [python_beta_type_design.md](./python_beta_type_design.md) — type audit
- [python_type_audit_checklist.md](./python_type_audit_checklist.md) — checklist (33 types deleted, affects file contents below)
- [python_beta_api_proposal.md](./python_beta_api_proposal.md) — public API surface + `GenkitBaseModel` serialization fix + `define_*` accepts raw types (schema/extract internalized)
- [GENKIT_CLASS_DESIGN.md](../GENKIT_CLASS_DESIGN.md) — Genkit class

---

## Current state

```
genkit/                          7 sub-packages, 73 .py files
├── ai/                          god object + helpers (5 files, 4,500 lines)
├── aio/                         async utilities (4 files)
├── blocks/                      domain types (14 files, 7,800 lines)
│   └── formats/                 output format impls
├── core/                        framework internals (15 files, 5,500 lines)
│   ├── action/                  Action class, context, types
│   └── trace/                   OTel exporters/processors
├── lang/                        deprecation helpers (1 file)
├── types/                       barrel re-export
├── web/                         ASGI server management (8 files)
│   └── manager/
├── __init__.py                  public API barrel
├── codec.py                     JSON serialization helpers
├── model_types.py               GenerationCommonConfig + api_key helpers
└── testing.py                   test doubles
```

### Problems

1. **`blocks/` doesn't exist in Go or JS.** Both put domain types in `ai/`.
   Python's extra layer creates the question "does this go in `ai/` or `blocks/`?"

2. **Orphan packages.** `aio/` (4 files), `lang/` (1 file), `types/` (barrel).
   None earn their existence as top-level packages.

3. **Giant files.** `blocks/prompt.py` (2,446 lines), `ai/_registry.py` (1,680),
   `ai/_aio.py` (1,164). JS/Go equivalents are 600–900 lines.

4. **No boundary enforcement.** Plugins import from `genkit.core.action._action`,
   `genkit.blocks.model`, `genkit.ai._runtime` — deep internal paths. No `__all__`
   on most `__init__.py` files.

5. **Loose root files.** `codec.py` and `model_types.py` are orphans that belong
   in `core/` and `ai/` respectively.

---

## Proposed structure

```
genkit/
├── __init__.py              public API barrel (__all__ defined)
├── ai/                      AI domain types + Genkit class
│   ├── __init__.py          public exports (__all__ defined)
│   ├── prompt.py            ExecutablePrompt + define_prompt
│   ├── streaming.py         GenerateStreamResponse
│   ├── model.py             GenerateResponseWrapper, ChunkWrapper, MessageWrapper,
│   │                        ModelReference, GenerationCommonConfig, define_model,
│   │                        resolve_api_key, compute_usage_stats
│   ├── document.py          Document, RankedDocument
│   ├── retriever.py         RetrieverRef, define_retriever, etc. (RetrieverOptions deleted — kwargs)
│   ├── embedding.py         Embedder, EmbedderRef, define_embedder (EmbedderOptions deleted — kwargs)
│   ├── reranker.py          RerankerRef, define_reranker (RerankerOptions deleted — kwargs)
│   ├── evaluator.py         EvaluatorRef, define_evaluator
│   ├── tools.py             ToolRunContext, ToolInterruptError, define_tool
│   ├── resource.py          resource actions, define_resource
│   ├── formats/             output format system
│   │   ├── types.py         FormatDef, Formatter, FormatterConfig
│   │   ├── json.py, text.py, jsonl.py, enum.py, array.py
│   ├── _internal/
│   │   ├── _genkit.py       Genkit class body (from ai/_aio.py)
│   │   ├── _genkit_base.py  Genkit __init__, server startup (from ai/_base_async.py)
│   │   ├── _dotprompt.py    dotprompt template engine — render_*, file loading, PromptCache
│   │   ├── _generate.py     generate() orchestration, tool loop (from blocks/generate.py)
│   │   ├── _middleware.py    model middleware execution
│   │   └── _messages.py     message construction helpers
│
├── core/                    framework primitives (not AI-specific)
│   ├── __init__.py          public exports (__all__ defined)
│   ├── action.py            Action, ActionRunContext, ActionMetadata, ActionKind,
│   │                         ActionResponse, ActionMetadataKey (flattened —
│   │                         absorbs action_types.py, 18 consumers, same concept)
│   ├── error.py             GenkitError, UserFacingError, StatusCodes, Status,
│   │                         http_status_code (absorbs status_types.py — only consumer)
│   ├── plugin.py            Plugin ABC
│   ├── _internal/
│   │   ├── _typing.py       auto-generated schema types (DO NOT EDIT header).
│   │   │                     60+ BaseModel classes. Re-exported via genkit/__init__.py
│   │   │                     and domain sub-modules. Nobody imports this directly.
│   │   ├── _base.py         GenkitBaseModel (Pydantic base with exclude_none + by_alias defaults)
│   │   ├── _compat.py       StrEnum (3.10), override (3.11), wait_for (3.10) backfills
│   │   │                     (absorbs aio/_compat.py — dies when min Python ≥ 3.12)
│   │   ├── _registry.py     Registry class
│   │   ├── _server.py       ServerSpec (reflection API config — moved from ai/)
│   │   ├── _context.py      RequestData, ContextMetadata
│   │   ├── _tracing.py      tracing setup, span creation, dev UI exporter,
│   │   │                     RealtimeSpanProcessor (~350 lines merged from
│   │   │                     tracing.py + default_exporter.py + realtime_processor.py.
│   │   │                     AdjustingTraceExporter + RedactedSpan moved to
│   │   │                     telemetry plugin — 5 plugins import, 0 core files do.)
│   │   ├── _http_client.py  HTTP client cache (per-event-loop httpx.AsyncClient — 8 plugins use)
│   │   ├── _environment.py  EnvVar, GenkitEnvironment, is_dev_environment()
│   │   ├── _aio.py          Channel, run_async, run_loop, ensure_async, iter_over_async
│   │   │                     (~500 lines merged from all 4 aio/* files)
│   │   ├── _schema.py       to_json_schema (internal — define_* accepts types directly)
│   │   ├── _extract.py      extract_json, extract_items (internal — only used by formats/)
│   │   ├── _logging.py      get_logger (structlog wrapper — trim 20-method Protocol to ~7)
│   │   ├── _constants.py    GENKIT_VERSION, GENKIT_CLIENT_HEADER
│   │   ├── _flow.py         FlowWrapper (~50 lines — users never construct, returned by @ai.flow())
│   │   ├── _background.py   BackgroundAction (2 internal consumers, not re-exported top-level)
│   │   └── _dap.py          DynamicActionProvider (1 internal consumer, not re-exported top-level)
│
├── tracing.py               tracer, add_custom_exporter (public — matches JS genkit/tracing)
│
├── _web/                    dev server only (all internal)
│   ├── _reflection.py       Dev UI reflection API (moved from core/). Starlette ASGI app
│   │                         exposing /api/actions, /api/runAction, etc. Only consumer is
│   │                         the runtime startup code that mounts it on uvicorn.
│   └── _runtime.py          RuntimeManager — writes .genkit/runtimes/ files
│
│   DELETED: web/manager/ (~1,500 lines, 7 types)
│   ServerManager, ASGIServerAdapter, UvicornAdapter, GranianAdapter,
│   SignalHandler, ServerLifecycle, ServerConfig, AbstractBaseServer,
│   ports.py, info.py — all unused by framework/plugins. Only consumer
│   was one sample (web-multi-server). The reflection server uses raw
│   uvicorn directly (~15 lines in _base_async.py). No abstraction needed.
│
└── testing.py               ProgrammableModel, EchoModel, StaticResponseModel
```

### What changed

| Change | Details |
|---|---|
| **Delete `blocks/`** | All files move into `ai/`. Domain types live where Go/JS put them. |
| **Delete `aio/`** | `Channel` + loop utils → `core/_internal/_aio.py` |
| **Delete `lang/`** | `deprecations.py` → inline into google-genai plugin (only consumer). |
| **Delete `types/`** | Barrel re-export removed. `genkit/__init__.py` handles this. |
| **Delete `web/manager/`** | ~1,500 lines of unused multi-server orchestration. Reflection server uses raw uvicorn (~15 lines). |
| **Delete `core/flows.py`** | `create_flows_asgi_app()` — auto-exposes flows as HTTP endpoints. Firebase Cloud Functions pattern that doesn't fit Python (Cloud Functions uses Flask, not ASGI; no `onCallGenkit` for Python). Users should use FastAPI/Flask instead. JS has this (`startFlowServer`) because the Express ecosystem aligns; Python's doesn't. ~370 lines. |
| **Rename `web/` → `_web/`** | Prefix signals "internal, don't import". Now just reflection + runtime. |
| **Move `core/reflection.py` → `_web/`** | It's a Starlette ASGI app, not a core primitive. Breaks `core/` → `web/` cycle. |
| **Delete `codec.py`** | `dump_dict`/`dump_json` die with `GenkitBaseModel` (see [python_beta_api_proposal.md §5](./python_beta_api_proposal.md)). Third-party `BaseModel` fallback inlined into `_base.py`. |
| **Delete `model_types.py`** | `GenerationCommonConfig` → `ai/model.py`. API key helpers renamed to `resolve_api_key` and exposed from `model.py`. `get_basic_usage_stats` renamed to `compute_usage_stats`. |
| **Merge `action_types.py` into `action.py`** | 95 lines, same 18 consumers, same concept. `ActionKind`, `ActionResponse`, `ActionMetadataKey` live alongside `Action`. |
| **Merge `status_types.py` into `error.py`** | Only consumer is `error.py`. `StatusCodes`, `Status`, `http_status_code` are tightly coupled with the error hierarchy. |
| **Move `FlowWrapper` → `_internal/`** | `ai/_registry.py` → `core/_internal/_flow.py`. ~50 lines, 2 consumers, users never construct directly (returned by `@ai.flow()`). |
| **Move `BackgroundAction` → `_internal/`** | `blocks/background_model.py` → `core/_internal/_background.py`. Not re-exported top-level, only 2 internal consumers. `genkit.model` sub-module re-exports it for plugin authors. |
| **Move `DynamicActionProvider` → `_internal/`** | `blocks/dap.py` → `core/_internal/_dap.py`. Not re-exported top-level, single internal consumer (`ai/_registry.py`). |
| **Split `prompt.py`** | 2,446 → ~600 (prompt.py) + ~200 (streaming.py) + ~800 (_dotprompt.py) |
| **Move `typing.py` → `_internal/`** | `core/typing.py` → `core/_internal/_typing.py`. Auto-generated 60+ `BaseModel` classes. `core/` is not a public import path — public types are re-exported from `genkit/__init__.py` and domain sub-modules. The file is pure plumbing. |
| **Internalize `schema.py` + `extract.py`** | Both move to `core/_internal/`. `define_*` functions accept raw Python types so no plugin needs `to_json_schema`. `extract_json` has zero plugin consumers — only used by `formats/`. JS exports both publicly but nobody imports them there either. See [python_beta_api_proposal.md §6](./python_beta_api_proposal.md). |
| **Dissolve `ai/_registry.py`** | define_* functions move to their domain files (like Go). `define_model` → `ai/model.py`, `define_retriever` → `ai/retriever.py`, etc. Genkit method stubs stay in `ai/_internal/_genkit.py`. `_registry.py` ceases to exist. |
| **Add `_internal/`** | Pydantic v2 pattern: private implementation behind `_internal/` |
| **Add `__all__`** | Every public `__init__.py` declares its exports |

## Plugin import paths — before and after

### Model plugin (e.g., google-genai gemini.py)

```python
# Before (6 deep imports):
from genkit.ai import ActionRunContext, GENKIT_CLIENT_HEADER
from genkit.blocks.model import get_basic_usage_stats
from genkit.codec import dump_dict, dump_json
from genkit.core.error import GenkitError, StatusName
from genkit.core.tracing import tracer
from genkit.core.typing import GenerationCommonConfig, Message, ...

# After (2-3 imports — top-level genkit, genkit.ai, genkit.tracing):
from genkit import GenkitError, GENKIT_CLIENT_HEADER
from genkit.tracing import tracer
from genkit.ai import (
    ActionRunContext, GenerationCommonConfig,
    Message, compute_usage_stats,
)
```

### Retriever plugin (e.g., vertex-ai vector_search.py)

```python
# Before (5 deep imports):
from genkit.ai import Genkit
from genkit.blocks.document import Document
from genkit.blocks.retriever import retriever_action_metadata
from genkit.core.action.types import ActionKind
from genkit.core.schema import to_json_schema

# After (1 import — define_retriever accepts types directly, no manual to_json_schema):
from genkit import Genkit, Document, ActionKind
```

### Telemetry plugin (e.g., observability)

```python
# Before (3 deep imports):
from genkit.core.environment import is_dev_environment
from genkit.core.trace.adjusting_exporter import AdjustingTraceExporter
from genkit.core.tracing import add_custom_exporter

# After (2 imports — AdjustingTraceExporter moves to telemetry plugin):
from genkit import is_dev_environment
from genkit.tracing import add_custom_exporter
```

---

## Circular import fix: `core/` → `_web/` cycle

**Problem.** Today `core/` has a hidden dependency on `web/`:

- `core/reflection.py` imports `genkit.web.manager` (it **is** a Starlette ASGI app)
- `core/flows.py` imports `genkit.web.manager` (it **is** a Starlette ASGI app)
- `web/` modules import from `genkit.core.*`

This creates a package-level cycle: `core/ ↔ web/`.

**Root cause.** Both `reflection.py` and `flows.py` are 100% HTTP server
code — Starlette routes, ASGI apps, request/response handling. They ended
up in `core/` by accident, not because they provide core primitives.

**Fix.**

- `core/reflection.py` → move to `_web/reflection.py`
- `core/flows.py` → **delete** (see "What changed" table — Firebase pattern
  that doesn't fit Python; users should use FastAPI/Flask)

```
_web/
├── _reflection.py   ← was core/reflection.py
└── _runtime.py      ← RuntimeManager
```

### Additional cross-package violations to fix

**`core/plugin.py` → `blocks/` (becomes `core/` → `ai/` after reorg).**
The `Plugin` base class has two convenience methods — `model(name)` and
`embedder(name)` — that do deferred runtime imports of `ModelReference` and
`EmbedderRef` from `blocks/`. After the reorg, `blocks/` merges into `ai/`,
creating a `core/ → ai/` layering violation.

Fix: **restore the original async resolve-based helpers, add `embedder()`.** The
current methods (added in #4278) construct `ModelReference`/`EmbedderRef` objects,
which requires importing from `blocks/`. The original version (from #4132) called
`self.resolve(ActionKind.MODEL, name)` and returned `Action` — no imports from
`blocks/` or `ai/`, zero layering violation. Matches JS's
`GenkitPluginV2Instance.model()`. The `embedder()` method gets the same treatment.
Both are async, return `Action | None`, and only use types already in `core/`.

**`ai/_base_async.py` → `web/manager/_ports.py`.**
Imports `find_free_port_sync` — a 15-line stdlib socket utility. After the
reorg, `web/manager/` is deleted.

Fix: move `find_free_port_sync` to `core/_internal/_ports.py`. It's pure
stdlib (`socket.bind`), no dependencies.

### After all fixes

The dependency graph is strictly one-directional:

```
_web/  →  ai/  →  core/
  └────────────────↗
```

`core/` has zero imports from `_web/` or `ai/`. Clean layering.

---

## Boundary enforcement

### 1. `__all__` on every public `__init__.py`

```python
# genkit/__init__.py  (the ONLY public import path for most users)
__all__ = [
    'Genkit', 'Document', 'GenkitError', 'UserFacingError',
    'GenerateResponse', 'StreamResponse', 'GenerateResponseChunk',
    'ExecutablePrompt', 'Message', 'Role',
    'Part', 'TextPart', 'MediaPart', 'Media',
    'ToolRunContext', 'ToolInterruptError', 'ToolChoice',
    'RequestData', 'ContextProvider',
    'GENKIT_VERSION', 'GENKIT_CLIENT_HEADER', 'is_dev_environment',
    'Plugin', 'Action', 'ActionMetadata', 'ActionKind', 'StatusCodes',
    # ... ~34 symbols (see python_beta_api_proposal.md §1)
]

# genkit/tracing.py  (telemetry plugin authors)
__all__ = ['tracer', 'add_custom_exporter']

# genkit/model.py, genkit/retriever.py, etc.  (domain sub-modules for plugin authors)
# Each defines __all__ with its domain types.
```

**No public `genkit.core` or `genkit.ai` import paths.** `core/` and `ai/` are
internal package structure — `genkit/__init__.py` re-exports everything users need.
Domain sub-modules (`genkit.model`, `genkit.retriever`, etc.) are for plugin authors
who need wire-format types not in the top-level barrel.

### 2. `import-linter` in CI

```ini
# .importlinter
[importlinter]
root_package = genkit

[importlinter:contract:layers]
name = Package layers
type = layers
layers =
    genkit._web
    genkit.ai
    genkit.core

[importlinter:contract:no-internal-from-plugins]
name = Plugins must not import _internal
type = forbidden
source_modules =
    genkit.plugins
forbidden_modules =
    genkit.ai._internal
    genkit.core._internal
```

### 3. `_internal/` convention

Following Pydantic v2's pattern. The split works like this:

**Files at package level (e.g. `core/action.py`, `core/error.py`):**
- Clean abstractions within the package — the "logical public API" of that sub-package
- Listed in the sub-package's `__init__.py` `__all__`
- Other SDK modules import from here: `from genkit.core.action import Action`
- Can still have private helpers (`_foo()`) inside the file — normal Python
- Signals to SDK developers: "this is a stable abstraction"

**Files in `_internal/` (e.g. `core/_internal/_registry.py`, `core/_internal/_typing.py`):**
- Implementation machinery — can change between versions without notice
- NOT listed in the sub-package's `__init__.py` `__all__`
- Other SDK modules import directly when needed: `from genkit.core._internal._base import GenkitBaseModel`
- `import-linter` prevents plugins from importing these paths
- Signals to SDK developers: "this is plumbing, handle with care"

Since there's **no public `genkit.core`** import path anyway, the split is primarily
about signaling intent to other developers working on the SDK itself. External users
import from `genkit`, `genkit.model`, `genkit.tracing`, etc. — they never see either level.

---

## File size targets

| File | Current | Target | How |
|---|---|---|---|
| `blocks/prompt.py` | 2,446 | ~600 | Split into prompt.py + streaming.py + _dotprompt.py (render_*, file loading, PromptCache) |
| `ai/_registry.py` | 1,680 | **0 (deleted)** | define_* functions move to domain files (model.py, retriever.py, etc.). Genkit method stubs absorbed into _genkit.py. File ceases to exist. |
| `ai/_aio.py` | 1,164 | ~800 | Rename to _genkit.py, extract server startup to _genkit_base.py |
| `blocks/generate.py` | 1,088 | ~600 | Extract tool loop to _generate.py, keep public generate function |
| `core/_internal/_typing.py` | 1,066 | 1,066 | Auto-generated, don't touch. Add DO NOT EDIT header. Moved to `_internal/`. |

Target: no hand-written file over 800 lines. Matches Go/JS norms.

---

## Migration path

This is a **one-time refactor** with minimal logic changes. Most of the diff is
file moves and import path updates. The API changes are:

- `define_*` functions accept `type | dict | None` (see [§6](./python_beta_api_proposal.md))
- `GenkitBaseModel` replaces `dump_dict`/`dump_json` (see [§5](./python_beta_api_proposal.md))
- `to_json_schema` and `extract_json` become internal
- Public import paths change from `genkit.core.*` / `genkit.blocks.*` to
  `from genkit import ...` and domain sub-modules (`genkit.model`, etc.)

The structural diff is:

1. Move files
2. Update import paths (find-and-replace across plugins)
3. Add `__all__` to `__init__.py` files
4. Split 3 oversized files

### Order of operations

1. **Add `__all__` to existing `__init__.py` files** — zero-risk, clarifies
   public API immediately. Can land as its own PR.

2. **Merge `blocks/` into `ai/`** — the big structural move. Update all
   import paths. One PR.

3. **Move `FlowWrapper`, `BackgroundAction`, `DynamicActionProvider` to `core/`** —
   small cross-language alignment fix. One PR.

4. **Kill orphans** — delete `aio/`, `lang/`, `types/`, move root files.
   One PR.

5. **Create `_internal/` directories** — move implementation files behind
   the boundary. Update internal imports. One PR.

6. **Rename `web/` → `_web/`, move `core/reflection.py` into `_web/`,
   delete `core/flows.py`** — breaks the `core/ ↔ web/` circular
   dependency and removes the unused flows server. One PR.

7. **Split oversized files** — `prompt.py`, `_registry.py`, `generate.py`.
   One PR each.

8. **Add `import-linter` to CI** — one PR, enforces the new structure going
   forward.

Each step is independently shippable and independently revertible.
