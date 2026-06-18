# Typed model handles for Python config — 1-pager

**Issue:** genkit-ai/genkit#5553 (Tier 2)
**Status:** proposal — decisions resolved, scoped to a pragmatic first cut
**Scope:** Python SDK + the big-3 plugins (google-genai, compat-oai, anthropic)

## Problem

App developers pass model config as a plain dict — that's the ergonomic path, and it
already works at runtime (the dict flows through the plugin's Pydantic schema to the
provider SDK). What's missing is **type safety on that dict**: nothing tells the editor
which keys a given model accepts.

```python
await ai.generate(
    model='googleai/gemini-2.0-flash',
    config={'temperatur': 0.7},   # typo — no error, no autocomplete
)
```

`ai.generate(model=...)` takes a bare `str`, so the type checker has no idea what the
config shape should be. `config` is typed `dict[str, object] | ModelConfig` — wide open.

## Goal — the syntax we're trying to achieve

When the model is known, `config={...}` should autocomplete its keys and flag typos —
without forcing the developer to build a Pydantic object, and without changing the
runtime passthrough that already works:

```python
flash = GoogleAI.model('gemini-2.0-flash')           # a typed, inert handle
await ai.generate(
    model=flash,
    prompt='...',
    config={'temperature': 0.7, 'code_execution': True},  # ✅ autocomplete + typo check
)
```

## How we landed on `.model(...)`

We worked through the options and eliminated the ones that don't hold up:

1. **Infer config from the bare `model='...'` string.** Impossible — `model` is a plain
   `str`, the type checker has nothing to bind a config type to.

2. **Make `ai.generate` generic per model.** Explodes the overload set and would force
   `genkit` core to import every plugin's config types (core↔plugin coupling). Rejected.

3. **A typed model handle, minted by the plugin.** The plugin is the only place that
   statically knows `gemini-2.0-flash → GeminiConfig`, so the typed accessor lives there.
   The handle carries the model name (runtime) and a config type (static);
   `ai.generate(model=handle, ...)` inherits that config type. This is the winner.

That leaves one shape decision: what the handle actually carries. We make `.model()` a
**classmethod** that returns an **inert** handle — just the model name + a phantom config
type, with no plugin instance, no `ai`, and no registry. Resolution happens later, inside
`ai.generate(model=handle)`:

```python
flash = GoogleAI.model('gemini-2.0-flash')   # classmethod, inert, no instance/registry
await ai.generate(model=flash, config={'temperature': 0.7, 'code_execution': True})
```

Inert is the deliberate choice here, over a handle you call `.generate()` on directly:

- **Statically mintable.** Carrying only a name + phantom type lets a `classmethod` produce
  the handle with no context. The moment a handle has to call `.generate()` itself it must
  hold a live registry — so it can't be static, and the developer is forced to keep a
  plugin/`ai` instance around. Inert keeps `Genkit(plugins=[GoogleAI()])` inline working.
- **One execution path.** `ai.generate` already owns resolution, middleware, tracing, and
  context. A bound `flash.generate()` would either duplicate that machinery or just delegate
  back to it — extra API surface for no new capability.
- Bare-string `ai.generate(model='...', config=...)` is untouched and stays loosely typed.

(Two alternatives rejected: a **registry-bound** handle that enables `flash.generate()`
sugar — it forces the handle to carry a registry and the developer to keep a plugin instance;
and a core `ai.model(token)` accessor — it loses the bare-string ergonomics by requiring a
wrapped token.)

## The one bottleneck — Python can't pattern-match names into types

Here's the wall we hit, and it's a genuine language limitation, not a design miss.

We wanted the plugin author to write the binding **once**, dynamically — something like
"every `gemini-*` string maps to `GeminiConfigDict`." **Python cannot express that with
type safety.**

```
  GoogleAI.model('gemini-FUTURE-MODEL')
                │
                ├─ RUNTIME:  is_gemini(name) → GeminiConfigSchema   ✅ a function works fine
                │
                └─ STATIC:   type checker can't run is_gemini(), can't match a prefix
                             → no autocomplete for the config
```

- Python's `Literal` is **exact strings only**. There is no "any string starting with
  `gemini-`" type. A regex/prefix function runs at *runtime*; the type checker never
  executes it.
- So the **only** mechanism that gives `GoogleAI.model('name')` a precise config type is an
  `@overload` whose parameter is an **enumerated `Literal`** of exact model names.

### Why this is a Python-specific cost

Two type-system features would make this free, and Python's type system has neither.

The first is **compiler-inferred dict types**: a system that can derive an object type
directly from a schema gets the config shape for nothing. Python can't — a Pydantic model
has no dict-view at the type level, so we have to write a separate `TypedDict` to describe
the same fields.

The second is **template-literal types**: a system that can express a pattern like
`gemini-${string}` as a real type can match model-name prefixes statically. Python's
`Literal` is exact strings only, so a prefix can't carry a config type and the names have to
be enumerated one by one.

Both gaps are the root cause of the bottleneck above.

**Net:** in Python, the plugin author must **explicitly enumerate the model names** that
get precise typing. There's no dynamic prefix escape.

## Workaround

So with that context, we only get type safety for explicitly enumerated models. Here's
our proposal, given the constraints:

1. **Enumerate the common models in the big-3 plugins — from a YAML manifest.** We list the
   names people actually use (the current Gemini / GPT / Claude families) and their family
   in one declarative file, and a generator emits the model refs. That covers the
   overwhelming majority of real calls with full autocomplete, and enumeration becomes a
   data-entry task, not hand-written overloads (see *Generating the binding* below).

2. **Loose fallback for everything else.** A `str` fallback overload means any
   un-enumerated or brand-new model name **still returns a working handle** — just typed
   as the wide base config instead of the family-specific one. Nothing ever breaks; you
   only lose the precise autocomplete.

3. **User-supplied annotation as an escape hatch.** For a model the plugin hasn't listed,
   the developer can opt in explicitly:
   `flash: ModelRef[GeminiConfigDict] = GoogleAI.model('gemini-future')`.

So the failure mode of the whole scheme is "base-config autocomplete instead of
family-specific" — never "can't call the model."

## Design

Three pieces, in order: the handle type and the `generate` overloads that read it; how
config types are grained; and how a per-plugin manifest generates the runtime and static
halves from one source.

### 1. The handle and the `generate` overloads

The core addition is a generic handle whose type parameter is the config shape:

```python
ConfigT = TypeVar('ConfigT', bound=Mapping[str, object], default=Mapping[str, object])

@dataclass(frozen=True)
class ModelRef(Generic[ConfigT]):
    name: str          # 'googleai/gemini-2.0-flash' — the only runtime payload
    # ConfigT is phantom: static-only, drives config autocomplete via ai.generate.
    # inert — carries no registry; ai.generate(model=handle) does the resolution.
```

`ai.generate` gains an overload for the handle path while the string path stays loose:

```python
@overload
async def generate(self, *, model: ModelRef[ConfigT],
                    config: ConfigT | None = None, ...): ...
@overload
async def generate(self, *, model: str | None = None,
                    config: Mapping[str, object] | ModelConfig | None = None, ...): ...
```

On the handle path, a dict is checked against the family `TypedDict` (`ConfigT`) for
autocomplete + typo flagging. Runtime `generate` just reads `model.name`; everything else is
unchanged.

### 2. Config types are grained by family

Config types are carved by **family**, mirroring the existing Pydantic schemas 1:1 — a
different config shape *is* a different family, so there's no new taxonomy:

| Family | Config type | Pydantic source |
|---|---|---|
| Gemini (text/multimodal) | `GeminiConfigDict` | `GeminiConfigSchema` |
| Gemini TTS | `GeminiTtsConfigDict` | `GeminiTtsConfigSchema` |
| Gemini image | `GeminiImageConfigDict` | `GeminiImageConfigSchema` |
| Veo (video) | `VeoConfigDict` | `VeoConfigSchema` |
| Imagen | `ImagenConfigDict` | `ImagenConfigSchema` |

This grain already exists at runtime today, in the resolver that picks a schema by name:

```python
# google-genai, get_model_config_schema(name) — already exists
if is_tts_model(name):   return GeminiTtsConfigSchema
if is_image_model(name): return GeminiImageConfigSchema
if is_gemma_model(name): return GemmaConfigSchema
return GeminiConfigSchema
```

### 3. One manifest generates both layers

The binding has two halves that must agree: a **runtime** resolver (name → Pydantic schema,
to validate the dict) and a **static** surface (name → `*ConfigDict`, to drive autocomplete).

```
                      name string  ─────►  config schema
  ──────────────────────────────────────────────────────────────
  RUNTIME   get_model_config_schema(name)  → Pydantic schema (validates the dict)
  STATIC    .model() overloads             → *ConfigDict     (drives autocomplete)
```

Rather than hand-write both (and pay the "enumerate the names" tax twice), each plugin owns
a **model manifest** — one declarative file a generator expands into both halves, so they
can't drift.

**The manifest owns the full `ModelInfo`.** It's not just the name→family→config binding —
it carries the complete per-model metadata (label, version, stage, `supports`) that's
hand-written and scattered across `ModelInfo` entries today, so codegen *replaces* those
entries rather than sitting beside them. One source for the runtime registry and the static
typing both.

```yaml
# plugins/google-genai/models.yaml
plugin: googleai
families:
  gemini:
    config_schema: GeminiConfigSchema      # Pydantic source → GeminiConfigDict
    models:
      - name: gemini-2.0-flash
        label: Gemini 2.0 Flash
        stage: stable
        supports:
          multiturn: true
          tools: true
          media: true
          systemRole: true
      - name: gemini-2.5-flash
        label: Gemini 2.5 Flash
        supports:
          multiturn: true
          tools: true
          media: true
  gemini-tts:
    config_schema: GeminiTtsConfigSchema
    models:
      - name: gemini-2.5-flash-tts
        supports:
          multiturn: false
  veo:
    config_schema: VeoConfigSchema
    models:
      - name: veo-2.0
        stage: preview
```

A generator (extending the existing `schema_to_typing.py` machinery) reads the manifest and
emits **one generated module per plugin**:

```
  models.yaml ──► generate_model_refs.py ──► _generated_models.py
                                              ├─ MODEL_INFO            (runtime registry — replaces hand-written ModelInfo entries)
                                              ├─ get_model_config_schema()  (runtime resolver — replaces the predicate chain)
                                              ├─ GeminiModelName = Literal['gemini-2.0-flash', ...]   (static)
                                              └─ GoogleAI.model() overloads → ModelRef[GeminiConfigDict]  (static)
```

The `*ConfigDict` types come from the Pydantic schemas named in the manifest (the Tier-1
`schema_to_typing` step), so the manifest only references a schema by name — it never
restates the config shape.

**What the author writes by hand for model refs: nothing.** The whole `.model()` accessor
— the typed `@overload` signatures *and* its uniform body — is generated, because the body
doesn't branch per model (the plugin prefix comes from the manifest's `plugin:` field). The
plugin class just inherits the generated mixin:

```python
# _generated_models.py (generated — do not edit)
class GoogleAIGeneratedModels:
    @overload
    @classmethod
    def model(cls, name: GeminiModelName, /) -> ModelRef[GeminiConfigDict]: ...
    @overload
    @classmethod
    def model(cls, name: str, /) -> ModelRef[Mapping[str, object]]: ...   # loose fallback
    @classmethod
    def model(cls, name, /):
        return ModelRef(name=name if '/' in name else f'googleai/{name}')

# the plugin (hand-written) just mixes it in — no model-ref code of its own:
class GoogleAI(Plugin, GoogleAIGeneratedModels):
    ...   # existing plugin setup (client, registration, …)
```

So the author maintains exactly two things, both data: the **Pydantic config schema** (the
config shape) and the **YAML manifest** (names, families, and full per-model `ModelInfo`).
Both are written today — the manifest just consolidates the `ModelInfo` entries that are
currently hand-coded and scattered. The hand-written `ModelInfo` table and the
`is_tts_model`/`is_image_model` predicate chain **go away**; codegen owns them. Adding a
model is a one-line YAML edit + regenerate.

**Drift is guarded, not trusted:** regeneration is wired into the build so **CI fails if
`_generated_models.py` is stale** vs the manifest (the pattern `schema_to_typing.py` already
uses for `_typing.py`), and a test asserts each generated `*ConfigDict` matches the fields of
the Pydantic schema it came from.

## What does NOT change

- Plugin authors keep writing **Pydantic** config schemas (validation, Dev UI, aliases).
- Dict passthrough + `extra='allow'` escape hatch stays exactly as-is — even inside a
  typed family, an unlisted key validates at runtime and reaches the SDK.
- Bare-string `ai.generate(model='...')` keeps working with loose config typing.
- No snake↔camel conversion layer (Pydantic aliases already handle it).

