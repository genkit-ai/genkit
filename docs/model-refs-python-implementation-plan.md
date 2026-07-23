# Model Refs (Python) — Implementation + Swarm Plan

Companion to [`model-refs-python.md`](./model-refs-python.md) (design) and
[`model-ref-docstrings-audit.md`](./model-ref-docstrings-audit.md) (docs checklist).

This doc is *what* to build, *which files*, and *how* to run a small agent swarm
in isolated git worktrees. Decisions in the design doc are **frozen** — agents
implement; they do not reopen them.

**Model default:** Composer 2.5 (`composer-2.5-fast`). Escalate only if an agent
starts inventing a unified `model(name)` overload API instead of family helpers.

**Orchestration:** parent agent integrates between waves. Ping Jeff when Wave 3
is DONE (or if hard-blocked on a real judgment call).

---

## 1. Principles

1. **One agent = one track = one worktree = one branch.**
2. **Don't parallelize across a dependency edge.** Wave 2 waits for Track 0 merge.
3. **Decisions are frozen** (see §3). Stop and ask the parent if ambiguous.
4. **Parent agent merges.** Children leave DONE summaries; no PRs/merges from kids.
5. **2–3 agents max per wave.**
6. **Every agent self-checks:** `ruff format` + `ruff check` on touched files, plus
   the track's pytest slice. Typecheck the touched package if the track changes
   public signatures.
7. **Ping Jeff once:** Wave 3 DONE (or unblock needed).

---

## 2. Waves & worktrees

```
Wave 1 (1 agent)              Wave 2 (parallel, 3 agents)         Wave 3 (1 agent)
─────────────────────────     ───────────────────────────────     ────────────────
T0  Core ModelRef +           T1  generate/prompt typing          T4  docs/samples
    ModelConfigDict +             (needs T0)                          (needs T1+T2)
    model_ref factory         T2  google-genai family helpers
                                  (needs T0)
                              T3  anthropic + ollama helpers
                                  (needs T0)
```

### Why this shape?

- T0 is the shared foundation (`ModelRef[ConfigT]`, `model_ref(..., config_schema=)`).
  Nothing typed/helper-related lands before it.
- T1 / T2 / T3 touch **different packages** after T0 → safe to parallelize.
- T4 rewrites hero examples only after APIs exist.
- Not 50 micro-tasks; not one mega-agent owning core+google+docs.

### Worktree layout

| Worktree path | Branch | Wave | Track |
|---|---|---|---|
| `~/Desktop/genkit-mr-t0` | `jh/mr-track-0-core` | 1 | T0 — core |
| `~/Desktop/genkit-mr-t1` | `jh/mr-track-1-generate` | 2 | T1 — generate/prompt |
| `~/Desktop/genkit-mr-t2` | `jh/mr-track-2-google` | 2 | T2 — google-genai |
| `~/Desktop/genkit-mr-t3` | `jh/mr-track-3-plugins` | 2 | T3 — anthropic/ollama |
| `~/Desktop/genkit-mr-t4` | `jh/mr-track-4-docs` | 3 | T4 — docs/samples |

### Base setup (once)

Use a **clean** anchor — prefer `genkit-ai/main` or `origin/main`, not a dirty
local agents branch. The design/plan docs live on `py-model-refs-doc`; either
cherry-pick/copy them into each worktree or pass absolute paths in prompts.

```bash
BASE=/Users/jeffhuang/Desktop
REPO=/Users/jeffhuang/Desktop/genkit   # or genkit-model-refs-doc if that's the clean clone

git -C "$REPO" fetch genkit-ai
git -C "$REPO" fetch origin
MAIN_SHA=$(git -C "$REPO" rev-parse genkit-ai/main)
echo "swarm base: $MAIN_SHA"

# Wave 1
git -C "$REPO" worktree add -b jh/mr-track-0-core \
  "$BASE/genkit-mr-t0" "$MAIN_SHA"
```

After T0 merges to integration branch `jh/mr-wave1`:

```bash
INTEGRATION=jh/mr-wave1

git -C "$REPO" worktree add -b jh/mr-track-1-generate \
  "$BASE/genkit-mr-t1" "$INTEGRATION"
git -C "$REPO" worktree add -b jh/mr-track-2-google \
  "$BASE/genkit-mr-t2" "$INTEGRATION"
git -C "$REPO" worktree add -b jh/mr-track-3-plugins \
  "$BASE/genkit-mr-t3" "$INTEGRATION"
```

After T1+T2+T3 merge to `jh/mr-wave2`:

```bash
git -C "$REPO" worktree add -b jh/mr-track-4-docs \
  "$BASE/genkit-mr-t4" jh/mr-wave2
```

---

## 3. Frozen decisions (do not reopen)

From [`model-refs-python.md`](./model-refs-python.md):

| ID | Decision |
|---|---|
| D1 | `ModelRef` is `Generic[ConfigT]` over the plugin's **Pydantic** config (`bound=BaseModel`), `frozen=True` |
| D2 | `ModelConfigDict` TypedDict for common dict-literal autocomplete |
| D3 | `Mapping[str, Any]` pass-thru arm stays (typos not hard-rejected on raw dicts) |
| D4 | Impl `config` accepts `BaseModel` (not only `ModelConfig`) because Imagen/Veo configs subclass plain `BaseModel` |
| D5 | **Family helpers** (`gemini_model`, …) are the typed public API; name is `KnownX \| str`; no prefix types in Python; OK to deviate from JS `googleAI.model()` |
| D6 | Optional known `Literal`s = **autocomplete only**; never gate runtime |
| D7 | No plugin-level unified overloaded `model(name)` that tries to narrow return type from the string |
| D8 | New Google version strings must work with **no** Genkit release (open `str` on family helpers) |

**Implementation freezes (not in design doc, but fixed for the swarm):**

| ID | Decision |
|---|---|
| F1 | Add public aliases `GeminiConfig = GeminiConfigSchema`, `ImagenConfig = ImagenConfigSchema`, etc. in package `__init__` — **do not** mass-rename schema classes in model files |
| F2 | Family helpers default `namespace="googleai"`; accept optional `namespace: str \| None = "googleai"` (callers pass `"vertexai"` when needed) |
| F3 | Skip optional loose plugin `model_ref(name) -> ModelRef[BaseModel]` in v1 — family helpers only |
| F4 | If `ConfigT` is already used by `ModelRequest` with `bound=ModelConfig`, **do not widen that bound**. Introduce a separate TypeVar for `ModelRef` (e.g. keep `ModelRequest`'s TypeVar as-is; give `ModelRef` its own `ConfigT` / `ModelRefConfigT` bound to `BaseModel`). Prefer the public name `ConfigT` on `ModelRef` if you can avoid breaking `ModelRequest` by renaming the request one internally |
| F5 | Runtime unwrap: before `resolve_model`, if `model` is a `ModelRef`, use `model.name`. If the ref carries a default `config` and the call didn't pass `config`, merge/use the ref's config (match existing `model_ref(..., config=)` field behavior) |
| F6 | Docstring audit hero syntax = `gemini_model(...)`, **not** `model_ref("gemini-...")` as the typed path |

---

## 4. Track specs

### T0 — Core `ModelRef` + `ModelConfigDict` + `model_ref` (Wave 1)

**Goal:** Make the veneer type and factory match the design so plugins/generate can depend on them.

**Files (only these):**

| Path | Change |
|---|---|
| `py/packages/genkit/src/genkit/_core/_model.py` | Generic frozen `ModelRef`; TypeVar hygiene (F4); optionally define `ModelConfigDict` here **or** next to `ModelConfig` |
| `py/packages/genkit/src/genkit/_ai/_model.py` | `model_ref(..., *, config_schema=...)` stamps schema; typed `config: ConfigT \| None` |
| `py/packages/genkit/src/genkit/model/__init__.py` | Re-export `ModelConfigDict` if new |
| `py/packages/genkit/src/genkit/plugin_api/__init__.py` | Same if this is the plugin-author import path |
| `py/packages/genkit/tests/genkit/...` | **New** unit tests for `model_ref` / `ModelRef` (see below) |

**Do not touch:** `_aio.py`, google-genai, samples, docs.

**Target shapes:**

```python
# _model.py (illustrative)
from typing import Generic, TypeVar
from pydantic import BaseModel, ConfigDict

ConfigT = TypeVar("ConfigT", bound=BaseModel, covariant=True, default=ModelConfig)

class ModelRef(BaseModel, Generic[ConfigT]):
    model_config = ConfigDict(frozen=True)
    name: str
    config_schema: type[ConfigT] | None = None
    info: ModelInfo | None = None  # keep existing info typing style
    version: str | None = None
    config: ConfigT | None = None  # was dict; prefer typed default config on the ref


class ModelConfigDict(TypedDict, total=False):
    """Common knobs for dict-literal autocomplete on ai.generate(config={...})."""
    version: str
    temperature: float
    max_output_tokens: int
    top_k: int
    top_p: float
    stop_sequences: list[str]
    api_key: str
```

Mirror fields from `GenerationCommonConfig` / `ModelConfig` — stay in sync with
that model, not with every provider knob.

```python
# _ai/_model.py (illustrative)
def model_ref(
    name: str,
    *,
    config_schema: type[ConfigT] | None = None,
    namespace: str | None = None,
    info: ModelInfo | None = None,
    version: str | None = None,
    config: ConfigT | None = None,
) -> ModelRef[ConfigT]:
    final_name = (
        f"{namespace}/{name}"
        if namespace and not name.startswith(f"{namespace}/")
        else name
    )
    return ModelRef(
        name=final_name,
        config_schema=config_schema,
        info=info,
        version=version,
        config=config,
    )
```

**Tests (minimum):**

```python
def test_model_ref_stamps_namespace_and_schema():
    ref = model_ref("gemini-pro-latest", namespace="googleai", config_schema=ModelConfig)
    assert ref.name == "googleai/gemini-pro-latest"
    assert ref.config_schema is ModelConfig

def test_model_ref_idempotent_namespace():
    ref = model_ref("googleai/gemini-pro-latest", namespace="googleai")
    assert ref.name == "googleai/gemini-pro-latest"

def test_model_ref_is_frozen():
    ref = model_ref("x", namespace="ns")
    with pytest.raises(Exception):
        ref.name = "y"  # pydantic ValidationError / FrozenInstanceError
```

**Self-check:**

```bash
cd py
uv run ruff format packages/genkit/src/genkit/_core/_model.py packages/genkit/src/genkit/_ai/_model.py
uv run ruff check packages/genkit/src/genkit/_core/_model.py packages/genkit/src/genkit/_ai/_model.py
uv run pytest packages/genkit/tests -k model_ref -q
```

**DONE summary must include:** TypeVar choice (F4), whether `config` on `ModelRef` stayed `dict` or became `ConfigT | None`, and export paths.

---

### T1 — `generate` / `prompt` accept `ModelRef` (Wave 2)

**Depends on:** T0 merged.

**Goal:** Overloads + runtime unwrap so `ai.generate(model=gemini_model(...), config=GeminiConfig(...))` typechecks and runs. Bare strings keep working (overload 2).

**Files (only these):**

| Path | Change |
|---|---|
| `py/packages/genkit/src/genkit/_ai/_aio.py` | Overloads + impl: widen `model` / `config`; unwrap `ModelRef` |
| `py/packages/genkit/src/genkit/_ai/_prompt.py` | `PromptConfig.model` / `to_generate_action_options` accept/unwrap `ModelRef` |
| `py/packages/genkit/tests/genkit/...` | Runtime tests: generate with a `ModelRef` (use test echo model or existing test doubles) |

**Do not touch:** google-genai package, docstring audit (T4), anthropic/ollama.

**Overload sketch (add alongside existing `output_schema` overloads — do not delete those):**

```python
# Conceptual — merge with existing output_schema overload matrix carefully.
# Pattern: when model is ModelRef[ConfigT], config is ConfigT | Mapping[str, Any] | None
#          when model is str | None, config is BaseModel | ModelConfigDict | Mapping[str, Any] | None

@overload
async def generate(
    self,
    *,
    model: ModelRef[ConfigT],
    config: ConfigT | Mapping[str, Any] | None = None,
    # ... existing keyword-only params ...
) -> ModelResponse[Any]: ...

@overload
async def generate(
    self,
    *,
    model: str | None = None,
    config: BaseModel | ModelConfigDict | Mapping[str, Any] | None = None,
    # ...
) -> ModelResponse[Any]: ...
```

Same idea for `generate_stream` if it duplicates the `model`/`config` params.

**Runtime unwrap helper (put near generate impl, private):**

```python
def _resolve_model_arg(
    model: str | ModelRef[BaseModel] | None,
    config: BaseModel | Mapping[str, Any] | None,
) -> tuple[str | None, BaseModel | Mapping[str, Any] | None]:
    if isinstance(model, ModelRef):
        resolved = model.name
        if config is None and model.config is not None:
            config = model.config
        return resolved, config
    return model, config
```

Wire this **before** `registry.resolve_model` / building `GenerateActionOptions`.
`GenerateActionOptionsData.model` stays a **string** on the wire — unwrap at the veneer.

**Tests (minimum):**

- `ai.generate(model=model_ref("testEcho", ...), prompt="hi")` (or whatever the test double name is) succeeds.
- Passing `ModelRef` with default `config=` applies when call omits `config`.
- String model path unchanged: `model="...", config={"temperature": 0.1}`.

**Self-check:** ruff on touched files; pytest generate/prompt slices; run package typecheck if that's how CI gates `_aio.py`.

**DONE summary:** how overloads interact with existing `output_schema` overloads; any pyright ignore you needed (justify).

---

### T2 — Google GenAI family helpers (Wave 2)

**Depends on:** T0 merged. Does **not** depend on T1 for compiling helpers, but examples in tests may use string generate until T1 lands — prefer unit-testing the helpers' return values only.

**Goal:** Public `gemini_model` / `imagen_model` / `veo_model` / … returning `ModelRef[ThatConfig]`.

**Files:**

| Path | Change |
|---|---|
| `py/packages/genkit-google-genai/src/genkit_google_genai/models/_model_refs.py` (**new**) *or* `model_refs.py` if you prefer public module — pick one; underscore module + re-export from `__init__` is fine | Family helpers + `Known*` Literals |
| `py/packages/genkit-google-genai/src/genkit_google_genai/__init__.py` | Export helpers + `GeminiConfig` / `ImagenConfig` aliases (F1); update module docstring lightly (full docstring pass is T4) |
| `py/packages/genkit-google-genai/src/genkit_google_genai/models/imagen.py` | Add `is_imagen_model(name)` if missing (`startswith("imagen-")` or existing `image` convention — **prefer `imagen-` prefix**, don't conflate with Gemini image models) |
| `py/packages/genkit-google-genai/tests/...` | Helper unit tests |

**Reuse existing prefix helpers** in `models/gemini.py`, `veo.py`, `lyria.py` — do not duplicate rules.

**Families to ship in v1 (minimum set):**

| Helper | Config alias | Notes |
|---|---|---|
| `gemini_model` | `GeminiConfig` (= `GeminiConfigSchema`) | standard text Gemini |
| `gemini_tts_model` | `GeminiTtsConfig` (= `GeminiTtsConfigSchema`) | optional if cheap; else document that TTS uses gemini helper + TTS schema — **prefer dedicated helper** for correct `ConfigT` |
| `gemini_image_model` | `GeminiImageConfig` | Gemini native image (not Imagen) |
| `gemma_model` | Gemma config schema | if schema exists as distinct type |
| `imagen_model` | `ImagenConfig` | |
| `veo_model` | `VeoConfig` | |
| `lyria_model` | `LyriaConfig` | |

If TTS/image/gemma helpers balloon the track, **minimum bar** is: `gemini_model`, `imagen_model`, `veo_model`, `lyria_model` + aliases. Add TTS/image/gemma helpers if schemas are already distinct (they are).

**Target shape:**

```python
# models/_model_refs.py
from typing import Literal
from genkit.model import model_ref, ModelRef  # or plugin_api — match package convention
from genkit_google_genai.models.gemini import GeminiConfigSchema

GeminiConfig = GeminiConfigSchema  # if alias lives here; else only in __init__

KnownGemini = Literal[
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
    # include the known list already used in SUPPORTED_MODELS / docs — autocomplete sugar only
]

def gemini_model(
    name: KnownGemini | str,
    *,
    namespace: str = "googleai",
    config: GeminiConfig | None = None,
) -> ModelRef[GeminiConfig]:
    return model_ref(
        name,
        config_schema=GeminiConfig,
        namespace=namespace,
        config=config,
    )
```

**Do not:**

- Build exhaustive exact-name → schema dicts as the source of truth.
- Add `googleAI.model()`-style string-narrowing overloads.
- Require the name to appear in `KnownGemini` at runtime.

**Tests:**

```python
def test_gemini_model_unknown_version_still_typed_schema():
    ref = gemini_model("gemini-flash-pro-whatever-99")
    assert ref.name == "googleai/gemini-flash-pro-whatever-99"
    assert ref.config_schema is GeminiConfigSchema

def test_imagen_model_namespace_vertex():
    ref = imagen_model("imagen-3.0-generate-001", namespace="vertexai")
    assert ref.name.startswith("vertexai/")
```

**Self-check:** ruff; pytest google-genai helper tests; ensure `__all__` exports helpers + aliases.

---

### T3 — Anthropic + Ollama family helpers (Wave 2)

**Depends on:** T0 merged.

**Goal:** Same pattern, one helper each (single config class per plugin today).

**Files:**

| Path | Change |
|---|---|
| `py/packages/genkit-anthropic/src/genkit_anthropic/__init__.py` (and/or small new module) | `claude_model(name: KnownClaude \| str, ...) -> ModelRef[AnthropicConfig]` |
| `py/packages/genkit-ollama/src/genkit_ollama/__init__.py` (and/or small new module) | `ollama_model(name: str, ...) -> ModelRef[OllamaConfig]` (Literals optional) |
| Tests under each package | Namespace + schema stamp |

**Shapes:**

```python
def claude_model(
    name: KnownClaude | str,
    *,
    namespace: str = "anthropic",
    config: AnthropicConfig | None = None,
) -> ModelRef[AnthropicConfig]:
    return model_ref(name, config_schema=AnthropicConfig, namespace=namespace, config=config)

def ollama_model(
    name: str,
    *,
    namespace: str = "ollama",
    config: OllamaConfig | None = None,
) -> ModelRef[OllamaConfig]:
    return model_ref(name, config_schema=OllamaConfig, namespace=namespace, config=config)
```

**Do not** redesign plugin registration or config schemas.

**Self-check:** ruff + each package's pytest slice for the new helpers.

---

### T4 — Docs, samples, docstring audit (Wave 3)

**Depends on:** T1 + T2 merged (T3 nice-to-have for anthropic/ollama README bits).

**Goal:** Hero examples show family helpers + typed configs. Walk
[`model-ref-docstrings-audit.md`](./model-ref-docstrings-audit.md) and flip checkboxes.

**Files (representative — follow the audit list):**

| Area | Paths |
|---|---|
| Core heroes | `_aio.py` docstrings (generate examples), `genkit` / `py/README.md`, `packages/genkit/README.md` |
| Google | `genkit_google_genai/__init__.py`, `google.py` class docs, `models/veo.py` / `lyria.py` examples, package README |
| Community | anthropic / ollama package docstrings + READMEs if present |
| Samples | Prefer 1–2 primary samples (e.g. a google-genai sample + menu/hello) — **not** a blind rewrite of every sample in `py/samples/` |

**Hero pattern to use everywhere typed config matters:**

```python
from genkit_google_genai import GeminiConfig, gemini_model

resp = await ai.generate(
    model=gemini_model("gemini-flash-latest"),
    config=GeminiConfig(temperature=0.7),
    prompt="...",
)
```

Bare-string examples may remain for the "90% dict config" story (design use case 1).

**Fix the audit doc itself:** replace any `model_ref("gemini-...")` *typed-path* guidance with `gemini_model(...)`.

**Do not** change runtime behavior beyond docstring/sample edits.

**Self-check:** ruff on touched files; smoke-import samples if trivial; mark audit checkboxes done in the same commit.

---

## 5. Per-track agent prompts

Copy/paste. Set `PLAN` to this file's absolute path:

`/Users/jeffhuang/Desktop/genkit-model-refs-doc/docs/model-refs-python-implementation-plan.md`

Also attach / point at:

`/Users/jeffhuang/Desktop/genkit-model-refs-doc/docs/model-refs-python.md`

### T0

```
You are implementing Track T0 only from PLAN (path above).
Design doc decisions D1–D8 and freezes F1–F6 are law.

Branch: jh/mr-track-0-core
Worktree: only this worktree.

Scope: Core ModelRef Generic+frozen, ModelConfigDict, model_ref(config_schema=), exports, unit tests.
Files: _core/_model.py, _ai/_model.py, model/__init__.py, plugin_api/__init__.py, new tests.
Do NOT touch _aio.py, google-genai, samples, or docs.

F4: do not break ModelRequest's TypeVar bound.
When done: ruff + pytest -k model_ref. Leave DONE summary. No PR/merge.
```

### T1

```
You are implementing Track T1 only from PLAN.
Branch: jh/mr-track-1-generate
Base already includes merged T0.

Scope: generate/generate_stream/prompt accept ModelRef; overloads per design; runtime unwrap (F5).
Files: _aio.py, _prompt.py, tests. Do NOT touch google-genai or docs.
Preserve existing output_schema overloads.
When done: ruff + generate/prompt pytest slice. DONE summary. No PR/merge.
```

### T2

```
You are implementing Track T2 only from PLAN.
Branch: jh/mr-track-2-google
Base already includes merged T0.

Scope: family helpers gemini_model/imagen_model/veo_model/lyria_model (+ TTS/image/gemma if schemas distinct).
Public aliases GeminiConfig/ImagenConfig (F1). Known* | str params (D5/D6).
No unified model(name) string-narrowing API (D7).
Files: genkit-google-genai only. Unit-test helpers; don't depend on T1.
When done: ruff + google-genai helper tests. DONE summary. No PR/merge.
```

### T3

```
You are implementing Track T3 only from PLAN.
Branch: jh/mr-track-3-plugins
Base already includes merged T0.

Scope: claude_model + ollama_model only. No registration redesign.
When done: ruff + package tests. DONE summary. No PR/merge.
```

### T4

```
You are implementing Track T4 only from PLAN.
Branch: jh/mr-track-4-docs
Base includes merged T1+T2 (+T3 if present).

Scope: docstring audit + READMEs + a few samples. Hero = gemini_model + GeminiConfig.
Update model-ref-docstrings-audit.md checkboxes / target syntax (F6).
No behavior changes. DONE summary. No PR/merge.
```

---

## 6. Integration & merge order

1. Merge **T0** → `jh/mr-wave1` (or main).
2. Rebase/create T1/T2/T3 from that tip; merge all three → `jh/mr-wave2`
   (order among T1/T2/T3 doesn't matter if file ownership held).
3. Merge **T4** → integration / PR branch.
4. Open one PR to `main` from the integration tip (human or parent).

Conflict hotspots (avoid by ownership):

| File | Owner |
|---|---|
| `_model.py` / `_ai/_model.py` | T0 only |
| `_aio.py` / `_prompt.py` | T1 only |
| `genkit_google_genai/**` | T2 only (T4 may touch docstrings later) |
| anthropic / ollama packages | T3 only (T4 docstrings later) |

If T4 and T2 both edit `genkit_google_genai/__init__.py`, T4 rebases last and only changes docstrings/examples.

---

## 7. Status table (parent fills in)

| Track | Worktree | Branch | Agent | Status | Notes |
|---|---|---|---|---|---|
| T0 | `~/Desktop/genkit-mr-t0` | `jh/mr-track-0-core` | done | merged → `jh/mr-wave1` | `11cefc933` |
| T1 | `~/Desktop/genkit-mr-t1` | `jh/mr-track-1-generate` | relaunched | finishing uncommitted work | prior agent provider-errored |
| T2 | `~/Desktop/genkit-mr-t2` | `jh/mr-track-2-google` | done | ready to merge | `e49b88413` |
| T3 | `~/Desktop/genkit-mr-t3` | `jh/mr-track-3-plugins` | done | ready to merge | `61f74069d` |
| T4 | | `jh/mr-track-4-docs` | | blocked on T1+T2 | |

---

## 8. Out of scope (explicit)

- JS/Go changes
- Renaming `GeminiConfigSchema` → `GeminiConfig` across the codebase (aliases only)
- Interactions / Veo background-model work (separate swarm)
- Making bare `model="googleai/..."` infer `GeminiConfig` in the type checker (impossible without prefix types; by design)
- Exhaustive known-model registries as a release gate
