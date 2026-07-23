# Model Refs (Python)

Tie a `ModelRef` to the plugin's Pydantic config model and type the config parameter as a small union:

```python
ConfigT = TypeVar('ConfigT', bound=BaseModel, covariant=True, default=ModelConfig)


class ModelRef(BaseModel, Generic[ConfigT]):
    """Frozen, generic reference to a model."""

    model_config = ConfigDict(frozen=True)  # Read-only!
    name: str
    config_schema: type[ConfigT] | None = None


# ai.generate() overload 1: typed ref
model: ModelRef[ConfigT]
config: ConfigT | Mapping[str, Any] | None

# ai.generate() overload 2: string / default
model: str | None = None
config: BaseModel | ModelConfigDict | Mapping[str, Any] | None

# ai.generate() implementation (ConfigT erased to its bound, BaseModel)
model: str | ModelRef[BaseModel] | None = None
config: BaseModel | ModelConfigDict | Mapping[str, Any] | None
```

The plugin's Pydantic config class is the single source of truth; the `ModelRef` carries it as `ConfigT`, so `ai.generate` binds the config type from the model automatically.

## Use Cases

### 1. Common knobs as a plain dict, with autocomplete — no imports needed
The 90% case stays a one-liner, and the IDE still suggests `temperature`, `top_k`, …

```python
resp = await ai.generate(
    model="googleai/gemini-2.5-flash",
    config={"temperature": 0.7},  # 'temperature', 'top_k', ... autocomplete here
)
```

### 2. Provider-specific config, fully typed and validated
Build the ref with a **family helper** (`gemini_model`, `imagen_model`, …) and construct the config — full autocomplete on the provider's own fields, wrong types caught immediately.

```python
from genkit_google_genai import GeminiConfig, gemini_model

resp = await ai.generate(
    model=gemini_model("gemini-pro-latest"),  # ModelRef[GeminiConfig]
    config=GeminiConfig(
        temperature=0.7,
        safety_settings=[...],  # provider-specific, autocompleted
    ),
)
```

### 3. Pass-thru for anything not (yet) modeled
New or exotic provider knobs go straight through — never blocked by the type system.

```python
resp = await ai.generate(
    model=gemini_model("gemini-pro-latest"),
    config={
        "some_brand_new_provider_flag": True
    },  # accepted, passed through
)
```

## Design Decisions

### Decision 1 — the `ConfigT` arm: tie the ref to a Pydantic model
A dict literal can only be checked against a `TypedDict`, never a Pydantic model. So the only way to autocomplete provider-specific keys in a raw dict is a per-model `TypedDict`. But configs are already Pydantic models (for runtime validation), and you can't derive a static `TypedDict` from one — so that route means two parallel definitions per model, hand-synced and prone to drift.

Instead `ModelRef` is generic over the Pydantic model (`ModelRef[ConfigT]`): the config type rides along with the ref, the model stays the single source of truth, and provider-specific autocomplete + validation come from constructing it — `GeminiConfig(...)`.
*Tradeoff*: provider-specific keys autocomplete on the constructor, not in a raw dict (dict literals only autocomplete the common keys, below).

### Decision 2 — the `ModelConfigDict` arm: common keys as a dict
The 90% case only touches common knobs (`temperature`, `top_k`, …). One small, stable, hand-written `ModelConfigDict` gives those dict-literal autocomplete on both paths — no config-class import required for the easy case.

### Decision 3 — the `Mapping[str, Any]` arm: pass-thru
Callers must be able to send keys we haven't modeled yet. `Mapping[str, Any]` accepts any dict, so arbitrary config flows straight through to the provider.

### Decision 4 — Bound `ConfigT` with `BaseModel` (accept any Pydantic class)
`ConfigT` only means something in the overloads, where it binds to the passed ref's config type. The single impl body isn't generic, so the `TypeVar` erases to its bound, `BaseModel` — and the impl must accept everything the overloads do, so `config` needs a `BaseModel` arm.
The bound is `BaseModel`, not `ModelConfig`, because some configs (`ImagenConfigSchema`, `VeoConfigSchema`) subclass plain `BaseModel`.

### Decision 5 — Family helpers carry the type; names stay open `str`
Python cannot express “any string starting with `gemini-`” in the type system (no template-literal types). A single `model(name)` typed only via exhaustive `Literal[...]` lists would force a plugin release every time Google ships a new version string — just to keep return types correct.

So each plugin exports **per-family helpers**. The helper encodes the family; the name is an open `str`:

```python
gemini_model("gemini-flash-pro-whatever")  # -> ModelRef[GeminiConfig], no SDK release
imagen_model("imagen-99.0-generate-001")   # -> ModelRef[ImagenConfig]
```

Optional `Literal[...] | str` on the name param may list known names for **autocomplete only**. It must not gate runtime: any `str` still returns the family config type.

A Genkit release is required only for a **new family / new config schema / new prefix rule** — never for a new version string inside an existing family.

*Intentional deviation from JS:* JS can narrow `googleAI.model(name)` with `` `gemini-${string}` `` overloads. Python can't, so family helpers (`gemini_model`, …) are the typed public surface instead of one overloaded `model()`.

### Doesn't `Mapping` kill the autocomplete?
No — autocomplete and type-checking are separate signals:
- **Autocomplete**: the language server aggregates suggestions across all union arms. The `TypedDict` arm contributes its keys; `Mapping` adds none but removes none. (Verified against `pyright-langserver`: `config={` still offers `'temperature'`, `'top_k'`, …)
- **Type-checking**: a union passes if any arm matches, and `Mapping` matches every dict — so a typo isn't flagged.

*Net*: suggestions on the common keys, any dict allowed. The only cost is no hard typo-rejection on raw dicts (drop `Mapping` if you ever want strictness instead).

## Plugin surface: family helpers

Each plugin exports one helper per model family — instead of a pile of per-version module symbols (`gemini_25_flash`, …) or a single string-overloaded `model()`.

```python
from genkit_google_genai import GeminiConfig, gemini_model

resp = await ai.generate(
    model=gemini_model("gemini-pro-latest"),  # -> ModelRef[GeminiConfig]
    config=GeminiConfig(temperature=0.7, safety_settings=[...]),
)
```

Here’s what the plugin code might look like.

**Runtime:** prefix → family config (not an exact-name map). Unknown version strings in a known family still get that family's schema:

```python
def config_schema_for(name: str) -> type[BaseModel]:
    if is_tts_model(name):
        return GeminiTtsConfig
    if is_image_model(name):
        return GeminiImageConfig
    if is_gemini_model(name):
        return GeminiConfig
    if name.startswith("imagen-"):
        return ImagenConfig
    return ModelConfig  # unknown family -> common knobs only
```

Deep Research / Antigravity family helpers land with the Interactions-backed models (follow-up), not in the initial ModelRef PR.

**Types:** family helper + open `str`. Known-name `Literal`s are autocomplete sugar in the same param type (no `@overload` — return type doesn't change). Python has no “string prefixed with …” types, so the family helper is what makes the return type correct:

```python
KnownGemini = Literal[
    "gemini-2.5-flash", "gemini-flash-latest", "gemini-pro-latest"
]
KnownImagen = Literal["imagen-3", "imagen-4"]


def gemini_model(name: KnownGemini | str) -> ModelRef[GeminiConfig]:
    return model_ref(
        name,
        config_schema=GeminiConfig,
        namespace="googleai",
    )


def imagen_model(name: KnownImagen | str) -> ModelRef[ImagenConfig]:
    return model_ref(
        name,
        config_schema=ImagenConfig,
        namespace="googleai",
    )
```

Same pattern for `veo_model`, `lyria_model`, etc.

An optional unified `model_ref(name: str) -> ModelRef[BaseModel]` may exist for convenience (runtime prefix dispatch, loose return type). Callers who want `GeminiConfig` in the type checker use `gemini_model`.

## Framework: core `model_ref`

`model_ref` is the SDK's generic `genkit.model.model_ref` — it stamps name/schema onto a `ModelRef` and binds `ConfigT` from `config_schema`. Family helpers call this; app code usually calls the helpers.

```python
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
