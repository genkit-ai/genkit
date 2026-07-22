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
Build the ref with the plugin's `model_ref(...)` helper (name autocompletes, no magic string) and construct the config — full autocomplete on the provider's own fields, wrong types caught immediately.

```python
from genkit_google_genai import GeminiConfig, gemini_model

gemini_model("gemini-pro-latest")
imagen_model("...")


resp = await ai.generate(
    model=model_ref("gemini-pro-latest"),  # ModelRef[GeminiConfig]
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
    model=model_ref("gemini-pro-latest"),
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

### Doesn't `Mapping` kill the autocomplete?
No — autocomplete and type-checking are separate signals:
- **Autocomplete**: the language server aggregates suggestions across all union arms. The `TypedDict` arm contributes its keys; `Mapping` adds none but removes none. (Verified against `pyright-langserver`: `config={` still offers `'temperature'`, `'top_k'`, …)
- **Type-checking**: a union passes if any arm matches, and `Mapping` matches every dict — so a typo isn't flagged.

*Net*: suggestions on the common keys, any dict allowed. The only cost is no hard typo-rejection on raw dicts (drop `Mapping` if you ever want strictness instead).

## Plugin-author surface: a `model_ref(...)` helper

Each plugin exports one free function that returns a typed ref by name — instead of a pile of module symbols (`gemini_25_flash`, …). The name autocompletes and the return stays typed:

```python
from genkit_google_genai import GeminiConfig, model_ref

resp = await ai.generate(
    model=model_ref("gemini-pro-latest"),  # -> ModelRef[GeminiConfig]
    config=GeminiConfig(temperature=0.7, safety_settings=[...]),
)
```

Here’s what the plugin code might look like:

Literal overloads dispatch the return type on the name (the `googleai/` prefix is baked in, so callers pass the bare name):
```python
MODEL_SCHEMAS: dict[str, type[BaseModel]] = {
    "gemini-2.5-flash": GeminiConfig,
    "gemini-flash-latest": GeminiConfig,
    "gemini-pro-latest": GeminiConfig,
    "imagen-3": ImagenConfig,
    "imagen-4": ImagenConfig,
}


def lookup(name: str) -> type[BaseModel]:
    return _MODEL_SCHEMAS.get(name, ModelConfig)  # unknown -> common config


GEMINI_MODEL_FAMILY = Literal[
    "gemini-2.5-flash", "gemini-flash-latest", "gemini-pro-latest"
]
IMAGEN_MODEL_FAMILY = Literal["imagen-3", "imagen-4"]


def gemini_model(name: GEMINI_MODEL_FAMILY) -> ModelRef[GeminiConfig]: ...


def imagen_model(name: IMAGEN_MODEL_FAMILY) -> ModelRef[ImagenConfig]: ...
```

And here’s what the framework code looks like:

`core_model_ref` is the SDK's generic `genkit.model.model_ref` — it stamps name/schema onto a `ModelRef` and binds `ConfigT` from `config_schema`:
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
