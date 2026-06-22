# Model Config Typing RFC — Part 3: Advanced Codegen & TypedDict Architecture

This document describes the underlying generic abstractions, code-generation pipeline, and architectural separation of compile-time static typing and runtime validation in the Genkit Python SDK.

---

## 1. Generic Abstraction and Phantom Generics

To carry static type information from the plugin call site to the execution layer without introducing runtime performance overhead, the SDK utilizes the generic `ModelRef` class:

```python
# genkit/ai/types.py (Core)
from typing import Generic, TypeVar
from dataclasses import dataclass
from genkit.ai import CommonModelConfigDict

# ConfigT is bound to CommonModelConfigDict, defaulting to it for fallback safety
ConfigT = TypeVar('ConfigT', bound=CommonModelConfigDict, default=CommonModelConfigDict)

@dataclass(frozen=True)
class ModelRef(Generic[ConfigT]):
    """An inert reference to a model.
    
    This class carries only a string name at runtime. The ConfigT generic 
    parameter is a "phantom" type that exists solely for static analysis 
    and drives inline configuration autocomplete inside ai.generate().
    """
    name: str  # e.g. 'googleai/gemini-2.0-flash'
```

### Inert Design Rationale
The `ModelRef` is designed to be an **inert data token**. It does not hold a reference to the registry, a client instance, or the parent `ai` class. This ensures that the handle can be instantiated statically as a classmethod *before* the registry is initialized or the plugin is registered. The resolution of the actual model action and middleware occurs exclusively during the execution of `ai.generate()`.

---

## 2. Declarative Model Manifest Specification (`models.yaml`)

To prevent schema drift and eliminate the manual overhead of writing and maintaining duplicate overload signatures, each plugin defines its supported models in a declarative manifest:

```yaml
# plugins/google-genai/models.yaml
plugin: googleai
families:
  gemini:
    config_schema: GeminiConfigSchema      # Reference to the Pydantic schema
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
    config_schema: GeminiTtsConfigSchema   # Reference to TTS-specific schema
    models:
      - name: gemini-2.5-flash-tts
        supports:
          multiturn: false
```

The manifest serves as the single source of truth for all model metadata (including label, stage, and capability flags), replacing the scattered, hand-coded `ModelInfo` tables in the source files.

---

## 3. Build-Time Code Generation Pipeline

During the build and CI processes, a code generator (`generate_model_refs.py`) parses the manifest and generates the corresponding static and runtime layers:

```
  models.yaml ──► generate_model_refs.py ──► _generated_models.py
                                              ├─ MODEL_INFO            (runtime registry)
                                              ├─ get_model_config_schema()  (runtime resolver)
                                              ├─ GeminiModelName = Literal['gemini-2.0-flash', ...]
                                              └─ GoogleAI.model() overloads → ModelRef[GeminiConfigDict]
```

### Step A: Compiling Pydantic Schemas to TypedDicts
The generator utilizes the `schema_to_typing.py` tool to parse the Pydantic schemas referenced in the manifest and output them as Python `TypedDict`s. 

To support generic helper functions and avoid duplicate field definitions, **every generated `TypedDict` is written to inherit from `CommonModelConfigDict`**:

```python
# _generated_models.py (generated — do not edit)
from genkit.ai import CommonModelConfigDict

class GeminiConfigDict(CommonModelConfigDict, total=False):
    # Gemini-specific keys only. Common keys are inherited from CommonModelConfigDict.
    google_safety_settings: list[SafetySettingDict]
    google_thinking_config: ThinkingConfigDict
```

### Step B: Generating the Plugin Accessor Overloads
The generator outputs a mixin class containing the statically typed `@overload` signatures for the plugin's `.model()` accessor:

```python
# _generated_models.py (generated — do not edit)
from typing import overload, Literal
from genkit.ai import ModelRef, CommonModelConfigDict

# Literal type of all Gemini text model names
GeminiModelName = Literal['gemini-2.0-flash', 'gemini-2.5-flash']
GeminiTtsModelName = Literal['gemini-2.5-flash-tts']

class GoogleAIGeneratedModels:
    @overload
    @classmethod
    def model(cls, name: GeminiModelName, /) -> ModelRef[GeminiConfigDict]: ...
    
    @overload
    @classmethod
    def model(cls, name: GeminiTtsModelName, /) -> ModelRef[GeminiTtsConfigDict]: ...
    
    @overload
    @classmethod
    def model(cls, name: str, /) -> ModelRef[CommonModelConfigDict]: ...  # Safe fallback
    
    @classmethod
    def model(cls, name, /):
        # Implementation adds the plugin prefix at runtime
        return ModelRef(name=name if '/' in name else f'googleai/{name}')
```

The hand-written plugin class inherits from this generated mixin, requiring no manual boilerplate:

```python
# plugins/google_genai/__init__.py (hand-written)
from ._generated_models import GoogleAIGeneratedModels

class GoogleAI(Plugin, GoogleAIGeneratedModels):
    # Existing plugin initialization, client setup, etc.
    ...
```

---

## 4. Boundary Separation: Compile-Time vs. Runtime

This architecture maintains a strict separation of concerns between **static developer ergonomics** and **production runtime safety**, leveraging the distinct capabilities of `TypedDict` and Pydantic:

```
  DEVELOPMENT (IDE / Static Analysis)               PRODUCTION (Runtime Execution)
  ───────────────────────────────────               ──────────────────────────────
  
    config = {                                       ai.generate(config)
      'temperature': 0.7,                                     │
      'google_safety_settings': [...]                         ▼
    }                                                Model Action Boundary
                                                              │
         [Validated by TypedDict]                             ▼
           • No runtime overhead                       [Validated by Pydantic]
           • Instant autocomplete                      • Active type coercion
           • Typo-flagging in editor                   • Out-of-bounds checks
                                                       • Dev UI schema generation
```

*   **Compile-Time (IDE DX):** The type checker (`pyright`/`mypy`) validates the inline configuration dictionary against the generated `TypedDict` hierarchy. This provides instant autocomplete and typo-flagging in the editor without introducing runtime overhead or requiring class instantiation.
*   **Runtime (Framework Engine):** The raw dictionary payload is passed through the execution pipeline. At the model action boundary, the engine retrieves the Pydantic schema (`GeminiConfigSchema`) and **actively validates and coerces the dictionary**, ensuring complete production safety.
