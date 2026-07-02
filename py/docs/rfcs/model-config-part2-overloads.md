# Model Config Typing RFC — Part 2: The `ai.generate` Overloads Spec

This document provides the technical specification for the updated `ai.generate()` method signatures in the Genkit Python SDK. It describes how the static type-inference engine resolves the input configuration type and the output schema type.

---

## 1. Type Inference Dimensions

The `ai.generate()` method signature must support static type inference across two independent dimensions:

1.  **Input Configuration (Dimension 1):** Resolves the configuration keys based on whether the model is specified as a bare string (`str`) or a statically typed model handle (`ModelRef`).
2.  **Output Structure (Dimension 2):** Resolves the return type based on whether an explicit structured output schema (`output_schema`) is provided.

To support all valid combinations of these two dimensions without losing type safety, the overload set for `ai.generate()` is structured as four distinct signatures.

---

## 2. Overload Signatures Specification

The following four overloads are defined in `genkit/_ai/_aio.py` (and mirrored in the synchronous client interface), replacing the existing signatures:

```python
# genkit/_ai/_aio.py
from typing import overload, TypeVar, Any, Sequence
from genkit.ai import CommonModelConfigDict, ModelRef

# Type variables for generic resolution
OutputT = TypeVar('OutputT')
ConfigT = TypeVar('ConfigT', bound=CommonModelConfigDict, default=CommonModelConfigDict)

# =============================================================================
# OVERLOAD 1: ModelRef Handle + Explicit Output Schema
# =============================================================================
@overload
async def generate(
    self,
    *,
    model: ModelRef[ConfigT],
    config: ConfigT | None = None,
    output_schema: type[OutputT],
    prompt: str | list[Part] | None = None,
    system: str | list[Part] | None = None,
    messages: list[Message] | None = None,
    tools: Sequence[str | Tool] | None = None,
    # ... other standard parameters
) -> ModelResponse[OutputT]: ...

# =============================================================================
# OVERLOAD 2: ModelRef Handle + Implicit Output Schema (Default Return)
# =============================================================================
@overload
async def generate(
    self,
    *,
    model: ModelRef[ConfigT],
    config: ConfigT | None = None,
    output_schema: None = None,
    prompt: str | list[Part] | None = None,
    system: str | list[Part] | None = None,
    messages: list[Message] | None = None,
    tools: Sequence[str | Tool] | None = None,
    # ... other standard parameters
) -> ModelResponse[Any]: ...

# =============================================================================
# OVERLOAD 3: Bare String Model + Explicit Output Schema
# =============================================================================
@overload
async def generate(
    self,
    *,
    model: str | None = None,
    config: CommonModelConfigDict | None = None,
    output_schema: type[OutputT],
    prompt: str | list[Part] | None = None,
    system: str | list[Part] | None = None,
    messages: list[Message] | None = None,
    tools: Sequence[str | Tool] | None = None,
    # ... other standard parameters
) -> ModelResponse[OutputT]: ...

# =============================================================================
# OVERLOAD 4: Bare String Model + Implicit Output Schema (Default Return)
# =============================================================================
@overload
async def generate(
    self,
    *,
    model: str | None = None,
    config: CommonModelConfigDict | None = None,
    output_schema: type | dict | None = None,
    prompt: str | list[Part] | None = None,
    system: str | list[Part] | None = None,
    messages: list[Message] | None = None,
    tools: Sequence[str | Tool] | None = None,
    # ... other standard parameters
) -> ModelResponse[Any]: ...
```

---

## 3. Static Type Resolution Flow

When a developer invokes `ai.generate()`, the static analysis engine (such as Pyright or Mypy) processes the arguments in the following sequence:

### Step 1: Input Model Evaluation
*   **Handle-Based Invocation:** If the `model` argument is passed a `ModelRef` object (e.g., the output of `GoogleAI.model(...)`), the type checker narrows the matching signatures to **Overloads 1 & 2**. The generic type parameter `ConfigT` is bound to the specific configuration dictionary type carried by that handle (e.g., `GeminiConfigDict`).
*   **String-Based Invocation:** If the `model` argument is passed a `str`, the type checker narrows the matching signatures to **Overloads 3 & 4**. The `config` parameter is constrained strictly to `CommonModelConfigDict`.

### Step 2: Input Configuration Validation
*   If Overloads 1 or 2 are matched, the `config` argument is validated against the bound `ConfigT`. The IDE provides autocomplete and verification for both the inherited common parameters and the model-specific parameters defined in that subclass.
*   If Overloads 3 or 4 are matched, the `config` argument is validated against `CommonModelConfigDict`. Only common parameters are accepted; any provider-specific parameters are flagged as type errors.

### Step 3: Output Schema Evaluation
*   If the `output_schema` argument is provided with a valid type (e.g., a Pydantic model class `type[OutputT]`), the type checker matches **Overload 1** (or Overload 3) and binds `OutputT` to that type. The return value is inferred as `ModelResponse[OutputT]`.
*   If the `output_schema` argument is omitted or `None`, the type checker matches **Overload 2** (or Overload 4) and infers the return value as `ModelResponse[Any]`.

---

## 4. API Surface Refinement

To simplify the public API surface area, the Pydantic `ModelConfig` class is removed from the public signatures of `ai.generate()`. 

1.  **Single Configuration Interface:** Developers configure models using a single, unified interface: the dictionary literal. This removes the need to choose between Pydantic instances and raw dictionaries at the call site.
2.  **Reduction of Import Overhead:** Developers do not need to import the `ModelConfig` class to write typed configurations.
3.  **Preservation of Runtime Validation:** This change is restricted to the static typing layer. At runtime, the Genkit execution engine still accepts the dictionary payload and validates/coerces it against the registered Pydantic schema at the model action boundary.
