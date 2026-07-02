# Model Config Typing RFC — Part 1: Code Examples

This document provides comparative code examples contrasting the current model configuration patterns in the Genkit Python SDK with the proposed TypedDict-based configuration design.

---

## 1. Common Configuration Path (Bare String Models)

This example demonstrates how common, cross-provider configuration parameters (such as `temperature` and `max_output_tokens`) are handled when invoking a model using a standard string identifier.

### Current Implementation
Developers must choose between passing an untyped dictionary (which lacks IDE autocomplete and compile-time verification) or instantiating a Pydantic configuration class (which requires explicit imports and constructor syntax).

```python
# Option A: Untyped Dictionary (No compile-time key validation or autocomplete)
await ai.generate(
    model='gemini-2.0-flash',
    config={
        'temperatur': 0.7,      # Typo is not flagged by static analysis
        'max_output_tokens': 100
    }
)

# Option B: Pydantic Configuration Instance (Type-safe but requires explicit imports)
from genkit import ModelConfig

await ai.generate(
    model='gemini-2.0-flash',
    config=ModelConfig(
        temperature=0.7,
        max_output_tokens=100
    )
)
```

### Proposed Implementation
The configuration is passed as an inline dictionary literal. The IDE's static analysis engine automatically autocompletes the keys and flags invalid fields at compile time, without requiring the developer to import or instantiate a configuration class.

```python
# Proposed: Inline TypedDict configuration
await ai.generate(
    model='gemini-2.0-flash',
    config={
        'temperature': 0.7,          # Autocompleted by IDE
        'max_output_tokens': 100,     # Autocompleted by IDE
        'temperatur': 0.7            # Flagged as type error by static analysis
    }
)
```

---

## 2. Model-Specific Configuration Path (Typed Handles)

This example demonstrates how provider-specific configuration parameters (such as Gemini's safety settings or OpenAI's reasoning configurations) are handled.

### Current Implementation
To configure provider-specific keys with type safety, developers must import the specific settings class and annotate a separate configuration variable before passing it to the constructor. Writing these keys inline in a dictionary literal bypasses call-site type-checking.

```python
# Option A: Inline Dictionary (Unvalidated provider-specific keys)
await ai.generate(
    model='gemini-2.0-flash',
    config={
        'temperature': 0.7,
        'google_safety_setting': [...]  # Typo is not flagged by static analysis
    }
)
```

### Proposed Implementation
Passing a statically typed model handle binds the corresponding configuration type to the invocation. The IDE autocompletes both the common parameters (inherited from the base type) and the provider-specific parameters within a single dictionary block.

```python
# Proposed: Model-specific configuration using a typed handle
from genkit.plugins.google_genai import GoogleAI

await ai.generate(
    model=GoogleAI.model('gemini-2.0-flash'),  # Binds GeminiConfigDict to the invocation
    config={
        'temperature': 0.7,                     # Autocompleted (inherited from base type)
        'google_safety_settings': [...],        # Autocompleted (provider-specific)
        'google_safety_setting': [...]         # Flagged as type error by static analysis
    }
)
```

---

## 3. Reusable Generic Configurations (TypedDict Inheritance)

This example demonstrates how developers can write reusable helper functions or middleware that accept and process configuration dictionaries across multiple model providers.

### Current Implementation
Because there is no shared `TypedDict` interface representing the intersection of common configuration parameters, generic functions must accept a wide dictionary type (such as `dict[str, Any]`), which disables type safety inside the helper.

```python
from typing import Any

# Helper function with wide dictionary type
def apply_corporate_limits(config: dict[str, Any]) -> dict[str, Any]:
    config['temperatur'] = 0.5  # Typo is not flagged by static analysis
    return config
```

### Proposed Implementation
Because all generated provider-specific configuration dictionaries inherit from the core `CommonModelConfigDict` base type, developers can use the base type as a standard type annotation. Static analysis verifies the keys both inside the helper and when passing provider-specific dictionaries to it.

```python
from genkit.ai import CommonModelConfigDict
from genkit.plugins.google_genai import GeminiConfigDict

# Helper function using the shared base type
def apply_corporate_limits(config: CommonModelConfigDict) -> CommonModelConfigDict:
    config['temperature'] = 0.5  # Autocompleted and validated inside the helper
    config['temperatur'] = 0.5   # Flagged as type error by static analysis
    return config

# Structural subtyping allows passing the provider-specific dictionary
gemini_config: GeminiConfigDict = {
    'temperature': 0.7,
    'google_safety_settings': [...]
}

# Accepted by static analysis due to inheritance relationship
safe_gemini_config = apply_corporate_limits(gemini_config)
```
