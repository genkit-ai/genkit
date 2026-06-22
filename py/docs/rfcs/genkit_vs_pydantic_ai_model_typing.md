# Architectural Comparison: Google-Specific Config Type Safety
## Genkit Python vs. Pydantic AI

This document provides an objective, technical comparison of model configuration type safety in Genkit Python and Pydantic AI. It outlines the strengths of Pydantic's approach, where that approach breaks down for model-specific configurations (such as Gemini's Google Search Grounding), and how the proposed Typed Model Handles design resolves these limitations.

---

## 1. The Common Case: Pydantic's TypedDict Success

For simple, cross-provider configuration parameters (such as `temperature` and `max_tokens`), Pydantic AI's use of a flat `TypedDict` (`ModelSettings`) is highly effective. It allows developers to configure models using a clean, inline dictionary literal:

```python
# Pydantic AI: Clean common config inline
agent = Agent(
    'google:gemini-2.5-flash',
    model_settings={'temperature': 0.7, 'max_tokens': 100}  # ✅ Statically verified inline
)
```

This approach has zero syntax friction: it requires no imports of configuration classes and no class instantiation boilerplate. 

**Genkit Proposal:** We acknowledge the elegance of this approach and are adopting it. In our new design, Genkit core defines a base `CommonModelConfigDict` `TypedDict` so that passing a plain dictionary to a bare string model or a generic function is similarly autocompleted and type-safe by default.

---

## 2. The Breakdown: Model-Specific Configurations

Pydantic's flat `TypedDict` approach breaks down when configuring advanced, provider-specific features that only exist for a particular model family, such as Gemini's **Google Search Grounding** (`google_search_grounding`) or **Safety Settings** (`google_safety_settings`).

Because Pydantic AI's constructor resolves models dynamically from a raw string (e.g., `'google:gemini-2.5-flash'`), the type checker cannot dynamically map the model provider to the configuration schema at the call site.
*   The constructor parameter `model_settings` is typed as the base `ModelSettings` class.
*   Because `ModelSettings` does not contain Google-specific keys, **writing `'google_search_grounding': True` inline will flag a compile-time error** (unrecognized key) in a strict type checker. 
*   To avoid flagging errors on other providers, the type checker must allow extra keys, which **completely deactivates call-site validation** for provider-specific configurations.

### The Solution: Typed Model References (`ModelRef[ConfigT]`)

To resolve this, the type checker must have a mechanism to dynamically bind the specific configuration schema to the model name at the call site. This requires a **typed model reference** (`ModelRef[ConfigT]`) that ties the configuration type to the model handle:

```python
# Genkit Proposal: Typed handle binds the configuration type inline
await ai.generate(
    model=GoogleAI.model('gemini-2.0-flash'),  # Binds ConfigT to GeminiConfigDict
    config={
        'temperature': 0.7,
        'google_search_grounding': True,       # ✅ Autocompleted & validated inline
        'google_safety_settings': [...]        # ✅ Autocompleted & validated inline
    }
)
```

By passing a typed handle instead of a bare string, the configuration dictionary's type is dynamically constrained, delivering perfect inline autocomplete and validation with zero user-facing boilerplate.

---

## 3. The Escape Hatch and Shared Type-Contamination Vulnerability

To work around their call-site breakdown and provide type safety for provider-specific keys, Pydantic AI offers an **escape hatch**: developers must manually import a provider-specific `TypedDict` (`GoogleModelSettings`), declare a separate variable, explicitly annotate it, and pass it to the constructor.

```python
# Pydantic AI Escape Hatch: Manual Type Casting
from pydantic_ai.models.google import GoogleModelSettings

# Manual annotation forces the type checker to validate the keys
settings: GoogleModelSettings = {
    'google_search_grounding': True,
    'temperature': 0.7
}
agent = Agent('google:gemini-2.5-flash', model_settings=settings)
```

### The Shared Vulnerability
This escape hatch is **structurally identical to how Genkit handles configuration today (before this proposal)**, where we allow developers to pass Pydantic schema class instances (such as `GeminiConfigSchema` or `OpenAIConfigSchema`) that inherit from a base `ModelConfig` class.

Both Pydantic AI's escape hatch and Genkit's current implementation suffer from the **exact same critical type-safety failure mode: Cross-Provider Type Contamination.**

Because the constructor parameter is typed widely as the base class (`ModelSettings` in Pydantic AI, `ModelConfig` in Genkit), **the type checker completely fails to protect developers from passing a completely different model's configuration.**

#### Example: Silent Contamination in both Pydantic AI & Current Genkit
If a developer refactors a model from OpenAI to Gemini, but mistakenly passes OpenAI-specific settings:

```python
# ❌ Pydantic AI: Silent Pass (IDE is happy, setting is silently discarded at runtime)
from pydantic_ai.models.openai import OpenAIChatModelSettings

settings: OpenAIChatModelSettings = {
    'openai_reasoning_effort': 'high',
    'temperature': 0.7
}
# Passing OpenAI settings to a Gemini agent is ALLOWED by the type checker!
agent = Agent('google:gemini-2.5-flash', model_settings=settings)


# ❌ Genkit Today: Silent Pass (IDE is happy, setting is silently discarded at runtime)
from genkit.plugins.openai import OpenAIConfigSchema

# Passing OpenAI Pydantic object to a Gemini model is ALLOWED by the type checker!
await ai.generate(
    model='gemini-2.0-flash',
    config=OpenAIConfigSchema(openai_reasoning_effort='high')
)
```

In both frameworks today, the type checker is silent, the application compiles without warnings, and **the invalid settings are silently discarded at runtime**, leading to invisible bugs in production.

---

## 4. The Unified Solution: Genkit's New Architecture

The proposed Typed Model Handle design resolves **both** Pydantic's call-site breakdown and the shared cross-provider contamination vulnerability completely.

By statically binding the configuration type to the model handle via generics, the type checker enforces a strict type boundary at the call site. Sibling configuration types (like `OpenAIConfigDict` and `GeminiConfigDict`) cannot be assigned to each other, ensuring absolute refactoring safety:

```python
# Proposed Genkit: Statically blocked at the call site
from genkit.plugins.google_genai import GoogleAI
from genkit.plugins.openai import OpenAIConfigDict

openai_config: OpenAIConfigDict = {
    'openai_reasoning_effort': 'high',
    'temperature': 0.7
}

# Sibling types do not match — blocked in the editor instantly!
await ai.generate(
    model=GoogleAI.model('gemini-2.0-flash'),  # Expects GeminiConfigDict
    config=openai_config                       # 🔴 pyright: Type Mismatch Error!
)
```

---

## 5. Conclusion

| Feature | Proposed Genkit (Our Design) | Pydantic AI | Current Genkit |
| :--- | :--- | :--- | :--- |
| **Common Config Inline DX** | **Safe & Ergonomic** (Base TypedDict) | **Safe & Ergonomic** (Base TypedDict) | **Unsafe** (Untyped dict) |
| **Model-Specific Inline DX** | **Safe & Ergonomic** (Family TypedDict) | **Unsafe** (Fails or bypasses linter) | **Unsafe** (Untyped dict) |
| **Type Contamination Guard** | **Strict** (Blocked by generic handle) | **None** (Allows wrong settings) | **None** (Allows wrong settings) |
| **API Cleanliness** | **High** (Single dictionary interface) | **Low** (Forced escape hatch boilerplate) | **Low** (Forced Pydantic class import) |

By adopting Pydantic's successful `TypedDict` pattern for the common case, and solving their provider-specific breakdown using **typed model handles**, Genkit Python delivers a unified, zero-boilerplate, and 100% type-safe configuration experience that is structurally superior to both Pydantic AI and the current Genkit SDK.
