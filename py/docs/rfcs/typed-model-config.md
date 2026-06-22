# Typed Model Configuration for Python — RFC Index

## 1. Executive Summary

This RFC proposes a statically typed, dictionary-based model configuration system for the Genkit Python SDK. 

Historically, developers had to choose between a completely untyped dictionary (which fails to flag typos or provide autocomplete in the IDE) and a verbose Pydantic `ModelConfig` class (which requires explicit imports and only supports common settings).

This proposal introduces a unified, zero-boilerplate **`TypedDict` hierarchy** that delivers TypeScript-grade inline autocomplete and compile-time key verification, while maintaining 100% of Pydantic's runtime validation and serialization power behind the scenes.

---

## 2. Detailed Proposal Structure

To make this design highly readable and digestible, the proposal is divided into three focused sections:

### 📖 **[Part 1: Code Examples](./model-config-part1-examples.md)**
*   **Topic:** Call-site Developer Experience (DX) and code comparisons.
*   **Content:** Side-by-side examples comparing the current syntax and static analysis behaviors with the proposed inline TypedDict design for:
    *   Common config parameters (bare strings).
    *   Model-specific config parameters (typed model handles).
    *   Reusable generic helpers and middleware (structural subtyping).

### 📝 **[Part 2: The `ai.generate` Overloads Spec](./model-config-part2-overloads.md)**
*   **Topic:** Public API interface and overload resolution.
*   **Content:** Thorough, precise technical specification of the **four new overload signatures** added to the `ai.generate()` method to simultaneously resolve input configuration typing and output schema typing. 
*   **Highlights:** Detailed step-by-step trace of how the IDE type compiler resolves arguments, and details on removing `ModelConfig` from the public signature to simplify the API surface area.

### 🛠️ **[Part 3: Advanced Codegen & TypedDict Architecture](./model-config-part3-codegen.md)**
*   **Topic:** System design, manifests, and the code-generation pipeline.
*   **Content:** Deep dive into the generic `ModelRef[ConfigT]` data token, the declarative `models.yaml` manifest, the build-time code generator, and the architectural division of labor between compile-time static types (`TypedDict`) and runtime validation (`Pydantic`).
