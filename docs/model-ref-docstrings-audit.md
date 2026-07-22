# ModelRef Docstrings & Examples Audit

This document tracks all user-facing hero docstrings, formatting guides, and code samples across the Python SDK that demonstrate model selection, generation, or configuration. These targets should be updated to showcase the typed `ModelRef[ConfigT]` design, `model_ref(...)` helpers, and typed Pydantic configuration constructors.

---

## 1. Core Runtime & API (`genkit`)

These docstrings currently show bare string model identifiers (e.g., `model='googleai/gemini-2.0-flash'`) or untyped dictionary configurations.

- [ ] **Runtime Generation Methods**
  - File: `py/packages/genkit/src/genkit/_ai/_runtime.py`
  - Targets: Docstrings for `generate()`, `generate_stream()`, `chat()`, and `embed()`.
  - Update: Demonstrate passing a `ModelRef` object vs plain string, and passing a typed Pydantic config model.

- [ ] **Async I/O Equivalents**
  - File: `py/packages/genkit/src/genkit/_ai/_aio.py`
  - Targets: Overload and method docstrings for async generation and streaming.
  - Update: Sync example signatures with `_runtime.py`.

- [ ] **Core Model Options & Helpers**
  - File: `py/packages/genkit/src/genkit/_core/_model.py`
  - Target: `GenerateOptions` and model reference construction utilities.
  - Update: Add examples of type-safe generation options using typed references.

---

## 2. Structured Output Formats (`genkit._ai._formats`)

All structured formatters currently demonstrate basic generation calls with default or bare string models in their class docstrings.

- [ ] **Array Formatter** (`py/packages/genkit/src/genkit/_ai/_formats/_array.py`)
- [ ] **Enum Formatter** (`py/packages/genkit/src/genkit/_ai/_formats/_enum.py`)
- [ ] **JSON Formatter** (`py/packages/genkit/src/genkit/_ai/_formats/_json.py`)
- [ ] **JSONL Stream Formatter** (`py/packages/genkit/src/genkit/_ai/_formats/_jsonl.py`)
- [ ] **Text Formatter** (`py/packages/genkit/src/genkit/_ai/_formats/_text.py`)
  - *Update Strategy:* Change `ai.generate(prompt=..., output_schema=...)` examples to highlight model helper functions where applicable, ensuring consistent usage style across formatting guides.

---

## 3. Middleware & Testing Plugins

- [ ] **Middleware Module Docstring** (`py/packages/genkit/src/genkit/middleware/__init__.py`)
  - Target: Module hero example showing `await ai.generate(...)`.
- [ ] **Testing Aids** (`py/packages/genkit/src/genkit/_ai/_testing.py`)
  - Target: Test helper docstrings demonstrating simulated model invocations (`model='testEcho'`, `model='testPM'`).

---

## 4. Provider Plugins

### Google GenAI & Vertex AI (`genkit-google-genai`)
This plugin is the primary benchmark for developer onboarding and should thoroughly showcase `gemini_model(...)`, `imagen_model(...)`, and typed Pydantic configs (`GeminiConfig(...)`, `ImagenConfig(...)`).

- [ ] **Package Landing Docstring**
  - File: `py/packages/genkit-google-genai/src/genkit_google_genai/__init__.py`
  - Targets: Module-level overview and hero code snippets for `GoogleAI` and `VertexAI` initialization and execution.
  - Current Syntax: `model='googleai/gemini-flash-latest'` / `model='vertexai/gemini-flash-latest'`.
  - Target Syntax: Demonstrate `model=model_ref("gemini-pro-latest")` and typed configuration objects.

- [ ] **Google & Vertex Plugin Implementation**
  - File: `py/packages/genkit-google-genai/src/genkit_google_genai/google.py`
  - Targets: `GoogleAI` and `VertexAI` class docstrings, as well as model registration tables and helper functions.

- [ ] **Specialized Models (Veo & Lyria)**
  - File: `py/packages/genkit-google-genai/src/genkit_google_genai/models/veo.py`
    - Target Syntax: Replace `model='googleai/veo-2.0-generate-001'` with typed Veo helper or `model_ref`.
  - File: `py/packages/genkit-google-genai/src/genkit_google_genai/models/lyria.py`
    - Target Syntax: Replace `model='vertexai/lyria-002'` with typed Lyria helper or `model_ref`.

### Other Community Provider Plugins
- [ ] **Anthropic Plugin** (`py/packages/genkit-anthropic/src/`)
  - Target: Docstrings demonstrating Claude model invocations (e.g., `anthropic/claude-haiku-4-5`).
- [ ] **Ollama Plugin** (`py/packages/genkit-ollama/src/`)
  - Target: Docstrings demonstrating local model execution.

---

## 5. Primary READMEs & Documentation
- [ ] **Top-Level SDK README** (`py/README.md`)
- [ ] **Core Genkit README** (`py/packages/genkit/README.md`)
- [ ] **Google GenAI README** (`py/packages/genkit-google-genai/README.md`)
