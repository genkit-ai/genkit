# ModelRef Docstrings & Examples Audit

This document tracks all user-facing hero docstrings, formatting guides, and code samples across the Python SDK that demonstrate model selection, generation, or configuration. These targets should be updated to showcase the typed `ModelRef[ConfigT]` design, **family helpers** (`gemini_model`, `claude_model`, …), and typed Pydantic configuration constructors.

**Hero syntax (typed path):** `gemini_model("gemini-flash-latest")` + `GeminiConfig(...)`, not `model_ref("gemini-...")` as the primary typed example.

---

## 1. Core Runtime & API (`genkit`)

- [x] ~~**Runtime Generation Methods**~~ (`py/packages/genkit/src/genkit/_ai/_runtime.py`)
  - **N/A:** this module manages CLI runtime metadata files, not `ai.generate`. No model-selection examples to update.

- [x] **Async I/O Equivalents**
  - File: `py/packages/genkit/src/genkit/_ai/_aio.py`
  - `generate` + `generate_stream` docstrings show dict-config quick path and typed hero examples.

- [x] **Core Model Options & Helpers**
  - Files: `py/packages/genkit/src/genkit/_core/_model.py` (`ModelRef`), `py/packages/genkit/src/genkit/_ai/_model.py` (`model_ref`)
  - Docstrings point at family helpers as the app-facing typed path.

---

## 2. Structured Output Formats (`genkit._ai._formats`)

- [x] **Array Formatter** (`_array.py`)
- [x] **Enum Formatter** (`_enum.py`)
- [x] **JSON Formatter** (`_json.py`)
- [x] **JSONL Stream Formatter** (`_jsonl.py`)
- [x] **Text Formatter** (`_text.py`)

Usage examples now include `model=gemini_model('gemini-flash-latest')`.

---

## 3. Middleware & Testing Plugins

- [x] **Middleware Module Docstring** (`py/packages/genkit/src/genkit/middleware/__init__.py`)
  - Hero uses `gemini_model('gemini-flash-latest')`.

- [x] **Testing Aids** (`py/packages/genkit/src/genkit/_ai/_testing.py`)
  - Module + `test_models` docs clarify test doubles stay string-named (`testEcho`, …); app code uses family helpers.

---

## 4. Provider Plugins

### Google GenAI & Vertex AI (`genkit-google-genai`)

- [x] **Package Landing Docstring** (`__init__.py`)
- [x] **Google & Vertex Plugin Implementation** (`google.py`)
- [x] **Specialized Models (Veo & Lyria)** (`veo.py`, `lyria.py`)

### Other Community Provider Plugins
- [x] **Anthropic Plugin** — `claude_model` + `AnthropicConfig`
- [x] **Ollama Plugin** — `ollama_model` + `OllamaConfig`

---

## 5. Primary READMEs & Documentation
- [x] **Top-Level SDK README** (`py/README.md`)
- [x] **Core Genkit README** (`py/packages/genkit/README.md`)
- [x] **Google GenAI README** (`py/packages/genkit-google-genai/README.md`)

---

## 6. Samples (T4 subset)

- [x] **google-genai-media** (`py/samples/google-genai-media/README.md`) — typed helper section; runtime code unchanged.
- [x] **gemini-code-execution** (`py/samples/gemini-code-execution/src/main.py`) — module docstring hero.

*Remaining samples under `py/samples/` still use bare model strings in code; update opportunistically or in a follow-up pass.*
