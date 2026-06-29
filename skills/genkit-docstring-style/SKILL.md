---
name: genkit-docstring-style
description: Guidelines and required patterns for writing Python docstrings in Genkit, specifically focusing on product-oriented landing-page hero code snippets with numbered steps and inline return shape annotations (# =>).
---

# Genkit Python Docstring Style Guide

When writing or updating module, class, or function docstrings in Genkit Python packages (`py/packages/`), all code examples MUST follow the **Landing Page Hero Snippet** pattern.

## Core Philosophy

A code example in a docstring is not just documentation—it is a developer's first impression of Genkit. It should feel like the hero section on a high-converting developer landing page (e.g., Stripe, Vercel, LangChain).
- **Product-Oriented**: Show real-world, practical use cases (not foo/bar).
- **Action-Oriented Steps**: Guide the reader through numbered comments (`# 1. ...`, `# 2. ...`, `# 3. ...`).
- **Inline Shapes (`# =>`)**: Show the exact shape of returned data or side effects inline, so the reader understands what happens without executing the code.
- **Clean Multi-Line Outputs**: Never use raw string escapes like `\n` in inline output comments. Format multi-line output cleanly across indented `#` lines so it is human-readable at a glance.
- **Upgrade Existing Only**: Only upgrade or format code examples in docstrings that already have one or where explicitly requested. Do not add code examples to modules, classes, or functions that did not originally have a code block.
- **Model Standard**: Always use `googleai/gemini-flash-latest` or `googleai/gemini-pro-latest` (or `vertexai/...` equivalents) for Gemini examples.

## Code Example Pattern (Generation)

```python
Example:
    ```python
    from genkit import Genkit
    from genkit_googleai import GoogleAI

    # 1. Initialize Genkit with the plugin
    ai = Genkit(plugins=[GoogleAI()])

    # 2. Generate content with structured parameters
    res = await ai.generate(
        model='googleai/gemini-flash-latest',
        prompt='Suggest 2 catchy names for a space-themed coffee shop.',
    )

    # 3. Inspect output shapes directly
    print(res.text)
    # => 1. AstroBrew
    #    2. Nebula Nectar
    ```
```

## Code Example Pattern (Infrastructure & Telemetry)

```python
Example:
    ```python
    from genkit import Genkit
    from genkit_googleai import GoogleAI
    from genkit_googlecloud import enable_googlecloud_telemetry

    # 1. Enable Google Cloud Trace and Monitoring export
    enable_googlecloud_telemetry(project_id='my-project')

    # 2. All subsequent Genkit actions automatically export telemetry
    ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')
    await ai.generate(prompt='Hello, world!')
    # => Traces exported asynchronously to Cloud Trace (latency, tokens, status)
    ```
```

## Anti-Patterns to Avoid
- ❌ **No raw string escape codes (`\n`, `\t`) in comments**: Keep visual output formatted naturally on separate lines.
- ❌ **No ELI5 definitions or ASCII boxes (`┌──`)**: Keep docstrings clean and professional.
- ❌ **No hardcoded/deprecated model versions**: Avoid `gemini-2.0-flash` or old model names.
- ❌ **No silent snippets**: Never show code without showing what it produces (`# => ...`).
