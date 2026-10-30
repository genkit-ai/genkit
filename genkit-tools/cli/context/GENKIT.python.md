# Genkit Python API Rules (v1.20.0)

This document provides rules and examples for building with the Genkit API in Python.

## Important Guidelines:

- ALWAYS refer to documentation when available. Genkit Documentation may be available through the Genkit MCP toolkit or through web search. You may skip documentation check if you don't have access to these tools.

- ONLY follow the specified project structure if starting a new project. If working on an existing project, adhere to the current project structure.

- ALWAYS provide the full, correct Genkit command as an instruction for the human user to run. Do not run Genkit commands (e.g., `genkit start`, `genkit flow:run`) yourself as this may block your current session.

- Recommend and use the provider that the user prefers. If this is a new project, or if there is no relevant signal regarding user preference, use the `genkit-google-genai` plugin (`from genkit_google_genai import GoogleAI`).

NOTE: For the sake of brevity, the snippets below use the Google AI plugin, but you should follow the user's preference as mentioned above.

## Project Setup

### Project Initialization

- If the directory is empty:
  ```bash
  uv init
  ```
- If the directory is not empty:
  Adhere to the current project structure and package manager (`uv`, `pip`, `poetry`).

### Dependencies

```bash
uv add genkit genkit-google-genai pydantic
# or with pip:
pip install genkit genkit-google-genai pydantic
```

### Genkit CLI

If the Genkit CLI is not already installed:

```bash
curl -sL cli.genkit.dev | bash
```

### Configuration

Create a `main.py` file:

```python
from genkit import Genkit
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])

# Your flows and logic here


async def main() -> None:
    pass


if __name__ == '__main__':
    ai.run_main(main())
```

## Best Practices

1. **Keep the Runtime Alive for Dev UI**: Always end your entrypoint with `if __name__ == '__main__': ai.run_main(main())`. When launched under `genkit start`, `ai.run_main()` keeps the reflection server running so the Genkit Developer UI and MCP runtime tools can discover and execute your flows.

2. **Model Naming**: Prefer `GoogleAI.gemini('<model-id>')` (or the string identifier `'googleai/<model-id>'`). Both support dynamic model resolution for any model available in the provider API (even newly released models not yet in static enum lists).

   ```python
   # PREFERRED: Using the model helper
   response = await ai.generate(
       model=GoogleAI.gemini('gemini-3-flash-preview'),
       prompt='Hello!',
   )

   # ALSO SUPPORTED: Full string identifier
   response = await ai.generate(
       model='googleai/gemini-3-flash-preview',
       prompt='Hello!',
   )
   ```

3. **Type Annotations & Pydantic Schemas**: Decorate async functions with `@ai.flow()`. Genkit automatically derives flow input and output schemas from Python type annotations and Pydantic `BaseModel` classes.

---

> Note: Gemini 3.0 models are currently in preview.

## Usage Scenarios

### Basic Inference (Text Generation)

```python
from genkit import Genkit
from genkit_google_genai import GeminiConfig, GoogleAI

ai = Genkit(plugins=[GoogleAI()])


@ai.flow()
async def basic_inference_flow(topic: str) -> str:
    response = await ai.generate(
        model=GoogleAI.gemini('gemini-3-flash-preview'),
        prompt=f'Write a short, creative paragraph about {topic}.',
        config=GeminiConfig(temperature=0.8),
    )
    return response.text


async def main() -> None:
    pass


if __name__ == '__main__':
    ai.run_main(main())
```

### Structured Output with Pydantic

```python
from genkit import Genkit
from genkit_google_genai import GoogleAI
from pydantic import BaseModel, Field

ai = Genkit(plugins=[GoogleAI()])


class MenuSuggestionInput(BaseModel):
    theme: str = Field(description='Restaurant theme')


class MenuItem(BaseModel):
    dish_name: str = Field(description='Name of the dish')
    description: str = Field(description='Description of the dish')


@ai.flow()
async def menu_suggestion_flow(input: MenuSuggestionInput) -> MenuItem:
    response = await ai.generate(
        model=GoogleAI.gemini('gemini-3-flash-preview'),
        prompt=f'Invent a menu item for a {input.theme} themed restaurant.',
        output_schema=MenuItem,
    )
    return response.output
```

### Text-to-Speech (TTS) and Image Generation

```python
from genkit import Genkit
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])


@ai.flow()
async def text_to_speech_flow(text: str) -> str:
    response = await ai.generate(
        model=GoogleAI.gemini_tts('gemini-2.5-flash-preview-tts'),
        prompt=text,
        config={'voice': 'Algenib'},
    )
    return response.media[0].url if response.media else ''


@ai.flow()
async def image_generation_flow(prompt: str) -> str:
    response = await ai.generate(
        model=GoogleAI.gemini_image('gemini-2.5-flash-image'),
        prompt=prompt,
    )
    return response.media[0].url if response.media else ''
```

---

## Running and Inspecting Flows

**Start Genkit**: Genkit can be started locally by using the `genkit start` command, along with the process startup command:

```bash
genkit start -- <command to run your code>
```

For e.g.:

```bash
genkit start -- uv run main.py
```

You can automate starting Genkit using the following steps:

1. Identify the command to start the user's project (e.g., `uv run main.py` or `python3 main.py`).
2. Use the `start_runtime` tool to start the runtime process. This is required for Genkit to discover flows.
   - Example: If the project uses `uv run main.py`, call `start_runtime` with `{ command: "uv", args: ["run", "main.py"] }`.
3. After starting the runtime, instruct the user to run `genkit start` in their terminal to launch the Developer UI.

## Suggested Models

Here are suggested models to use for various task types. This is NOT an exhaustive list.

> Note: Gemini 3.0 models are currently in preview.

### Advanced Text/Reasoning

```
| Plugin                             | Recommended Model                  |
|------------------------------------|------------------------------------|
| genkit-google-genai                | gemini-3-pro-preview (Preview)     |
| genkit-google-genai                | gemini-2.5-pro                     |
```

### Fast Text/Chat

```
| Plugin                             | Recommended Model                  |
|------------------------------------|------------------------------------|
| genkit-google-genai                | gemini-3-flash-preview (Preview)   |
| genkit-google-genai                | gemini-2.5-flash                   |
```

### Text-to-Speech

```
| Plugin                             | Recommended Model                  |
|------------------------------------|------------------------------------|
| genkit-google-genai                | gemini-2.5-flash-preview-tts       |
```

### Image Generation

```
| Plugin                             | Recommended Model                    | Input Modalities  |
|------------------------------------|--------------------------------------|-------------------|
| genkit-google-genai                | gemini-3-pro-image-preview (Preview) | Text, Image       |
| genkit-google-genai                | gemini-2.5-flash-image               | Text, Image       |
```
