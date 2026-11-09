# Genkit Python SDK

Build production-ready AI applications in Python with type-safe flows, structured outputs, and pluggable observability.

> **Building with a coding agent? Install the Genkit Python skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-python
> ```
>
> It teaches your agent the current Genkit Python APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

> **Upgrading from 0.12?** See the [0.12 → 0.13 migration guide](docs/migrating-from-0.12.md).

## Quick Start

1. **Create a project and install the SDK with a model provider:**
```bash
uv init my-genkit-app && cd my-genkit-app
uv add genkit genkit-google-genai
```

Using pip instead: `pip install genkit genkit-google-genai`.

2. **Set your API key** (get one from [Google AI Studio](https://aistudio.google.com/app/apikey)):
```bash
export GEMINI_API_KEY="your-api-key"
```

3. **Write `app.py`:**
```python
from genkit import Genkit
from genkit_google_genai import GoogleAI

# 1. Initialize Genkit with the Google AI (Gemini) plugin
ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))


# 2. Define a tool. Tools are async; wrap blocking I/O in asyncio.to_thread.
@ai.tool(description='Get current weather for a city')
async def get_weather(city: str) -> str:
    return f'Sunny, 72°F in {city}'


# 3. Define a flow. Flows take a single input argument.
@ai.flow()
async def plan_trip(destination: str) -> str:
    response = await ai.generate(
        prompt=f'Suggest activities in {destination} given the weather.',
        tools=[get_weather],
    )
    return response.text


# 4. Call the flow from the script entry point
if __name__ == '__main__':
    print(ai.run_main(plan_trip('Seattle')))
    # => Since it's sunny and 72°F in Seattle, here are some activities...
```

4. **Run it:**
```bash
uv run app.py
```

5. **Open the Developer UI** (needs Node.js and the Genkit CLI, `npm install -g genkit-cli`):
```bash
genkit start -- uv run app.py
```

`genkit start` sets `GENKIT_ENV=dev`, which turns on the reflection server. `ai.run_main()` keeps the process alive so the Dev UI can run `plan_trip` and show its traces. Without `genkit start`, `run_main()` returns as soon as the flow does.

## Why Genkit?

- **Type-Safe by Design:** Leverage Python type annotations and Pydantic models for structured inputs, outputs, and tool definitions.
- **Multi-Model Provider API:** Switch effortlessly between Google Gemini, Anthropic Claude, OpenAI, Ollama, Vertex AI, and Amazon Bedrock with a unified API.
- **Pluggable Observability:** Run your app under `genkit start -- uv run app.py` and the Developer UI Traces tab shows every flow, model, and tool call. Export to OpenTelemetry backends with `genkit-otel`.
- **Deploy Anywhere:** Expose flows as standard ASGI/WSGI applications compatible with FastAPI, Flask, Django, Cloud Run, or any serverless platform.

---

## Repository & Development Guidelines

This section covers onboarding and common development workflows for contributing to the Genkit Python SDK.

### Prerequisites
- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/):** Fast Python package and project manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **[just](https://github.com/casey/just#installation):** Modern command runner (`brew install just` or `cargo install just`)
- **[Node.js](https://nodejs.org/) and the Genkit CLI:** Required for `genkit start` and the Developer UI (`npm install -g genkit-cli`). The `genkit` Python package does not ship a `genkit` command.

### Workspace Structure
```
py/
├── bin/               # CI/CD and release automation scripts
├── docs/              # Playbooks and generated API reference templates
├── packages/          # Core framework and official integrations
├── samples/           # Runnable example applications and demos
├── scripts/           # Maintenance and verification scripts
├── tests/             # Cross-package integration test suites
├── justfile           # Command runner shortcuts (just py <command>)
├── noxfile.py         # Multi-version test automation (3.10–3.14)
├── pyproject.toml     # Workspace metadata and tool dependencies
└── uv.lock            # Resolved dependency lockfile
```

### Development Commands (`just py`)

From the repository root, run `just py <command>` (or `just <command>` in `py/`):

- **`sync`** — Install workspace dependencies (`uv sync`).
- **`lint`** — Run formatters, linters, and type checkers (maps to CI `lint-and-format` / `type-check`).
- **`fmt`** — Auto-format code and fix lint errors.
- **`test`** — Run unit tests (use `test-nox` to test Python 3.10–3.14 like CI).
- **`check`** — Validate workspace version consistency.

### Running Samples

To run example applications from `samples/`, navigate to a sample directory and launch the Genkit Developer UI:

```bash
cd samples/<sample-name>
genkit start -- uv run <entrypoint.py>
```

Open the Dev UI in your browser to interact with registered flows and agents directly.

### Documentation & Maintenance
- **API Reference:** For complete class and method signatures, see [python.api.genkit.dev](https://python.api.genkit.dev). Guides live at [genkit.dev](https://genkit.dev/docs/python/get-started/).
- **Migration Guides:** Upgrading between versions, see [docs/migrating-from-0.12.md](docs/migrating-from-0.12.md).
- **Contributing & Standards:** For coding conventions, commit guidelines, and type-checking rules, see [CONTRIBUTING.md](../CONTRIBUTING.md).
- **Release Playbook:** For maintainer release procedures, see [docs/release_playbook.md](docs/release_playbook.md).

## License
Apache 2.0
