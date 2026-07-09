# Genkit Python SDK

Build production-ready AI applications in Python with type-safe flows, structured outputs, and integrated observability.

## Quick Start

Get started in three simple steps:

1. **Install the SDK and your preferred model provider:**
```bash
uv add genkit genkit-google-genai
```

2. **Set your API key:**
```bash
export GEMINI_API_KEY="your-api-key"
```

3. **Create your AI application:**
```python
from genkit import Genkit
from genkit_google_genai import GoogleAI

# 1. Initialize Genkit with the Google AI (Gemini) plugin
ai = Genkit(plugins=[GoogleAI()])

# 2. Define a type-safe tool
@ai.tool(description="Get current weather for a city")
def get_weather(city: str) -> str:
    return f"Sunny, 72°F in {city}"

# 3. Define an observable flow
@ai.flow()
async def plan_trip(destination: str) -> str:
    response = await ai.generate(
        model="googleai/gemini-flash-latest",
        prompt=f"Suggest activities in {destination} given the weather.",
        tools=[get_weather],
    )
    return response.text  # => "Based on the sunny weather in Seattle..."
```

## Why Genkit?

- **Type-Safe by Design:** Leverage Python type annotations and Pydantic models for structured inputs, outputs, and tool definitions.
- **Multi-Model Provider API:** Switch effortlessly between Google Gemini, Anthropic Claude, OpenAI, Ollama, and Vertex AI with a unified API.
- **Integrated Observability:** Built-in OpenTelemetry tracing and evaluation metrics. Inspect spans and debug flows in real-time using the Genkit Developer UI (`genkit start`).
- **Deploy Anywhere:** Expose flows as standard ASGI/WSGI applications compatible with FastAPI, Flask, Django, Cloud Run, or any serverless platform.

---

## Repository & Development Guidelines

This section covers onboarding and common development workflows for contributing to the Genkit Python SDK.

### Prerequisites
- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/):** Fast Python package and project manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **[just](https://github.com/casey/just#installation):** Modern command runner (`brew install just` or `cargo install just`)

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
- **API Reference:** For complete class and method signatures, see [docs/index.md](docs/index.md).
- **Contributing & Standards:** For coding conventions, commit guidelines, and type-checking rules, see [CONTRIBUTING.md](../CONTRIBUTING.md).
- **Release Playbook:** For maintainer release procedures, see [docs/release_playbook.md](docs/release_playbook.md).

## License
Apache 2.0
