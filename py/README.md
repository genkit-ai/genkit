# Genkit Python SDK

Genkit is a framework for building AI-powered applications with type-safe flows, structured outputs, and integrated observability. This is the Python implementation that maintains feature parity with the JavaScript/TypeScript SDK.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Your Application                                    │
│                                                                             │
│   from genkit import Genkit                                                 │
│   ai = Genkit(plugins=[GoogleGenAI()], model=gemini_2_0_flash)             │
│                                                                             │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌───────────┐     │
│   │  Flows  │  │  Tools  │  │ Prompts │  │ Embedders│  │ Retrievers│     │
│   │@ai.flow │  │@ai.tool │  │.prompt  │  │ai.embed()│  │ai.retrieve│     │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬─────┘  └─────┬─────┘     │
│        │            │            │             │             │              │
│        └────────────┴────────────┴─────────────┴─────────────┘              │
│                                  │                                          │
│                          ┌───────▼────────┐                                 │
│                          │    Registry     │                                 │
│                          │  (all actions)  │                                 │
│                          └───────┬────────┘                                 │
│                                  │                                          │
│             ┌────────────────────┼────────────────────┐                     │
│             ▼                    ▼                    ▼                     │
│      ┌────────────┐     ┌──────────────┐     ┌──────────────┐             │
│      │   Plugins   │     │   Tracing     │     │  Reflection  │             │
│      │ (providers) │     │ (OpenTelemetry│     │  API (DevUI) │             │
│      └────────────┘     └──────────────┘     └──────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Plugin Architecture

Plugins are the primary extension mechanism in Genkit. They add model
providers, telemetry backends, vector stores, and other capabilities.
Every plugin implements the same abstract interface (`Plugin` base class)
and is loaded lazily by the `Registry`.

### Plugin Class Hierarchy

All plugins inherit from `genkit.core.plugin.Plugin` and implement three
abstract methods:

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                  Plugin (Abstract Base Class)                       │
  │                  genkit.core.plugin.Plugin                          │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  name: str                                                          │
  │  ─────────────────────────────────────                              │
  │  Plugin namespace (e.g., 'googleai', 'anthropic', 'ollama')        │
  │                                                                     │
  │  async init() → list[Action]            ← called once per plugin   │
  │  ─────────────────────────────────────                              │
  │  One-time initialization; returns actions to pre-register.          │
  │  Called lazily on first action resolution, NOT at registration.     │
  │                                                                     │
  │  async resolve(kind, name) → Action?    ← called per lookup        │
  │  ─────────────────────────────────────                              │
  │  Resolve a single action by kind and name. Returns None if         │
  │  this plugin doesn't handle the requested action.                   │
  │                                                                     │
  │  async list_actions() → list[ActionMetadata]  ← for Dev UI         │
  │  ─────────────────────────────────────                              │
  │  Advertise available actions without heavy initialization.          │
  │  Called by the Reflection API for DevUI action discovery.           │
  │                                                                     │
  │  model(name) → ModelReference           ← helper method            │
  │  embedder(name) → EmbedderRef           ← helper method            │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
                  ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │   GoogleAI   │ │  Anthropic   │ │    Ollama    │   ... etc.
        │  name =      │ │  name =      │ │  name =      │
        │  'googleai'  │ │  'anthropic' │ │  'ollama'    │
        └──────────────┘ └──────────────┘ └──────────────┘
```

### Plugin Lifecycle

Plugins go through four phases: registration, lazy initialization,
action resolution, and action discovery.

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      Plugin Lifecycle                               │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  Phase 1: REGISTRATION (at Genkit startup)                         │
  │  ─────────                                                         │
  │  ai = Genkit(plugins=[GoogleAI(), Anthropic()])                    │
  │       │                                                            │
  │       ├─► registry.register_plugin(GoogleAI())                     │
  │       └─► registry.register_plugin(Anthropic())                    │
  │           │                                                        │
  │           ▼                                                        │
  │       ┌────────────────────────────────────────┐                   │
  │       │  Registry._plugins                      │                   │
  │       │  ┌────────────────┬───────────────────┐ │                   │
  │       │  │ "googleai"     │ GoogleAI instance │ │                   │
  │       │  │ "anthropic"    │ Anthropic instance│ │                   │
  │       │  └────────────────┴───────────────────┘ │                   │
  │       └────────────────────────────────────────┘                   │
  │                                                                     │
  │  Phase 2: LAZY INIT (on first action resolution)                   │
  │  ──────────                                                        │
  │  await ai.generate(model='googleai/gemini-2.0-flash', ...)        │
  │       │                                                            │
  │       ▼                                                            │
  │  registry._ensure_plugin_initialized('googleai')                   │
  │       │                                                            │
  │       ▼                                                            │
  │  actions = await plugin.init()    ← called exactly once            │
  │       │                            (subsequent calls are no-ops)   │
  │       ▼                                                            │
  │  for action in actions:                                            │
  │      registry.register_action_instance(action)                     │
  │                                                                     │
  │  Phase 3: ACTION RESOLUTION (on each usage)                        │
  │  ─────────────────────                                             │
  │  await plugin.resolve(ActionKind.MODEL, 'googleai/gemini-2.0-flash')│
  │       │                                                            │
  │       ▼                                                            │
  │  Action instance returned and cached in registry                   │
  │                                                                     │
  │  Phase 4: ACTION DISCOVERY (for Dev UI)                            │
  │  ──────────────────                                                │
  │  await plugin.list_actions()                                       │
  │       │                                                            │
  │       ▼                                                            │
  │  ActionMetadata[] returned to Reflection API                       │
  │  (does NOT trigger init — must be fast and safe)                   │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
```

### How the Registry Resolves Actions

When you call `ai.generate(model='googleai/gemini-2.0-flash')`, the
registry uses a multi-step resolution algorithm:

```
  ai.generate(model="googleai/gemini-2.0-flash")
       │
       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  Step 1: CACHE HIT?                                              │
  │  Is "googleai/gemini-2.0-flash" already in registry._entries?    │
  │     ├── YES → return cached Action (fast path)                   │
  │     └── NO  → continue to Step 2                                 │
  ├──────────────────────────────────────────────────────────────────┤
  │  Step 2: NAMESPACED or UNPREFIXED?                               │
  │  Does the name contain "/"?                                      │
  │     │                                                            │
  │     ├── YES ("googleai/gemini-2.0-flash")                        │
  │     │    ├── Find plugin "googleai"                               │
  │     │    ├── await _ensure_plugin_initialized("googleai")         │
  │     │    ├── Check cache again (init may have registered it)     │
  │     │    └── await plugin.resolve(MODEL, "googleai/gemini-2.0")  │
  │     │                                                            │
  │     └── NO ("gemini-2.0-flash")                                  │
  │          ├── Try ALL plugins                                      │
  │          ├── If 1 match  → use it                                 │
  │          ├── If 2+ match → ValueError (ambiguous)                │
  │          └── If 0 match  → continue to Step 3                    │
  ├──────────────────────────────────────────────────────────────────┤
  │  Step 3: DYNAMIC ACTION PROVIDERS (fallback)                     │
  │  Try registered Dynamic Action Providers (e.g., MCP servers)     │
  │     ├── Found → register and return                              │
  │     └── Not found → return None                                  │
  └──────────────────────────────────────────────────────────────────┘
```

### Writing a Custom Plugin

```python
from genkit.core.plugin import Plugin
from genkit.core.action import Action, ActionMetadata
from genkit.core.action.types import ActionKind


class MyPlugin(Plugin):
    name = 'myplugin'

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def init(self) -> list[Action]:
        # Return actions to pre-register (optional)
        return []

    async def resolve(self, kind: ActionKind, name: str) -> Action | None:
        if kind == ActionKind.MODEL:
            return self._create_model(name)
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        return [
            ActionMetadata(kind=ActionKind.MODEL, name='myplugin/my-model'),
        ]

# Usage:
ai = Genkit(plugins=[MyPlugin(api_key='...')])
response = await ai.generate(model='myplugin/my-model', prompt='Hello!')
```

## How a Flow Executes

```
  ai.generate(prompt="Tell me a joke")
       │
       ▼
  ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
  │  1. Flow  │───►│ 2. Model  │───►│ 3. Tool? │───►│ 4. Model │
  │  starts   │    │  called   │    │  (if the │    │ responds │
  │  tracing  │    │  (Gemini, │    │  model   │    │  with    │
  │           │    │  Claude,  │    │  decides │    │  final   │
  │           │    │  etc.)    │    │  to use  │    │  answer  │
  │           │    │           │    │  one)    │    │          │
  └──────────┘    └───────────┘    └──────────┘    └──────────┘
       │                │               │               │
       └────────────────┴───────────────┴───────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Trace (every step   │
                    │   recorded for DevUI, │
                    │   Cloud Trace, etc.)  │
                    └──────────────────────┘
```

## Deploying Flows

Flows are just functions. You choose **how** to expose them over HTTP:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Two Ways to Serve Flows                               │
│                                                                             │
│  Option A: Built-in Flow Server              Option B: Framework Adapter    │
│  (zero config, all flows exposed)            (full control, per-route)      │
│                                                                             │
│  app = create_flows_asgi_app(                from flask import Flask        │
│      registry=ai.registry,                   app = Flask(__name__)          │
│      context_providers=[                                                    │
│          api_key('my-secret'),               @app.route('/joke')            │
│      ],                                      @genkit_flask_handler(ai)      │
│  )                                           @ai.flow()                     │
│                                              async def joke(topic):         │
│  # Exposes ALL flows as:                         return ai.generate(...)    │
│  # POST /tell_joke                                                          │
│  # POST /summarize                                                          │
│  # POST /translate                                                          │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Deploy Anywhere                                    │   │
│  │   Cloud Run · Firebase · Fly.io · AWS · Azure · K8s · Bare Metal    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Context Providers and Authentication

Context providers run **before** your flow, reading the HTTP request
and deciding whether to allow or reject it:

```
  curl POST /tell_joke  -H "Authorization: my-secret-key"
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │                   Flow Server                         │
  │                                                       │
  │  1. Parse request body                                │
  │       │                                               │
  │       ▼                                               │
  │  2. Context providers (one by one):                   │
  │       │                                               │
  │       ├── api_key('my-secret')                        │
  │       │     ├── ✅ Key matches → {auth: {api_key: …}} │
  │       │     ├── ❌ Wrong key   → 403 Permission Denied│
  │       │     └── ❌ No key      → 401 Unauthenticated  │
  │       │                                               │
  │       ├── (more providers, if any — results merge)    │
  │       │                                               │
  │       ▼                                               │
  │  3. Run flow with merged context                      │
  │       tell_joke("banana", context={auth: {api_key:…}})│
  │       │                                               │
  │       ▼                                               │
  │  4. Return response                                   │
  └──────────────────────────────────────────────────────┘
```

## Directory Structure

```
py/
├── packages/genkit/          # Core Genkit framework package
├── plugins/                  # Official plugins
│   ├── amazon-bedrock/       # Amazon Bedrock models + X-Ray telemetry (community)
│   ├── anthropic/            # Claude models
│   ├── azure/                # Azure AI telemetry (community)
│   ├── cloudflare-workers-ai/# Cloudflare Workers AI + OTLP telemetry (community)
│   ├── checks/               # Safety guardrails
│   ├── cohere/               # Cohere models (community)
│   ├── compat-oai/           # OpenAI-compatible APIs
│   ├── deepseek/             # DeepSeek models
│   ├── dev-local-vectorstore/# Local development vector store
│   ├── evaluators/           # RAGAS and custom evaluators
│   ├── firebase/             # Firebase integration + telemetry
│   ├── flask/                # Flask HTTP endpoints
│   ├── google-cloud/         # GCP telemetry (Cloud Trace, Logging)
│   ├── google-genai/         # Gemini, Imagen, Veo, Lyria, TTS
│   ├── huggingface/          # HuggingFace Inference API
│   ├── mcp/                  # Model Context Protocol
│   ├── microsoft-foundry/    # Azure AI Foundry (11,000+ models) (community)
│   ├── mistral/              # Mistral models
│   ├── observability/        # 3rd party telemetry (Sentry, Datadog, etc.)
│   ├── ollama/               # Local Ollama models
│   ├── vertex-ai/            # Model Garden + Vector Search
│   └── xai/                  # Grok models
├── samples/                  # Sample applications
└── testapps/                 # Test applications
```

## Setup Instructions

1. Install `uv` from https://docs.astral.sh/uv/getting-started/installation/

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install required tools using `uv`

```bash
uv tool install httpie
uv tool install mypy
uv tool install ruff
```

3. If you are using VSCode, install the `Ruff` extension from the marketplace to add linter support.

## Quick Start

```python
from genkit import Genkit
from genkit.plugins.google_genai import GoogleGenAI, gemini_2_0_flash

ai = Genkit(
    plugins=[GoogleGenAI()],
    model=gemini_2_0_flash,
)

response = await ai.generate(prompt="Tell me a joke")
print(response.text)
```

## Genkit Concepts

| Concept | What It Is (ELI5) | Where to Learn More |
|---------|-------------------|---------------------|
| **Genkit** | The main "control center" — you create one and use it to define flows, tools, and prompts. Think of it as the brain of your AI app. | `Genkit()` constructor |
| **Flow** | A function that does AI work. It calls models, uses tools, and its execution is fully traced so you can debug it. Flows are the unit you deploy. | `@ai.flow()` |
| **Model** | An AI model (Gemini, Claude, GPT, Llama, etc.) that generates text, images, or other content. You call it with `ai.generate()`. | `ai.generate()` |
| **Tool** | A function that a model can *choose* to call. For example, "look up the weather" or "search the database." You define it, and the model decides when to use it. | `@ai.tool()` |
| **Prompt** | A template for what to say to a model. Can include variables, system instructions, and output schemas. `.prompt` files use Handlebars syntax. | `ai.prompt()`, `.prompt` files |
| **Plugin** | A package that adds capabilities — model providers (Google, Anthropic), telemetry (Cloud Trace), vector stores, etc. You plug them in when creating `Genkit`. | `Genkit(plugins=[...])` |
| **Context Provider** | Middleware that runs *before* a flow is called via HTTP. It reads the request (headers, body) and either provides auth info to the flow or rejects the request. | `api_key()`, `create_flows_asgi_app()` |
| **Flow Server** | A built-in HTTP server that wraps your flows as API endpoints so `curl` (or any client) can call them. It's Genkit's simple way to deploy flows without a web framework. | `create_flows_asgi_app()` |
| **Registry** | The internal directory of all defined flows, tools, models, and prompts. The Dev UI and CLI read it to discover what's available. | `ai.registry` |
| **Action** | The low-level building block behind flows, tools, models, and prompts. Everything you define becomes an "action" in the registry with input/output schemas and tracing. | `genkit.core.action` |
| **Middleware** | Functions that wrap around model calls to add behavior — logging, caching, safety checks, or modifying requests/responses. Runs at the model level, not HTTP level. | `ai.define_model(use=[...])` |
| **Embedder** | A model that turns text into numbers (vectors) for similarity search. Used with vector stores for RAG (Retrieval-Augmented Generation). | `ai.embed()` |
| **Retriever** | A component that searches a vector store and returns relevant documents for a query. Used in RAG pipelines. | `ai.retrieve()` |
| **Indexer** | A component that stores documents into a vector store. The opposite of a retriever — it writes, while the retriever reads. | `ai.index()` |
| **Evaluator** | A tool that scores AI outputs for quality, safety, or correctness. Used for automated testing of AI behavior. | `ai.evaluate()` |
| **Dotprompt** | Genkit's prompt file format (`.prompt` files). Combines Handlebars templates + YAML frontmatter for model config, schemas, and few-shot examples. | `prompts/` directory |
| **Tracing** | Automatic recording of every step in a flow — model calls, tool invocations, timings. Visible in the Dev UI and exportable to Cloud Trace, Jaeger, etc. | OpenTelemetry integration |
| **Reflection API** | An internal HTTP/WebSocket API that lets the Dev UI inspect the registry, trigger actions, and stream results. Only active in development mode. | Dev UI (`genkit start`) |
| **Dynamic Action Provider** | A plugin that can register actions at runtime (not just at startup). Example: MCP servers that expose tools on-demand. | `ai.define_dynamic_action_provider()` |
| **Structured Output** | Asking a model to return data in a specific format (JSON matching a Pydantic model or schema). Genkit validates the output for you. | `ai.generate(output_schema=...)` |

## Plugin Categories

| Category | Plugins | Purpose |
|----------|---------|---------|
| **Model Providers** | google-genai, anthropic, amazon-bedrock, ollama, compat-oai, deepseek, xai, mistral, huggingface, microsoft-foundry, cloudflare-workers-ai, cohere | AI model access |
| **Telemetry** | google-cloud, amazon-bedrock, azure, firebase, cloudflare-workers-ai, observability | Distributed tracing & logging |
| **Vector Stores** | firebase, vertex-ai, dev-local-vectorstore | Embeddings storage & retrieval |
| **Safety** | checks, evaluators | Guardrails & evaluation |
| **Integrations** | flask, mcp | HTTP endpoints, tool protocols |

## Community Plugins

Some plugins are community-maintained and supported on a best-effort basis:

- **amazon-bedrock** - Amazon Bedrock models + AWS X-Ray telemetry
- **azure** - Azure Monitor / Application Insights telemetry
- **cloudflare-workers-ai** - Cloudflare Workers AI models + OTLP telemetry
- **cohere** - Cohere command models + reranking
- **microsoft-foundry** - Azure AI Foundry (11,000+ models)
- **observability** - Third-party backends (Sentry, Honeycomb, Datadog, etc.)

## Sample Catalog

Each sample demonstrates specific Genkit concepts. Use this table to find
examples of any feature:

| Sample | Key Concepts | Description |
|--------|-------------|-------------|
| **Model Providers** | | |
| `provider-google-genai-hello` | Model, Flow | Basic Gemini model usage |
| `provider-anthropic-hello` | Model, Flow | Claude model usage |
| `provider-ollama-hello` | Model, Flow | Local Ollama models |
| `provider-cohere-hello` | Model, Flow | Cohere models |
| `provider-compat-oai-hello` | Model, Flow | OpenAI-compatible APIs |
| `provider-deepseek-hello` | Model, Flow | DeepSeek models |
| `provider-xai-hello` | Model, Flow | Grok models |
| `provider-mistral-hello` | Model, Flow | Mistral models |
| `provider-huggingface-hello` | Model, Flow | HuggingFace Inference API |
| `provider-amazon-bedrock-hello` | Model, Flow, Telemetry | AWS Bedrock + X-Ray tracing |
| `provider-cloudflare-workers-ai-hello` | Model, Flow, Telemetry | Cloudflare Workers AI |
| `provider-microsoft-foundry-hello` | Model, Flow | Azure AI Foundry |
| **Google Cloud** | | |
| `provider-google-genai-vertexai-hello` | Model, Flow | Vertex AI models |
| `provider-google-genai-vertexai-image` | Model, Flow | Imagen image generation |
| `provider-google-genai-media-models-demo` | Model, Flow | TTS, STT, Veo, Lyria |
| `provider-google-genai-code-execution` | Model, Tool | Code execution sandbox |
| `provider-google-genai-context-caching` | Model, Flow | Context caching for long prompts |
| `provider-vertex-ai-model-garden` | Model, Flow | Model Garden access |
| `provider-vertex-ai-rerank-eval` | Retriever, Evaluator | Reranking + evaluation |
| `provider-vertex-ai-vector-search-*` | Embedder, Indexer, Retriever | Vector search |
| `provider-firestore-retriever` | Retriever | Firestore document retrieval |
| `provider-observability-hello` | Telemetry | Multi-backend tracing |
| **Framework Patterns** | | |
| `framework-prompt-demo` | Prompt, Dotprompt | Advanced prompt templates |
| `framework-format-demo` | Structured Output | JSON/enum output formatting |
| `framework-context-demo` | Context Provider, Flow | Auth context in flows |
| `framework-middleware-demo` | Middleware | Model-level request/response hooks |
| `framework-evaluator-demo` | Evaluator | Custom evaluation metrics |
| `framework-restaurant-demo` | Flow, Tool, Prompt | Multi-step agent with tools |
| `framework-dynamic-tools-demo` | Tool, Dynamic Action Provider | Runtime tool registration |
| `framework-tool-interrupts` | Tool | Human-in-the-loop tool approval |
| `framework-realtime-tracing-demo` | Tracing | Live trace streaming |
| `dev-local-vectorstore-hello` | Embedder, Retriever, Indexer | Local dev vector store |
| **Deployment** | | |
| `web-short-n-long` | Flow Server, Context Provider | Built-in flows server + `api_key()` auth |
| `web-flask-hello` | Flow, Plugin (Flask) | Flask framework integration |
| `web-multi-server` | Flow Server, Reflection API | Multiple Genkit instances |

See the [samples/README.md](samples/README.md) for running instructions.

## Running Tests

Run all unit tests:

```bash
uv run pytest .
```

Run tests for a specific plugin:

```bash
uv run pytest plugins/amazon-bedrock/
```

## Development

See [GEMINI.md](GEMINI.md) for detailed development guidelines, including:
- Code quality and linting requirements
- Type checking configuration
- Testing conventions
- Documentation standards

## License

Apache 2.0
