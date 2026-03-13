# Genkit Samples

This directory contains sample applications demonstrating various Genkit features and integrations.

## Sample Categories

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              GENKIT SAMPLE APPLICATIONS                                  │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   MODEL PROVIDERS (provider-*)              FRAMEWORK FEATURES (framework-*)             │
│   ────────────────────────────              ────────────────────────────────             │
│   ┌──────────────────────────────────┐      ┌──────────────────────────────────┐        │
│   │ provider-google-genai-hello      │      │ framework-context-demo           │        │
│   │ provider-google-genai-vertexai-  │      │ framework-dynamic-tools-demo     │        │
│   │   hello                          │      │ framework-evaluator-demo         │        │
│   │ provider-anthropic-hello         │      │ framework-format-demo            │        │
│   │ provider-amazon-bedrock-hello    │      │ framework-middleware-demo        │        │
│   │ provider-microsoft-foundry-hello │      │ framework-prompt-demo            │        │
│   │ provider-ollama-hello            │      │ framework-realtime-tracing-demo  │        │
│   │ provider-compat-oai-hello        │      │ framework-restaurant-demo        │        │
│   │ provider-deepseek-hello          │      │ framework-tool-interrupts        │        │
│   │ provider-xai-hello               │      └──────────────────────────────────┘        │
│   │ provider-cloudflare-workers-ai-  │                                                   │
│   │   hello                          │      WEB FRAMEWORKS (web-*)                      │
│   │ provider-mistral-hello           │      ──────────────────────────                  │
│   │ provider-huggingface-hello       │      ┌──────────────────────────────────┐        │
│   │ provider-cohere-hello            │      │ web-endpoints-hello              │        │
│   │ provider-checks-hello            │      │ web-fastapi-bugbot               │        │
│   │ provider-observability-hello     │      │ web-flask-hello                  │        │
│   │ provider-vertex-ai-model-garden  │      │ web-multi-server                 │        │
│   │ provider-vertex-ai-rerank-eval   │      │ web-short-n-long                 │        │
│   │ provider-firestore-retriever     │      └──────────────────────────────────┘        │
│   │ provider-google-genai-code-      │                                                   │
│   │   execution                      │                                                   │
│   │ provider-google-genai-context-   │      OTHER                                       │
│   │   caching                        │      ─────                                       │
│   │ provider-google-genai-vertexai-  │      ┌──────────────────────────────────┐        │
│   │   image                          │      │ dev-local-vectorstore-hello      │        │
│   │ provider-google-genai-media-     │      └──────────────────────────────────┘        │
│   │   models-demo                    │                                                   │
│   │ provider-vertex-ai-vector-       │                                                   │
│   │   search-bigquery                │                                                   │
│   │ provider-vertex-ai-vector-       │                                                   │
│   │   search-firestore               │                                                   │
│   └──────────────────────────────────┘                                                   │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

Each sample follows the same pattern:

```bash
# Navigate to any sample
cd py/samples/<sample-name>

# Run the sample (starts DevUI at http://localhost:4000)
./run.sh
```

## Sample List

### Model Provider Samples (`provider-*`)

| Sample | Plugin | Description |
|--------|--------|-------------|
| **provider-google-genai-hello** | google-genai | Gemini models with Google AI |
| **provider-google-genai-vertexai-hello** | vertex-ai | Gemini models with Vertex AI |
| **provider-anthropic-hello** | anthropic | Claude models directly |
| **provider-amazon-bedrock-hello** | amazon-bedrock | Claude, Llama, Nova via Bedrock |
| **provider-microsoft-foundry-hello** | microsoft-foundry | Azure AI Foundry models + Application Insights |
| **provider-ollama-hello** | ollama | Local models with Ollama |
| **provider-compat-oai-hello** | compat-oai | OpenAI-compatible APIs |
| **provider-deepseek-hello** | deepseek | DeepSeek V3 and R1 |
| **provider-xai-hello** | xai | Grok models |
| **provider-cloudflare-workers-ai-hello** | cloudflare-workers-ai | Cloudflare Workers AI + OTLP telemetry |
| **provider-mistral-hello** | mistral | Mistral models |
| **provider-huggingface-hello** | huggingface | HuggingFace Inference API models |
| **provider-cohere-hello** | cohere | Cohere models, embeddings, reranking |
| **provider-vertex-ai-model-garden** | vertex-ai | Third-party models via Vertex AI Model Garden |
| **provider-observability-hello** | observability | Sentry, Honeycomb, Datadog, etc. |
| **provider-checks-hello** | checks | Google Checks AI content safety guardrails |

### Provider Feature Samples (`provider-*`)

| Sample | Plugin | Description |
|--------|--------|-------------|
| **provider-google-genai-code-execution** | google-genai | Gemini server-side code execution |
| **provider-google-genai-context-caching** | google-genai | Context caching for long prompts |
| **provider-google-genai-vertexai-image** | vertex-ai | Image generation with Vertex AI Imagen |
| **provider-google-genai-media-models-demo** | google-genai | Media generation: TTS, Veo, Lyria, Imagen, Gemini Image, editing |
| **provider-vertex-ai-rerank-eval** | vertex-ai | Vertex AI rerankers and evaluators |
| **provider-vertex-ai-vector-search-bigquery** | vertex-ai | BigQuery with Vertex AI vectors |
| **provider-vertex-ai-vector-search-firestore** | vertex-ai | Firestore with Vertex AI vectors |
| **provider-firestore-retriever** | firebase | Firestore vector search retriever |

### Framework Feature Samples (`framework-*`)

| Sample | Features | Description |
|--------|----------|-------------|
| **framework-context-demo** | Context | Context propagation through flows, tools, and generate |
| **framework-dynamic-tools-demo** | Tools | Dynamic tool registration at runtime |
| **framework-evaluator-demo** | Evaluation | Custom evaluators and RAGAS |
| **framework-format-demo** | Formats | Output formatting and schemas |
| **framework-middleware-demo** | Middleware | Custom retry and logging middleware |
| **framework-prompt-demo** | Prompts | Dotprompt: Picoschema, roles, media, tools, variants, helpers, partials, `{{#each}}` |
| **framework-realtime-tracing-demo** | Telemetry | Real-time tracing visualization |
| **framework-restaurant-demo** | Tools, RAG | Restaurant menu ordering with tools |
| **framework-tool-interrupts** | Tools | Human-in-the-loop tool approval |

### Web Framework Samples (`web-*`)

| Sample | Features | Description |
|--------|----------|-------------|
| **web-endpoints-hello** | FastAPI, Litestar, Quart, gRPC | REST + gRPC endpoints with multi-cloud deploy, auto-telemetry |
| **web-fastapi-bugbot** | FastAPI | AI code reviewer with streaming SSE and Swagger UI |
| **web-flask-hello** | Flask | Flask HTTP endpoints with Genkit |
| **web-multi-server** | Litestar, Starlette | Multiple Genkit servers |
| **web-short-n-long** | ASGI | ASGI deployment with long-running flows |

### Other Samples

| Sample | Features | Description |
|--------|----------|-------------|
| **dev-local-vectorstore-hello** | Vector Store | Local development vector store |

## Feature Coverage Matrix

The table below tracks which capabilities each model provider sample exercises.
This is a living document — update it as new flows are added to samples.

> **Last audited**: 2026-02-12

| Sample | Basic | Stream | Tools | Struct | Vision | Embed | Code | Reasoning | TTS | Cache | PDF |
|--------|:-----:|:------:|:-----:|:------:|:------:|:-----:|:----:|:---------:|:---:|:-----:|:---:|
| **provider-amazon-bedrock-hello** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| **provider-anthropic-hello** | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | — | ✅ | ✅ |
| **provider-cloudflare-workers-ai-hello** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — |
| **provider-cohere-hello** | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | — | — | — | — |
| **provider-compat-oai-hello** | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | — | — |
| **provider-deepseek-hello** | ✅ | ✅ | ✅ | ✅ | — | — | ✅ | ✅ | — | — | — |
| **provider-google-genai-hello** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| **provider-google-genai-vertexai-hello** | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | — | — | — | — |
| **provider-huggingface-hello** | ✅ | ✅ | ✅ | ✅ | — | — | ✅ | — | — | — | — |
| **provider-microsoft-foundry-hello** | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | — | — | — |
| **provider-mistral-hello** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| **provider-ollama-hello** | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | — | — | — |
| **provider-xai-hello** | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | — | — | — |

**Legend**: ✅ = exercised in sample, — = plugin does not support this feature

All plugin-supported features are now exercised in their respective samples.
The matrix is complete — no remaining gaps (❌) exist.

## Environment Setup

Most samples require environment variables for API keys. Configure these before running samples.

### Model Provider API Keys

| Variable | Sample | Required | Description | Get Credentials |
|----------|--------|----------|-------------|-----------------|
| `GEMINI_API_KEY` | provider-google-genai-hello | Yes | Google AI Studio API key | [Google AI Studio](https://aistudio.google.com/apikey) |
| `ANTHROPIC_API_KEY` | provider-anthropic-hello | Yes | Anthropic API key | [Anthropic Console](https://console.anthropic.com/) |
| `AWS_REGION` | provider-amazon-bedrock-hello | Yes | AWS region (e.g., `us-east-1`) | [AWS Bedrock Regions](https://docs.aws.amazon.com/general/latest/gr/bedrock.html) |
| `AWS_ACCESS_KEY_ID` | provider-amazon-bedrock-hello | Yes* | AWS access key | [AWS IAM](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html) |
| `AWS_SECRET_ACCESS_KEY` | provider-amazon-bedrock-hello | Yes* | AWS secret key | [AWS IAM](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html) |
| `AZURE_AI_FOUNDRY_ENDPOINT` | provider-microsoft-foundry-hello | Yes | Azure AI Foundry endpoint | [Azure AI Foundry](https://ai.azure.com/) |
| `AZURE_AI_FOUNDRY_API_KEY` | provider-microsoft-foundry-hello | Yes* | Azure AI Foundry API key | [Azure AI Foundry](https://ai.azure.com/) |
| `OPENAI_API_KEY` | provider-compat-oai-hello | Yes | OpenAI API key | [OpenAI Platform](https://platform.openai.com/api-keys) |
| `DEEPSEEK_API_KEY` | provider-deepseek-hello | Yes | DeepSeek API key | [DeepSeek Platform](https://platform.deepseek.com/) |
| `XAI_API_KEY` | provider-xai-hello | Yes | xAI API key | [xAI Console](https://console.x.ai/) |
| `CLOUDFLARE_ACCOUNT_ID` | provider-cloudflare-workers-ai-hello | Yes | Cloudflare account ID | [Cloudflare Dashboard](https://dash.cloudflare.com/) |
| `CLOUDFLARE_API_TOKEN` | provider-cloudflare-workers-ai-hello | Yes | Cloudflare API token | [Cloudflare API Tokens](https://developers.cloudflare.com/fundamentals/api/get-started/create-token/) |
| `COHERE_API_KEY` | provider-cohere-hello | Yes | Cohere API key | [Cohere Dashboard](https://dashboard.cohere.com/api-keys) |

*Can use IAM roles, managed identity, or other credential providers instead.

### Telemetry Configuration

| Variable | Sample | Required | Description | Get Credentials |
|----------|--------|----------|-------------|-----------------|
| `GOOGLE_CLOUD_PROJECT` | framework-realtime-tracing-demo | Yes | GCP project ID | [GCP Console](https://console.cloud.google.com/) |
| `GOOGLE_APPLICATION_CREDENTIALS` | framework-realtime-tracing-demo | Yes* | Service account JSON path | [GCP IAM](https://cloud.google.com/docs/authentication/application-default-credentials) |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | provider-microsoft-foundry-hello | Yes | Azure App Insights connection string | [Azure Portal](https://learn.microsoft.com/azure/azure-monitor/app/create-workspace-resource) |
| `CF_OTLP_ENDPOINT` | provider-cloudflare-workers-ai-hello | No* | OTLP endpoint URL | Your OTLP backend |
| `CF_API_TOKEN` | provider-cloudflare-workers-ai-hello | No* | Bearer token for OTLP auth | Your OTLP backend |

*Only required if using OTLP telemetry export.

### Observability Plugin (Third-Party Backends)

The `provider-observability-hello` sample supports multiple backends. Configure based on your choice:

#### Sentry

| Variable | Required | Description | Get Credentials |
|----------|----------|-------------|-----------------|
| `SENTRY_DSN` | Yes | Sentry DSN (Data Source Name) | [Sentry Settings](https://docs.sentry.io/concepts/otlp/) > Projects > Client Keys |
| `SENTRY_ENVIRONMENT` | No | Environment name | - |

#### Honeycomb

| Variable | Required | Description | Get Credentials |
|----------|----------|-------------|-----------------|
| `HONEYCOMB_API_KEY` | Yes | Honeycomb API key | [Honeycomb Settings](https://docs.honeycomb.io/configure/environments/manage-api-keys/) |
| `HONEYCOMB_DATASET` | No | Dataset name (Classic only) | - |
| `HONEYCOMB_API_ENDPOINT` | No | API endpoint (US default) | - |

Endpoints: `https://api.honeycomb.io` (US), `https://api.eu1.honeycomb.io` (EU)

#### Datadog

| Variable | Required | Description | Get Credentials |
|----------|----------|-------------|-----------------|
| `DD_API_KEY` | Yes | Datadog API key | [Datadog Settings](https://app.datadoghq.com/organization-settings/api-keys) |
| `DD_SITE` | No | Site (default: `datadoghq.com`) | - |

Sites: `datadoghq.com`, `datadoghq.eu`, `us3.datadoghq.com`, `us5.datadoghq.com`, `ap1.datadoghq.com`

#### Grafana Cloud

| Variable | Required | Description | Get Credentials |
|----------|----------|-------------|-----------------|
| `GRAFANA_OTLP_ENDPOINT` | Yes | OTLP endpoint URL | [Grafana Cloud](https://grafana.com/) > My Account > Stack > OpenTelemetry |
| `GRAFANA_USER_ID` | Yes | Instance ID (numeric) | Same as above |
| `GRAFANA_API_KEY` | Yes | API key/token | Same as above |

#### Axiom

| Variable | Required | Description | Get Credentials |
|----------|----------|-------------|-----------------|
| `AXIOM_TOKEN` | Yes | Axiom API token | [Axiom Settings](https://app.axiom.co/settings/tokens) |
| `AXIOM_DATASET` | No | Dataset name (default: `genkit`) | - |

### Quick Setup Examples

```bash
# Google AI (provider-google-genai-hello)
export GEMINI_API_KEY="AIza..."

# Anthropic (provider-anthropic-hello)
export ANTHROPIC_API_KEY="sk-ant-..."

# AWS Bedrock (provider-amazon-bedrock-hello)
export AWS_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."

# Azure AI Foundry (provider-microsoft-foundry-hello)
export AZURE_AI_FOUNDRY_ENDPOINT="https://your-resource.services.ai.azure.com/"
export AZURE_AI_FOUNDRY_API_KEY="..."

# Azure Telemetry (provider-microsoft-foundry-hello)
export APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=...;IngestionEndpoint=..."

# Cloudflare Workers AI (provider-cloudflare-workers-ai-hello)
export CLOUDFLARE_ACCOUNT_ID="abc123..."
export CLOUDFLARE_API_TOKEN="..."

# Sentry (provider-observability-hello)
export SENTRY_DSN="https://abc123@o123456.ingest.us.sentry.io/4507654321"

# Honeycomb (provider-observability-hello)
export HONEYCOMB_API_KEY="..."

# Datadog (provider-observability-hello)
export DD_API_KEY="..."

# Grafana Cloud (provider-observability-hello)
export GRAFANA_OTLP_ENDPOINT="https://otlp-gateway-prod-us-central-0.grafana.net/otlp"
export GRAFANA_USER_ID="123456"
export GRAFANA_API_KEY="glc_..."

# Axiom (provider-observability-hello)
export AXIOM_TOKEN="xaat-..."

# Cohere (provider-cohere-hello)
export COHERE_API_KEY="..."
```

Each sample's README.md contains specific environment requirements.

## Cross-Language Sample Parity

> **Last audited**: 2026-02-12

The table below compares sample coverage across Python and JavaScript SDKs.
Python currently has **35 samples** covering more provider diversity
than JavaScript's **32 testapps + 9 top-level samples**.

### Parity Status

| JS Testapp / Sample | Python Equivalent | Status |
|---------------------|-------------------|:------:|
| `basic-gemini` | `provider-google-genai-hello` | ✅ |
| `anthropic` | `provider-anthropic-hello` | ✅ |
| `ollama` | `provider-ollama-hello` | ✅ |
| `compat-oai` | `provider-compat-oai-hello` | ✅ |
| `prompt-file` | `framework-prompt-demo` | ✅ |
| `context-caching` | `provider-google-genai-context-caching` | ✅ |
| `custom-evaluators`, `evals` | `framework-evaluator-demo` | ✅ |
| `format-tester` | `framework-format-demo` | ✅ |
| `express` | `web-flask-hello` | ✅ |
| `vertexai-vector-search-bigquery` | `provider-vertex-ai-vector-search-bigquery` | ✅ |
| `vertexai-vector-search-firestore` | `provider-vertex-ai-vector-search-firestore` | ✅ |
| `vertexai-modelgarden` | `provider-vertex-ai-model-garden` | ✅ |
| `vertexai-reranker` | `provider-vertex-ai-rerank-eval` | ✅ |
| `menu`, `docs-menu-rag` | `framework-restaurant-demo` | ✅ |
| `multimodal` | `provider-google-genai-media-models-demo` | ✅ |
| `mcp` | — | ❌ |
| `multiagents-demo` | — | ❌ |
| `rag` | — | ❌ |
| `dev-ui-gallery` | — | 🟡 |
| `durable-streaming` | — | 🟡 |
| `firebase-functions-sample1` | — | 🟡 |
| `next`, `esm` | — | 🟡 |
| `model-armor` | — | 🟡 |
| `model-tester` | — | 🟡 |
| `js-chatbot`, `js-coffee-shop`, etc. | — | 🟡 |

**Legend**: ✅ = parity achieved, ❌ = gap (should port), 🟡 = low priority or JS-specific

### Python-Only Samples (No JS Equivalent)

Python has significantly broader provider and framework coverage:

| Python Sample | Category |
|---------------|----------|
| `provider-amazon-bedrock-hello` | Model provider |
| `provider-microsoft-foundry-hello` | Model provider |
| `provider-deepseek-hello` | Model provider |
| `provider-xai-hello` | Model provider |
| `provider-cloudflare-workers-ai-hello` | Model provider |
| `provider-mistral-hello` | Model provider |
| `provider-huggingface-hello` | Model provider |
| `provider-observability-hello` | Telemetry (5 backends) |
| `provider-firestore-retriever` | Vector store |
| `provider-cohere-hello` | Model provider |
| `provider-google-genai-vertexai-image` | Image generation |
| `provider-checks-hello` | Safety (Checks AI) |
| `web-fastapi-bugbot` | Web (FastAPI + SSE) |
| `framework-middleware-demo` | Framework |
| `framework-realtime-tracing-demo` | Framework |
| `framework-context-demo` | Framework |
| `framework-dynamic-tools-demo` | Framework |
| `web-multi-server` | Web (Litestar + Starlette) |
| `web-short-n-long` | Web (ASGI long-running) |
| `genkit-chat` (testapp) | Full-stack testapp |

### Gaps to Close (Prioritized)

| Priority | Sample to Create | JS Reference | Why |
|:--------:|------------------|--------------|-----|
| 🔴 High | `framework-multiagent-demo` | `multiagents-demo` | Multi-agent orchestration with handoffs is a flagship feature |
| 🟡 Medium | `framework-mcp-demo` | `mcp` | Python has the MCP plugin but no sample demonstrating it |
| 🟡 Medium | `framework-rag-demo` | `rag` | End-to-end RAG pipeline (index → embed → retrieve → generate) |

## Creating New Samples

When creating new samples, follow these guidelines:

1. **Directory structure**:
   ```
   samples/<name>/
   ├── pyproject.toml
   ├── README.md
   ├── run.sh
   └── src/
       └── main.py
   ```

2. **Entry point**: Always use `ai.run_main(main())`:
   ```python
   import asyncio

   async def main():
       # Keep the server alive to handle requests.
       await asyncio.Event().wait()

   if __name__ == '__main__':
       ai.run_main(main())
   ```

3. **run.sh**: Use the standard pattern with watchdog:
   ```bash
   genkit start -- \
     uv tool run --from watchdog watchmedo auto-restart \
       -d src \
       -d ../../packages \
       -d ../../plugins \
       -p '*.py;*.prompt;*.json' \
       -R \
       -- uv run src/main.py "$@"
   ```

4. **Flow inputs**: Use Pydantic BaseModel with defaults for DevUI:
   ```python
   class HelloInput(BaseModel):
       name: str = Field(default='World', description='Name to greet')
   ```

See `py/GEMINI.md` for complete guidelines.
