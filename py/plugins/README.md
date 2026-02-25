# Genkit Plugins

This directory contains all official Genkit plugins for Python.

## Plugin Architecture

All plugins inherit from `genkit.core.plugin.Plugin` and implement three
async methods. The registry calls them lazily — `init()` runs only on
first use, not at registration time.

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                  Plugin (Abstract Base Class)                       │
  │                  genkit.core.plugin.Plugin                          │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  name: str              Plugin namespace (e.g., 'googleai')        │
  │                                                                     │
  │  async init()           One-time setup; returns actions to          │
  │    → list[Action]       pre-register. Called lazily on first use.   │
  │                                                                     │
  │  async resolve(kind, name)   Resolve a single action by kind/name. │
  │    → Action | None           Called on each action lookup.          │
  │                                                                     │
  │  async list_actions()   Advertise available actions (for Dev UI).   │
  │    → list[ActionMetadata]   Must be fast — no heavy initialization.│
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
                                  │
          ┌───────────┬───────────┼───────────┬───────────┐
          ▼           ▼           ▼           ▼           ▼
     GoogleAI   Anthropic     Ollama     Cohere       ... etc.
```

### Plugin Lifecycle

```
  ai = Genkit(plugins=[GoogleAI()])      ← Phase 1: REGISTER
       │
       ▼
  registry.register_plugin(GoogleAI())   (stored, not initialized)
       │
       ⋮  (later, on first use)
       │
  await ai.generate(model="googleai/gemini-2.0-flash", ...)
       │
       ▼
  registry._ensure_plugin_initialized()  ← Phase 2: LAZY INIT
       │
       ▼
  actions = await plugin.init()          (called exactly once)
  for action in actions:
      registry.register_action_instance(action)
       │
       ▼
  await plugin.resolve(MODEL, name)      ← Phase 3: RESOLVE
       │
       ▼
  Action cached in registry              (subsequent lookups skip init)
```

### Action Resolution Algorithm

```
  ai.generate(model="googleai/gemini-2.0-flash")
       │
       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  1. CACHE HIT?                                                   │
  │     Name already in registry._entries?                           │
  │     YES → return cached Action  │  NO → continue                │
  ├──────────────────────────────────────────────────────────────────┤
  │  2. NAMESPACED? (name contains "/")                              │
  │     YES → find plugin by prefix, init it, resolve via plugin    │
  │     NO  → try ALL plugins; 1 match = ok, 2+ = ValueError        │
  ├──────────────────────────────────────────────────────────────────┤
  │  3. DYNAMIC ACTION PROVIDERS (fallback)                          │
  │     Try MCP servers / dynamic providers → found? register it    │
  │     Not found → return None                                      │
  └──────────────────────────────────────────────────────────────────┘
```

## Plugin Categories

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           GENKIT PLUGIN ECOSYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   MODEL PROVIDERS                    TELEMETRY                                  │
│   ───────────────                    ─────────                                  │
│   ┌─────────────────────────┐        ┌─────────────────────────┐               │
│   │ google-genai            │        │ google-cloud            │               │
│   │ • Gemini, Imagen, Veo   │        │ • Cloud Trace           │               │
│   │ • Lyria, TTS            │        │ • Cloud Logging         │               │
│   └─────────────────────────┘        └─────────────────────────┘               │
│   ┌─────────────────────────┐        ┌─────────────────────────┐               │
│   │ anthropic               │        │ firebase                │               │
│   │ • Claude 3.5/4          │        │ • Firebase Telemetry    │               │
│   └─────────────────────────┘        └─────────────────────────┘               │
│   ┌─────────────────────────┐                                                  │
│   │ amazon-bedrock  🌐      │        INTEGRATIONS                              │
│   │ • Claude, Llama, Nova   │        ────────────                              │
│   │ • Titan, Mistral        │        ┌─────────────────────────┐               │
│   │ • X-Ray telemetry       │        │ flask                   │               │
│   └─────────────────────────┘        │ • HTTP endpoints        │               │
│   ┌─────────────────────────┐        └─────────────────────────┘               │
│   │ microsoft-foundry               │        ┌─────────────────────────┐               │
│   │ • GPT-4o, Claude, Llama │        │ mcp                     │               │
│   │ • 11,000+ models        │        │ • Model Context Protocol│               │
│   └─────────────────────────┘        └─────────────────────────┘               │
│   ┌─────────────────────────┐                                                  │
│   │ vertex-ai               │        VECTOR STORES                             │
│   │ • Model Garden          │        ─────────────                             │
│   │ • Vector Search         │        ┌─────────────────────────┐               │
│   └─────────────────────────┘        │ firebase                │               │
│   ┌─────────────────────────┐        │ • Firestore vectors     │               │
│   │ ollama                  │        └─────────────────────────┘               │
│   │ • Local models          │        ┌─────────────────────────┐               │
│   └─────────────────────────┘        │ vertex-ai               │               │
│   ┌─────────────────────────┐        │ • Vector Search         │               │
│   │ compat-oai              │        └─────────────────────────┘               │
│   │ • OpenAI API compatible │        ┌─────────────────────────┐               │
│   └─────────────────────────┘        │ dev-local-vectorstore   │               │
│   ┌─────────────────────────┐        │ • Local development     │               │
│   │ deepseek                │        └─────────────────────────┘               │
│   │ • DeepSeek V3, R1       │                                                  │
│   └─────────────────────────┘        SAFETY & EVALUATION                       │
│   ┌─────────────────────────┐        ───────────────────                       │
│   │ xai                     │        ┌─────────────────────────┐               │
│   │ • Grok models           │        │ checks                  │               │
│   └─────────────────────────┘        │ • Content moderation    │               │
│   ┌─────────────────────────┐        │ • Safety guardrails     │               │
│   │ mistral                 │        ┌─────────────────────────┐               │
│   │ • Mistral Large, Small  │        │ evaluators              │               │
│   │ • Codestral, Pixtral    │        │ • RAGAS metrics         │               │
│   └─────────────────────────┘        │ • Custom evaluators     │               │
│   ┌─────────────────────────┐        └─────────────────────────┘               │
│   │ huggingface             │                                                  │
│   │ • 1M+ open models       │                                                  │
│   │ • Inference providers   │                                                  │
│   └─────────────────────────┘                                                  │
│   ┌─────────────────────────┐                                                  │
│   │ cohere             🌐   │                                                  │
│   │ • Command R/R+          │                                                  │
│   │ • Embed, Rerank         │                                                  │
│   └─────────────────────────┘                                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## When to Use What

### Model Provider Selection

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    WHICH MODEL PROVIDER SHOULD I USE?                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   "I want the best multimodal AI"                                               │
│       → google-genai (Gemini 2.0)                                               │
│                                                                                 │
│   "I need Claude models"                                                        │
│       → anthropic (direct) OR amazon-bedrock OR microsoft-foundry                          │
│                                                                                 │
│   "I'm on AWS and want managed models"                                          │
│       → amazon-bedrock (Claude, Llama, Nova, Titan)                                │
│                                                                                 │
│   "I'm on Azure and want managed models"                                        │
│       → microsoft-foundry (GPT-4o, Claude, Llama, 11,000+ models)                       │
│                                                                                 │
│   "I'm on GCP and want third-party models"                                      │
│       → vertex-ai (Model Garden - Claude, Llama, etc.)                          │
│                                                                                 │
│   "I want to run models locally"                                                │
│       → ollama (Llama, Mistral, Phi, etc.)                                      │
│                                                                                 │
│   "I need OpenAI GPT models"                                                    │
│       → compat-oai (direct OpenAI) OR microsoft-foundry (via Azure)                     │
│                                                                                 │
│   "I want to use any OpenAI-compatible API"                                     │
│       → compat-oai (works with OpenRouter, Together, etc.)                      │
│                                                                                 │
│   "I need DeepSeek reasoning models"                                            │
│       → deepseek (V3, R1 reasoning)                                             │
│                                                                                 │
│   "I want Grok models"                                                          │
│       → xai                                                                     │
│                                                                                 │
│   "I want Cloudflare Workers AI models"                                         │
│       → cloudflare-workers-ai (Llama, Mistral, Qwen + OTLP telemetry)                    │
│                                                                                 │
│   "I want Mistral AI models (French AI)"                                        │
│       → mistral (mistral-large, codestral, pixtral)                             │
│                                                                                 │
│   "I want access to 1M+ open source models"                                     │
│       → huggingface (Inference API + 17 providers)                              │
│                                                                                 │
│   "I want Cohere command models + reranking"                                    │
│       → cohere (Command R/R+, Embed, Rerank)                                    │
│                                                                                 │
│   "I want one API for 500+ models from 60+ providers"                           │
│       → compat-oai with OpenRouter (works TODAY)                                │
│         OR openrouter plugin (COMING SOON - adds model discovery)               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Telemetry Selection

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        WHICH TELEMETRY PLUGIN SHOULD I USE?                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   NATIVE PLATFORM BACKENDS              THIRD-PARTY BACKENDS                    │
│   ────────────────────────              ────────────────────                    │
│                                                                                 │
│   ┌─────────┐  ┌─────────┐             ┌───────────────────┐                   │
│   │ amazon- │  │ google- │             │   observability   │               │
│   │ bedrock │  │ cloud   │             │   • Sentry        │                   │
│   │ • SigV4 │  │ • ADC   │             │   • Honeycomb     │                   │
│   │ • X-Ray │  │ • Trace │             │   • Datadog       │                   │
│   │         │  │ • Logs  │             │   • Grafana       │                   │
│   └────┬────┘  └────┬────┘             │   • Axiom         │                   │
│        │            │                   └─────────┬─────────┘                   │
│        ▼            ▼                             │                             │
│   ┌─────────┐  ┌─────────┐                        ▼                             │
│   │ X-Ray   │  │ Cloud   │             ┌───────────────────┐                   │
│   │ Console │  │ Trace   │             │  Any OTLP Backend │                   │
│   └─────────┘  └─────────┘             └───────────────────┘                   │
│                                                                                 │
│   ┌───────────────────┐  ┌─────────────┐                                        │
│   │ microsoft-foundry │  │cloudflare-workers-ai│                                │
│   │ • Models + AppIns │  │ • OTLP      │  • Models + Telemetry                   │
│   │ • Azure Telemetry │  │ • Token     │  • Single plugin                        │
│   └─────────┬─────────┘  └──────┬──────┘                                         │
│             │                   │    CAN'T BE REPLICATED      CAN BE REPLICATED  │
│             ▼                   ▼    WITH GENERIC OTLP        WITH GENERIC OTLP  │
│   ┌───────────────────┐  ┌─────────┐                                             │
│   │  App Insights     │  │  OTLP   │                                             │
│   └───────────────────┘  │ Backend │                                             │
│                          └─────────┘                                             │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   "I'm on AWS and want X-Ray"           → amazon-bedrock plugin                 │
│   "I'm on GCP and want Cloud Trace"     → google-cloud plugin                  │
│   "I'm on Azure and want App Insights"  → microsoft-foundry plugin              │
│   "I'm using Firebase"                  → firebase plugin (auto telemetry)     │
│                                                                                 │
│   "I want Sentry/Honeycomb/Datadog"     → observability plugin                 │
│   "I'm multi-cloud"                     → observability plugin                 │
│   "I want generic OTLP export"          → cloudflare-workers-ai plugin (combined)      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Vector Store Selection

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      WHICH VECTOR STORE SHOULD I USE?                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   "I'm just developing/testing locally"                                         │
│       → dev-local-vectorstore                                                   │
│                                                                                 │
│   "I need production vector search on Firebase"                                 │
│       → firebase (Firestore vector search)                                      │
│                                                                                 │
│   "I need enterprise-scale vector search on GCP"                                │
│       → vertex-ai (Vertex AI Vector Search + Firestore/BigQuery)                │
│                                                                                 │
│   "I want to use a third-party vector DB"                                       │
│       → Implement custom retriever (Pinecone, Weaviate, Chroma, etc.)           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Safety & Evaluation Selection

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  WHICH SAFETY/EVALUATION PLUGIN SHOULD I USE?                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   "I need content moderation / safety guardrails"                               │
│       → checks (Google Checks AI Safety)                                        │
│         • Real-time content classification                                      │
│         • Block harmful input/output                                            │
│         • 8 policy types (harassment, hate speech, etc.)                        │
│                                                                                 │
│   "I need to evaluate RAG quality"                                              │
│       → evaluators (RAGAS metrics)                                              │
│         • Faithfulness, relevancy, answer accuracy                              │
│         • Custom evaluation metrics                                             │
│                                                                                 │
│   "I need both safety AND quality evaluation"                                   │
│       → Use both: checks for guardrails, evaluators for quality                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Plugin List

### Model Providers

| Plugin | Models | Best For |
|--------|--------|----------|
| **google-genai** | Gemini, Imagen, Veo, Lyria | Multimodal AI, Google ecosystem |
| **anthropic** | Claude 3.5, Claude 4 | Direct Claude access |
| **amazon-bedrock** 🌐 | Claude, Llama, Nova, Titan | AWS managed models (community) |
| **microsoft-foundry** 🌐 | GPT-4o, Claude, Llama, 11,000+ | Azure AI, enterprise (community) |
| **vertex-ai** | Model Garden (Claude, Llama) | GCP third-party models |
| **ollama** | Llama, Mistral, Phi, etc. | Local/private deployment |
| **compat-oai** | Any OpenAI-compatible | OpenAI, OpenRouter, etc. |
| **deepseek** | DeepSeek V3, R1 | Reasoning, cost-effective |
| **xai** | Grok | X/Twitter integration |
| **cloudflare-workers-ai** 🌐 | Llama, Mistral, Qwen, Gemma | Cloudflare Workers AI + OTLP telemetry (community) |
| **mistral** | Mistral Large, Small, Codestral, Pixtral | French AI, efficient models, code generation |
| **huggingface** | 1M+ models via HF Hub | Open source models, inference providers |
| **cohere** 🌐 | Command R/R+, Embed, Rerank | Cohere models, embeddings, reranking (community) |

### Planned Model Providers

| Plugin | Models | Status | Notes |
|--------|--------|--------|-------|
| **openrouter** | 500+ models, 60+ providers | 🔜 Planned | Unified gateway (OpenAI, Anthropic, Google, etc.) |

> **Note:** OpenRouter is already usable today via `compat-oai` since it's OpenAI-compatible.
> A dedicated plugin would add model discovery, provider routing, and usage analytics.

### Telemetry

| Plugin | Backend | Features |
|--------|---------|----------|
| **google-cloud** | Cloud Trace, Logging | GCP native, log correlation |
| **amazon-bedrock** 🌐 | X-Ray | AWS native, SigV4 auth, built into model plugin (community) |
| **microsoft-foundry** 🌐 | Application Insights | Azure Monitor, trace correlation, built into model plugin (community) |
| **cloudflare-workers-ai** 🌐 | Any OTLP endpoint | Generic OTLP, Bearer auth, combined with models (community) |
| **observability** 🌐 | Sentry, Honeycomb, Datadog, Grafana, Axiom | 3rd party presets (community) |
| **firebase** | Firebase console | Auto-telemetry for Firebase apps |

### Integrations

| Plugin | Purpose |
|--------|---------|
| **fastapi** | Serve flows via FastAPI endpoints with lifespan management |
| **flask** | Serve flows via Flask HTTP endpoints |
| **mcp** | Model Context Protocol for tool integration |

### Safety & Evaluation

| Plugin | Purpose | Features |
|--------|---------|----------|
| **checks** | Content moderation & safety | Google Checks AI guardrails, 8 policy types |
| **evaluators** | Quality evaluation | RAGAS metrics, custom evaluators |

### Vector Stores

| Plugin | Backend | Scale |
|--------|---------|-------|
| **dev-local-vectorstore** | Local JSON | Development only |
| **firebase** | Firestore | Production, serverless |
| **vertex-ai** | Vertex AI Vector Search | Enterprise scale |

## Environment Variables Reference

All environment variables used by Genkit plugins. Configure these before running your application.

### Model Provider Environment Variables

| Variable | Plugin | Required | Description | Documentation |
|----------|--------|----------|-------------|---------------|
| `GEMINI_API_KEY` | google-genai | Yes | Google AI Studio API key | [Get API Key](https://aistudio.google.com/apikey) |
| `ANTHROPIC_API_KEY` | anthropic | Yes | Anthropic API key | [Anthropic Console](https://console.anthropic.com/) |
| `AWS_REGION` | amazon-bedrock | Yes | AWS region (e.g., `us-east-1`) | [AWS Regions](https://docs.aws.amazon.com/general/latest/gr/bedrock.html) |
| `AWS_ACCESS_KEY_ID` | amazon-bedrock | Yes* | AWS access key | [AWS Credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html) |
| `AWS_SECRET_ACCESS_KEY` | amazon-bedrock | Yes* | AWS secret key | [AWS Credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html) |
| `AZURE_AI_FOUNDRY_ENDPOINT` | microsoft-foundry | Yes | Azure AI Foundry endpoint URL | [Azure AI Foundry](https://ai.azure.com/) |
| `AZURE_AI_FOUNDRY_API_KEY` | microsoft-foundry | Yes* | Azure AI Foundry API key | [Azure AI Foundry](https://ai.azure.com/) |
| `OPENAI_API_KEY` | compat-oai | Yes | OpenAI API key | [OpenAI API Keys](https://platform.openai.com/api-keys) |
| `OPENAI_ORG_ID` | compat-oai | No | OpenAI organization ID | [OpenAI Settings](https://platform.openai.com/account/organization) |
| `DEEPSEEK_API_KEY` | deepseek | Yes | DeepSeek API key | [DeepSeek Platform](https://platform.deepseek.com/) |
| `XAI_API_KEY` | xai | Yes | xAI API key | [xAI Console](https://console.x.ai/) |
| `CLOUDFLARE_ACCOUNT_ID` | cloudflare-workers-ai | Yes | Cloudflare account ID | [Cloudflare Dashboard](https://dash.cloudflare.com/) |
| `CLOUDFLARE_API_TOKEN` | cloudflare-workers-ai | Yes | Cloudflare API token | [Cloudflare API Tokens](https://developers.cloudflare.com/fundamentals/api/get-started/create-token/) |
| `MISTRAL_API_KEY` | mistral | Yes | Mistral AI API key | [Mistral Console](https://console.mistral.ai/) |
| `HF_TOKEN` | huggingface | Yes | Hugging Face API token | [HF Tokens](https://huggingface.co/settings/tokens) |
| `COHERE_API_KEY` | cohere | Yes | Cohere API key | [Cohere Dashboard](https://dashboard.cohere.com/api-keys) |
| `CO_API_KEY` | cohere | Yes* | Cohere API key (alternative) | [Cohere Dashboard](https://dashboard.cohere.com/api-keys) |

*Can use IAM roles, managed identity, or other credential providers instead. For cohere, either `COHERE_API_KEY` or `CO_API_KEY` is accepted.

### Telemetry Environment Variables

#### Google Cloud Plugin

| Variable | Required | Description | Documentation |
|----------|----------|-------------|---------------|
| `GOOGLE_CLOUD_PROJECT` | Yes | GCP project ID | [GCP Projects](https://cloud.google.com/resource-manager/docs/creating-managing-projects) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes* | Path to service account JSON | [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) |
| `GCLOUD_PROJECT` | No | Alternative to `GOOGLE_CLOUD_PROJECT` | - |
| `FIREBASE_PROJECT_ID` | No | Firebase project ID (auto-detected) | - |

*Not required when running on GCP with default credentials.

#### Amazon Bedrock Plugin (X-Ray Telemetry)

| Variable | Required | Description | Documentation |
|----------|----------|-------------|---------------|
| `AWS_REGION` | Yes | AWS region for X-Ray | [AWS X-Ray](https://docs.aws.amazon.com/xray/latest/devguide/xray-sdk-python.html) |
| `AWS_ACCESS_KEY_ID` | Yes* | AWS access key | [AWS Credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html) |
| `AWS_SECRET_ACCESS_KEY` | Yes* | AWS secret key | [AWS Credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html) |
| `AWS_DEFAULT_REGION` | No | Fallback region | - |

*Can use IAM roles instead.

#### Microsoft Foundry Plugin (Azure Telemetry)

| Variable | Required | Description | Documentation |
|----------|----------|-------------|---------------|
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Yes | Application Insights connection string | [Azure Monitor OpenTelemetry](https://learn.microsoft.com/azure/azure-monitor/app/opentelemetry-configuration) |
| `AZURE_TENANT_ID` | No | Azure AD tenant ID | [Azure Identity](https://learn.microsoft.com/azure/developer/python/sdk/authentication-overview) |
| `AZURE_CLIENT_ID` | No | Azure AD client ID | - |
| `AZURE_CLIENT_SECRET` | No | Azure AD client secret | - |

#### Cloudflare Workers AI (cloudflare-workers-ai) Plugin

| Variable | Required | Description | Documentation |
|----------|----------|-------------|---------------|
| `CLOUDFLARE_ACCOUNT_ID` | Yes | Cloudflare account ID | [Cloudflare Dashboard](https://dash.cloudflare.com/) |
| `CLOUDFLARE_API_TOKEN` | Yes | API token for Workers AI | [Cloudflare API Tokens](https://developers.cloudflare.com/fundamentals/api/get-started/create-token/) |
| `CF_OTLP_ENDPOINT` | No* | OTLP endpoint URL (for telemetry) | [Cloudflare Workers Observability](https://developers.cloudflare.com/workers/observability/) |
| `CF_API_TOKEN` | No | API token for telemetry (Bearer auth) | [Cloudflare API Tokens](https://developers.cloudflare.com/fundamentals/api/get-started/create-token/) |

*Required only if using OTLP telemetry export.

#### Observability Plugin (Third-Party Backends)

##### Sentry

| Variable | Required | Description | Documentation |
|----------|----------|-------------|---------------|
| `SENTRY_DSN` | Yes | Sentry DSN (Data Source Name) | [Sentry DSN](https://docs.sentry.io/concepts/otlp/) |
| `SENTRY_ENVIRONMENT` | No | Environment name (production, staging) | [Sentry Configuration](https://docs.sentry.io/platforms/python/configuration/options/) |
| `SENTRY_RELEASE` | No | Release version identifier | - |

##### Honeycomb

| Variable | Required | Description | Documentation |
|----------|----------|-------------|---------------|
| `HONEYCOMB_API_KEY` | Yes | Honeycomb API key | [Honeycomb API Keys](https://docs.honeycomb.io/configure/environments/manage-api-keys/) |
| `HONEYCOMB_DATASET` | No | Dataset name (Classic only) | [Honeycomb Datasets](https://docs.honeycomb.io/send-data/python/opentelemetry-sdk/) |
| `HONEYCOMB_API_ENDPOINT` | No | API endpoint (default: US) | [Honeycomb Endpoints](https://docs.honeycomb.io/configure/environments/manage-api-keys/) |

Honeycomb endpoints:
- US (default): `https://api.honeycomb.io`
- EU: `https://api.eu1.honeycomb.io`

##### Datadog

| Variable | Required | Description | Documentation |
|----------|----------|-------------|---------------|
| `DD_API_KEY` | Yes | Datadog API key | [Datadog API Keys](https://docs.datadoghq.com/account_management/api-app-keys/) |
| `DD_SITE` | No | Datadog site (default: `datadoghq.com`) | [Datadog Sites](https://docs.datadoghq.com/getting_started/site/) |
| `DD_APP_KEY` | No | Datadog application key | - |

Datadog sites: `datadoghq.com`, `datadoghq.eu`, `us3.datadoghq.com`, `us5.datadoghq.com`, `ap1.datadoghq.com`

##### Grafana Cloud

| Variable | Required | Description | Documentation |
|----------|----------|-------------|---------------|
| `GRAFANA_OTLP_ENDPOINT` | Yes | Grafana Cloud OTLP endpoint | [Grafana Cloud OTLP](https://grafana.com/docs/grafana-cloud/monitor-applications/application-observability/setup/collector/opentelemetry-collector/) |
| `GRAFANA_USER_ID` | Yes | Grafana Cloud instance ID (numeric) | [Grafana Cloud Portal](https://grafana.com/docs/grafana-cloud/account-management/authentication-and-permissions/) |
| `GRAFANA_API_KEY` | Yes | Grafana Cloud API key | [Grafana Cloud API Keys](https://grafana.com/docs/grafana-cloud/account-management/authentication-and-permissions/create-api-key/) |

Find your credentials: My Account > [Stack] > OpenTelemetry > Configure

##### Axiom

| Variable | Required | Description | Documentation |
|----------|----------|-------------|---------------|
| `AXIOM_TOKEN` | Yes | Axiom API token | [Axiom API Tokens](https://axiom.co/docs/reference/tokens) |
| `AXIOM_DATASET` | No | Dataset name (default: `genkit`) | [Axiom Datasets](https://axiom.co/docs/reference/datasets) |
| `AXIOM_ORG_ID` | No | Organization ID | - |

#### Generic OpenTelemetry (Standard Variables)

| Variable | Required | Description | Documentation |
|----------|----------|-------------|---------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Yes | OTLP collector endpoint | [OTel SDK Environment Variables](https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/) |
| `OTEL_EXPORTER_OTLP_HEADERS` | No | Headers for authentication | - |
| `OTEL_SERVICE_NAME` | No | Service name for traces | - |

### Safety & Evaluation Environment Variables

| Variable | Plugin | Required | Description | Documentation |
|----------|--------|----------|-------------|---------------|
| `GOOGLE_CLOUD_PROJECT` | checks | Yes | GCP project with Checks API enabled | [Google Checks](https://developers.google.com/checks) |
| `GOOGLE_APPLICATION_CREDENTIALS` | checks | Yes* | Service account credentials | - |

## Installation

Each plugin is a separate package. Install only what you need:

```bash
# Model providers
pip install genkit-google-genai-plugin
pip install genkit-anthropic-plugin
pip install genkit-amazon-bedrock-plugin  # Also includes X-Ray telemetry
pip install genkit-microsoft-foundry-plugin

# Telemetry
pip install genkit-google-cloud-plugin

# Safety & Evaluation
pip install genkit-checks-plugin
pip install genkit-evaluators-plugin

# Integrations
pip install genkit-plugin-fastapi
pip install genkit-flask-plugin
pip install genkit-mcp-plugin
pip install genkit-cohere-plugin
```

## Quick Start

```python
from genkit.ai import Genkit
from genkit.plugins.google_genai import GoogleAI

# Initialize with your chosen plugin
ai = Genkit(
    plugins=[GoogleAI()],
    model="googleai/gemini-2.0-flash",
)

@ai.flow()
async def hello(name: str) -> str:
    response = await ai.generate(prompt=f"Say hello to {name}")
    return response.text
```

## Plugin Dependency Graph

Shows how plugins relate to each other and the core `genkit` package. Most
plugins are independent leaf nodes; only a few have inter-plugin dependencies.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PLUGIN DEPENDENCY GRAPH                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                           ┌──────────┐                                           │
│                           │  genkit   │ (core SDK)                               │
│                           └─────┬────┘                                           │
│                                 │                                                │
│              ┌──────────────────┼──────────────────┐                             │
│              │                  │                   │                             │
│              ▼                  ▼                   ▼                             │
│   ┌──────────────────┐ ┌──────────────┐ ┌───────────────────┐                   │
│   │   compat-oai     │ │ google-genai │ │ All other plugins │                   │
│   │ (OpenAI compat)  │ │              │ │ (independent)     │                   │
│   └────────┬─────────┘ └──────┬───────┘ └───────────────────┘                   │
│            │                  │                                                   │
│     ┌──────┴──────┐          │                                                   │
│     │             │          │                                                   │
│     ▼             ▼          ▼                                                   │
│ ┌─────────┐ ┌──────────┐ ┌──────────┐                                           │
│ │deepseek │ │vertex-ai │ │  flask   │                                           │
│ │(extends)│ │(Model    │ │(uses     │                                           │
│ │         │ │ Garden)  │ │ google-  │                                           │
│ │         │ │          │ │ genai)   │                                           │
│ └─────────┘ └──────────┘ └──────────┘                                           │
│                                                                                  │
│   INDEPENDENT PLUGINS (no inter-plugin dependencies):                            │
│   ─────────────────────────────────────────────────                               │
│   google-genai, anthropic, amazon-bedrock, microsoft-foundry,                    │
│   ollama, xai, mistral, huggingface, cloudflare-workers-ai,                      │
│   cohere, google-cloud, firebase, observability, mcp, fastapi,                   │
│   evaluators, dev-local-vectorstore, checks                                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key relationships:**
- **`compat-oai`** provides the shared OpenAI-compatible model layer (chat, image, TTS, STT)
- **`deepseek`** extends `compat-oai` with reasoning model detection and param validation
- **`vertex-ai`** (Model Garden) uses `compat-oai` for third-party model support
- **`flask`** has a dev dependency on `google-genai` for its sample

## Cross-Language Plugin Coverage

> **Last audited**: 2026-02-12

The table below compares plugin availability across Python and JavaScript SDKs.
Python currently has **22 plugins** vs JavaScript's **17 plugins**, with broader
model provider diversity.

### Model Providers

| Plugin | Python | JavaScript | Notes |
|--------|:------:|:----------:|-------|
| Google GenAI (Gemini, Imagen, Veo, Lyria) | ✅ | ✅ | |
| Vertex AI (Model Garden, Vector Search) | ✅ | ✅ | |
| Anthropic (Claude) | ✅ | ✅ | |
| Ollama | ✅ | ✅ | |
| OpenAI-Compatible (compat-oai) | ✅ | ✅ | |
| Amazon Bedrock | ✅ | — | Python-only; community 🌐 |
| Microsoft Foundry (Azure AI) | ✅ | — | Python-only; community 🌐 |
| DeepSeek | ✅ | — | Python-only |
| xAI (Grok) | ✅ | — | Python-only |
| Cloudflare Workers AI | ✅ | — | Python-only; community 🌐 |
| Mistral | ✅ | — | Python-only |
| HuggingFace | ✅ | — | Python-only |
| Cohere | ✅ | — | Python-only; community 🌐 |

### Telemetry & Observability

| Plugin | Python | JavaScript | Notes |
|--------|:------:|:----------:|-------|
| Google Cloud (Trace, Logging) | ✅ | ✅ | |
| Firebase | ✅ | ✅ | |
| Observability (Sentry, Honeycomb, Datadog, Grafana, Axiom) | ✅ | — | Python-only; community 🌐 |

### Integrations

| Plugin | Python | JavaScript | Notes |
|--------|:------:|:----------:|-------|
| MCP (Model Context Protocol) | ✅ | ✅ | |
| FastAPI | ✅ | — | Python-only |
| Flask | ✅ | — | Python-only |
| Express | — | ✅ | JS-only |
| Next.js | — | ✅ | JS-only |

### Vector Stores

| Plugin | Python | JavaScript | Notes |
|--------|:------:|:----------:|-------|
| Dev Local Vectorstore | ✅ | ✅ | |
| Firebase (Firestore vectors) | ✅ | ✅ | |
| Vertex AI Vector Search | ✅ | ✅ | |
| Chroma | — | ✅ | JS-only |
| Pinecone | — | ✅ | JS-only |
| Cloud SQL PG | — | ✅ | JS-only |

### Safety & Evaluation

| Plugin | Python | JavaScript | Notes |
|--------|:------:|:----------:|-------|
| Evaluators (RAGAS) | ✅ | ✅ | |
| Checks (Content Safety) | ✅ | ✅ | |

### Other

| Plugin | Python | JavaScript | Notes |
|--------|:------:|:----------:|-------|
| LangChain | — | ✅ | JS-only |

## Further Reading

- [Contributing Guide](../engdoc/contributing/)
