# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Cloudflare Workers AI plugin for Genkit.

This plugin provides access to Cloudflare Workers AI models and OTLP telemetry
export through the Genkit framework. Cloudflare Workers AI runs AI models at
the edge, close to users, providing low-latency inference with global availability.

Features:
    - **Text Generation**: Chat and instruction-following models (Llama, Qwen, Mistral)
    - **Embeddings**: Text embeddings for semantic search (BGE models)
    - **OTLP Telemetry**: Export traces to any OTLP-compatible backend

Documentation Links:
    - Cloudflare Workers AI: https://developers.cloudflare.com/workers-ai/
    - Model Catalog: https://developers.cloudflare.com/workers-ai/models/
    - REST API: https://developers.cloudflare.com/workers-ai/get-started/rest-api/
    - Workers Observability: https://developers.cloudflare.com/workers/observability/

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Workers AI          │ AI models that run on Cloudflare's edge servers.   │
    │                     │ Like having AI helpers in data centers worldwide.  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Account ID          │ Your unique Cloudflare identifier. Like your       │
    │                     │ customer number at a store.                        │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ API Token           │ Your secret key to access the API. Like a password │
    │                     │ that proves you're allowed to use the service.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ OTLP Telemetry      │ Export traces via OpenTelemetry Protocol to any    │
    │                     │ compatible backend (Grafana, Honeycomb, etc.).     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Model ID            │ The specific AI model to use. Format is like       │
    │                     │ @cf/meta/llama-3.1-8b-instruct.                    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Edge Computing      │ Running code close to users geographically. Like   │
    │                     │ having mini-servers in every city instead of one.  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Streaming           │ Getting the response word-by-word as it's          │
    │                     │ generated, instead of waiting for all of it.       │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    User Request
         │
         ▼
    ┌───────────────────┐     ┌───────────────────────────────────┐
    │  Genkit Flow      │────▶│  CF Workers AI Plugin             │
    │  (your app)       │     │  (cloudflare-workers-ai)                  │
    └───────────────────┘     └─────────────┬─────────────────────┘
                                            │
                              ┌─────────────┴─────────────┐
                              │                           │
                              ▼                           ▼
               ┌────────────────────────┐    ┌────────────────────────┐
               │  Cloudflare Workers AI │    │  OTLP Telemetry        │
               │  (models + embedders)  │    │  (traces export)       │
               └────────────────────────┘    └────────────────────────┘

Plugin Architecture::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    CF Workers AI Plugin (cloudflare-workers-ai)                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  __init__.py - Plugin Entry Point                                        │
    │  ├── CloudflareWorkersAI - Plugin class                                          │
    │  ├── cloudflare_name() - Helper to create qualified model names          │
    │  ├── cloudflare_model() - Convenience alias for cloudflare_name          │
    │  ├── add_cloudflare_telemetry() - Enable OTLP telemetry export           │
    │  └── Pre-defined model references (llama_3_1_8b, bge_base_en, etc.)      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  telemetry/tracing.py - OTLP Telemetry                                   │
    │  ├── add_cloudflare_telemetry() - Configure trace export                 │
    │  ├── CfTelemetry - Telemetry manager class                               │
    │  └── Bearer token authentication support                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  models/model.py - Model Implementation                                  │
    │  ├── CfModel - Handles text generation                                   │
    │  ├── generate() - Non-streaming generation                               │
    │  └── _generate_streaming() - SSE-based streaming                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  embedders/embedder.py - Embedder Implementation                         │
    │  └── CfEmbedder - Handles text embeddings (BGE models)                   │
    └─────────────────────────────────────────────────────────────────────────┘

Example Usage::

    from genkit import Genkit
    from genkit.plugins.cloudflare_workers_ai import CloudflareWorkersAI, cloudflare_model, add_cloudflare_telemetry

    # Enable OTLP telemetry export (optional)
    add_cloudflare_telemetry()

    ai = Genkit(
        plugins=[CloudflareWorkersAI()],
        model=cloudflare_model('@cf/meta/llama-3.1-8b-instruct'),
    )

    # Simple generation
    response = await ai.generate(prompt='Tell me a joke.')
    print(response.text)

    # With streaming
    async for chunk in ai.generate_stream(prompt='Write a story.'):
        print(chunk.text, end='')

Authentication:
    The plugin requires two environment variables:

    - ``CLOUDFLARE_ACCOUNT_ID``: Your Cloudflare account ID
    - ``CLOUDFLARE_API_TOKEN``: API token with Workers AI permissions

    For telemetry:

    - ``CF_OTLP_ENDPOINT``: OTLP traces endpoint URL
    - ``CF_API_TOKEN``: API token for Bearer authentication (optional)

    These can also be passed directly to plugin/function constructors.

Implementation Notes & Edge Cases
---------------------------------

**Media URL Fetching (Cloudflare-Specific Requirement)**

Unlike other AI providers (Anthropic, OpenAI, Google GenAI) that accept media URLs
directly in their APIs, Cloudflare Workers AI **only accepts base64 data URIs** for
multimodal models like Llama 4 Scout. From the Cloudflare docs:

    "url string - image uri with data (e.g. data:image/jpeg;base64,/9j/...).
    HTTP URL will not be accepted"

The plugin automatically fetches images from URLs and converts them to base64::

    # User provides:
    MediaPart(media=Media(url='https://example.com/image.jpg'))

    # Plugin sends to Cloudflare:
    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,...'}}

**User-Agent Header for External URLs**

Some servers (notably Wikipedia/Wikimedia) block requests without a proper
``User-Agent`` header, returning HTTP 403 Forbidden. The plugin includes::

    'User-Agent': 'Genkit/1.0 (https://github.com/firebase/genkit; genkit@google.com)'

**Tool Calling Format**

For models supporting function calling, Cloudflare expects tool calls as JSON
serialized in the assistant message's ``content`` field::

    # Assistant requesting a tool call:
    {'role': 'assistant', 'content': '{"name": "get_weather", "arguments": {...}}'}

    # Tool response:
    {'role': 'tool', 'name': 'get_weather', 'content': '{"temperature": "72°F"}'}

This differs from OpenAI's ``tool_calls`` array format.

**Tool Input Schema Wrapping**

Cloudflare requires tool parameters to be object schemas. Primitive schemas
(e.g., ``{'type': 'string'}``) are automatically wrapped::

    {'type': 'object', 'properties': {'input': {...}}, 'required': ['input']}

**Server-Sent Events (SSE) Streaming**

Streaming uses SSE format. Each event contains a JSON payload with the
generated text chunk::

    async for line in response.aiter_lines():
        if line.startswith('data: '):
            chunk = json.loads(line[6:])
            yield chunk['response']

**Model ID Format**

Model IDs use hierarchical format: ``@cf/<provider>/<model>``.
Example: ``@cf/meta/llama-3.1-8b-instruct``.

Trademark Notice:
    This is a community plugin and is not officially supported by Cloudflare.
    "Cloudflare", "Workers", "CF", and related marks are trademarks of
    Cloudflare, Inc.
"""

from genkit.plugins.cloudflare_workers_ai.models.model_info import (
    SUPPORTED_CF_MODELS as SUPPORTED_CF_MODELS,
    SUPPORTED_EMBEDDING_MODELS as SUPPORTED_EMBEDDING_MODELS,
)
from genkit.plugins.cloudflare_workers_ai.plugin import (
    CLOUDFLARE_WORKERS_AI_PLUGIN_NAME as CLOUDFLARE_WORKERS_AI_PLUGIN_NAME,
    CloudflareWorkersAI as CloudflareWorkersAI,
    cloudflare_model as cloudflare_model,
    cloudflare_name as cloudflare_name,
)
from genkit.plugins.cloudflare_workers_ai.telemetry import (
    add_cloudflare_telemetry as add_cloudflare_telemetry,
)
from genkit.plugins.cloudflare_workers_ai.typing import CloudflareConfig as CloudflareConfig

# Pre-defined model references for convenience
# Text Generation Models
llama_3_3_70b = cloudflare_name('@cf/meta/llama-3.3-70b-instruct-fp8-fast')
llama_3_1_8b = cloudflare_name('@cf/meta/llama-3.1-8b-instruct')
llama_3_1_8b_fast = cloudflare_name('@cf/meta/llama-3.1-8b-instruct-fast')
llama_4_scout = cloudflare_name('@cf/meta/llama-4-scout-17b-16e-instruct')
mistral_7b = cloudflare_name('@hf/mistral/mistral-7b-instruct-v0.2')
qwen_14b = cloudflare_name('@cf/qwen/qwen1.5-14b-chat-awq')

# Embedding Models
bge_base_en = cloudflare_name('@cf/baai/bge-base-en-v1.5')
bge_large_en = cloudflare_name('@cf/baai/bge-large-en-v1.5')
bge_small_en = cloudflare_name('@cf/baai/bge-small-en-v1.5')

__all__ = [
    # Telemetry
    'add_cloudflare_telemetry',
    # Plugin name
    'CLOUDFLARE_WORKERS_AI_PLUGIN_NAME',
    # Model registries
    'SUPPORTED_CF_MODELS',
    'SUPPORTED_EMBEDDING_MODELS',
    # Plugin class
    'CloudflareWorkersAI',
    # Config
    'CloudflareConfig',
    # Helpers
    'cloudflare_model',
    'cloudflare_name',
    # Pre-defined model references
    'bge_base_en',
    'bge_large_en',
    'bge_small_en',
    'llama_3_1_8b',
    'llama_3_1_8b_fast',
    'llama_3_3_70b',
    'llama_4_scout',
    'mistral_7b',
    'qwen_14b',
]
