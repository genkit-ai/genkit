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

"""Mistral AI plugin for Genkit.

This plugin provides integration with Mistral AI's models for the
Genkit framework. Mistral AI offers powerful, efficient language models
spanning text, vision, audio, code, and reasoning capabilities.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Mistral AI          │ French AI company known for efficient, powerful    │
    │                     │ open-weight models. Great balance of speed/quality.│
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Mistral Large 3     │ Most capable model. Multimodal (text + vision).   │
    │                     │ Best for complex reasoning and coding tasks.      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Mistral Medium 3.1  │ Frontier-class multimodal model. Premium tier.    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Mistral Small 3.2   │ Fast and efficient with vision support.           │
    │                     │ Great for everyday tasks.                         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Magistral           │ Reasoning models. Think step-by-step before       │
    │                     │ answering. Best for math and logic problems.      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Codestral/Devstral  │ Specialized coding models. Optimized for code     │
    │                     │ generation, completion, and SWE agent tasks.      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Voxtral             │ Audio models. Can understand spoken language       │
    │                     │ and transcribe audio files.                       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Pixtral             │ Legacy vision models. Superseded by Large 3+.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Function Calling    │ Model can call your tools/functions. Like giving   │
    │                     │ the AI a toolbox to help answer questions.         │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  HOW MISTRAL PROCESSES YOUR REQUEST                     │
    │                                                                         │
    │    Your Code                                                            │
    │    ai.generate(prompt="Explain quantum computing...")                   │
    │         │                                                               │
    │         │  (1) Request goes to Mistral plugin                           │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Mistral        │   Adds API key, formats request                  │
    │    │  Plugin         │   (Mistral SDK format)                           │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) HTTPS to api.mistral.ai                              │
    │             ▼                                                           │
    │    ════════════════════════════════════════════════════                 │
    │             │  Internet                                                 │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Mistral API    │   Model processes your prompt                    │
    │    │  (Large/Small)  │   Supports tools, vision, streaming              │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (3) Response with generated text                         │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Your App       │   response.text = "Quantum computing is..."      │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Architecture Overview::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        Mistral AI Plugin                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Plugin Entry Point (__init__.py)                                       │
    │  ├── Mistral - Plugin class                                             │
    │  ├── mistral_name() - Helper to create namespaced model names           │
    │  └── DEFAULT_MISTRAL_API_URL - API endpoint constant                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  plugin.py - Plugin Implementation                                      │
    │  ├── Mistral class (registers models + embedders)                       │
    │  └── Configuration and API key handling                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  models.py - Model Implementation                                       │
    │  ├── MistralModel (generation logic)                                    │
    │  ├── Message conversion (Genkit <-> Mistral)                            │
    │  ├── Multimodal support (images, audio)                                 │
    │  └── Streaming support                                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  embeddings.py - Embedder Implementation                                │
    │  ├── MistralEmbedder (embedding logic)                                  │
    │  └── Supports mistral-embed and codestral-embed                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  model_info.py - Model Registry                                         │
    │  ├── SUPPORTED_MODELS (30+ models across 8 categories)                  │
    │  └── Model capabilities and metadata                                    │
    └─────────────────────────────────────────────────────────────────────────┘

Supported Models:
    Generalist (with vision):
        - mistral-large-latest: Mistral Large 3 — most capable multimodal
        - mistral-medium-latest: Mistral Medium 3.1 — frontier-class
        - mistral-small-latest: Mistral Small 3.2 — fast and efficient

    Compact (Ministral 3, with vision):
        - ministral-14b-latest: 14B params, best-in-class compact
        - ministral-8b-latest: 8B params, efficient
        - ministral-3b-latest: 3B params, edge deployment

    Reasoning (Magistral):
        - magistral-medium-latest: Frontier reasoning
        - magistral-small-latest: Efficient reasoning

    Code (Codestral/Devstral):
        - codestral-latest: Code completion
        - devstral-large-latest: Devstral 2 (code agent)
        - devstral-small-latest: Devstral Small 2

    Audio (Voxtral):
        - voxtral-small-latest: Audio chat
        - voxtral-mini-latest: Mini audio chat

    Vision (Legacy Pixtral):
        - pixtral-large-latest: Pixtral Large
        - pixtral-12b-latest: Pixtral 12B

Supported Embedders:
    - mistral-embed: 1024-dimensional text embeddings for RAG and search
    - codestral-embed-2505: Code-specific embeddings

Example:
    ```python
    from genkit import Genkit
    from genkit.plugins.mistral import Mistral

    # Uses MISTRAL_API_KEY env var or pass api_key explicitly
    ai = Genkit(plugins=[Mistral()], model='mistral/mistral-large-latest')

    response = await ai.generate(prompt='Hello, Mistral!')
    print(response.text)
    ```

Caveats:
    - Requires MISTRAL_API_KEY environment variable or api_key parameter
    - Model names are prefixed with 'mistral/' (e.g., 'mistral/mistral-large-latest')
    - Uses official Mistral Python SDK (mistralai)

See Also:
    - Mistral AI documentation: https://docs.mistral.ai/
    - Mistral models: https://docs.mistral.ai/getting-started/models/
    - Mistral API Reference: https://docs.mistral.ai/api/
    - Genkit documentation: https://genkit.dev/
"""

from .embeddings import SUPPORTED_EMBEDDING_MODELS, MistralEmbedConfig
from .model_info import SUPPORTED_MISTRAL_MODELS
from .models import MISTRAL_PLUGIN_NAME, mistral_name
from .plugin import Mistral

# Official Mistral API endpoint
DEFAULT_MISTRAL_API_URL = 'https://api.mistral.ai'

__all__ = [
    'DEFAULT_MISTRAL_API_URL',
    'MISTRAL_PLUGIN_NAME',
    'Mistral',
    'MistralEmbedConfig',
    'SUPPORTED_EMBEDDING_MODELS',
    'SUPPORTED_MISTRAL_MODELS',
    'mistral_name',
]
