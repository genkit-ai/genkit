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

"""Hugging Face plugin for Genkit.

This plugin provides integration with Hugging Face's Inference API and
Inference Providers, giving access to 1,000,000+ models through a unified
interface.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Hugging Face        │ The "GitHub for AI models". Hosts millions of      │
    │                     │ models you can use through their API.              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Inference API       │ HF's free API to run models. Like a free trial    │
    │                     │ for AI models with rate limits.                    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Inference Providers │ 17+ partner services (Cerebras, Groq, Together)    │
    │                     │ accessible through one HF API. Pick the fastest!   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ InferenceClient     │ The Python SDK class that talks to HF's API.       │
    │                     │ Like a universal remote for all HF models.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ HF_TOKEN            │ Your API key for Hugging Face. Get one free at     │
    │                     │ huggingface.co/settings/tokens.                    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Model ID            │ The model's address on HF, like "meta-llama/       │
    │                     │ Llama-3.3-70B-Instruct". Owner/model-name format.  │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  HOW HUGGING FACE PROCESSES YOUR REQUEST                │
    │                                                                         │
    │    Your Code                                                            │
    │    ai.generate(model='huggingface/meta-llama/Llama-3.3-70B-Instruct')   │
    │         │                                                               │
    │         │  (1) Request goes to HuggingFace plugin                       │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  HuggingFace    │   Adds HF_TOKEN, formats request                 │
    │    │  Plugin         │   (OpenAI-compatible format)                     │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) HTTPS to api-inference.huggingface.co                │
    │             ▼                                                           │
    │    ════════════════════════════════════════════════════                 │
    │             │  Internet                                                 │
    │             ▼                                                           │
    │    ┌─────────────────┐     ┌─────────────────┐                          │
    │    │  HF Inference   │ OR  │ Inference       │  (Cerebras, Groq, etc.)  │
    │    │  API            │     │ Provider        │                          │
    │    └────────┬────────┘     └────────┬────────┘                          │
    │             │                       │                                   │
    │             └───────────┬───────────┘                                   │
    │                         │                                               │
    │                         │  (3) Response with generated text             │
    │                         ▼                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Your App       │   response.text = "Here's my answer..."          │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Architecture Overview::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        Hugging Face Plugin                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Plugin Entry Point (__init__.py)                                       │
    │  ├── HuggingFace - Plugin class                                         │
    │  ├── huggingface_name() - Helper to create namespaced model names       │
    │  └── HUGGINGFACE_PLUGIN_NAME - Plugin identifier                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  plugin.py - Plugin Implementation                                      │
    │  ├── HuggingFace class (registers models)                               │
    │  └── Configuration and API token handling                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  models.py - Model Implementation                                       │
    │  ├── HuggingFaceModel (generation logic)                                │
    │  ├── Message conversion (Genkit <-> HF)                                 │
    │  └── Streaming support                                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  model_info.py - Model Registry                                         │
    │  ├── POPULAR_MODELS (Llama, Mistral, Qwen, etc.)                        │
    │  └── Model capabilities and metadata                                    │
    └─────────────────────────────────────────────────────────────────────────┘

Popular Models:
    - meta-llama/Llama-3.3-70B-Instruct: Meta's latest Llama model
    - mistralai/Mistral-Small-24B-Instruct-2501: Mistral's efficient model
    - Qwen/Qwen2.5-72B-Instruct: Alibaba's powerful multilingual model
    - deepseek-ai/DeepSeek-R1: DeepSeek's reasoning model
    - google/gemma-2-27b-it: Google's open Gemma model

Example:
    ```python
    from genkit import Genkit
    from genkit.plugins.huggingface import HuggingFace

    # Uses HF_TOKEN env var or pass token explicitly
    ai = Genkit(
        plugins=[HuggingFace()],
        model='huggingface/meta-llama/Llama-3.3-70B-Instruct',
    )

    response = await ai.generate(prompt='Hello, Hugging Face!')
    print(response.text)
    ```

Caveats:
    - Requires HF_TOKEN environment variable or token parameter
    - Model names are prefixed with 'huggingface/' (e.g., 'huggingface/meta-llama/...')
    - Free tier has rate limits; consider HF Pro for higher limits
    - Some models may require accepting terms on huggingface.co first

See Also:
    - Hugging Face Hub: https://huggingface.co/
    - Inference API docs: https://huggingface.co/docs/api-inference/
    - Inference Providers: https://huggingface.co/docs/inference-providers/
    - Genkit documentation: https://genkit.dev/
"""

from .model_info import POPULAR_HUGGINGFACE_MODELS
from .models import HUGGINGFACE_PLUGIN_NAME, huggingface_name
from .plugin import HuggingFace

__all__ = [
    'HUGGINGFACE_PLUGIN_NAME',
    'POPULAR_HUGGINGFACE_MODELS',
    'HuggingFace',
    'huggingface_name',
]
