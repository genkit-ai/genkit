# Copyright 2025 Google LLC
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


"""Google GenAI plugin for Genkit.

This plugin provides integration with Google's generative AI models through
either Google AI (Gemini API) or Google Cloud Vertex AI. Any model ID served by
the Gemini API (including newly released models) is resolved lazily on demand
when ``ai.generate()`` is called, without needing pre-registration or SDK
updates.

Specifying Models (``ModelRef`` vs. Prefixed String):
    Both ``Genkit(model=...)`` and ``ai.generate(model=...)`` accept either form:

    - **Typed ``ModelRef`` (recommended)**: ``GoogleAI.gemini_model('gemini-flash-latest')``
      (or ``VertexAI.gemini_model('gemini-4-argon')``) — pass the bare model ID;
      returns a ``ModelRef[GeminiConfig]`` with the ``'googleai/'`` (or
      ``'vertexai/'``) namespace stamped automatically and static config
      type-checking.
    - **Prefixed string shorthand**: ``'googleai/gemini-flash-latest'`` (or
      ``'vertexai/gemini-4-argon'``) — when passing a plain ``str`` to
      ``Genkit(model=...)`` or ``ai.generate(model=...)``, the ``'googleai/'``
      (or ``'vertexai/'``) prefix is required; passing an unprefixed string
      raises ``NOT_FOUND``.

Example:
    Using GoogleAI (Gemini API):

    ```python
    from genkit import Genkit
    from genkit_google_genai import GoogleAI

    # 1. Initialize Genkit with the GoogleAI plugin
    ai = Genkit(plugins=[GoogleAI()])

    # 2. Generate content using dynamic model discovery
    res = await ai.generate(
        model=GoogleAI.gemini_model('gemini-flash-latest'),
        prompt='Suggest 2 catchy names for a space coffee shop.',
    )

    # 3. Inspect output shapes directly
    print(res.text)
    # => 1. AstroBrew
    #    2. Nebula Nectar
    ```

    Using VertexAI (Google Cloud):

    ```python
    from genkit import Genkit
    from genkit_google_genai import VertexAI

    # 1. Initialize with your GCP project and location
    ai = Genkit(plugins=[VertexAI(project='my-project', location='us-central1')])

    # 2. Generate content with Gemini on Vertex AI
    res = await ai.generate(
        model=VertexAI.gemini_model('gemini-4-argon'),
        prompt='Explain quantum entanglement in one sentence.',
    )

    # 3. Inspect output shapes directly
    print(res.text)
    # => "Quantum entanglement occurs when paired particles remain linked..."
    ```

Requirements:
    - GoogleAI requires the ``GEMINI_API_KEY`` environment variable or explicit ``api_key``.
    - VertexAI requires Google Cloud Application Default Credentials (ADC) or explicit credentials.

See Also:
    - Gemini API: https://ai.google.dev/
    - Vertex AI: https://cloud.google.com/vertex-ai
"""

from genkit_google_genai._google import (
    GoogleAI,
    VertexAI,
)
from genkit_google_genai._models._antigravity import AntigravityConfig
from genkit_google_genai._models._deep_research import DeepResearchConfig
from genkit_google_genai._models._embedder import EmbeddingConfig
from genkit_google_genai._models._gemini import (
    GeminiConfig,
    GeminiImageConfig,
    GeminiTtsConfig,
    GemmaConfig,
)
from genkit_google_genai._models._interactions_lyria import LyriaConfig
from genkit_google_genai._models._veo import VeoConfig
from genkit_google_genai._models._virtual_try_on import VirtualTryOnConfig

__all__ = [
    'AntigravityConfig',
    'DeepResearchConfig',
    'EmbeddingConfig',
    'GeminiConfig',
    'GeminiImageConfig',
    'GeminiTtsConfig',
    'GemmaConfig',
    'GoogleAI',
    'LyriaConfig',
    'VeoConfig',
    'VertexAI',
    'VirtualTryOnConfig',
]
