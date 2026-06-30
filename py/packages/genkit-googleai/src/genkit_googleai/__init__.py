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
either Google AI (Gemini API) or Google Cloud Vertex AI. It dynamically discovers
and registers available models and embedders at runtime.

Example:
    Using GoogleAI (Gemini API):

    ```python
    from genkit import Genkit
    from genkit_googleai import GoogleAI

    # 1. Initialize Genkit with the GoogleAI plugin
    ai = Genkit(plugins=[GoogleAI()])

    # 2. Generate content using dynamic model discovery
    res = await ai.generate(
        model='googleai/gemini-flash-latest',
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
    from genkit_googleai import VertexAI

    # 1. Initialize with your GCP project and location
    ai = Genkit(plugins=[VertexAI(project='my-project', location='us-central1')])

    # 2. Generate content with Gemini Pro on Vertex AI
    res = await ai.generate(
        model='vertexai/gemini-pro-latest',
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

from genkit_googleai.google import (
    GoogleAI,
    VertexAI,
)
from genkit_googleai.models.embedder import (
    EmbeddingTaskType,
    GeminiEmbeddingModels,
    VertexEmbeddingModels,
)
from genkit_googleai.models.gemini import (
    GeminiConfigSchema,
    GeminiImageConfigSchema,
    GeminiTtsConfigSchema,
    GoogleAIGeminiVersion,
    VertexAIGeminiVersion,
)
from genkit_googleai.models.imagen import ImagenVersion
from genkit_googleai.models.lyria import LyriaConfig, LyriaVersion
from genkit_googleai.models.veo import VeoConfig, VeoVersion


def package_name() -> str:
    """Get the package name for the Vertex AI plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit_googleai'


__all__ = [
    'EmbeddingTaskType',
    'GeminiConfigSchema',
    'GeminiEmbeddingModels',
    'GeminiImageConfigSchema',
    'GeminiTtsConfigSchema',
    'GoogleAI',
    'GoogleAIGeminiVersion',
    'ImagenVersion',
    'LyriaConfig',
    'LyriaVersion',
    'VeoConfig',
    'VeoVersion',
    'VertexAI',
    'VertexAIGeminiVersion',
    'VertexEmbeddingModels',
    'package_name',
]
