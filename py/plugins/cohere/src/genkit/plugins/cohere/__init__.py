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

"""Genkit Cohere AI plugin (Community).

Community-maintained plugin — not an official Google or Cohere product.
Use of Cohere's API is subject to `Cohere's Terms of Use
<https://cohere.com/terms-of-use>`_ and `Privacy Policy
<https://cohere.com/privacy>`_.

Provides integration with Cohere's AI models for the Genkit framework,
including:

- **Chat models**: Command A, Command A Vision, Command A Reasoning,
  Command A Translate, Command R+, Command R, and Command R7B for text
  generation, tool calling, and structured output.
- **Embedding models**: Embed v4.0, Embed English v3.0,
  Embed Multilingual v3.0, and lightweight variants for semantic
  search and RAG pipelines.

Quick Start::

    from genkit.ai import Genkit
    from genkit.plugins.cohere import Cohere, cohere_name

    ai = Genkit(
        plugins=[Cohere()],
        model=cohere_name('command-a-03-2025'),
    )

    response = await ai.generate(prompt='Hello, Cohere!')

See:
    - https://docs.cohere.com/docs/models
    - https://dashboard.cohere.com/api-keys
"""

from genkit.plugins.cohere.converters import (
    FINISH_REASON_MAP,
    CohereV2Message,
    convert_messages,
    convert_response,
    convert_tools,
    convert_usage,
    extract_content_delta_text,
    extract_finish_reason,
    extract_tool_call_delta_args,
    extract_tool_call_start,
    get_response_format,
    parse_tool_arguments,
)
from genkit.plugins.cohere.embeddings import CohereEmbedConfig, CohereEmbedder
from genkit.plugins.cohere.model_info import (
    SUPPORTED_COHERE_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    ModelInfo,
    ModelSupports,
)
from genkit.plugins.cohere.models import (
    COHERE_PLUGIN_NAME,
    CohereConfig,
    CohereModel,
    cohere_name,
)
from genkit.plugins.cohere.plugin import Cohere


def package_name() -> str:
    """Return the fully qualified package name.

    Returns:
        The package name string.
    """
    return 'genkit.plugins.cohere'


__all__ = [
    # Plugin
    'Cohere',
    # Models
    'CohereConfig',
    'CohereModel',
    'COHERE_PLUGIN_NAME',
    'cohere_name',
    # Embeddings
    'CohereEmbedConfig',
    'CohereEmbedder',
    # Model metadata
    'ModelInfo',
    'ModelSupports',
    'SUPPORTED_COHERE_MODELS',
    'SUPPORTED_EMBEDDING_MODELS',
    # Converters
    'CohereV2Message',
    'FINISH_REASON_MAP',
    'convert_messages',
    'convert_response',
    'convert_tools',
    'convert_usage',
    'extract_content_delta_text',
    'extract_finish_reason',
    'extract_tool_call_delta_args',
    'extract_tool_call_start',
    'get_response_format',
    'parse_tool_arguments',
    # Utilities
    'package_name',
]
