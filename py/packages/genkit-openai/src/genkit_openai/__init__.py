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


"""OpenAI-compatible model provider for Genkit.

This plugin provides integration with OpenAI and any OpenAI-compatible API
endpoints (such as Azure OpenAI, Together AI, or Anyscale) using the official
OpenAI Python SDK.

Example:
    ```python
    from genkit import Genkit
    from genkit_openai import OpenAI

    # 1. Initialize Genkit with OpenAI plugin
    ai = Genkit(plugins=[OpenAI()])

    # 2. Generate content using GPT-4o
    res = await ai.generate(
        model='openai/gpt-4o',
        prompt='Suggest 2 catchy names for an AI newsletter.',
    )

    # 3. Inspect output shapes directly
    print(res.text)
    # => 1. Prompt Daily
    #    2. Neural Notes
    ```

Requirements:
    - ``OpenAI`` requires ``OPENAI_API_KEY`` or an explicit ``api_key``.
    - ``DeepSeek`` requires ``DEEPSEEK_API_KEY`` or an explicit ``api_key``.

See Also:
    - OpenAI documentation: https://platform.openai.com/docs/
"""

from .deepseek import DeepSeek, DeepSeekConfig, deepseek_model
from .openai_plugin import OpenAI, openai_model
from .typing import OpenAIConfig


def package_name() -> str:
    """The package name for the OpenAI-compatible model provider."""
    return 'genkit_openai'


__all__ = [
    'DeepSeek',
    'DeepSeekConfig',
    'OpenAI',
    'OpenAIConfig',
    'deepseek_model',
    'openai_model',
    'package_name',
]
