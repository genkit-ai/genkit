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

"""Ollama plugin for Genkit.

This plugin provides integration with Ollama for running local LLMs and text
embedders directly on your own infrastructure.

Prerequisites:
    - Install Ollama: https://ollama.ai/
    - Ensure the Ollama server is running (default: ``http://localhost:11434``).
    - Pull target models locally (e.g., ``ollama pull llama3.2``).

Example:
    ```python
    from genkit import Genkit
    from genkit_ollama import Ollama

    ai = Genkit(
        plugins=[Ollama(models=['llama3.2'])],
        model='ollama/llama3.2',
    )
    response = await ai.generate(prompt='Hello, Llama!')
    print(response.text)
    ```

See Also:
    - Ollama documentation: https://ollama.ai/
"""

from genkit_ollama.plugin_api import Ollama, ollama_name


def package_name() -> str:
    """Get the package name for the Ollama plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit_ollama'


__all__ = [
    'Ollama',
    'ollama_name',
    'package_name',
]
