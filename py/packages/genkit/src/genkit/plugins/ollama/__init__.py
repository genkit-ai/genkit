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

This plugin provides integration with Ollama for running local LLMs. Ollama
allows you to run models like Llama, Mistral, and others on your own hardware.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Ollama              │ Software that runs AI models on YOUR computer.    │
    │                     │ Like having a mini ChatGPT at home.               │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Local LLM           │ An AI that runs offline on your machine.          │
    │                     │ No internet needed, your data stays private.      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Llama               │ Meta's open-source AI model. Like a free          │
    │                     │ version of ChatGPT you can run yourself.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Model Pull          │ Download a model to your computer. Like           │
    │                     │ installing an app before you can use it.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Server URL          │ Where Ollama listens for requests. Default        │
    │                     │ is localhost:11434 (your own computer).           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ GGUF                │ File format for AI models. Like .mp3 for          │
    │                     │ music, but for AI brains.                         │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   HOW OLLAMA RUNS AI ON YOUR COMPUTER                   │
    │                                                                         │
    │    Your Code                                                            │
    │    ai.generate(prompt="Hello!")                                         │
    │         │                                                               │
    │         │  (1) Request goes to Ollama plugin                            │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Ollama Plugin  │   Formats request for Ollama API                 │
    │    │                 │                                                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) HTTP to localhost:11434 (your computer!)             │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Ollama Server  │   Loads model into RAM/GPU                       │
    │    │  (on your PC)   │   (first request may be slow)                    │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (3) Model processes on YOUR hardware                     │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Llama/Mistral  │   CPU or GPU does the thinking                   │
    │    │  Model (local)  │   No data leaves your machine!                   │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (4) Response streamed back                               │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Your App       │   response.text = "Hello! How can I help?"       │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Architecture Overview::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          Ollama Plugin                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Plugin Entry Point (__init__.py)                                       │
    │  ├── Ollama - Plugin class                                              │
    │  └── ollama_name() - Helper to create namespaced model names            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  plugin_api.py - Plugin Implementation                                  │
    │  ├── Ollama class (registers models and embedders)                      │
    │  └── Configuration for server URL and models                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  models.py - Model Implementation                                       │
    │  ├── OllamaModel (chat/generate API integration)                        │
    │  ├── Request/response conversion                                        │
    │  └── Streaming support                                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  embedders.py - Embedding Implementation                                │
    │  └── OllamaEmbedder (embedding API integration)                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  constants.py - Default Configuration                                   │
    │  └── DEFAULT_OLLAMA_SERVER_URL                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Overview:
    The Ollama plugin connects Genkit to locally running Ollama models.
    This is ideal for development, privacy-sensitive applications, or
    when you want to run models without cloud dependencies.

Prerequisites:
    - Install Ollama: https://ollama.ai/
    - Pull a model: ``ollama pull llama3.2`` or ``ollama pull mistral``
    - Ollama server running (default: http://localhost:11434)

Key Components:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Component         │ Purpose                                             │
    ├───────────────────┼─────────────────────────────────────────────────────┤
    │ Ollama            │ Plugin class to register with Genkit                │
    │ ollama_name()     │ Helper to create namespaced model names             │
    └───────────────────┴─────────────────────────────────────────────────────┘

Example:
    Basic usage:

    ```python
    from genkit import Genkit
    from genkit.plugins.ollama import Ollama

    # Configure with model name and optional server URL
    ai = Genkit(
        plugins=[Ollama(models=['llama3.2', 'mistral'])],
        model='ollama/llama3.2',
    )

    response = await ai.generate(prompt='Hello, Llama!')
    print(response.text)

    # Use a specific model
    response = await ai.generate(
        model='ollama/mistral',
        prompt='Write a haiku about coding',
    )
    ```

Caveats:
    - Requires Ollama installed and running locally
    - Model names are prefixed with 'ollama/' (e.g., 'ollama/llama3.2')
    - Performance depends on local hardware

See Also:
    - Ollama documentation: https://ollama.ai/
    - Genkit documentation: https://genkit.dev/
"""

from genkit.plugins.ollama.plugin_api import Ollama, ollama_name


def package_name() -> str:
    """Get the package name for the Ollama plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit.plugins.ollama'


__all__ = [
    package_name.__name__,
    Ollama.__name__,
    ollama_name.__name__,
]
