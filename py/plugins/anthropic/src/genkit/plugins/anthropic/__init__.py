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

"""Anthropic plugin for Genkit.

This plugin provides integration with Anthropic's Claude models for the
Genkit framework. It registers Claude models as Genkit actions, enabling
text generation operations.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Claude              │ Anthropic's AI assistant. Like a helpful friend   │
    │                     │ who's great at explaining things and writing.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Sonnet              │ The "just right" Claude model. Good at most       │
    │                     │ tasks without being too slow or expensive.        │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Haiku               │ The fast & cheap Claude model. Perfect for        │
    │                     │ quick tasks like classification or summaries.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Opus                │ The most capable Claude. For complex tasks        │
    │                     │ like research, analysis, or creative writing.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ API Key             │ Your password to use Claude. Keep it secret!      │
    │                     │ Set as ANTHROPIC_API_KEY environment variable.    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ System Prompt       │ Instructions that shape Claude's personality.     │
    │                     │ Like giving a new employee their job description. │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Tool Calling        │ Claude can use functions you define. Like         │
    │                     │ giving it a calculator or search engine to use.   │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     HOW CLAUDE PROCESSES YOUR REQUEST                   │
    │                                                                         │
    │    Your Code                                                            │
    │    ai.generate(prompt="Explain quantum computing")                      │
    │         │                                                               │
    │         │  (1) Request goes to Anthropic plugin                         │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Anthropic      │   Plugin adds API key to request                 │
    │    │  Plugin         │                                                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) Converts Genkit format → Claude Messages API         │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  AnthropicModel │   Handles message roles, images,                 │
    │    │                 │   tools, and streaming                           │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (3) HTTPS request to api.anthropic.com                   │
    │             ▼                                                           │
    │    ════════════════════════════════════════════════════                 │
    │             │  Internet                                                 │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Anthropic      │   Claude thinks about your prompt                │
    │    │  Claude API     │   and generates a response                       │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (4) Response parsed back to Genkit format                │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Your App       │   response.text = "Quantum computing..."         │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Architecture Overview::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        Anthropic Plugin                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Plugin Entry Point (__init__.py)                                       │
    │  ├── Anthropic - Plugin class                                           │
    │  └── anthropic_name() - Helper to create namespaced model names         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  plugin.py - Plugin Implementation                                      │
    │  ├── Anthropic class (registers models)                                 │
    │  └── Client initialization with Anthropic SDK                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  models.py - Model Implementation                                       │
    │  ├── AnthropicModel (Messages API integration)                          │
    │  ├── Request/response conversion                                        │
    │  └── Streaming support                                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  model_info.py - Model Registry                                         │
    │  ├── SUPPORTED_MODELS (claude-3.5-sonnet, opus, haiku, etc.)            │
    │  └── Model capabilities and metadata                                    │
    └─────────────────────────────────────────────────────────────────────────┘

Overview:
    The Anthropic plugin adds support for Claude models to Genkit. It uses
    the official Anthropic Python SDK and registers models that can be used
    with ai.generate() and other Genkit generation methods.

Supported Models:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Model                     │ Description                                 │
    ├───────────────────────────┼─────────────────────────────────────────────┤
    │ claude-3-5-sonnet-latest  │ Balanced performance and capability         │
    │ claude-3-5-haiku-latest   │ Fast and cost-effective                     │
    │ claude-3-opus-latest      │ Most capable, complex tasks                 │
    │ claude-sonnet-4-20250514  │ Latest Sonnet model                         │
    └───────────────────────────┴─────────────────────────────────────────────┘

Key Components:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Component           │ Purpose                                           │
    ├─────────────────────┼───────────────────────────────────────────────────┤
    │ Anthropic           │ Plugin class to register with Genkit              │
    │ anthropic_name()    │ Helper to create namespaced model names           │
    └─────────────────────┴───────────────────────────────────────────────────┘

Example:
    Basic usage:

    ```python
    from genkit import Genkit
    from genkit.plugins.anthropic import Anthropic

    # Uses ANTHROPIC_API_KEY env var or pass api_key explicitly
    ai = Genkit(
        plugins=[Anthropic()],
        model='anthropic/claude-3-5-sonnet-latest',
    )

    response = await ai.generate(prompt='Hello, Claude!')
    print(response.text)

    # With custom configuration
    response = await ai.generate(
        model='anthropic/claude-3-5-haiku-latest',
        prompt='Write a haiku about AI',
        config={'temperature': 0.7, 'max_tokens': 100},
    )
    ```

    With tools:

    ```python
    @ai.tool()
    def get_weather(city: str) -> str:
        return f'Weather in {city}: Sunny, 72°F'


    response = await ai.generate(
        model='anthropic/claude-3-5-sonnet-latest',
        prompt='What is the weather in Paris?',
        tools=['get_weather'],
    )
    ```

Caveats:
    - Requires ANTHROPIC_API_KEY environment variable or api_key parameter
    - Model names are prefixed with 'anthropic/' (e.g., 'anthropic/claude-3-5-sonnet-latest')
    - Anthropic models may have different tool calling behavior than Google models

See Also:
    - Anthropic documentation: https://docs.anthropic.com/
    - Genkit documentation: https://genkit.dev/
"""

from genkit.plugins.anthropic.plugin import Anthropic, anthropic_name

__all__ = ['Anthropic', 'anthropic_name']
