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

"""Minimal entry point for ollama model conformance testing.

Ollama runs locally so no API key is required, but an Ollama server
must be running (default: http://localhost:11434).

Usage:
    genkit dev:test-model --from-file model-conformance.yaml -- uv run conformance_entry.py

Env:
    OLLAMA_HOST: Optional. Ollama server URL (default: http://localhost:11434).
"""

import asyncio

from genkit.ai import Genkit
from genkit.plugins.ollama import Ollama
from genkit.plugins.ollama.models import ModelDefinition

ai = Genkit(
    plugins=[
        Ollama(
            models=[
                ModelDefinition(name='gemma3'),
            ],
        )
    ],
)


async def main() -> None:
    """Keep the process alive for the test runner."""
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
