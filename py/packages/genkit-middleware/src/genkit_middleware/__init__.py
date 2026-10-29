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

"""Genkit middleware plugin.

Provides concrete middleware implementations:

* ``Retry`` — retries model calls on transient errors with exponential
  backoff.
* ``Fallback`` — falls back to alternative models on failure.
* ``ToolApproval`` — requires approval before executing tools.
* ``Skills`` — exposes a ``SKILL.md`` library as system prompts plus a
  ``use_skill`` tool.
* ``Filesystem`` — sandboxed filesystem operations (list / read / write /
  edit).

Import the classes you need and pass instances into ``use=[...]``; that works
without registering anything. Register the ``Middleware()`` plugin to list
them in the Dev UI and to name them in a ``.prompt`` file's ``use:``.

``Artifacts`` only works inside an agent, so it lives in
``genkit_middleware.exp`` alongside the other experimental agent APIs.
"""

from genkit.middleware import Middleware as GenkitMiddleware
from genkit.plugin_api import Action, ActionKind, ActionMetadata, Plugin
from genkit_middleware._fallback import Fallback
from genkit_middleware._filesystem import Filesystem
from genkit_middleware._retry import Retry
from genkit_middleware._skills import Skills
from genkit_middleware._tool_approval import ToolApproval


class Middleware(Plugin):
    """Plugin that registers Retry, Fallback, ToolApproval, Skills, and Filesystem.

    Registering lists them in the Dev UI and lets a ``.prompt`` file name them
    in ``use:``. Passing an instance in ``use=[...]`` works either way.

    ``Filesystem`` has no default root — supply ``root_dir`` when
    constructing an instance, for example
    ``Filesystem(root_dir='./workspace')``.

    Example:
        ```python
        from genkit import Genkit
        from genkit_google_genai import GoogleAI
        from genkit_middleware import Middleware, Retry

        # 1. Register middleware plugin
        ai = Genkit(plugins=[GoogleAI(), Middleware()])

        # 2. Generate with automatic retry resilience
        res = await ai.generate(
            model=GoogleAI.gemini('gemini-flash-latest'),
            prompt='Summarize quantum computing.',
            use=[Retry(max_retries=3)],
        )

        # 3. Inspect output
        print(res.text)
        # => Quantum computing uses quantum mechanics for complex calculations...
        ```
    """

    name = 'genkit-middleware'

    async def init(self) -> list[Action]:
        """No actions; this plugin only contributes middleware."""
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolves nothing."""
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        """Lists nothing."""
        return []

    def list_middleware(self) -> list[type[GenkitMiddleware]]:
        """Retry, Fallback, ToolApproval, Skills and Filesystem."""
        return [Retry, Fallback, ToolApproval, Skills, Filesystem]


__all__ = [
    'Fallback',
    'Filesystem',
    'Middleware',
    'Retry',
    'Skills',
    'ToolApproval',
]
