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

Import the classes you need and pass instances into ``use=[...]``:

    from genkit.plugins.middleware import Retry, Fallback

    response = await ai.generate(
        prompt='Hello',
        use=[
            Retry(max_retries=5),
            Fallback(models=['googleai/gemini-2.5-pro']),
        ],
    )

Or register all five with the ``Middleware`` plugin so they appear in
the Dev UI.
"""

from genkit.middleware import MiddlewareDesc
from genkit.plugin_api import Action, ActionKind, ActionMetadata, Plugin, new_middleware
from genkit.plugins.middleware._fallback import Fallback
from genkit.plugins.middleware._filesystem import Filesystem
from genkit.plugins.middleware._retry import Retry
from genkit.plugins.middleware._skills import Skills
from genkit.plugins.middleware._tool_approval import ToolApproval


class Middleware(Plugin):
    """Plugin that registers Retry, Fallback, ToolApproval, Skills, and Filesystem.

    Registers all five middleware descriptors so they show up in the Dev
    UI.

    ``Filesystem`` has no default root — supply ``root_dir`` when
    constructing an instance, for example
    ``Filesystem(root_dir='./workspace')``.

    Usage:
        from genkit.plugins.middleware import Middleware, Retry, Skills

        ai = Genkit(plugins=[GoogleAI(), Middleware()])
        await ai.generate(
            prompt='Hello',
            use=[Retry(max_retries=5), Skills(skill_paths=['skills'])],
        )
    """

    name = 'genkit-middleware'

    async def init(self) -> list[Action]:
        """No actions to register; this plugin only contributes middleware."""
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """No dynamic actions to resolve."""
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        """No actions to list."""
        return []

    def list_middleware(self) -> list[MiddlewareDesc]:
        """Return descriptors for all middleware exposed by this plugin."""
        return [
            new_middleware(Retry),
            new_middleware(Fallback),
            new_middleware(ToolApproval),
            new_middleware(Skills),
            new_middleware(Filesystem),
        ]


__all__ = [
    'Fallback',
    'Filesystem',
    'Middleware',
    'Retry',
    'Skills',
    'ToolApproval',
]
