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
* ``Artifacts`` — ``read_artifact`` / ``write_artifact`` plus artifact listing in the system prompt.

Import the classes you need and pass instances into ``use=[...]``.
See below for an example.
"""

from genkit.plugin_api import MiddlewarePlugin, new_middleware
from genkit.plugins.middleware._artifacts import Artifacts
from genkit.plugins.middleware._fallback import Fallback
from genkit.plugins.middleware._filesystem import Filesystem
from genkit.plugins.middleware._retry import Retry
from genkit.plugins.middleware._skills import Skills
from genkit.plugins.middleware._tool_approval import ToolApproval

_MIDDLEWARE_DESCS = [
    new_middleware(
        Retry,
        name='retry',
        description='Retries model calls on transient failures with exponential backoff',
    ),
    new_middleware(
        Fallback,
        name='fallback',
        description='Falls back to alternative models on failure',
    ),
    new_middleware(
        ToolApproval,
        name='tool_approval',
        description='Requires approval before executing tools',
    ),
    new_middleware(
        Skills,
        name='skills',
        description='Provides access to skill library for specialized instructions',
    ),
    new_middleware(
        Filesystem,
        name='filesystem',
        description='Sandboxed filesystem operations',
    ),
    new_middleware(
        Artifacts,
        name='artifacts',
        description='read_artifact and write_artifact tools with session artifact listing in system prompt',
    ),
]


class Middleware(MiddlewarePlugin):
    """Plugin that registers Retry, Fallback, ToolApproval, Skills, Filesystem, and Artifacts.

    Registers all six middleware descriptors so they show up in the Dev
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
    middleware = list(_MIDDLEWARE_DESCS)


__all__ = [
    'Artifacts',
    'Fallback',
    'Filesystem',
    'Middleware',
    'Retry',
    'Skills',
    'ToolApproval',
]
