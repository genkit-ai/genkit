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

Import the classes you need and pass instances into ``use=[...]``.
See below for an example.
"""

from genkit.plugin_api import MiddlewarePlugin, new_middleware
from genkit_middleware._fallback import Fallback
from genkit_middleware._filesystem import Filesystem
from genkit_middleware._retry import Retry
from genkit_middleware._skills import Skills
from genkit_middleware._tool_approval import ToolApproval

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
]


class Middleware(MiddlewarePlugin):
    """Plugin that registers Retry, Fallback, ToolApproval, Skills, and Filesystem.

    Registers all five middleware descriptors so they show up in the Dev
    UI.

    ``Filesystem`` has no default root — supply ``root_dir`` when
    constructing an instance, for example
    ``Filesystem(root_dir='./workspace')``.

    Example:
        ```python
        from genkit import Genkit
        from genkit_googleai import GoogleAI
        from genkit_middleware import Middleware, Retry

        # 1. Register middleware plugin
        ai = Genkit(plugins=[GoogleAI(), Middleware()])

        # 2. Generate with automatic retry resilience
        res = await ai.generate(
            model='googleai/gemini-flash-latest',
            prompt='Summarize quantum computing.',
            use=[Retry(max_retries=3)],
        )

        # 3. Inspect output
        print(res.text)
        # => Quantum computing uses quantum mechanics for complex calculations...
        ```
    """

    name = 'genkit-middleware'
    middleware = list(_MIDDLEWARE_DESCS)


__all__ = [
    'Fallback',
    'Filesystem',
    'Middleware',
    'Retry',
    'Skills',
    'ToolApproval',
]
