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

- ``Retry``        — retries model calls on transient errors with exponential backoff
- ``Fallback``     — falls back to alternative models on failure
- ``ToolApproval`` — requires approval before executing tools
- ``Skills``       — exposes a SKILL.md library as system prompts + ``use_skill`` tool
- ``Filesystem``   — sandboxed filesystem operations (list/read/write/edit)

Quick start — register all five at once:

    from genkit.plugins.middleware import middleware_bundle

    ai = Genkit(plugins=[middleware_bundle()])
"""

from genkit import middleware_plugin
from genkit.middleware import new_middleware
from genkit.plugins.middleware._fallback import Fallback
from genkit.plugins.middleware._filesystem import Filesystem
from genkit.plugins.middleware._retry import Retry
from genkit.plugins.middleware._skills import Skills
from genkit.plugins.middleware._tool_approval import ToolApproval


def middleware_bundle() -> object:
    """Return a plugin that registers Retry, Fallback, ToolApproval, Skills, and Filesystem.

    Registers all five middleware descriptors under bare names (``retry``,
    ``fallback``, ``tool_approval``, ``skills``, ``filesystem``) so they can be
    referenced by name in ``generate(use=[MiddlewareRef(...)])`` calls.

    ``Filesystem`` has no default root: you must supply ``root_dir`` in the ref
    config (or pass a ``Filesystem(root_dir=...)`` instance inline).  Resolving
    ``Filesystem`` with empty config raises a validation error.

    Usage::

        ai = Genkit(plugins=[middleware_bundle()])
        # Then reference by name in generate():
        await ai.generate(use=[MiddlewareRef(name='retry', config={'max_retries': 5})])
    """
    return middleware_plugin([
        new_middleware(Retry),
        new_middleware(Fallback),
        new_middleware(ToolApproval),
        new_middleware(Skills),
        new_middleware(Filesystem),
    ])


__all__ = [
    'Fallback',
    'Filesystem',
    'Retry',
    'Skills',
    'ToolApproval',
    'middleware_bundle',
    'new_middleware',
]
