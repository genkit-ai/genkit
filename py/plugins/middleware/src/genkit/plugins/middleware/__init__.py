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
- Retry: Retries model calls on transient errors with exponential backoff
- Fallback: Falls back to alternative models on failure
- ToolApproval: Requires approval before executing tools
- Skills: Exposes skill library as system prompts
- Filesystem: Sandboxed filesystem operations
"""

from genkit.plugins.middleware._fallback import Fallback
from genkit.plugins.middleware._filesystem import Filesystem
from genkit.plugins.middleware._retry import Retry
from genkit.plugins.middleware._skills import Skills
from genkit.plugins.middleware._tool_approval import ToolApproval

__all__ = ['Retry', 'Fallback', 'ToolApproval', 'Skills', 'Filesystem']
