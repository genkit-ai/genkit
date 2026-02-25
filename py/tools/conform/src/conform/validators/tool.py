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

"""Tool request validator: has-tool-request.

Canonical JS source (keep in sync):
    genkit-tools/cli/src/commands/dev-test-model.ts — VALIDATORS['has-tool-request']
"""

from __future__ import annotations

import json
from typing import Any

from conform.validators import ValidationError, register
from conform.validators.helpers import get_message_content


@register('has-tool-request')
def has_tool_request(
    response: dict[str, Any],
    arg: str | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> None:
    """Check that the response contains a tool request, optionally for a specific tool."""
    content = get_message_content(response)
    if not content or not isinstance(content, list):
        raise ValidationError(f'Response missing message content. Full response: {json.dumps(response, indent=2)}')
    tool_request = next((c for c in content if 'toolRequest' in c), None)
    if not tool_request:
        raise ValidationError(f'Model did not return a tool request. Content: {json.dumps(content, indent=2)}')
    if arg and tool_request.get('toolRequest', {}).get('name') != arg:
        actual_name = tool_request.get('toolRequest', {}).get('name')
        raise ValidationError(f"Expected tool request '{arg}', got '{actual_name}'")
