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

"""Response helpers shared across validators.

These mirror the helper functions in the JS canonical source:
    genkit-tools/cli/src/commands/dev-test-model.ts
"""

from __future__ import annotations

from typing import Any


def get_message_content(response: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract the message content array from a GenerateResponse."""
    message = response.get('message') or (
        response.get('candidates', [{}])[0].get('message') if response.get('candidates') else None
    )
    if message is None:
        return None
    return message.get('content')


def get_message_text(response: dict[str, Any]) -> str | None:
    """Extract the first text part from the response."""
    content = get_message_content(response)
    if not content or not isinstance(content, list):
        return None
    for part in content:
        if 'text' in part:
            return part['text']
    return None


def get_media_part(response: dict[str, Any]) -> dict[str, Any] | None:
    """Find the first media part in the response."""
    content = get_message_content(response)
    if not content or not isinstance(content, list):
        return None
    for part in content:
        if 'media' in part:
            return part
    return None
