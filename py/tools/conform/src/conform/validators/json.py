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

"""JSON validator: valid-json.

Canonical JS source (keep in sync):
    genkit-tools/cli/src/commands/dev-test-model.ts — VALIDATORS['valid-json']
"""

from __future__ import annotations

import json as json_mod
from typing import Any

from conform.validators import ValidationError, register
from conform.validators.helpers import get_message_content


@register('valid-json')
def valid_json(
    response: dict[str, Any],
    arg: str | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> None:
    """Check that response text is valid JSON."""
    content = get_message_content(response)
    if not content or not isinstance(content, list):
        raise ValidationError(f'Response missing message content. Full response: {json_mod.dumps(response, indent=2)}')
    text_part = next((c for c in content if 'text' in c), None)
    if not text_part:
        raise ValidationError(
            f'Model did not return text content for JSON. Content: {json_mod.dumps(content, indent=2)}'
        )
    try:
        json_mod.loads(text_part['text'])
    except (json_mod.JSONDecodeError, TypeError) as exc:
        raise ValidationError(f'Response text is not valid JSON. Text: {text_part["text"]}') from exc
