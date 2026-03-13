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

"""Streaming validators: stream-text-includes, stream-has-tool-request, stream-valid-json.

These validators check both the streaming chunks and the final response.

Canonical JS source (keep in sync):
    genkit-tools/cli/src/commands/dev-test-model.ts — VALIDATORS
"""

from __future__ import annotations

import json
from typing import Any

from conform.validators import VALIDATORS, ValidationError, register


@register('stream-text-includes')
def stream_text_includes(
    response: dict[str, Any],
    arg: str | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> None:
    """Check that streamed text includes the expected substring."""
    if not chunks:
        raise ValidationError('Streaming expected but no chunks were received')

    streamed_text = ''.join(
        part.get('text', '') for chunk in chunks for part in (chunk.get('content') or []) if 'text' in part
    )

    if arg and arg not in streamed_text:
        raise ValidationError(f"Streaming response did not include '{arg}'")


@register('stream-has-tool-request')
def stream_has_tool_request(
    response: dict[str, Any],
    arg: str | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> None:
    """Check that streamed chunks contain a tool request."""
    if not chunks:
        raise ValidationError('Streaming expected but no chunks were received')

    has_tool = any('toolRequest' in part for chunk in chunks for part in (chunk.get('content') or []))
    if not has_tool:
        raise ValidationError('No tool request found in the streamed chunks')

    if arg:
        VALIDATORS['has-tool-request'](response, arg)


@register('stream-valid-json')
def stream_valid_json(
    response: dict[str, Any],
    arg: str | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> None:
    """Check that streamed text forms valid JSON."""
    if not chunks:
        raise ValidationError('Streaming expected but no chunks were received')

    streamed_text = ''.join(
        part.get('text', '') for chunk in chunks for part in (chunk.get('content') or []) if 'text' in part
    )

    if not streamed_text.strip():
        raise ValidationError('Streamed response contained no text')
    try:
        json.loads(streamed_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValidationError(f'Streamed text is not valid JSON: {streamed_text}') from exc
