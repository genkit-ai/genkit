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

"""Text validators: text-includes, text-starts-with, text-not-empty.

Canonical JS source (keep in sync):
    genkit-tools/cli/src/commands/dev-test-model.ts — VALIDATORS
"""

from __future__ import annotations

from typing import Any

from conform.validators import ValidationError, register
from conform.validators.helpers import get_message_text


@register('text-includes')
def text_includes(
    response: dict[str, Any],
    arg: str | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> None:
    """Check that response text includes the expected substring (case-insensitive)."""
    text = get_message_text(response)
    if not text or (arg and arg.lower() not in text.lower()):
        raise ValidationError(f"Response text does not include '{arg}'. Text: {text}")


@register('text-starts-with')
def text_starts_with(
    response: dict[str, Any],
    arg: str | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> None:
    """Check that response text starts with the expected prefix."""
    text = get_message_text(response)
    if not text or (arg and not text.strip().startswith(arg)):
        raise ValidationError(f"Response text does not start with '{arg}'. Text: {text}")


@register('text-not-empty')
def text_not_empty(
    response: dict[str, Any],
    arg: str | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> None:
    """Check that response text is not empty."""
    text = get_message_text(response)
    if not text or not text.strip():
        raise ValidationError('Response text is empty')
