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

"""Reasoning validator: reasoning.

Canonical JS source (keep in sync):
    genkit-tools/cli/src/commands/dev-test-model.ts — VALIDATORS['reasoning']
"""

from __future__ import annotations

from typing import Any

from conform.validators import ValidationError, register
from conform.validators.helpers import get_message_content


@register('reasoning')
def reasoning(
    response: dict[str, Any],
    arg: str | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> None:
    """Check that the response contains reasoning content."""
    content = get_message_content(response)
    if not content or not isinstance(content, list):
        raise ValidationError('Response is missing message content')
    has_reasoning = any('reasoning' in p for p in content)
    if not has_reasoning:
        raise ValidationError('reasoning content not found')
