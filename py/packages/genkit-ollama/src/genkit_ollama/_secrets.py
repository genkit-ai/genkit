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

"""Per-request API key on ``context.secrets`` for the Ollama plugin.

A tenant key travels with the call, not the config, so it never lands in a
trace: ``context={'secrets': {'api_key': tenant_key}}``. Hosted Ollama reads
it as ``Authorization: Bearer <key>``. A key on ``config`` (including
``config.extra``) is rejected by Genkit before the model action runs.

Same rules as ``genkit_openai._secrets``.
"""

from collections.abc import Mapping
from typing import cast

from genkit import GenkitError

SECRETS_HINT = "Pass the key as context={'secrets': {'api_key': ...}}."

_KEY_SLOTS = ('api_key', 'apiKey')
_BAD_KEY_CHARS = ('\r', '\n', '\0', ' ', '\t')


def _clean_key(value: object, slot: str) -> str:
    if not isinstance(value, str):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets.{slot} must be a string. {SECRETS_HINT}',
        )
    cleaned = value.strip()
    if not cleaned:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets.{slot} is blank. {SECRETS_HINT}',
        )
    if any(c in cleaned for c in _BAD_KEY_CHARS):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets.{slot} contains invalid whitespace or control characters. {SECRETS_HINT}',
        )
    return cleaned


def context_api_key(context: Mapping[str, object] | None) -> str | None:
    """Read the per-request key from ``context.secrets``.

    ``api_key`` is the documented slot; ``apiKey`` works too. ``secrets``
    without either runs on the plugin's headers, so apps can keep other
    secrets there. A key that is set but blank or not a string raises.
    Top-level context entries like ``context['api_key']`` are ignored, so
    app-level auth context providers and keys meant for other providers don't
    break an Ollama call made inside the same flow.

    Args:
        context: The action context (``ctx.context``).

    Returns:
        The trimmed key, or None when ``context.secrets`` has no key.

    Raises:
        GenkitError: ``INVALID_ARGUMENT`` when ``secrets`` is not a mapping or
            the key is not a usable string.
    """
    if not context:
        return None
    secrets = context.get('secrets')
    if secrets is None:
        return None
    if not isinstance(secrets, Mapping):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets must be a dict. {SECRETS_HINT}',
        )
    bag = cast(Mapping[str, object], secrets)
    for slot in _KEY_SLOTS:
        value = bag.get(slot)
        if value is not None:
            return _clean_key(value, slot)
    return None
