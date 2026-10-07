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

"""Per-request API key on ``context.secrets``.

A tenant key travels with the call, not the config, so it never lands in a
trace: ``context={'secrets': {'api_key': tenant_key}}``.
"""

from collections.abc import Mapping
from typing import Any, cast

from genkit import GenkitError

SECRETS_SLOT = "Pass the key as context={'secrets': {'api_key': ...}}."


def misplaced_key_error() -> GenkitError:
    return GenkitError(
        status='INVALID_ARGUMENT',
        message=f'API key belongs in context.secrets, not config or the top-level context. {SECRETS_SLOT}',
    )


def missing_key_error() -> GenkitError:
    return GenkitError(
        status='FAILED_PRECONDITION',
        message=(
            'OpenAI needs an API key: set OPENAI_API_KEY or pass OpenAI(api_key=...), '
            "or send a per-request key as context={'secrets': {'api_key': ...}}."
        ),
    )


def reject_config_api_key(config: object) -> None:
    """Raise when ``request.config`` carries an API key.

    ``ModelConfig`` declares ``api_key``, so ``OpenAIConfig`` accepts it even
    with unknown keys forbidden. Left alone it would be dropped and the call
    would run on the plugin's key. Chat models get an ``OpenAIConfig``; image,
    speech and transcription models get the config as a dict. ``config.extra``
    isn't checked: it goes out as request body fields as written.
    """
    if config is None:
        return
    if isinstance(config, Mapping):
        bag = cast(Mapping[str, object], config)
        has_key = bag.get('api_key') is not None or bag.get('apiKey') is not None
    else:
        has_key = getattr(config, 'api_key', None) is not None
    if has_key:
        raise misplaced_key_error()


def string_secret(value: object) -> str:
    """The key with surrounding whitespace trimmed, or raise when it can't be a key."""
    if not isinstance(value, str):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets.api_key must be a string. {SECRETS_SLOT}',
        )
    cleaned = value.strip()
    if not cleaned:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets.api_key is blank. {SECRETS_SLOT}',
        )
    if any(c in cleaned for c in ('\r', '\n', '\0', ' ', '\t')):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets.api_key contains invalid whitespace or control characters. {SECRETS_SLOT}',
        )
    return cleaned


def context_api_key(context: dict[str, Any]) -> str | None:
    """Read the per-request key from ``context.secrets``.

    ``api_key`` is the documented slot; ``apiKey`` works too. ``secrets``
    without either runs on the plugin's key, so apps can keep other secrets
    there. A key that is set but blank or not a string raises. A key on the
    top-level context or ``context['config']`` raises, because quietly
    running that call on the plugin's own key would bill the wrong account.
    """
    nested = context.get('config')
    if isinstance(nested, dict) and (nested.get('api_key') is not None or nested.get('apiKey') is not None):
        raise misplaced_key_error()
    if context.get('api_key') is not None or context.get('apiKey') is not None:
        raise misplaced_key_error()

    if 'secrets' not in context:
        return None
    secrets = context['secrets']
    if not isinstance(secrets, dict):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets must be a dict. {SECRETS_SLOT}',
        )
    for slot in ('api_key', 'apiKey'):
        value = secrets.get(slot)
        if value is not None:
            return string_secret(value)
    return None
