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
trace: ``context={'secrets': {'api_key': tenant_key}}``. Hosted Ollama reads
it as ``Authorization: Bearer <key>``.
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


def reject_config_api_key(config: object) -> None:
    """Raise when ``request.config`` carries an API key.

    ``ModelConfig`` declares ``api_key``, so ``OllamaConfig`` accepts it even
    with unknown keys forbidden. Left alone it would be dropped and the call
    would run on the plugin's header. ``config.extra`` isn't checked: it goes
    out as sampler options as written.
    """
    if config is None:
        return
    if isinstance(config, Mapping):
        bag = cast(Mapping[str, object], config)
        has_key = bag.get('api_key') is not None or bag.get('apiKey') is not None
    else:
        extra = getattr(config, 'model_extra', None) or {}
        has_key = (
            getattr(config, 'api_key', None) is not None
            or extra.get('api_key') is not None
            or extra.get('apiKey') is not None
        )
    if has_key:
        raise misplaced_key_error()


def string_secret(value: object) -> str | None:
    if value is None or value == '':
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if any(c in cleaned for c in ('\r', '\n', '\0', ' ', '\t')):
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'context.secrets.api_key contains invalid whitespace or control characters. {SECRETS_SLOT}',
            )
        return cleaned
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=f'context.secrets.api_key must be a string. {SECRETS_SLOT}',
    )


def context_api_key(context: dict[str, Any]) -> str | None:
    """Read the per-request key from ``context.secrets``.

    ``api_key`` is the documented slot; ``apiKey`` works too. A key on the
    top-level context or ``context['config']`` raises, because quietly
    running that call on the plugin's own header would bill the wrong account.
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
    key = string_secret(secrets.get('api_key'))
    if key is None:
        key = string_secret(secrets.get('apiKey'))
    if key is None:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets is set but has no api_key. {SECRETS_SLOT}',
        )
    return key
