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

Config is recorded in traces and secrets aren't, so a tenant key only
counts when it arrives as ``context={'secrets': {'api_key': tenant}}``.
"""

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
            'Anthropic needs an API key: set ANTHROPIC_API_KEY or pass Anthropic(api_key=...), '
            "or send a per-request key as context={'secrets': {'api_key': ...}}."
        ),
    )


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

    ``api_key`` is the documented slot; ``apiKey`` is accepted too. Secrets
    without either key return None, so the call uses the plugin's key; apps
    keep other secrets there. A key on ``context['config']`` or the top-level
    context raises, so it can't quietly fall through to the plugin's own key.
    """
    nested = context.get('config')
    if isinstance(nested, dict) and _bag_has_api_key(cast(dict[str, Any], nested)):
        raise misplaced_key_error()
    if _bag_has_api_key(context):
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
    return key


def _bag_has_api_key(bag: dict[str, Any]) -> bool:
    return bag.get('api_key') is not None or bag.get('apiKey') is not None


def reject_request_config_api_key(config: object) -> None:
    """A key on ``request.config`` raises instead of authenticating the call."""
    if config is None:
        return
    if isinstance(config, dict):
        has_key = _bag_has_api_key(cast(dict[str, Any], config))
    else:
        has_key = getattr(config, 'api_key', None) is not None
    if has_key:
        raise misplaced_key_error()
