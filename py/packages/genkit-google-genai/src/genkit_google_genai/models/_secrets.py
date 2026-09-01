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

The ticket and the generate span are not where a tenant key lives. Callers
pass it again on the run: ``context={'secrets': {'api_key': tenant}}``.
"""

from typing import Any, cast

from genkit import GenkitError

SECRETS_SLOT = "Pass the key as context={'secrets': {'api_key': ...}}."


def misplaced_key_error() -> GenkitError:
    return GenkitError(
        status='INVALID_ARGUMENT',
        message=f'API key belongs in context.secrets, not config or the top-level context. {SECRETS_SLOT}',
    )


def string_secret(value: object) -> str | None:
    if value is None or value == '':
        return None
    if isinstance(value, str):
        return value
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=f'context.secrets.api_key must be a string. {SECRETS_SLOT}',
    )


def context_api_key(context: dict[str, Any]) -> str | None:
    """Read the per-request key from ``context.secrets``.

    Documented slot is ``api_key``; ``apiKey`` is accepted so either
    spelling works. A key on ``context['config']`` or the top-level
    context is rejected so it cannot silently fall through to the
    plugin client. ``generate(..., config={'api_key': ...})`` is the
    same leftover — reject it on the request too.
    """
    extra = context.get('config')
    if isinstance(extra, dict) and (extra.get('api_key') is not None or extra.get('apiKey') is not None):
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


def _bag_has_api_key(bag: dict[str, Any]) -> bool:
    return bag.get('api_key') is not None or bag.get('apiKey') is not None


def request_config_has_api_key(config: object) -> bool:
    """True when the generate/start request still carries a tenant key."""
    if config is None:
        return False
    if isinstance(config, dict):
        return _bag_has_api_key(cast(dict[str, Any], config))
    extra = getattr(config, 'model_extra', None)
    if isinstance(extra, dict) and _bag_has_api_key(cast(dict[str, Any], extra)):
        return True
    if getattr(config, 'api_key', None) is not None or getattr(config, 'apiKey', None) is not None:
        return True
    dump = getattr(config, 'model_dump', None)
    if callable(dump):
        try:
            dumped = dump(by_alias=True)
        except TypeError:
            dumped = dump()
        return isinstance(dumped, dict) and _bag_has_api_key(cast(dict[str, Any], dumped))
    return False


def reject_request_config_api_key(config: object) -> None:
    """A key on request.config belongs in context.secrets."""
    if request_config_has_api_key(config):
        raise misplaced_key_error()
