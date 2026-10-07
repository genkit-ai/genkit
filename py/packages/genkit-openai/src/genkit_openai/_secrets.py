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

"""Per-request API key on ``context.secrets`` for the OpenAI plugin.

A tenant key travels with the call, not the config, so it never lands in a
trace: ``context={'secrets': {'api_key': tenant_key}}``.
"""

from collections.abc import Mapping
from typing import cast

from genkit import GenkitError

SECRETS_HINT = "Pass the key as context={'secrets': {'api_key': ...}}."

_KEY_SLOTS = ('api_key', 'apiKey')
_BAD_KEY_CHARS = ('\r', '\n', '\0', ' ', '\t')


def misplaced_api_key_error() -> GenkitError:
    """The ``INVALID_ARGUMENT`` error for an API key found in config."""
    return GenkitError(
        status='INVALID_ARGUMENT',
        message=f'API key belongs in context.secrets, not config. {SECRETS_HINT}',
    )


def _has_key(bag: Mapping[str, object]) -> bool:
    return any(bag.get(slot) is not None for slot in _KEY_SLOTS)


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
    without either runs on the plugin's key, so apps can keep other secrets
    there. A key that is set but blank or not a string raises. Top-level
    context entries like ``context['api_key']`` are ignored so app-level
    auth context providers are untouched.

    Args:
        context: The action context (``ctx.context``).

    Returns:
        The trimmed key, or None when ``context.secrets`` has no key.

    Raises:
        GenkitError: ``INVALID_ARGUMENT`` when ``secrets`` is not a mapping,
            the key is not a usable string, or the background poll config
            (``context['config']``) carries a key.
    """
    if not context:
        return None
    # check_operation/cancel_operation fold their config into context['config'].
    poll_config = context.get('config')
    if isinstance(poll_config, Mapping) and _has_key(cast(Mapping[str, object], poll_config)):
        raise misplaced_api_key_error()

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


def reject_config_api_key(config: object) -> None:
    """Raise when a request config carries an API key.

    Checks ``api_key`` / ``apiKey`` on a config dict or model, on a model's
    undeclared fields, and inside ``extra``. A key in any of those would be
    traced, and a key in ``extra`` would also go to the provider as a body
    field while the call authenticates with the plugin's key.

    Args:
        config: ``request.config`` as the model received it.

    Raises:
        GenkitError: ``INVALID_ARGUMENT`` when a key is present.
    """
    if config is None:
        return
    bags: list[object]
    if isinstance(config, Mapping):
        top = cast(Mapping[str, object], config)
        bags = [top, top.get('extra')]
    else:
        if any(getattr(config, slot, None) is not None for slot in _KEY_SLOTS):
            raise misplaced_api_key_error()
        bags = [getattr(config, 'model_extra', None), getattr(config, 'extra', None)]
    for bag in bags:
        if isinstance(bag, Mapping) and _has_key(cast(Mapping[str, object], bag)):
            raise misplaced_api_key_error()
