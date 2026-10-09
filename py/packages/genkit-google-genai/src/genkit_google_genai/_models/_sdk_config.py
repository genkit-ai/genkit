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

"""Dump a family config and translate SDK ValidationErrors into Genkit errors."""

import copy
from typing import Any, cast

from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from genkit import GenkitError


def copy_http_options(opts: genai_types.HttpOptions) -> genai_types.HttpOptions:
    """Copy ``opts`` so plugin edits never reach the caller's object.

    Dict fields get fresh containers. Transport objects (``httpx_client``,
    ``httpx_async_client``, ``aiohttp_client``) stay shared: they hold locks
    and sockets that can't be deep-copied, and the caller expects every
    request to go through them.
    """
    update: dict[str, Any] = {}
    for field in ('headers', 'client_args', 'async_client_args'):
        value = getattr(opts, field)
        if value is not None:
            update[field] = dict(value)
    if opts.extra_body is not None:
        update['extra_body'] = copy.deepcopy(opts.extra_body)
    if opts.retry_options is not None:
        update['retry_options'] = opts.retry_options.model_copy(deep=True)
    return opts.model_copy(update=update)


def unexpected_config_error(*, action_name: str) -> GenkitError:
    """Fail when the dump leaf sees a config Action did not produce."""
    return GenkitError(
        status='INVALID_ARGUMENT',
        message=f'{action_name}: config must be the family schema instance',
    )


def dump_family_config(
    *,
    config: object,
    expected_type: type[BaseModel],
    action_name: str,
) -> dict[str, Any] | None:
    """Dump a typed family config to a snake_case dict for the SDK.

    Action already turned the caller's config into the family instance. A dict
    here means that did not happen, so we fail rather than generate with no knobs.
    """
    if config is None:
        return None
    if not isinstance(config, expected_type):
        raise unexpected_config_error(action_name=action_name)
    dumped = config.model_dump(exclude_none=True, by_alias=False)
    return dumped or None


def split_sdk_fields(
    dumped: dict[str, Any],
    sdk_type: type[BaseModel],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a dump into fields the SDK type knows and leftover extras."""
    fields = sdk_type.model_fields
    known = {key: value for key, value in dumped.items() if key in fields}
    leftovers = {key: value for key, value in dumped.items() if key not in fields}
    return known, leftovers


def attach_leftovers(
    config: Any,  # noqa: ANN401
    leftovers: dict[str, Any],
    *,
    nest: str,
) -> Any:  # noqa: ANN401
    """Put leftover keys on extra_body so a declared field the SDK doesn't type yet still reaches the API.

    The typed google-genai request rejects unknowns, so leftovers ride on the
    HTTP body instead of being dropped or rejected here.
    """
    if not leftovers:
        return config
    http = config.http_options or genai_types.HttpOptions()
    extra = dict(http.extra_body or {})
    bucket = dict(extra.get(nest) or {})
    bucket.update(leftovers)
    extra[nest] = bucket
    http.extra_body = extra
    config.http_options = http
    return config


# Body fields Genkit builds from the request, compared the way google-genai's
# extra_body merge aligns keys (underscores dropped, lowercased), so
# `Contents` or `SYSTEM_INSTRUCTION` can't slip past.
GEMINI_MANAGED_BODY_FIELDS = frozenset({'contents', 'systeminstruction', 'tools', 'toolconfig', 'cachedcontent'})
# generationConfig fields Genkit sets from the request's output config.
GEMINI_MANAGED_GENERATION_FIELDS = frozenset({'responseschema', 'responsejsonschema', 'responsemimetype'})
# Veo's predictLongRunning body: `instances` carries the prompt and input media.
VEO_MANAGED_BODY_FIELDS = frozenset({'instances'})


def _wire_key(key: str) -> str:
    return key.replace('_', '').lower()


def _managed_extra_field(
    extra: dict[str, Any], *, body_fields: frozenset[str], generation_fields: frozenset[str]
) -> str | None:
    for key, value in extra.items():
        wire = _wire_key(key)
        if wire in body_fields:
            return key
        if wire == 'generationconfig' and isinstance(value, dict):
            for inner in cast(dict[str, Any], value):
                if _wire_key(inner) in generation_fields:
                    return f'{key}.{inner}'
    return None


def _existing_wire_key(target: dict[str, Any], key: str) -> str | None:
    """The key already in ``target`` that google-genai would treat as the same name."""
    wire = _wire_key(key)
    for existing in target:
        if _wire_key(existing) == wire:
            return existing
    return None


def _deep_merge(target: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(target)
    for key, value in update.items():
        existing = _existing_wire_key(merged, key)
        if existing is None:
            merged[key] = value
            continue
        current = merged[existing]
        if isinstance(current, dict) and isinstance(value, dict):
            merged[existing] = _deep_merge(cast(dict[str, Any], current), cast(dict[str, Any], value))
        else:
            merged[existing] = value
    return merged


def attach_config_extra(
    config: Any,  # noqa: ANN401
    extra: dict[str, Any] | None,
    *,
    action_name: str,
    managed_body_fields: frozenset[str],
    managed_generation_fields: frozenset[str] = frozenset(),
) -> Any:  # noqa: ANN401
    """Send ``config.extra`` verbatim at the top level of the request body.

    Keys are wire names (``labels``, ``generationConfig``, ...). google-genai
    merges ``extra_body`` into the built request recursively, so
    ``{'generationConfig': {'newKnob': 1}}`` adds one key instead of replacing
    the block, and a colliding leaf wins over the declared field. Fields
    Genkit builds from the request (``managed_body_fields``, and
    ``managed_generation_fields`` under ``generationConfig``) are rejected.
    """
    if not extra:
        return config
    field = _managed_extra_field(extra, body_fields=managed_body_fields, generation_fields=managed_generation_fields)
    if field is not None:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                f'{action_name}: extra field {field!r} is built by Genkit from the request '
                'and cannot be set from config'
            ),
        )
    http = config.http_options or genai_types.HttpOptions()
    http.extra_body = _deep_merge(dict(http.extra_body or {}), extra)
    config.http_options = http
    return config


def keep_client_extra_body(config: Any, client_http_options: object) -> Any:  # noqa: ANN401
    """Layer a request's extra_body over the plugin's instead of replacing it.

    google-genai swaps in a per-request ``extra_body`` wholesale, so without
    this a request carrying ``extra`` would drop ``GoogleAI(http_options=
    HttpOptions(extra_body=...))``. The request's keys win.
    """
    request_body = config.http_options.extra_body if config.http_options else None
    client_body = getattr(client_http_options, 'extra_body', None)
    if request_body and isinstance(client_body, dict):
        config.http_options.extra_body = _deep_merge(cast(dict[str, Any], client_body), request_body)
    return config


def sdk_config_error(*, action_name: str, error: ValidationError) -> GenkitError:
    """Name the action and the field the SDK rejected on a known typed field."""
    loc = ()
    errors = error.errors()
    if errors:
        loc = errors[0].get('loc') or ()
    key = str(loc[0]) if loc else 'config'
    return GenkitError(
        status='INVALID_ARGUMENT',
        message=f'{action_name}: invalid config field {key}',
        cause=error,
    )
