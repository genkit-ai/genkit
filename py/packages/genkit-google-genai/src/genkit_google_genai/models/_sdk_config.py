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

from typing import Any

from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from genkit import GenkitError


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
    """Put leftover keys on extra_body so a newly supported field can reach the API.

    Family schemas keep extras. The typed google-genai request rejects unknowns,
    so leftovers ride on the HTTP body instead of being dropped or rejected here.
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
