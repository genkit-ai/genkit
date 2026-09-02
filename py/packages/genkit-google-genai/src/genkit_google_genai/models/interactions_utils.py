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

"""Shared helpers for Google AI Interactions-backed models."""

from __future__ import annotations

import os
from typing import Any

from genkit import GenkitError
from genkit_google_genai._interactions.options import ClientOptions
from genkit_google_genai.models._routing import strip_ref_prefixes
from genkit_google_genai.models._secrets import context_api_key

# Snake_case: callers dump config with by_alias=False before we strip these.
CLIENT_OPTION_KEYS = frozenset({
    'api_key',
    'base_url',
    'api_version',
    'custom_headers',
    'timeout',
    'experimental_debug_traces',
})


def get_api_key_from_env() -> str | None:
    """Read a Gemini API key from common environment variables."""
    return os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENAI_API_KEY')


def calculate_api_key(
    plugin_api_key: str | None,
    request_api_key: str | None,
) -> str:
    """Resolve the plugin/env API key when context.secrets did not supply one."""
    api_key = request_api_key or plugin_api_key or get_api_key_from_env()
    if not api_key:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=(
                'Please pass in the API key or set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.\n'
                'For more details see https://genkit.dev/docs/plugins/google-genai/'
            ),
        )
    return api_key


def api_key_for_context(context: dict[str, Any], plugin_api_key: str | None) -> str:
    """Tenant key from context.secrets, else the plugin/env key."""
    return context_api_key(context) or calculate_api_key(plugin_api_key, None)


def extract_version(model_name: str) -> str:
    """Bare model id for the wire. Pasted action keys still have models/googleai/ on them."""
    return strip_ref_prefixes(model_name)


def remove_client_option_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Drop client-only config keys before passthrough to the wire payload."""
    return {key: value for key, value in config.items() if key not in CLIENT_OPTION_KEYS}


def client_overrides_from_config(
    *,
    base_url: str | None = None,
    api_version: str | None = None,
    timeout: float | None = None,
    custom_headers: dict[str, str] | None = None,
) -> ClientOptions:
    """Lift per-request transport knobs out of a model config into ClientOptions."""
    return ClientOptions(
        base_url=base_url,
        api_version=api_version,
        timeout=timeout,
        custom_headers=custom_headers,
    )


def partition_keys(
    payload: dict[str, Any],
    *groups: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """Split payload into one dict per key group, then a remainder of leftovers.

    Does not mutate payload. Keys listed in any group are claimed for that group
    (or dropped from the remainder even when absent), so callers can peel a
    dumped model config into agent_config / create options / tool fields /
    passthrough extras without colliding.
    """
    claimed: set[str] = set()
    for keys in groups:
        claimed.update(keys)
    parts = tuple({key: payload[key] for key in keys if key in payload} for keys in groups)
    remainder = {key: value for key, value in payload.items() if key not in claimed}
    return (*parts, remainder)


def require_interaction_steps(steps: list[Any]) -> list[Any]:
    """Reject an empty Interactions input before we pay for a round trip."""
    if not steps:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='Missing input.',
        )
    return steps


def lowercase_choice(value: object) -> object:
    """Accept AUTO/NONE-style labels the same as auto/none."""
    return value.lower() if isinstance(value, str) else value


def lowercase_choice_list(value: object) -> object:
    """Lowercase each string in a response_modalities list."""
    if not isinstance(value, list):
        return value
    return [item.lower() if isinstance(item, str) else item for item in value]
