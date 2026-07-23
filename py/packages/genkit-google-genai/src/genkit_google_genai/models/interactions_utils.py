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
from typing import Any, cast

from pydantic import BaseModel

from genkit import GenkitError, Message
from genkit_google_genai._interactions.types import ClientOptions, ResponseModality

_CLIENT_OPTION_KEYS = frozenset({
    'api_key',
    'apiKey',
    'base_url',
    'baseUrl',
    'api_version',
    'apiVersion',
    'custom_headers',
    'customHeaders',
    'timeout',
    'experimental_debug_traces',
    'experimentalDebugTraces',
})


def get_api_key_from_env() -> str | None:
    """Read a Gemini API key from common environment variables."""
    return os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENAI_API_KEY')


def calculate_api_key(
    plugin_api_key: str | None,
    request_api_key: str | None,
) -> str:
    """Resolve the effective API key for an Interactions call."""
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


def config_as_dict(config: Any) -> dict[str, Any]:  # noqa: ANN401
    """Normalize model config to a plain dict."""
    if config is None:
        return {}
    if isinstance(config, BaseModel):
        return config.model_dump(by_alias=True, exclude_none=True)
    if isinstance(config, dict):
        return dict(config)
    return {}


def extract_version(model_name: str) -> str:
    """Return the bare model version from a namespaced model name."""
    if '/' in model_name:
        return model_name.split('/', 1)[1]
    return model_name


def downgrade_system_messages(messages: list[Message]) -> list[Message]:
    """Map system turns to user turns for agents that reject system instructions."""
    downgraded = [message.model_copy(deep=True) for message in messages]
    for message in downgraded:
        if message.role == 'system':
            message.role = 'user'
    return downgraded


def merge_client_options(
    base: ClientOptions,
    config: dict[str, Any],
) -> ClientOptions:
    """Apply per-request client overrides from model config."""
    merged = cast(ClientOptions, dict(base))
    if base_url := config.get('baseUrl') or config.get('base_url'):
        merged['base_url'] = str(base_url)
    if api_version := config.get('apiVersion') or config.get('api_version'):
        merged['api_version'] = str(api_version)
    if custom_headers := config.get('customHeaders') or config.get('custom_headers'):
        merged['custom_headers'] = dict(custom_headers)
    if isinstance(config.get('timeout'), (int, float)):
        merged['timeout'] = float(config['timeout'])
    return merged


def remove_client_option_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Drop client-only config keys before passthrough to the wire payload."""
    return {key: value for key, value in config.items() if key not in _CLIENT_OPTION_KEYS}


def client_options_for_operation(client_options: ClientOptions) -> ClientOptions:
    """Persist only reconstructable client settings on an Operation (DB10)."""
    return cast(ClientOptions, dict(client_options))


def response_modalities_from_config(
    config: dict[str, Any],
    *,
    default: list[ResponseModality] | None = None,
) -> list[ResponseModality] | None:
    """Convert config responseModalities to wire lowercase modalities."""
    raw = config.get('responseModalities') or config.get('response_modalities')
    if raw is None:
        return default
    return [str(item).lower() for item in raw]  # type: ignore[misc]
