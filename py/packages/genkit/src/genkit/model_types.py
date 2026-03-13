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

"""Python SDK-specific model type extensions.

Extends auto-generated core model types with Python SDK runtime fields and
shared helpers for plugin authors.
"""

from collections.abc import Mapping
from typing import cast

from pydantic import Field

from genkit.core.typing import GenerationCommonConfig as CoreGenerationCommonConfig


class GenerationCommonConfig(CoreGenerationCommonConfig):
    """Common generation config with Python SDK runtime extensions."""

    api_key: str | None = Field(
        default=None,
        alias='apiKey',
        description='API Key to use for the model call, overrides API key provided in plugin config.',
    )


def get_request_api_key(config: GenerationCommonConfig | Mapping[str, object] | object | None) -> str | None:
    """Extract a request-scoped API key from config.

    Supports both typed config objects and dict payloads with either snake_case
    or camelCase keys.
    """
    if config is None:
        return None

    if isinstance(config, GenerationCommonConfig):
        return config.api_key

    if isinstance(config, Mapping):
        config_mapping = cast(Mapping[str, object], config)
        api_key = config_mapping.get('api_key') or config_mapping.get('apiKey')
        if isinstance(api_key, str) and api_key:
            return api_key
    else:
        # Defensive fallback for plugin-specific config classes that inherit from
        # GenerationCommonConfig or expose an api_key attribute.
        api_key_attr = getattr(config, 'api_key', None)
        if isinstance(api_key_attr, str) and api_key_attr:
            return api_key_attr

    return None


def get_effective_api_key(
    config: GenerationCommonConfig | Mapping[str, object] | object | None,
    plugin_api_key: str | None,
) -> str | None:
    """Resolve effective API key using request-over-plugin precedence."""
    return get_request_api_key(config) or plugin_api_key


__all__ = ['GenerationCommonConfig', 'get_request_api_key', 'get_effective_api_key']
