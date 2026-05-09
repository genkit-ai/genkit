# Copyright 2025 Google LLC
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

"""Model type definitions for the Genkit framework."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from pydantic import BaseModel

from genkit._core._action import (
    Action,
    ActionKind,
    ActionRunContext,
    get_func_description,
)
from genkit._core._model import (
    Message,
    ModelConfig,
    ModelMiddleware,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    get_basic_usage_stats,
    text_from_content,
    text_from_message,
)
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._typing import ActionMetadata, ModelInfo

# Type alias for model functions (must be async)
# Use ctx.send_chunk() for streaming
ModelFn = Callable[[ModelRequest, ActionRunContext], Awaitable[ModelResponse[Any]]]


def model_action_metadata(
    name: str,
    info: dict[str, object] | None = None,
    config_schema: type | dict[str, Any] | None = None,
) -> ActionMetadata:
    """Create ActionMetadata for a model action."""
    info = info if info is not None else {}
    return ActionMetadata(
        action_type=ActionKind.MODEL,
        name=name,
        input_json_schema=to_json_schema(ModelRequest),
        output_json_schema=to_json_schema(ModelResponse),
        metadata={'model': {**info, 'customOptions': to_json_schema(config_schema) if config_schema else None}},
    )


def model_ref(
    name: str,
    namespace: str | None = None,
    info: ModelInfo | None = None,
    version: str | None = None,
    config: dict[str, object] | None = None,
) -> ModelRef:
    """Create a ModelRef, optionally prefixing name with namespace."""
    # Logic: if (options.namespace && !name.startsWith(options.namespace + '/'))
    final_name = f'{namespace}/{name}' if namespace and not name.startswith(f'{namespace}/') else name

    return ModelRef(name=final_name, info=info, version=version, config=config)


def define_model(
    registry: Registry,
    name: str,
    fn: ModelFn,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    info: ModelInfo | None = None,
    description: str | None = None,
) -> Action:
    """Register a custom model action."""
    # Build model options dict
    model_options: dict[str, object] = {}

    # Start with info if provided
    if info:
        model_options.update(info.model_dump())

    # Check if metadata has model info
    if metadata and 'model' in metadata:
        existing = metadata['model']
        if isinstance(existing, dict):
            existing_dict = cast(dict[str, object], existing)
            for key, value in existing_dict.items():
                if isinstance(key, str) and key not in model_options:
                    model_options[key] = value

    # Default label to name if not set
    if 'label' not in model_options or not model_options['label']:
        model_options['label'] = name

    # Add config schema if provided
    if config_schema:
        model_options['customOptions'] = to_json_schema(config_schema)

    # Build the final metadata dict
    model_meta: dict[str, object] = metadata.copy() if metadata else {}
    model_meta['model'] = model_options

    model_description = get_func_description(fn, description)
    return registry.register_action(
        name=name,
        kind=ActionKind.MODEL,
        fn=fn,
        metadata=model_meta,
        description=model_description,
    )


# =============================================================================
# Model config types (from model_types.py)
# =============================================================================


def get_request_api_key(config: Mapping[str, object] | ModelConfig | object | None) -> str | None:
    """Extract API key from config (snake_case or camelCase)."""
    if config is None:
        return None

    if isinstance(config, ModelConfig):
        return config.api_key

    if isinstance(config, Mapping):
        config_mapping = cast(Mapping[str, object], config)
        api_key = config_mapping.get('api_key')
        if isinstance(api_key, str) and api_key:
            return api_key
    else:
        # Defensive fallback for plugin-specific config classes that inherit from
        # ModelConfig or expose an api_key attribute.
        api_key_attr = getattr(config, 'api_key', None)
        if isinstance(api_key_attr, str) and api_key_attr:
            return api_key_attr

    return None


def get_effective_api_key(
    config: Mapping[str, object] | ModelConfig | object | None,
    plugin_api_key: str | None,
) -> str | None:
    """Return request API key if set, otherwise plugin API key."""
    return get_request_api_key(config) or plugin_api_key
