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
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from pydantic import BaseModel

from genkit._core._action import (
    Action,
    ActionKind,
    ActionRunContext,
    get_func_description,
)
from genkit._core._error import GenkitError
from genkit._core._model import (
    Message,
    ModelConfig,
    ModelRef,
    ModelRefConfigT,
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

# Veneer-facing argument shapes. Internals resolve these into ResolvedModel.
ModelArg: TypeAlias = str | ModelRef[BaseModel]
ConfigArg: TypeAlias = BaseModel | Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class ResolvedModel:
    """Concrete wire model name + config dict after veneer normalization."""

    name: str
    config: dict[str, Any]


def normalize_config(*, config: object) -> dict[str, Any]:
    """Turn a Pydantic config, mapping, or None into a plain dict."""
    if config is None:
        return {}
    if isinstance(config, BaseModel):
        return config.model_dump(exclude_unset=True)
    return dict(cast(Mapping[str, Any], config))


def concrete_config_schema(*candidates: object) -> type[BaseModel] | None:
    """Return the first concrete BaseModel subclass among candidates."""
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, BaseModel) and candidate is not BaseModel:
            return candidate
    return None


def resolve_model_name(
    *,
    model: str | None,
    registry: Registry,
    message: str = 'No model configured.',
) -> str:
    """Return an explicit model name or the registry default; error if neither exists."""
    name = model if model is not None else cast(str | None, registry.lookup_value('defaultModel', 'defaultModel'))
    if not name:
        raise GenkitError(status='INVALID_ARGUMENT', message=message)
    return name


def resolve_model_ref(*, model: ModelRef[Any], config: dict[str, Any]) -> ResolvedModel:
    """Merge call-time config over a ModelRef's default config into a wire ResolvedModel."""
    merged: dict[str, Any] = {}

    # 1. Start with defaults defined on the ModelRef (e.g. temperature=0.7)
    if model.config is not None:
        merged.update(normalize_config(config=model.config))

    # 2. Call-time config overrides defaults (e.g. temperature=0.2, top_k=0.9)
    merged.update(config)

    # 3. Find concrete Pydantic schema (e.g. GeminiConfig or ModelConfig)
    schema = concrete_config_schema(
        model.config_schema,
        type(model.config) if model.config is not None else None,
    )

    # 4. Validate types against Pydantic schema & omit unset None fields
    if schema is not None and merged:
        return ResolvedModel(
            name=model.name,
            config=schema.model_validate(merged).model_dump(exclude_unset=True),
        )

    # 5. Fallback for untyped/raw models
    return ResolvedModel(name=model.name, config=merged)


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
    *,
    config_schema: type[ModelRefConfigT],
    namespace: str | None = None,
    info: ModelInfo | None = None,
    version: str | None = None,
    config: ModelRefConfigT | None = None,
) -> ModelRef[ModelRefConfigT]:
    """Create a ModelRef, optionally prefixing name with namespace."""
    final_name = f'{namespace}/{name}' if namespace and not name.startswith(f'{namespace}/') else name

    return ModelRef(
        name=final_name,
        config_schema=config_schema,
        info=info,
        version=version,
        config=config,
    )


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
        model_options.update(info.model_dump(by_alias=True, exclude_none=True))

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
