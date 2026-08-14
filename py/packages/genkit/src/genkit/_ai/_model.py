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
    ModelConfig as ModelConfig,
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
ConfigArg: TypeAlias = Mapping[str, Any] | BaseModel


@dataclass(frozen=True, kw_only=True)
class ResolvedModel:
    """Concrete wire model name + config dict after veneer normalization."""

    name: str
    config: dict[str, Any]


def normalize_config(*, config: object) -> dict[str, Any]:
    """Convert a config object or dict into a mergeable dict.

    ``generate()`` runs both a ModelRef's default config and the per-call
    ``config=`` through this, then overlays them. Call-time keys win;
    ``None`` means don't send that field. Keys stay snake_case so
    ``ModelConfig(max_output_tokens=100)`` and
    ``config={'max_output_tokens': 200}`` hit the same key.
    """
    if config is None:
        return {}
    if isinstance(config, BaseModel):
        # Overlay needs the keys the caller wrote, including explicit None.
        # Skip fields they never set. Don't camelCase — maxOutputTokens
        # would miss a dict override on max_output_tokens.
        dumped = config.model_dump(exclude_unset=True, exclude_none=False, by_alias=False)
        # api_key is left out of JSON on purpose; copy it back so a
        # per-request key still reaches the plugin.
        for name in config.model_fields_set:
            if name not in dumped:
                dumped[name] = getattr(config, name)
        return dumped
    if isinstance(config, Mapping):
        return dict(cast(Mapping[str, Any], config))
    raise TypeError(f'Unsupported config type: {type(config).__name__}')


def resolve_model_name(
    *,
    model: str | None,
    registry: Registry,
    message: str = 'No model configured.',
) -> str:
    """Return an explicit model name or the registry default; error if neither exists."""
    name = model if model is not None else cast(str | None, registry.lookup_value('defaultModel', 'defaultModel'))
    if not name or not isinstance(name, str):
        raise GenkitError(status='INVALID_ARGUMENT', message=message)
    return name


def resolve_model_ref(*, model: ModelRef[Any], config: dict[str, Any]) -> ResolvedModel:
    """Merge a ModelRef's defaults with per-call config for the plugin request.

    Precedence (lowest to highest): ``ref.version``, ``ref.config``, then
    call-time ``config``. Each layer overwrites keys from the layers below.

    The merged dict is forwarded to the plugin as-is — no schema validation or
    key stripping at this layer. ``ModelRef`` typing catches config mistakes at
    construction; at call time we still pass unknown keys through so plugins can
    accept new provider options before the SDK schema is updated.

    An explicitly set ``None`` clears a value inherited from a lower layer.
    After merge, ``None`` values are removed so plugins see a missing key rather
    than ``null``.
    """
    merged: dict[str, Any] = {}
    if model.version is not None:
        merged['version'] = model.version
    if model.config is not None:
        merged.update(normalize_config(config=model.config))
    merged.update(config)
    merged = {k: v for k, v in merged.items() if v is not None}
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
