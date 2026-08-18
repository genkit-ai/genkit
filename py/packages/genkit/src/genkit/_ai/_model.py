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

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, TypeAlias, cast, get_args, get_origin, get_type_hints

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
    ModelConfigDict,
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


@dataclass(frozen=True, kw_only=True)
class ResolvedModel:
    """Concrete wire model name + config dict after veneer normalization."""

    name: str
    config: dict[str, Any]


def config_field_names(schema: type[BaseModel]) -> dict[str, str]:
    """Map each field name and alias to the Python field name."""
    names: dict[str, str] = {}
    for name, field in schema.model_fields.items():
        names[name] = name
        if field.alias:
            names[field.alias] = name
    return names


def fold_config_aliases(*, config: dict[str, Any], schema: type[BaseModel]) -> dict[str, Any]:
    """Rewrite schema aliases to field names. Unknown keys stay as written."""
    names = config_field_names(schema)
    return {names.get(key, key): value for key, value in config.items()}


def overlay_config(*, layers: list[dict[str, Any]], schema: type[BaseModel]) -> dict[str, Any]:
    """Fold each layer, last layer wins, drop ``None``.

    ``maxOutputTokens`` and ``max_output_tokens`` are the same slot. Keys
    the schema does not know pass through.
    """
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update(fold_config_aliases(config=layer, schema=schema))
    return {key: value for key, value in merged.items() if value is not None}


def normalize_config(*, config: object) -> dict[str, Any]:
    """Dump a config object or dict. Does not fold or merge.

    Pydantic dumps the Python field names, including explicit ``None``.
    Dict keys stay as written. ``api_key`` is copied back when dump omits it.
    """
    if config is None:
        return {}
    if isinstance(config, BaseModel):
        dumped = config.model_dump(exclude_unset=True, exclude_none=False, by_alias=False)
        # api_key is left out of JSON on purpose; copy it back so a
        # per-request key still reaches the plugin.
        for name in config.model_fields_set:
            if name not in dumped:
                dumped[name] = getattr(config, name)
        return dumped
    if isinstance(config, Mapping):
        return dict(cast(Mapping[str, Any], config))
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=f'config is {type(config).__name__}, expected Mapping or BaseModel.',
    )


def resolve_model_arg(
    *,
    model: object | None,
    registry: Registry,
    message: str = 'No model configured.',
) -> ModelArg:
    """Return the explicit model or the registry default (name or ModelRef).

    An empty string is treated as omitted so ``model=os.getenv('MODEL')``
    still picks up the constructor default when the env var is unset.
    An empty constructor default is omitted the same way: not a model
    name, and not a type error.
    Anything else that is not a name or ModelRef is a hard error — a
    leftover int or action must not silently run the default model.
    """
    if isinstance(model, ModelRef):
        return cast(ModelArg, model)
    if isinstance(model, str) and model:
        return model
    if model is not None and model != '':
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'model is {type(model).__name__}, expected str or ModelRef.',
        )
    resolved = registry.lookup_value('defaultModel', 'defaultModel')
    if isinstance(resolved, ModelRef):
        return cast(ModelArg, resolved)
    if isinstance(resolved, str) and resolved:
        return resolved
    if resolved is not None and resolved != '':
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(f'defaultModel is {type(resolved).__name__}, expected str or ModelRef.'),
        )
    raise GenkitError(status='INVALID_ARGUMENT', message=message)


def resolve_model_name(
    *,
    model: object | None,
    registry: Registry,
    message: str = 'No model configured.',
) -> str:
    """Return a wire model name, unwrapping a ModelRef default if needed."""
    resolved = resolve_model_arg(model=model, registry=registry, message=message)
    return resolved.name if isinstance(resolved, ModelRef) else resolved


def resolve_call_model(
    *,
    model: object | None,
    config: object = None,
    registry: Registry,
    message: str = 'No model configured.',
) -> ResolvedModel:
    """Resolve a name or stored ModelRef plus call-time config.

    ``generate()`` / prompts with no ``model=`` still apply a registry
    default ref's version and config. The merged bag is a dict so overlay
    can happen; ModelRequest is what turns it back into an object.

    The outgoing bag has no ``None`` values — name or ref — so the plugin
    sees a missing key rather than null.
    """
    resolved = resolve_model_arg(model=model, registry=registry, message=message)
    normalized = normalize_config(config=config)
    if isinstance(resolved, ModelRef):
        return resolve_model_ref(model=resolved, config=normalized)
    return ResolvedModel(
        name=resolved,
        config={key: value for key, value in normalized.items() if value is not None},
    )


def resolve_model_ref(*, model: ModelRef[Any], config: dict[str, Any]) -> ResolvedModel:
    """Dump layers, overlay, return name + bag.

    Lowest to highest: ``ref.version``, dumped ``ref.config``, call
    ``config``. No validation — unknown keys pass through.
    """
    layers: list[dict[str, Any]] = []
    if model.version is not None:
        layers.append({'version': model.version})
    if model.config is not None:
        layers.append(normalize_config(config=model.config))
    layers.append(config)
    return ResolvedModel(
        name=model.name,
        config=overlay_config(layers=layers, schema=model.config_schema),
    )


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


def _check_request_annotation(name: str, fn: ModelFn) -> None:
    """Reject model fns whose request annotation is not a ModelRequest class.

    Unions like ``ModelRequest[X] | None`` are an antipattern: generate() never
    passes None, and a non-class annotation silently disables typed-request
    construction (the request falls back to the untyped carrier + rebuild).
    Fail fast at definition time with an actionable message instead.
    """
    try:
        hints = get_type_hints(fn, include_extras=True)
        params = list(inspect.signature(fn).parameters)
    except Exception:  # noqa: BLE001 - unresolvable annotations: let Action handle it
        return
    if not params:
        return
    ann = hints.get(params[0])
    if ann is None:
        return  # unannotated stays allowed
    if get_origin(ann) is Annotated:
        ann = get_args(ann)[0]
    # Hand-written ModelRequest subclasses also pass this check. That is
    # incidental — generate() only builds bare ModelRequest and
    # ModelRequest[Config], so subclassing is not a supported surface.
    if isinstance(ann, type) and issubclass(ann, ModelRequest):
        return
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=(
            f"Model '{name}': the request parameter must be annotated as ModelRequest "
            f'or ModelRequest[YourConfig], got {ann!r}. Unions such as '
            f"'ModelRequest[X] | None' are not allowed: generate() never passes None, "
            f'and non-class annotations disable typed-request construction.'
        ),
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
    _check_request_annotation(name, fn)
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
        for key in ('api_key', 'apiKey'):
            api_key = config_mapping.get(key)
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
