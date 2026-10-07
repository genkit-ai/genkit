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
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Annotated, Any, TypeAlias, cast, get_args, get_origin, get_type_hints

from pydantic import AliasChoices, BaseModel, ValidationError

from genkit._core._action import (
    Action,
    ActionKind,
    ActionRunContext,
    get_func_description,
)
from genkit._core._background import BackgroundAction
from genkit._core._error import GenkitError, RuntimeErrorReason
from genkit._core._logger import get_logger
from genkit._core._model import (
    Message,
    ModelConfig as ModelConfig,
    ModelConfigDict,
    ModelRef,
    ModelRefConfigT,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    config_type_path,
    get_basic_usage_stats,
    reject_config_api_key,
    text_from_content,
    text_from_message,
)
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._typing import ActionMetadata, GenerationCommonConfig, ModelInfo

# Type alias for model functions (must be async)
# Use ctx.send_chunk() for streaming
ModelFn = Callable[[ModelRequest, ActionRunContext], Awaitable[ModelResponse[Any]]]

logger = get_logger(__name__)

# Veneer-facing argument shapes. Internals resolve these into ResolvedModel.
# ModelArg is also the constructor default, stored as a registry value the
# Dev UI lists as JSON, so it stays a name or ModelRef.
ModelArg: TypeAlias = str | ModelRef[BaseModel]
# Call sites also take the action define_model returned on this instance.
CallModelArg: TypeAlias = str | ModelRef[BaseModel] | Action


@dataclass(frozen=True, kw_only=True)
class ResolvedModel:
    """Concrete wire model name + config dict after veneer normalization."""

    name: str
    config: dict[str, Any]
    # Config class this call accepts, if the model declared one.
    config_schema: type[BaseModel] | None = None


def python_config_schema(schema: object) -> type[BaseModel] | None:
    """The class a call's config is checked against, or None for no check."""
    return schema if isinstance(schema, type) and issubclass(schema, BaseModel) else None


def ref_defers_to_registered_class(schema: type[BaseModel] | None) -> bool:
    """True when the ref named plain ModelConfig, so the model's class is used."""
    return schema is ModelConfig or schema is GenerationCommonConfig


def config_field_names(schema: type[BaseModel]) -> dict[str, str]:
    """Map each field name and alias to the Python field name."""
    names: dict[str, str] = {}
    for name, field in schema.model_fields.items():
        names[name] = name
        if field.alias:
            names[field.alias] = name
        accepted = field.validation_alias
        if isinstance(accepted, str):
            names[accepted] = name
        elif isinstance(accepted, AliasChoices):
            for choice in accepted.choices:
                if isinstance(choice, str):
                    names[choice] = name
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
    Dict keys stay as written. Fields marked ``exclude=True`` are copied back.
    """
    if config is None:
        return {}
    if isinstance(config, BaseModel):
        dumped = config.model_dump(exclude_unset=True, exclude_none=False, by_alias=False)
        # a plugin can keep a client-only setting out of JSON with exclude=True;
        # copy it back so the setting the caller passed still reaches the plugin.
        for name in config.model_fields_set:
            if name not in dumped:
                dumped[name] = getattr(config, name)
        return dumped
    if isinstance(config, Mapping):
        return dict(cast(Mapping[str, Any], config))
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=f'config is {type(config).__name__}, expected Mapping or BaseModel.',
        reason=RuntimeErrorReason.INVALID_INPUT,
    )


def _name_or_ref(model: object) -> str | ModelRef[BaseModel] | None:
    """Unwrap a name or ModelRef. Other values stay None."""
    if isinstance(model, ModelRef):
        return cast(ModelRef[BaseModel], model)
    if isinstance(model, str) and model:
        return model
    return None


def _registered_action_name(*, action: Action, kind: ActionKind, registry: Registry) -> str:
    """The action's name if this registry holds this exact object under it.

    A name lookup alone would run whatever this registry has under that name:
    another Genkit instance's model, or a later define_model that replaced it.
    """
    if registry.registered_action(kind, action.name) is not action:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                f"model action '{action.name}' is not the one registered on this Genkit instance. "
                'Pass the object this instance returned, or the model name.'
            ),
            reason=RuntimeErrorReason.INVALID_INPUT,
        )
    return action.name


def _model_action_name(*, model: object, registry: Registry) -> str | None:
    """Name of a define_model action registered here; None if not an action."""
    if isinstance(model, BackgroundAction):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f"model is background model '{model.name}'. Pass it to generate_operation.",
            reason=RuntimeErrorReason.INVALID_INPUT,
        )
    if not isinstance(model, Action):
        return None
    if model.kind != ActionKind.MODEL:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f"model is {model.kind} action '{model.name}', expected a model.",
            reason=RuntimeErrorReason.INVALID_INPUT,
        )
    return _registered_action_name(action=model, kind=ActionKind.MODEL, registry=registry)


def background_model_name(*, model: BackgroundAction[Any], registry: Registry) -> str:
    """Name of a define_background_model result registered on this registry."""
    return _registered_action_name(action=model.start_action, kind=ActionKind.BACKGROUND_MODEL, registry=registry)


def resolve_model_arg(
    *,
    model: object | None,
    registry: Registry,
    message: str = 'No model configured.',
) -> str | ModelRef[BaseModel]:
    """Return the explicit model or the registry default (name or ModelRef).

    An empty string is treated as omitted so ``model=os.getenv('MODEL')``
    still picks up the constructor default when the env var is unset.
    An empty constructor default is omitted the same way: not a model
    name, and not a type error.
    A define_model action registered on this registry is the same as its
    name. Another instance's action, or one since replaced, is an error.
    Anything else that is not a name, ModelRef, or model action is a hard
    error — an int or other wrong type must not silently run the default model.
    """
    explicit = _name_or_ref(model)
    if explicit is not None:
        return explicit
    action_name = _model_action_name(model=model, registry=registry)
    if action_name is not None:
        return action_name
    if model is not None and model != '':
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(f'model is {type(model).__name__}, expected str, ModelRef, or a model action.'),
            reason=RuntimeErrorReason.INVALID_INPUT,
        )
    resolved = registry.lookup_value('defaultModel', 'defaultModel')
    default = _name_or_ref(resolved)
    if default is not None:
        name = default.name if isinstance(default, ModelRef) else default
        logger.debug('no model specified, using default model', model=name)
        return default
    if resolved is not None and resolved != '':
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(f'defaultModel is {type(resolved).__name__}, expected str or ModelRef.'),
            reason=RuntimeErrorReason.INVALID_INPUT,
        )
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=message,
        reason=RuntimeErrorReason.MODEL_NOT_FOUND,
    )


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


async def resolve_for_generate(
    *,
    model: object | None,
    config: object = None,
    registry: Registry,
    message: str = 'No model configured.',
) -> ResolvedModel:
    """Name, config bag, and the config class this generate will check against.

    A plugin class on a ModelRef is the class this call checks. Plain
    ``ModelConfig`` on a ref means the same as the model name: check
    against the class the model registered.
    """
    resolved = resolve_call_model(model=model, config=config, registry=registry, message=message)
    reject_config_api_key(resolved.config)
    if resolved.config_schema is not None and not ref_defers_to_registered_class(resolved.config_schema):
        return resolved
    action = await registry.resolve_model(resolved.name)
    raw = getattr(action, '_config_schema', None) if action is not None else None
    return replace(resolved, config_schema=python_config_schema(raw))


def config_schema_at_define(*, model: object | None, registry: Registry) -> tuple[str | None, type[BaseModel] | None]:
    """Name and class a define-time typed config is checked against.

    A ModelRef already has the class. A string name only sees an already-registered
    model — define_prompt is sync, so plugin resolve waits until generate.
    """
    omitted = model is None or model == ''
    if omitted:
        default = registry.lookup_value('defaultModel', 'defaultModel')
        if default is None or default == '':
            return None, None
    resolved = resolve_model_arg(model=model, registry=registry, message='No model configured.')
    if isinstance(resolved, ModelRef):
        return resolved.name, python_config_schema(resolved.config_schema)
    action = registry.registered_action(ActionKind.MODEL, resolved)
    raw = getattr(action, '_config_schema', None) if action is not None else None
    return resolved, python_config_schema(raw)


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
        config_schema=python_config_schema(model.config_schema),
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


def claims_long_running(*, model_options: dict[str, object]) -> bool:
    """True when this model metadata asked for a background job."""
    supports = model_options.get('supports')
    if isinstance(supports, dict):
        supports_dict = cast(dict[str, object], supports)
        return bool(supports_dict.get('longRunning'))
    return False


def model(
    name: str,
    fn: ModelFn,
    *,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    info: ModelInfo | None = None,
    description: str | None = None,
) -> Action:
    """Build a model action without registering it.

    Plugin ``init`` / ``resolve`` return this. ``define_model`` registers it.
    The config class stays on the action so ``generate(model='name', config=)``
    can isinstance-check a Pydantic instance against a string model name.
    """
    model_options: dict[str, object] = {}

    if info:
        model_options.update(info.model_dump(by_alias=True, exclude_none=True))

    if metadata and 'model' in metadata:
        existing = metadata['model']
        if isinstance(existing, dict):
            existing_dict = cast(dict[str, object], existing)
            for key, value in existing_dict.items():
                if isinstance(key, str) and key not in model_options:
                    model_options[key] = value

    if 'label' not in model_options or not model_options['label']:
        model_options['label'] = name

    if claims_long_running(model_options=model_options):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f"define_model '{name}' cannot set longRunning. Use define_background_model.",
        )

    if config_schema:
        model_options['customOptions'] = to_json_schema(config_schema)

    model_meta: dict[str, object] = metadata.copy() if metadata else {}
    model_meta['model'] = model_options

    return Action(
        kind=ActionKind.MODEL,
        name=name,
        fn=fn,
        metadata=model_meta,
        description=get_func_description(fn, description),
        config_schema=config_schema,
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
    action = model(
        name,
        fn,
        config_schema=config_schema,
        metadata=metadata,
        info=info,
        description=description,
    )
    registry.register_action_from_instance(action)
    return action


def assert_correct_config_class(
    *,
    config: object,
    schema: type[BaseModel] | None,
    model: str | None = None,
) -> None:
    """A typed config object has to belong to the model this call hits.

    Dicts stay legal. Omit / ``None`` skip this. A model with no Python
    class (JSON-only or unset) cannot be checked.
    """
    if not isinstance(config, BaseModel):
        return
    if schema is None or isinstance(config, schema):
        return
    body = f'config must be {config_type_path(schema)} or a mapping, got {config_type_path(type(config))}'
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=f'{model}: {body}' if model else body,
        reason=RuntimeErrorReason.INVALID_INPUT,
    )


def check_config_dict(*, config: object, schema: type[BaseModel] | None, model: str) -> None:
    """A dict config has to fit the model's class before anything is sent.

    Layers merge by top-level key, so a missing top-level field is fine
    here — another layer may supply it. A nested object is sent whole, so
    a missing field inside one raises. ``None`` means "clear the default"
    and isn't checked.
    """
    if schema is None or not isinstance(config, Mapping):
        return
    layer = {key: value for key, value in cast(Mapping[str, Any], config).items() if value is not None}
    try:
        schema.model_validate(layer)
    except ValidationError as e:
        problems = [err for err in e.errors() if not (err['type'] == 'missing' and len(err['loc']) == 1)]
        if not problems:
            return
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'{model}: {_describe_config_problems(problems, layer=layer, schema=schema)}',
            reason=RuntimeErrorReason.INVALID_INPUT,
            cause=e,
        ) from e


def _describe_config_problems(
    problems: Sequence[Mapping[str, Any]], *, layer: Mapping[str, Any], schema: type[BaseModel]
) -> str:
    # pydantic binds one spelling of a setting and calls the other unknown;
    # the caller didn't misspell anything, they wrote the setting twice.
    names = config_field_names(schema)
    repeated: dict[str, list[str]] = {}
    unknown: list[str] = []
    for err in problems:
        if err['type'] != 'extra_forbidden':
            continue
        key = _config_path(err['loc'])
        field = names.get(key) if len(err['loc']) == 1 else None
        spellings = [k for k in layer if field and names.get(k) == field]
        if field and len(spellings) > 1:
            repeated[field] = sorted(spellings, key=lambda k: k != field)
        else:
            unknown.append(key)
    parts = [f'{_join_words(spellings)} are the same setting; pass one' for spellings in repeated.values()]
    if unknown:
        keys = ', '.join(repr(key) for key in unknown)
        noun = 'key' if len(unknown) == 1 else 'keys'
        parts.append(f"unknown config {noun} {keys}; put provider-only settings in config['extra']")
    parts.extend(
        f'config {_config_path(err["loc"])!r}: {err["msg"]}' for err in problems if err['type'] != 'extra_forbidden'
    )
    return '; '.join(parts)


def _join_words(words: list[str]) -> str:
    """`a and b`, or `a, b, and c` for three or more."""
    if len(words) <= 2:
        return ' and '.join(words)
    return f'{", ".join(words[:-1])}, and {words[-1]}'


def _config_path(loc: tuple[int | str, ...]) -> str:
    return '.'.join(str(part) for part in loc)


def check_call_config(*, config: object, schema: type[BaseModel] | None, model: str) -> None:
    """Call-time config check: a typed object's class and a dict's keys and values."""
    assert_correct_config_class(config=config, schema=schema, model=model)
    check_config_dict(config=config, schema=schema, model=model)
