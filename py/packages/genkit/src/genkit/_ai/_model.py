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

from pydantic import BaseModel, ValidationError

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
    check_config_dict as check_config_dict,
    config_field_names as config_field_names,
    config_type_path,
    fold_config_aliases as fold_config_aliases,
    get_basic_usage_stats,
    normalize_config as normalize_config,
    overlay_config as overlay_config,
    reject_config_api_key,
    text_from_content,
    text_from_message,
    validate_config_dict,
)
from genkit._core._registry import Registry
from genkit._core._schema import custom_options_schema, to_json_schema
from genkit._core._typing import ActionMetadata, GenerationCommonConfig, ModelInfo, Operation

# Type alias for model functions (must be async)
# Use ctx.send_chunk() for streaming
ModelFn = Callable[[ModelRequest, ActionRunContext], Awaitable[ModelResponse[Any]]]

logger = get_logger(__name__)

# Veneer-facing argument shapes. Internals resolve these into ResolvedModel.
# ModelArg is also the constructor default, stored as a registry value the
# Dev UI lists as JSON, so it stays a name or ModelRef.
ModelArg: TypeAlias = str | ModelRef[BaseModel]


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


def background_model_name(*, model: BackgroundAction, registry: Registry) -> str:
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
    schema: type[BaseModel] | None = None,
) -> ResolvedModel:
    """Resolve a name or stored ModelRef plus call-time config.

    ``generate()`` / prompts with no ``model=`` still apply a registry
    default ref's version and config. The merged bag is a dict so overlay
    can happen; ModelRequest is what turns it back into an object.

    An explicit ``None`` is a value and reaches the model. ``schema`` is the
    class aliases fold to (see call_config_class); a ref's own class when
    omitted.
    """
    resolved = resolve_model_arg(model=model, registry=registry, message=message)
    if isinstance(resolved, ModelRef):
        return resolve_model_ref(model=resolved, config=normalize_config(config=config), schema=schema)
    return ResolvedModel(name=resolved, config=layer_call_config(call=config, schema=schema))


def layer_call_config(
    *,
    call: object,
    version: str | None = None,
    ref_config: object = None,
    schema: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """``ref.version < ref.config < call``: the merge generate, embed and evaluate share.

    Each layer is the fields it set (normalize_config, deep-copied), folded
    to the keys ``schema`` accepts. An explicit ``None`` is a value: it wins
    over lower layers and goes to validation.
    """
    layers: list[dict[str, Any]] = []
    if version is not None:
        layers.append({'version': version})
    if ref_config is not None:
        layers.append(normalize_config(config=ref_config))
    layers.append(normalize_config(config=call))
    return overlay_config(layers=layers, schema=schema)


def call_config_class(
    *,
    name: str,
    kind: str,
    ref_schema: type[BaseModel] | None,
    action_schema: type[BaseModel] | None,
) -> type[BaseModel] | None:
    """The class a call's config is checked against and folded to.

    The definition owns the class. A ref's config_schema only types the call
    site, so it has to name the action's class (plain ModelConfig /
    GenerationCommonConfig on a model ref defer to it). With no action class,
    the ref's class is the only one there is.
    """
    if ref_schema is None or ref_defers_to_registered_class(ref_schema):
        return action_schema
    if action_schema is None or ref_schema is action_schema:
        return ref_schema
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=(
            f"{kind} '{name}' takes config {config_type_path(action_schema)}, but the ref's config_schema is "
            f'{config_type_path(ref_schema)}. Use {action_schema.__name__} on the ref, or pass the name.'
        ),
        reason=RuntimeErrorReason.INVALID_INPUT,
    )


def resolve_call_config(
    *,
    name: str,
    kind: str,
    action_schema: type[BaseModel] | None,
    call: object,
    ref_schema: type[BaseModel] | None = None,
    version: str | None = None,
    ref_config: object = None,
) -> tuple[type[BaseModel] | None, dict[str, Any]]:
    """The class a call checks against and its layered config, for embed and evaluate.

    call_config_class, check_call_config on the call's config,
    layer_call_config, then check_merged_config: the same steps generate takes.
    """
    schema = call_config_class(name=name, kind=kind, ref_schema=ref_schema, action_schema=action_schema)
    check_call_config(config=call, schema=schema, model=name)
    config = layer_call_config(call=call, version=version, ref_config=ref_config, schema=schema)
    check_merged_config(config=config, schema=schema, model=name)
    return schema, config


async def resolve_for_generate(
    *,
    model: object | None,
    config: object = None,
    registry: Registry,
    message: str = 'No model configured.',
) -> ResolvedModel:
    """Name, config bag, and the config class this generate will check against.

    The model's definition-time class is the class this call checks. A
    ModelRef's config_schema must name it (see call_config_class), and the
    ref's layers fold aliases to it.
    """
    arg = resolve_model_arg(model=model, registry=registry, message=message)
    name = arg.name if isinstance(arg, ModelRef) else arg
    action = await registry.resolve_model(name)
    schema = call_config_class(
        name=name,
        kind='model',
        ref_schema=python_config_schema(arg.config_schema) if isinstance(arg, ModelRef) else None,
        action_schema=python_config_schema(action.config_schema) if action is not None else None,
    )
    resolved = resolve_call_model(model=arg, config=config, registry=registry, message=message, schema=schema)
    reject_config_api_key(resolved.config)
    return replace(resolved, config_schema=schema)


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
    return resolved, python_config_schema(action.config_schema) if action is not None else None


def resolve_model_ref(
    *, model: ModelRef[Any], config: dict[str, Any], schema: type[BaseModel] | None = None
) -> ResolvedModel:
    """Dump layers, overlay, return name + bag.

    Lowest to highest: ``ref.version``, dumped ``ref.config``, call
    ``config``. No validation — unknown keys pass through. Aliases fold to
    ``schema``, the ref's own class when omitted.
    """
    return ResolvedModel(
        name=model.name,
        config=layer_call_config(
            call=config,
            version=model.version,
            ref_config=model.config,
            schema=schema or model.config_schema,
        ),
        config_schema=python_config_schema(model.config_schema),
    )


def model_action_metadata(
    name: str,
    info: dict[str, object] | None = None,
    config_schema: type | dict[str, Any] | None = None,
    *,
    background: bool = False,
) -> ActionMetadata:
    """Create ActionMetadata for a model action.

    With ``background=True`` the metadata describes a background model's start
    action: it takes a ``ModelRequest`` and returns an ``Operation`` to poll.
    """
    info = info if info is not None else {}
    model = {**info, 'customOptions': custom_options_schema(config_schema) if config_schema else None}
    if not background:
        return ActionMetadata(
            action_type=ActionKind.MODEL,
            name=name,
            input_json_schema=to_json_schema(ModelRequest),
            output_json_schema=to_json_schema(ModelResponse),
            metadata={'model': model},
        )
    config_class = python_config_schema(config_schema)
    # A runtime class, not a type expression: the schema is ModelRequest[ThatClass].
    request = cast(Any, ModelRequest)[config_class] if config_class is not None else ModelRequest
    return ActionMetadata(
        action_type=ActionKind.BACKGROUND_MODEL,
        name=name,
        input_json_schema=to_json_schema(request),
        output_json_schema=to_json_schema(Operation),
        metadata={'model': model, 'type': 'background-model'},
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
    It comes from ``config_schema`` or the fn's ``ModelRequest[Cfg]``
    annotation; given both, they must be the same class.
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
        model_options['customOptions'] = custom_options_schema(config_schema)

    model_meta: dict[str, object] = metadata.copy() if metadata else {}
    model_meta['model'] = model_options

    action = Action(
        kind=ActionKind.MODEL,
        name=name,
        fn=fn,
        metadata=model_meta,
        description=get_func_description(fn, description),
        config_schema=config_schema,
    )
    # Annotation only: the Dev UI form comes from ModelRequest[Cfg].
    if 'customOptions' not in model_options and action.config_schema is not None:
        model_options['customOptions'] = custom_options_schema(action.config_schema)
    return action


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


def check_call_config(*, config: object, schema: type[BaseModel] | None, model: str) -> None:
    """Call-time config check: a typed object's class and a dict's keys and values."""
    assert_correct_config_class(config=config, schema=schema, model=model)
    check_config_dict(config=config, schema=schema, model=model)


def check_merged_config(*, config: Mapping[str, Any], schema: type[BaseModel] | None, model: str) -> None:
    """The caller's layers together have to fit the class, class defaults filling the rest.

    Runs before the action, so a missing required field raises
    INVALID_ARGUMENT here, with the same message the action boundary uses,
    instead of a failed response from inside generate.
    """
    if schema is not None:
        validate_config_dict(config=config, schema=schema, label=model)


def _config_values(config: BaseModel) -> dict[str, Any]:
    """``config`` as plain data, nested models included, in the shape validation takes.

    ``BaseModel.model_dump`` skips GenkitModel's ``exclude_none`` default (an
    explicit ``None`` is a value too) and its fallback serializer, so a value
    of the wrong type comes back as itself and fails validation.
    """
    return BaseModel.model_dump(config, by_alias=True, round_trip=True, exclude_none=False, warnings=False)


class MiddlewareConfigCheck:
    """Checks ``request.config`` each time a middleware hands the request to the next layer.

    generate turns the call's config into the model's class once, so every
    middleware and the model see the same shape. Middleware changes fields on
    that object. A dict, ``None``, or another class put back into
    ``request.config`` would reach inner layers as that shape instead, so the
    handoff raises, naming the middleware that did it.

    Plain assignment (``config.temperature = 'hot'``), ``model_copy(update=...)``,
    and edits inside a field (``config.stop_sequences.append(5)``,
    ``config.thinking.budget_tokens = 10``) don't validate. Each handoff dumps
    the config and compares it with the dump from the last check that passed.
    If anything differs, the whole dump is validated, so nested bounds and
    model validators see the config as the next layer will. Fields that
    changed are stored back parsed (``config.task_budget = {'total': 1}``
    reaches the model as ``TaskBudget``). Untouched fields keep their objects,
    so a non-idempotent validator doesn't compound across layers.

    A failed check is remembered with the config it saw. When an outer layer
    catches the error and calls next again with that config unchanged, the
    same error is raised, still naming the layer that made the change. Once
    the config differs, it's checked again.
    """

    def __init__(self, *, config: BaseModel, schema: type[BaseModel], model: str) -> None:
        """Start from ``config`` as generate built it; ``schema`` is the model's config class."""
        self._schema = schema
        self._model = model
        self._checked = _config_values(config)
        self._rejected: list[tuple[object, GenkitError]] = []

    def check(self, config: object, middleware: str) -> None:
        """Raise a GenkitError naming ``middleware`` if ``config`` can't go to the next layer.

        On success, changed fields on ``config`` hold their parsed values.
        """
        dumped = _config_values(config) if isinstance(config, self._schema) else None
        typed = dumped is not None
        # Middleware edits a config of the right class in place, so it's
        # matched by its keys and values. Anything else is matched by identity.
        seen: object = (frozenset(vars(config)), dumped) if typed else config
        for bad, err in self._rejected:
            if (bad == seen) if typed else (bad is seen):
                raise err
        try:
            self._check(config, dumped, f"{self._model}: middleware '{middleware}'")
        except GenkitError as err:
            self._rejected.append((seen, err))
            raise
        self._rejected.clear()

    def _check(self, config: object, dumped: dict[str, Any] | None, prefix: str) -> None:
        if not isinstance(config, self._schema) or dumped is None:
            got = (
                'None'
                if config is None
                else type(config).__name__
                if type(config).__module__ == 'builtins'
                else config_type_path(type(config))
            )
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'{prefix} replaced request.config with {got}; change fields on request.config instead',
                reason=RuntimeErrorReason.INVALID_INPUT,
            )
        cls = type(config)
        values: dict[str, Any] = vars(config)
        if cls.model_config.get('extra') == 'forbid':
            unknown = [key for key in values if key not in cls.model_fields]
            if unknown:
                keys = ', '.join(repr(key) for key in unknown)
                noun = 'key' if len(unknown) == 1 else 'keys'
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message=f"{prefix} set unknown config {noun} {keys}; put provider-only settings in config['extra']",
                    reason=RuntimeErrorReason.INVALID_INPUT,
                )
        if dumped == self._checked:
            return
        keys_by_name = {name: f.serialization_alias or f.alias or name for name, f in cls.model_fields.items()}
        names_by_key = {key: name for name, key in keys_by_name.items()}
        missing = object()
        changed = [
            name for name, key in keys_by_name.items() if dumped.get(key, missing) != self._checked.get(key, missing)
        ]
        try:
            parsed = cls.model_validate(dumped)
        except ValidationError as e:
            errors = e.errors()
            loc = errors[0]['loc'] if errors else ()
            if loc:
                where = repr('.'.join([names_by_key.get(str(loc[0]), str(loc[0])), *(str(part) for part in loc[1:])]))
            else:
                # A model validator: blame the fields this layer changed.
                where = ', '.join(repr(name) for name in changed) or 'values'
            msg = errors[0]['msg'] if errors else str(e)
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'{prefix} set config {where}: {msg}',
                reason=RuntimeErrorReason.INVALID_INPUT,
                cause=e,
            ) from e
        # Writing ``__dict__`` directly keeps ``model_fields_set`` as the
        # middleware left it, so ``exclude_unset`` dumps don't change.
        for name in changed:
            values[name] = parsed.__dict__[name]
        self._checked = _config_values(config) if changed else dumped
