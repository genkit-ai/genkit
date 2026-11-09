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
from dataclasses import dataclass, replace
from typing import Annotated, Any, TypeAlias, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ValidationError

from genkit._ai._background_model import BackgroundModel
from genkit._core._action import (
    Action,
    ActionKind,
    ModelContext,
    get_func_description,
)
from genkit._core._config_check import (
    check_config_values,
    config_field_names,
    config_values,
    fold_config_aliases as fold_config_aliases,
    layer_call_config,
    normalize_config as normalize_config,
    overlay_config as overlay_config,
)
from genkit._core._error import (
    ErrorReason,
    GenkitError,
    validation_error_details,
    validation_input_is_printable,
    validation_problem,
)
from genkit._core._logger import get_logger
from genkit._core._model import (
    Message,
    ModelConfig as ModelConfig,
    ModelConfigDict,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    config_type_path,
    declared_config_type,
    reject_config_api_key,
    text_from_content,
    text_from_message,
)
from genkit._core._registry import Registry
from genkit._core._schema import custom_options_schema, fixed_json_schema
from genkit._core._typing import GenerationCommonConfig, ModelInfo

# Type alias for model functions (must be async)
# Use ctx.send_chunk() for streaming
ModelFn = Callable[[ModelRequest, ModelContext], Awaitable[ModelResponse[Any]]]
"""Signature of the function passed to ``define_model``.

An async function taking the ``ModelRequest`` and a ``ModelContext``; stream
with ``ctx.send_chunk(...)`` and return the final ``ModelResponse``.
"""

logger = get_logger(__name__)

# Veneer-facing argument shapes. Internals resolve these into ResolvedModel.
# ModelArg is also the constructor default, stored as a registry value the
# Dev UI lists as JSON, so it stays a name or ModelRef.
ModelArg: TypeAlias = str | ModelRef[Any]


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


def annotated_config_class(action: object | None) -> type[BaseModel] | None:
    """The class on the model function's ``ModelRequest[...]`` annotation, if any."""
    input_class = getattr(action, 'input_class', None) if action is not None else None
    return declared_config_type(input_class) if isinstance(input_class, type) else None


def _name_or_ref(model: object) -> str | ModelRef[Any] | None:
    """Unwrap a name or ModelRef. Other values stay None."""
    if isinstance(model, ModelRef):
        return cast(ModelRef[Any], model)
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
            reason=ErrorReason.INVALID_INPUT,
        )
    return action.name


def _model_action_name(*, model: object, registry: Registry) -> str | None:
    """Name of a define_model action registered here; None if not an action."""
    if isinstance(model, BackgroundModel):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f"model is background model '{model.name}'. Pass it to generate_operation.",
            reason=ErrorReason.INVALID_INPUT,
        )
    if not isinstance(model, Action):
        return None
    if model.kind != ActionKind.MODEL:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f"model is {model.kind} action '{model.name}', expected a model.",
            reason=ErrorReason.INVALID_INPUT,
        )
    return _registered_action_name(action=model, kind=ActionKind.MODEL, registry=registry)


def background_model_name(*, model: BackgroundModel, registry: Registry) -> str:
    """Name of a define_background_model result registered on this registry."""
    return _registered_action_name(action=model.start_action, kind=ActionKind.BACKGROUND_MODEL, registry=registry)


def resolve_model_arg(
    *,
    model: object | None,
    registry: Registry,
    message: str = 'No model configured.',
) -> str | ModelRef[Any]:
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
            reason=ErrorReason.INVALID_INPUT,
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
            reason=ErrorReason.INVALID_INPUT,
        )
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=message,
        reason=ErrorReason.MODEL_NOT_FOUND,
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
    # A plain ModelConfig / GenerationCommonConfig ref defers to the model's class.
    if ref_schema is None or ref_schema is ModelConfig or ref_schema is GenerationCommonConfig:
        return action_schema
    if action_schema is None or ref_schema is action_schema:
        return ref_schema
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=(
            f"{kind} '{name}' takes config {config_type_path(action_schema)}, but the ref's config_schema is "
            f'{config_type_path(ref_schema)}. Use {action_schema.__name__} on the ref, or pass the name.'
        ),
        reason=ErrorReason.INVALID_INPUT,
    )


def clear_on_hop(
    *,
    prompt_config: object,
    call_config: object,
    schema: type[BaseModel] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prompt and call bags minus the keys a hop's call-level ``None`` clears.

    A Gemini prompt called with ``model='other', config={'thinkingConfig': None}``
    drops thinkingConfig from both bags, so ``other`` never sees a key its
    class doesn't declare. A key is cleared only when the call sets it to
    ``None``, ``schema`` doesn't declare it, and the prompt's config has it
    (spellings fold through the prompt's typed class). A ``None`` for a key the
    prompt doesn't have (``'temprature'``) stays and fails the unknown-key check.
    """
    prompt_bag = normalize_config(config=prompt_config)
    call_bag = normalize_config(config=call_config)
    if schema is None or not isinstance(call_config, Mapping):
        return prompt_bag, call_bag
    target = config_field_names(schema)
    spelling = config_field_names(type(prompt_config)) if isinstance(prompt_config, BaseModel) else {}
    prompt_keys = {spelling.get(key, key) for key in prompt_bag}
    cleared = {
        spelling.get(key, key)
        for key, value in call_bag.items()
        if value is None and key not in target and spelling.get(key, key) in prompt_keys
    }
    if not cleared:
        return prompt_bag, call_bag
    return (
        {key: value for key, value in prompt_bag.items() if spelling.get(key, key) not in cleared},
        {key: value for key, value in call_bag.items() if spelling.get(key, key) not in cleared},
    )


async def resolve_for_generate(
    *,
    model: object | None,
    config: object = None,
    prompt_config: object = None,
    hopping: bool = False,
    registry: Registry,
    message: str = 'No model configured.',
) -> ResolvedModel:
    """Name, config bag, and the config class this generate will check against.

    The model's definition-time class is the class this call checks. A
    ModelRef's config_schema must name it (see call_config_class), and the
    ref's layers fold aliases to it. When ``prompt_config`` is passed, ``config``
    overlays it before overlaying the ref's layers. ``hopping`` (the call
    named a model other than the prompt's) runs clear_on_hop first.
    """
    arg = resolve_model_arg(model=model, registry=registry, message=message)
    name = arg.name if isinstance(arg, ModelRef) else arg
    action = await registry.resolve_model(name)
    ref_schema = python_config_schema(arg.config_schema) if isinstance(arg, ModelRef) else None
    schema = call_config_class(
        name=name,
        kind='model',
        ref_schema=ref_schema,
        action_schema=python_config_schema(action.config_schema) if action is not None else None,
    )
    if prompt_config is not None:
        if hopping:
            prompt_bag, call_bag = clear_on_hop(prompt_config=prompt_config, call_config=config, schema=schema)
        else:
            prompt_bag, call_bag = normalize_config(config=prompt_config), normalize_config(config=config)
        config = overlay_config(layers=[prompt_bag, call_bag], schema=schema or ref_schema)
    resolved = resolve_call_model(model=arg, config=config, registry=registry, message=message, schema=schema)
    reject_config_api_key(resolved.config)
    return replace(resolved, config_schema=schema)


async def check_config_before_turn(*, model: ModelArg, config: object, registry: Registry) -> None:
    """Raise before the turn when a value in ``config`` can't reach the model.

    Checks against ``action.config_schema`` (from the ``ModelRequest[...]``
    annotation or ``config_schema=``). A missing top-level field passes:
    ``wrap_model`` middleware can still fill it, and the model action reports
    it if nothing does.
    """
    model_name = model.name if isinstance(model, ModelRef) else model
    action = await registry.resolve_model(model_name)
    if action is not None:
        check_config_values(config=config, schema=action.config_schema, model=model_name)


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


def _model_options(
    *,
    name: str,
    info: ModelInfo | None,
    config_schema: type | dict[str, Any] | None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build ``metadata['model']`` for ``Model``."""
    options: dict[str, object] = info.model_dump(by_alias=True, exclude_none=True) if info else {}
    existing = metadata.get('model') if metadata else None
    if isinstance(existing, dict):
        for key, value in cast(dict[str, object], existing).items():
            if isinstance(key, str) and key not in options:
                options[key] = value
    if not options.get('label'):
        options['label'] = name
    if claims_long_running(model_options=options):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f"model '{name}' cannot set longRunning. Use define_background_model.",
        )
    if config_schema:
        options['customOptions'] = custom_options_schema(config_schema)
    return options


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


def _model_io_schemas() -> tuple[dict[str, object], dict[str, object]]:
    """Every model publishes ModelRequest/ModelResponse; config shows in customOptions."""
    return fixed_json_schema(ModelRequest), fixed_json_schema(ModelResponse)


class Model(Action[ModelRequest, ModelResponse, ModelResponseChunk]):
    """A model action returned from a plugin's ``init`` / ``resolve`` / ``list_actions`` or ``define_model``.

    Plugin ``init`` / ``resolve`` / ``list_actions`` return this without registering it;
    ``define_model`` registers it.
    The config class stays on the action so ``generate(model='name', config=...)``
    can ``isinstance``-check a Pydantic instance against a string model name.
    It comes from ``config_schema`` or the fn's ``ModelRequest[Cfg]``
    annotation; given both, they must be the same class.
    """

    def __init__(
        self,
        name: str,
        fn: ModelFn,
        *,
        config_schema: type[BaseModel] | dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        info: ModelInfo | None = None,
        description: str | None = None,
    ) -> None:
        """Build a model action."""
        model_options = _model_options(name=name, info=info, config_schema=config_schema, metadata=metadata)
        model_meta: dict[str, object] = metadata.copy() if metadata else {}
        model_meta['model'] = model_options
        super().__init__(
            kind=ActionKind.MODEL,
            name=name,
            fn=fn,
            metadata=model_meta,
            description=get_func_description(fn, description),
            config_schema=config_schema,
            _io_schemas=_model_io_schemas(),
        )
        # Annotation only: the Dev UI form comes from ModelRequest[Cfg].
        if 'customOptions' not in model_options and self.config_schema is not None:
            model_options['customOptions'] = custom_options_schema(self.config_schema)


def define_model(
    registry: Registry,
    name: str,
    fn: ModelFn,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    info: ModelInfo | None = None,
    description: str | None = None,
) -> Model:
    """Register a custom model action."""
    _check_request_annotation(name, fn)
    action = Model(
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
        reason=ErrorReason.INVALID_INPUT,
    )


def check_call_config(*, config: object, schema: type[BaseModel] | None, model: str) -> None:
    """Call-time config check: a typed object's class and a dict's keys and values."""
    assert_correct_config_class(config=config, schema=schema, model=model)
    check_config_values(config=config, schema=schema, model=model)


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
        self._checked = config_values(config)
        self._rejected: list[tuple[object, GenkitError]] = []

    def check(self, config: object, middleware: str) -> None:
        """Raise a GenkitError naming ``middleware`` if ``config`` can't go to the next layer.

        On success, changed fields on ``config`` hold their parsed values.
        """
        dumped = config_values(config) if isinstance(config, self._schema) else None
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
                reason=ErrorReason.INVALID_INPUT,
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
                    reason=ErrorReason.INVALID_INPUT,
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
            # A model validator has no path, and its input is the whole config.
            show_input = bool(loc) and validation_input_is_printable(e)
            msg = validation_problem(errors[0], show_input=show_input) if errors else str(e)
            error = GenkitError(
                status='INVALID_ARGUMENT',
                message=f'{prefix} set config {where}: {msg}',
                reason=ErrorReason.INVALID_INPUT,
                details={'errors': validation_error_details(e)},
            )
            # The message names the problem; cause= would append it again.
            raise error from e
        # Writing ``__dict__`` directly keeps ``model_fields_set`` as the
        # middleware left it, so ``exclude_unset`` dumps don't change.
        for name in changed:
            values[name] = parsed.__dict__[name]
        self._checked = config_values(config) if changed else dumped
