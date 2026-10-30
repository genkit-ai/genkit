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

"""BackgroundModel: a ``BackgroundAction[ModelRequest, ModelResponse]``.

Mirrors Go ``ai.BackgroundModel`` over ``core.BackgroundActionDef``. Adds
``ModelInfo`` / config schema metadata, request config validation on start,
and the "pass response.operation" guard on poll handles.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, ClassVar, cast

from pydantic import BaseModel

from genkit._core._action import Action, ActionKind, ModelContext
from genkit._core._background import (
    BackgroundAction,
    missing_operation_error,
    operation_actions,
    operation_context,
    operation_key,
    require_operation,
    resolve_operation_action,
    validate_operation,
)
from genkit._core._config_check import check_config_json_schema, check_request_config
from genkit._core._error import GenkitError, RuntimeErrorReason, StatusName
from genkit._core._model import ModelRequest, ModelResponse
from genkit._core._operation import Operation
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._typing import ModelInfo

# Spelled out, not CheckOpFn[ModelResponse]: Operation[ModelResponse] is a
# concrete pydantic class at runtime, so the generic alias can't be subscripted.
# A fn may return a plain Operation; its output is validated as a ModelResponse.
StartModelOpFn = Callable[[ModelRequest, ModelContext], Awaitable[Operation]]
CheckModelOpFn = Callable[[Operation[ModelResponse], ModelContext], Awaitable[Operation]]
CancelModelOpFn = Callable[[Operation[ModelResponse], ModelContext], Awaitable[Operation]]

# Keys a ModelResponse dump carries. An Operation dump always has 'id'.
_MODEL_RESPONSE_KEYS = frozenset({'message', 'finishReason', 'finish_reason', 'operation'})


def as_model_operation(operation: Operation, *, name: str, status: StatusName) -> Operation[ModelResponse]:
    """The same handle as an ``Operation[ModelResponse]``, with every field validated."""
    return validate_operation(operation, output_type=ModelResponse, name=name, status=status)


def reject_model_response(value: object) -> None:
    """A ModelResponse (or its dump) is the most common wrong thing to poll."""
    if isinstance(value, ModelResponse):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='got ModelResponse; pass response.operation',
            reason=RuntimeErrorReason.INVALID_INPUT,
        )
    if isinstance(value, Mapping) and 'id' not in value and not _MODEL_RESPONSE_KEYS.isdisjoint(value):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='got a ModelResponse dump; pass response.operation',
            reason=RuntimeErrorReason.INVALID_INPUT,
        )


class BackgroundModel(BackgroundAction[ModelRequest, ModelResponse]):
    """A background model: ``start`` returns an ``Operation[ModelResponse]``.

    Pass the handle to ``check`` until it's done. A finished job's ``output``
    is a ``ModelResponse``. ``label`` and ``config_schema`` are read back from
    the start action, so a model looked up from the registry has them too.
    """

    _noun: ClassVar[str] = 'Background model'
    _cancel_unsupported_reason: ClassVar[RuntimeErrorReason | None] = RuntimeErrorReason.UNSUPPORTED_BY_MODEL

    label: str
    config_schema: type[BaseModel] | dict[str, Any] | None

    def _bind(self, start: Action, check: Action, cancel: Action | None) -> None:
        super()._bind(start, check, cancel)
        model_meta = start.metadata.get('model')
        label = cast(dict[str, object], model_meta).get('label') if isinstance(model_meta, dict) else None
        self.label = label if isinstance(label, str) and label else start.name
        self.config_schema = start.config_schema

    def _require_handle(self, value: object) -> Operation[ModelResponse]:
        reject_model_response(value)
        return super()._require_handle(value)

    async def start(
        self,
        input: ModelRequest | None = None,  # noqa: A002
        *,
        context: dict[str, Any] | None = None,
    ) -> Operation[ModelResponse]:
        """Start a background operation.

        Args:
            input: The input request.
            context: Optional run context. Per-request keys go in
                ``context['secrets']``.

        Returns:
            An Operation with an ID to track the job.
        """
        return await super().start(input, context=context)


def ensure_model_operation(*, response: object, name: str) -> Operation[ModelResponse]:
    """A background model start fn returns an Operation, not a dict."""
    if isinstance(response, Operation):
        return as_model_operation(response, name=name, status='INTERNAL')
    raise missing_operation_error(name=name)


def background_model(
    name: str,
    *,
    start: StartModelOpFn,
    check: CheckModelOpFn,
    cancel: CancelModelOpFn | None = None,
    info: ModelInfo | None = None,
    config_schema: type[BaseModel] | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    description: str | None = None,
) -> BackgroundModel:
    """Build a background model without registering it.

    Plugin ``init`` / ``resolve`` return this. ``define_background_model``
    registers the start / check / cancel actions.
    """
    model_meta: dict[str, Any] = metadata.copy() if metadata else {}
    model_options: dict[str, Any] = {}
    if info:
        model_options.update(info.model_dump(by_alias=True, exclude_none=True))

    # generate_operation looks at this flag. A background model is a
    # poll-handle model, so the flag is set when the action is built.
    supports = model_options.get('supports')
    supports = dict(supports) if isinstance(supports, dict) else {}
    supports['longRunning'] = True
    model_options['supports'] = supports

    label = model_options.get('label') or name
    model_options['label'] = label
    if config_schema:
        model_options['customOptions'] = to_json_schema(config_schema)
    model_meta['model'] = model_options

    if isinstance(config_schema, dict):
        check_config_json_schema(config_schema, model=name)

    def check_start_input(request: object, action_name: str) -> None:
        # Same check as a regular model action: the whole config, after every
        # middleware, before the provider starts an operation.
        if config_schema:
            check_request_config(request, schema=config_schema, model=action_name)

    start_action, check_action, cancel_action = operation_actions(
        BackgroundModel,
        name,
        kind=ActionKind.BACKGROUND_MODEL,
        start=start,
        check=check,
        cancel=cancel,
        metadata=model_meta,
        description=description or f'Background model: {label}',
        label=label,
        config_schema=config_schema,
        check_start_input=check_start_input,
    )
    # Annotation only: the Dev UI form comes from ModelRequest[Cfg]. Start,
    # check and cancel share this dict, so all three see it.
    if 'customOptions' not in model_options and start_action.config_schema is not None:
        model_options['customOptions'] = to_json_schema(start_action.config_schema)
    return BackgroundModel(start_action, check_action, cancel_action)


def define_background_model(
    registry: Registry,
    name: str,
    *,
    start: StartModelOpFn,
    check: CheckModelOpFn,
    cancel: CancelModelOpFn | None = None,
    info: ModelInfo | None = None,
    config_schema: type[BaseModel] | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    description: str | None = None,
) -> BackgroundModel:
    """Register a background model for long-running AI operations."""
    model = background_model(
        name,
        start=start,
        check=check,
        cancel=cancel,
        info=info,
        config_schema=config_schema,
        metadata=metadata,
        description=description,
    )
    for each in model.actions:
        registry.register_action_from_instance(each)
    return model


async def lookup_background_model(registry: Registry, key: str) -> BackgroundModel | None:
    """Look up a background model by its start action key."""
    return await BackgroundModel.lookup(registry, key)


def _is_model_key(key: str | None) -> bool:
    """True for ``/background-model/...``, and for a missing or unparseable key.

    Those last two fail either way; treating them as model handles keeps the
    error order and wording background models have always had.
    """
    if not key:
        return True
    parts = key.split('/', 2)
    return not (len(parts) == 3 and parts[0] == '' and parts[1] != ActionKind.BACKGROUND_MODEL)


async def resolve_operation(
    registry: Registry,
    value: object,
) -> tuple[BackgroundAction[Any, Any], Operation[Any]]:
    """The background action that owns a handle, and the handle validated for it.

    The handle's ``action`` key picks the class: a background model validates
    ``output`` as a ModelResponse, any other kind as ``Any`` (its check action
    still validates against its own output type). Every check runs before an
    action span opens.
    """
    reject_model_response(value)
    key = operation_key(value)
    if _is_model_key(key):
        # Shape and output first, then the lookup: a typo'd key or a bad
        # output is named before "not found".
        require_operation(value=value, output_type=ModelResponse)
        bg_model = await resolve_operation_action(
            registry, key, lookup=lookup_background_model, noun='background model'
        )
        return bg_model, bg_model._require_handle(value)
    require_operation(value=value)
    action = await resolve_operation_action(registry, key, lookup=BackgroundAction.lookup)
    return action, action._require_handle(value)


async def check_operation(
    registry: Registry,
    operation: Operation | Mapping[str, Any],
    *,
    context: dict[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Operation[Any]:
    """Check any background operation; see :func:`resolve_operation`.

    Raises:
        GenkitError: If the handle is not an Operation or a dict that
            validates as one, is missing action, or the action is not found.
    """
    action, handle = await resolve_operation(registry, operation)
    return await action.check(handle, context=operation_context(context=context, config=config))


async def cancel_operation(
    registry: Registry,
    operation: Operation | Mapping[str, Any],
    *,
    context: dict[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Operation[Any]:
    """Cancel any background operation; see :func:`resolve_operation`.

    Raises:
        GenkitError: If the handle is invalid, the action is not found, or
            cancel is not implemented (UNIMPLEMENTED).
    """
    action, handle = await resolve_operation(registry, operation)
    return await action.cancel(handle, context=operation_context(context=context, config=config))
