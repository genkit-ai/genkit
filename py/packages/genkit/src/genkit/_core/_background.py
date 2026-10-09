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

"""Background model definitions for the Genkit framework."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from functools import wraps
from typing import Any

from pydantic import BaseModel

from genkit._core._action import Action, ActionKind, ActionRunContext, get_current_context
from genkit._core._error import GenkitError, RuntimeErrorReason
from genkit._core._model import ModelRequest, ModelResponse
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._typing import (
    ModelInfo,
    Operation,
)


def _make_action_key(action_type: ActionKind | str, name: str) -> str:
    """Create an action key in format: /{action_type}/{name}.

    Args:
        action_type: The action type (e.g., 'background-model').
        name: The action name.

    Returns:
        Action key in format /{action_type}/{name}.
    """
    return f'/{action_type}/{name}'


def stamp_operation_action(*, operation: Operation, name: str) -> None:
    """A handle needs the start action key so check/cancel can find the job."""
    if operation.action:
        return
    operation.action = _make_action_key(ActionKind.BACKGROUND_MODEL, name)


def _operation_action(
    *,
    kind: ActionKind,
    model_name: str,
    suffix: str = '',
    fn: Callable[..., Awaitable[Operation]],
    description: str,
    metadata: dict[str, object],
    config_schema: type[BaseModel] | dict[str, Any] | None = None,
) -> Action:
    """An Action for start/check/cancel that stamps the returned Operation.

    ``fn``'s signature is still the Action's (``metadata_fn``), so
    ``(ctx, request)`` and ``(request, ctx)`` both work: the wrapper forwards
    through ``params.call``. The stamp is the start action key, so a caller
    who passes the Operation back reaches the right check/cancel.

    The action is named ``{model_name}{suffix}``. The registry may add the
    plugin prefix after this is built, so the key is read from the action's
    live name minus ``suffix`` (e.g. ``/check``) when it runs.
    """

    # wraps keeps fn's annotations on the wrapper, e.g. ModelRequest[VeoConfig].
    @wraps(fn)
    async def run_and_stamp(input: object, ctx: ActionRunContext) -> Operation:  # noqa: A002
        op = await action.params.call(fn, input, ctx)
        if isinstance(op, Operation):
            stamp_operation_action(operation=op, name=action.name.removesuffix(suffix))
        return op

    action = Action(
        kind=kind,
        name=f'{model_name}{suffix}',
        fn=run_and_stamp,
        metadata_fn=fn,
        metadata=metadata,
        description=description,
        config_schema=config_schema,
    )
    return action


StartModelOpFn = Callable[[ModelRequest, ActionRunContext], Awaitable[Operation]]
CheckModelOpFn = Callable[[Operation, ActionRunContext], Awaitable[Operation]]
CancelModelOpFn = Callable[[Operation, ActionRunContext], Awaitable[Operation]]


def operation_context(
    *,
    context: dict[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Fold check/cancel ``config=`` into the context bag the plugin reads.

    ``config`` here is client knobs (``base_url``, ``location``), not video
    settings. A per-request key lives in ``context['secrets']``. Top-level
    ``config=`` wins when both are set so the caller's explicit override is
    what the plugin sees.

    Supplying only ``config=`` keeps the current action context. An explicit
    ``context={}`` still clears it. Both omitted returns ``None`` so
    ``Action.run`` inherits directly.
    """
    if context is None and config is None:
        return None
    folded = dict(context if context is not None else (get_current_context() or {}))
    if config is not None:
        folded['config'] = dict(config)
    return folded


class BackgroundAction:
    """A handle over a background model's start, check and cancel actions.

    Built on registered actions but isn't itself an ``Action``: each of
    start, check and cancel has its own registry key.
    ``start`` returns an Operation; pass it to ``check`` until it's done.

    Not generic: start, check and cancel always return an ``Operation``, and
    ``Operation.output`` is untyped, so a type parameter would carry nothing.
    If ``Operation`` gets a typed output, add one with a default
    (``TypeVar('T', default=ModelResponse)``) so a bare ``BackgroundAction``
    still type-checks.

    Attributes:
        __action: Action metadata.
        start_action: Action to start the operation.
        check_action: Action to check operation status.
        cancel_action: Optional action to cancel operations.
        supports_cancel: Whether this action supports cancellation.
    """

    def __init__(
        self,
        start_action: Action,
        check_action: Action,
        cancel_action: Action | None = None,
    ) -> None:
        """Initialize a BackgroundAction.

        Args:
            start_action: Action to start the operation.
            check_action: Action to check operation status.
            cancel_action: Optional action to cancel the operation.
        """
        self.start_action = start_action
        self.check_action = check_action
        self.cancel_action = cancel_action

        # Store action metadata
        self.__action = {
            'name': start_action.name,
            'description': start_action.description,
            'actionType': start_action.kind,
            'metadata': start_action.metadata,
        }

    @property
    def name(self) -> str:
        """The name of the background action."""
        return self.start_action.name

    @property
    def supports_cancel(self) -> bool:
        """Whether this background action supports cancellation."""
        return self.cancel_action is not None

    @property
    def actions(self) -> list[Action]:
        """The start, check and (if any) cancel actions, for registering or listing.

        A background model is three registered actions, not one. Register all
        of them, or a poll fails later with the check action not found.
        """
        actions = [self.start_action, self.check_action]
        if self.cancel_action is not None:
            actions.append(self.cancel_action)
        return actions

    async def start(
        self,
        input: ModelRequest | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> Operation:
        """Start a background operation.

        Args:
            input: The input request.
            context: Optional run context. Per-request keys go in
                ``context['secrets']``.

        Returns:
            An Operation with an ID to track the job.
        """
        # Same pocket as check/cancel — a tenant key on start has to
        # reach the plugin, not die on this wrapper.
        result = await self.start_action.run(input, context=context)
        return _ensure_operation(response=result.response, name=self.start_action.name)

    async def check(
        self,
        operation: Operation,
        *,
        context: dict[str, Any] | None = None,
    ) -> Operation:
        """Check the status of a background operation.

        Args:
            operation: The operation to check.
            context: Optional run context (secrets, folded client config).

        Returns:
            Updated Operation with current status.

        Raises:
            GenkitError: INVALID_ARGUMENT if ``operation`` is not a live
                ``Operation`` (e.g. a dump or a ``ModelResponse``).
        """
        operation = require_operation(value=operation)
        result = await self.check_action.run(operation, context=context)
        return _ensure_operation(response=result.response, name=self.check_action.name)

    async def cancel(
        self,
        operation: Operation,
        *,
        context: dict[str, Any] | None = None,
    ) -> Operation:
        """Cancel a background operation.

        Args:
            operation: The operation to cancel.
            context: Optional run context (secrets, folded client config).

        Returns:
            Updated Operation reflecting cancellation attempt.

        Raises:
            GenkitError: UNIMPLEMENTED if this action does not implement
                cancel, INVALID_ARGUMENT if ``operation`` is not a live
                ``Operation``.
        """
        operation = require_operation(value=operation)
        # Raising here is deliberate: returning the operation unchanged would
        # make "this model can't cancel" indistinguishable from "cancelled".
        if self.cancel_action is None:
            raise GenkitError(
                status='UNIMPLEMENTED',
                message=f'Background action {operation.action} does not support cancellation.',
                reason=RuntimeErrorReason.UNSUPPORTED_BY_MODEL,
            )
        result = await self.cancel_action.run(operation, context=context)
        return _ensure_operation(response=result.response, name=self.cancel_action.name)


class MissingOperationError(GenkitError):
    """A background action lost the operation handle it created."""


def missing_operation_error(*, name: str) -> MissingOperationError:
    """The caller asked for a handle and this action did not return one."""
    return MissingOperationError(
        status='FAILED_PRECONDITION',
        message=f"'{name}' did not return an operation.",
    )


def _ensure_operation(*, response: object, name: str) -> Operation:
    """A start/check/cancel fn returns an Operation, not a dict."""
    if isinstance(response, Operation):
        return response
    raise missing_operation_error(name=name)


def background_model(
    name: str,
    start: StartModelOpFn,
    check: CheckModelOpFn,
    *,
    cancel: CancelModelOpFn | None = None,
    label: str | None = None,
    info: ModelInfo | None = None,
    config_schema: type[BaseModel] | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    description: str | None = None,
) -> BackgroundAction:
    """Build a background model without registering it.

    Plugin ``init`` / ``resolve`` return this. ``define_background_model``
    registers the start / check / cancel actions.
    """
    # Build model metadata
    model_meta: dict[str, Any] = metadata.copy() if metadata else {}
    model_options: dict[str, Any] = {}

    if info:
        model_options.update(info.model_dump(by_alias=True, exclude_none=True))

    # generate_operation looks at this flag. A background model is a
    # poll-handle model, so the flag is set when the action is built.
    supports = model_options.get('supports')
    if not isinstance(supports, dict):
        supports = {}
    else:
        supports = dict(supports)
    supports['longRunning'] = True
    model_options['supports'] = supports

    # Precedence: explicit label argument > info.label > fallback to model name
    label = label or model_options.get('label') or name
    model_options['label'] = label

    if config_schema:
        model_options['customOptions'] = to_json_schema(config_schema)

    model_meta['model'] = model_options

    # Build output schema metadata
    output_schema_meta = to_json_schema(ModelResponse)
    model_meta['outputSchema'] = output_schema_meta

    start_action = _operation_action(
        kind=ActionKind.BACKGROUND_MODEL,
        model_name=name,
        fn=start,
        metadata=model_meta,
        description=description or f'Background model: {label}',
        config_schema=config_schema,
    )
    # Annotation only: the Dev UI form comes from ModelRequest[Cfg].
    if 'customOptions' not in model_options and start_action.config_schema is not None:
        model_options['customOptions'] = to_json_schema(start_action.config_schema)

    check_action = _operation_action(
        kind=ActionKind.CHECK_OPERATION,
        model_name=name,
        suffix='/check',
        fn=check,
        metadata={'outputSchema': output_schema_meta},
        description=f'Check operation status for {label}',
    )

    cancel_action = None
    if cancel is not None:
        cancel_action = _operation_action(
            kind=ActionKind.CANCEL_OPERATION,
            model_name=name,
            suffix='/cancel',
            fn=cancel,
            metadata={'outputSchema': output_schema_meta},
            description=f'Cancel operation for {label}',
        )

    return BackgroundAction(
        start_action=start_action,
        check_action=check_action,
        cancel_action=cancel_action,
    )


def define_background_model(
    registry: Registry,
    name: str,
    start: StartModelOpFn,
    check: CheckModelOpFn,
    cancel: CancelModelOpFn | None = None,
    label: str | None = None,
    info: ModelInfo | None = None,
    config_schema: type[BaseModel] | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    description: str | None = None,
) -> BackgroundAction:
    """Register a background model for long-running AI operations."""
    action = background_model(
        name,
        start,
        check,
        cancel=cancel,
        label=label,
        info=info,
        config_schema=config_schema,
        metadata=metadata,
        description=description,
    )
    for each in action.actions:
        registry.register_action_from_instance(each)
    return action


async def lookup_background_action(
    registry: Registry,
    key: str,
) -> BackgroundAction | None:
    """Look up a background action by its action key.

    Matches JS lookupBackgroundAction from js/core/src/background-action.ts.

    The key format is /{actionType}/{name}, e.g., /background-model/video-gen.

    Args:
        registry: The registry to search in.
        key: The action key (e.g., '/background-model/video-gen').

    Returns:
        The BackgroundAction if found, None otherwise.
    """
    # Look up the start action
    start_action = await registry.resolve_action_by_key(key)
    if start_action is None:
        return None

    # Extract action name from key: /{actionType}/{name} -> {name}
    # JS: const actionName = key.substring(key.indexOf('/', 1) + 1);
    parts = key.split('/', 2)  # ['', 'background-model', 'name']
    if len(parts) < 3:
        return None
    action_name = parts[2]

    # Look up check action: /check-operation/{name}/check
    check_key = f'/check-operation/{action_name}/check'
    check_action = await registry.resolve_action_by_key(check_key)
    if check_action is None:
        return None

    # Look up cancel action (optional): /cancel-operation/{name}/cancel
    cancel_key = f'/cancel-operation/{action_name}/cancel'
    cancel_action = await registry.resolve_action_by_key(cancel_key)

    return BackgroundAction(
        start_action=start_action,
        check_action=check_action,
        cancel_action=cancel_action,
    )


def require_operation(*, value: object) -> Operation:
    """A poll handle is an Operation. A dump or generate() box is not."""
    if isinstance(value, Operation):
        return value
    if isinstance(value, ModelResponse):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='got ModelResponse; pass response.operation',
            reason=RuntimeErrorReason.INVALID_INPUT,
        )
    if isinstance(value, Mapping):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='got a dump; pass Operation.model_validate(...)',
            reason=RuntimeErrorReason.INVALID_INPUT,
        )
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=f'got {type(value).__name__}, expected Operation',
        reason=RuntimeErrorReason.INVALID_INPUT,
    )


async def resolve_operation_action(
    registry: Registry,
    operation: Operation,
) -> BackgroundAction:
    """Turn a poll handle into the background action that owns it."""
    operation = require_operation(value=operation)
    if not operation.action:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='Provided operation is missing original request information',
            reason=RuntimeErrorReason.INVALID_INPUT,
        )

    try:
        background_action = await lookup_background_action(registry, operation.action)
    except ValueError as e:
        # operation.action is caller data (often reloaded from storage), so a
        # mangled key is the caller's bad argument, not an internal failure.
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'Failed to resolve background action from original request: {operation.action}',
            reason=RuntimeErrorReason.ACTION_NOT_FOUND,
        ) from e
    if background_action is None:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'Failed to resolve background action from original request: {operation.action}',
            reason=RuntimeErrorReason.ACTION_NOT_FOUND,
        )
    return background_action


async def check_operation(
    registry: Registry,
    operation: Operation,
    *,
    context: dict[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Operation:
    """Check the status of a background operation.

    Args:
        registry: The registry to look up actions from.
        operation: The poll handle.
        context: Optional run context. Per-request keys go in
            ``context['secrets']``.
        config: Optional client knobs (``base_url``, ``location``). Folded
            into ``context['config']`` for the plugin.

    Returns:
        Updated Operation with current status.

    Raises:
        GenkitError: If the handle is missing action, or the action is
            not found.
    """
    background_action = await resolve_operation_action(registry, operation)
    return await background_action.check(
        operation,
        context=operation_context(context=context, config=config),
    )


async def cancel_operation(
    registry: Registry,
    operation: Operation,
    *,
    context: dict[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> Operation:
    """Cancel a background operation.

    Args:
        registry: The registry to look up actions from.
        operation: The poll handle.
        context: Optional run context. Per-request keys go in
            ``context['secrets']``.
        config: Optional client knobs (``base_url``, ``location``). Folded
            into ``context['config']`` for the plugin.

    Returns:
        Updated Operation reflecting the cancel attempt.

    Raises:
        GenkitError: If the handle is missing action, the action is not
            found, or cancel is not implemented (UNIMPLEMENTED, raised by
            ``BackgroundAction.cancel``).
    """
    background_action = await resolve_operation_action(registry, operation)
    return await background_action.cancel(
        operation,
        context=operation_context(context=context, config=config),
    )
