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
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from genkit._core._action import Action, ActionKind, ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import ModelRequest, ModelResponse
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._typing import (
    ModelInfo,
    Operation,
)

# Type variable for operation output
OutputT = TypeVar('OutputT')


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


StartModelOpFn = Callable[[ModelRequest, ActionRunContext], Awaitable[Operation]]
CheckModelOpFn = Callable[[Operation], Awaitable[Operation]]
CancelModelOpFn = Callable[[Operation], Awaitable[Operation]]


class BackgroundAction(Generic[OutputT]):
    """A background action that can run for a long time.

    Unlike regular actions, background actions can run for extended periods.
    The returned operation can be used to check status and retrieve the response.

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

    async def start(
        self,
        input: ModelRequest | None = None,
        options: dict[str, Any] | None = None,
    ) -> Operation:
        """Start a background operation.

        Args:
            input: The input request.
            options: Optional run options.

        Returns:
            An Operation with an ID to track the job.
        """
        result = await self.start_action.run(input)
        return _ensure_operation(response=result.response, name=self.start_action.name)

    async def check(self, operation: Operation) -> Operation:
        """Check the status of a background operation.

        Args:
            operation: The operation to check.

        Returns:
            Updated Operation with current status.

        Raises:
            GenkitError: INVALID_ARGUMENT if ``operation`` is not a live
                ``Operation`` (e.g. a dump or a ``ModelResponse``).
        """
        operation = require_operation(value=operation)
        result = await self.check_action.run(operation)
        return _ensure_operation(response=result.response, name=self.check_action.name)

    async def cancel(self, operation: Operation) -> Operation:
        """Cancel a background operation.

        Args:
            operation: The operation to cancel.

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
            )
        result = await self.cancel_action.run(operation)
        return _ensure_operation(response=result.response, name=self.cancel_action.name)


def missing_operation_error(*, name: str) -> GenkitError:
    """The caller asked for a handle and this action did not return one."""
    return GenkitError(
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
) -> BackgroundAction[ModelResponse]:
    """Build a background model without registering it.

    Plugin ``init`` / ``resolve`` return this. ``define_background_model``
    registers the start / check / cancel actions.
    """
    action_key = _make_action_key(ActionKind.BACKGROUND_MODEL, name)

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

    # Wrap the start function to add the action key and timing.
    # Keep the caller's request annotation (ModelRequest[FamilyConfig]) so
    # Action still types the config bag as that family.
    @wraps(start)
    async def wrapped_start(request: ModelRequest, ctx: ActionRunContext) -> Operation:
        op = await start(request, ctx)
        # The handle needs this key so check/cancel can find the job later.
        op.action = action_key
        return op

    # Wrap the check function (no ctx parameter)
    async def wrapped_check(op: Operation, ctx: ActionRunContext) -> Operation:
        updated = await check(op)
        # Preserve action key
        updated.action = action_key
        return updated

    start_action = Action(
        kind=ActionKind.BACKGROUND_MODEL,
        name=name,
        fn=wrapped_start,
        metadata_fn=start,
        metadata=model_meta,
        description=description or f'Background model: {label}',
        config_schema=config_schema,
    )

    check_action = Action(
        kind=ActionKind.CHECK_OPERATION,
        name=f'{name}/check',
        fn=wrapped_check,
        metadata={'outputSchema': output_schema_meta},
        description=f'Check operation status for {label}',
    )

    cancel_action = None
    if cancel is not None:
        cancel_fn = cancel

        async def wrapped_cancel(op: Operation, ctx: ActionRunContext) -> Operation:
            cancelled = await cancel_fn(op)
            cancelled.action = action_key
            return cancelled

        cancel_action = Action(
            kind=ActionKind.CANCEL_OPERATION,
            name=f'{name}/cancel',
            fn=wrapped_cancel,
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
) -> BackgroundAction[ModelResponse]:
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
    registry.register_action_from_instance(action.start_action)
    registry.register_action_from_instance(action.check_action)
    if action.cancel_action is not None:
        registry.register_action_from_instance(action.cancel_action)
    return action


async def lookup_background_action(
    registry: Registry,
    key: str,
) -> BackgroundAction[ModelResponse] | None:
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
        )
    if isinstance(value, Mapping):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='got a dump; pass Operation.model_validate(...)',
        )
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message=f'got {type(value).__name__}, expected Operation',
    )


async def resolve_operation_action(
    registry: Registry,
    operation: Operation,
) -> BackgroundAction[ModelResponse]:
    """Turn a poll handle into the background action that owns it."""
    operation = require_operation(value=operation)
    if not operation.action:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='Provided operation is missing original request information',
        )

    try:
        background_action = await lookup_background_action(registry, operation.action)
    except ValueError as e:
        # operation.action is caller data (often reloaded from storage), so a
        # mangled key is the caller's bad argument, not an internal failure.
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'Failed to resolve background action from original request: {operation.action}',
        ) from e
    if background_action is None:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'Failed to resolve background action from original request: {operation.action}',
        )
    return background_action


async def check_operation(
    registry: Registry,
    operation: Operation,
) -> Operation:
    """Check the status of a background operation.

    Args:
        registry: The registry to look up actions from.
        operation: The poll handle.

    Returns:
        Updated Operation with current status.

    Raises:
        GenkitError: If the handle is missing action, or the action is
            not found.
    """
    background_action = await resolve_operation_action(registry, operation)
    return await background_action.check(operation)


async def cancel_operation(
    registry: Registry,
    operation: Operation,
) -> Operation:
    """Cancel a background operation.

    Args:
        registry: The registry to look up actions from.
        operation: The poll handle.

    Returns:
        Updated Operation reflecting the cancel attempt.

    Raises:
        GenkitError: If the handle is missing action, the action is not
            found, or cancel is not implemented (UNIMPLEMENTED, raised by
            ``BackgroundAction.cancel``).
    """
    background_action = await resolve_operation_action(registry, operation)
    return await background_action.cancel(operation)
