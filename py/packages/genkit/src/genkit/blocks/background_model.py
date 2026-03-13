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

"""Background model definitions for the Genkit framework.

Background models are long-running AI operations that don't complete immediately.
They are used for tasks like video generation (Veo) or large image generation (Imagen)
that may take seconds or minutes to complete.

Why Background Models?
    Regular models return results synchronously - you call generate() and get a
    response. But some AI tasks (video generation, complex rendering) can take
    minutes. Background models solve this by:

    1. Returning immediately with an operation ID
    2. Allowing you to poll for completion
    3. Optionally supporting cancellation

Architecture:
    A background model consists of three actions registered together.
    The naming convention matches the JS implementation:

    ```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      BackgroundAction                                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
    │  │   start_action   │   │   check_action   │   │  cancel_action   │    │
    │  │ /background-     │   │ /check-operation │   │ /cancel-operation│    │
    │  │  model/{name}    │   │  /{name}/check   │   │  /{name}/cancel  │    │
    │  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘    │
    │           │                      │                      │ (optional)   │
    └───────────┼──────────────────────┼──────────────────────┼──────────────┘
                │                      │                      │
                ▼                      ▼                      ▼
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │  StartModelOpFn  │   │  CheckModelOpFn  │   │ CancelModelOpFn  │
    │  (user-provided) │   │  (user-provided) │   │  (user-provided) │
    └──────────────────┘   └──────────────────┘   └──────────────────┘
    ```

Key Concepts:
    +---------------+----------------------------------------------------------------+
    | Term          | Description                                                    |
    +---------------+----------------------------------------------------------------+
    | Operation     | A long-running task with an ID, status, and eventual result   |
    | start()       | Initiates the background operation, returns an Operation      |
    | check()       | Polls the operation status, returns updated Operation         |
    | cancel()      | Attempts to cancel a running operation (if supported)         |
    +---------------+----------------------------------------------------------------+

Workflow:
    ```
    ┌─────────┐         ┌─────────────┐         ┌─────────────┐
    │ Client  │         │  Background │         │   Backend   │
    │  Code   │         │    Model    │         │   Service   │
    └────┬────┘         └──────┬──────┘         └──────┬──────┘
         │                     │                       │
         │  1. start(request)  │                       │
         │────────────────────►│  submit job           │
         │                     │──────────────────────►│
         │                     │        job_id         │
         │  Operation(id=X)    │◄──────────────────────│
         │◄────────────────────│                       │
         │                     │                       │
         │  2. check(op)       │                       │
         │────────────────────►│  get status(X)        │
         │                     │──────────────────────►│
         │                     │   status: processing  │
         │  Operation(done=F)  │◄──────────────────────│
         │◄────────────────────│                       │
         │                     │                       │
         │  ... (poll loop)    │                       │
         │                     │                       │
         │  3. check(op)       │                       │
         │────────────────────►│  get status(X)        │
         │                     │──────────────────────►│
         │                     │   status: complete    │
         │  Operation(done=T,  │◄──────────────────────│
         │    output=result)   │                       │
         │◄────────────────────│                       │
         │                     │                       │
    ```

Example:
    >>> # Define a background model for video generation
    >>> async def start_video(request: GenerateRequest, ctx) -> Operation:
    ...     job_id = await video_api.submit(request.messages[0].content[0].text)
    ...     return Operation(id=job_id, done=False)
    >>> async def check_video(op: Operation, ctx) -> Operation:
    ...     status = await video_api.get_status(op.id)
    ...     if status.complete:
    ...         return Operation(id=op.id, done=True, output={...})
    ...     return Operation(id=op.id, done=False)
    >>> ai.define_background_model(
    ...     name='my-video-model',
    ...     start=start_video,
    ...     check=check_video,
    ... )

See Also:
    - JS implementation: js/core/src/background-action.ts
    - JS model wrapper: js/ai/src/model.ts (defineBackgroundModel)
    - Sample: py/samples/background-model-demo/
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from genkit.codec import dump_dict
from genkit.core.action import Action, ActionRunContext
from genkit.core.action.types import ActionKind
from genkit.core.registry import Registry
from genkit.core.schema import to_json_schema
from genkit.core.typing import (
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    Operation,
)

# Type variable for operation output
OutputT = TypeVar('OutputT')


def _make_action_key(action_type: ActionKind | str, name: str) -> str:
    """Create an action key matching JS format: /{actionType}/{name}.

    Args:
        action_type: The action type (e.g., 'background-model').
        name: The action name.

    Returns:
        Action key in format /{actionType}/{name}.
    """
    return f'/{action_type}/{name}'


# Type aliases for background model functions matching JS signatures
# JS: start: (input, options) => Promise<Operation<OutputT>>
StartModelOpFn = Callable[[GenerateRequest, ActionRunContext], Awaitable[Operation]]
# JS: check: (input: Operation<OutputT>) => Promise<Operation<OutputT>>
CheckModelOpFn = Callable[[Operation], Awaitable[Operation]]
# JS: cancel?: (input: Operation<OutputT>) => Promise<Operation<OutputT>>
CancelModelOpFn = Callable[[Operation], Awaitable[Operation]]


class BackgroundAction(Generic[OutputT]):
    """A background action that can run for a long time.

    Unlike regular actions, background actions can run for extended periods.
    The returned operation can be used to check status and retrieve the response.

    This class matches the JS BackgroundAction interface from
    js/core/src/background-action.ts.

    Attributes:
        __action: Action metadata (matches JS __action property).
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

        # Match JS __action property structure
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
        input: GenerateRequest | None = None,
        options: dict[str, Any] | None = None,
    ) -> Operation:
        """Start a background operation.

        Matches JS: start(input?, options?) => Promise<Operation<OutputT>>

        Args:
            input: The input request.
            options: Optional run options.

        Returns:
            An Operation with an ID to track the job.
        """
        result = await self.start_action.arun(input)
        return _ensure_operation(result.response)

    async def check(self, operation: Operation) -> Operation:
        """Check the status of a background operation.

        Matches JS: check(operation) => Promise<Operation<OutputT>>

        Args:
            operation: The operation to check.

        Returns:
            Updated Operation with current status.
        """
        result = await self.check_action.arun(operation)
        return _ensure_operation(result.response)

    async def cancel(self, operation: Operation) -> Operation:
        """Cancel a background operation.

        Matches JS: cancel(operation) => Promise<Operation<OutputT>>

        If cancellation is not supported, returns the operation unchanged
        (matching JS behavior).

        Args:
            operation: The operation to cancel.

        Returns:
            Updated Operation reflecting cancellation attempt.
        """
        if self.cancel_action is None:
            # Match JS behavior: return operation unchanged if cancel not supported
            return operation
        result = await self.cancel_action.arun(operation)
        return _ensure_operation(result.response)


def _ensure_operation(response: Any) -> Operation:  # noqa: ANN401
    """Convert response to Operation type."""
    if isinstance(response, Operation):
        return response
    if isinstance(response, dict):
        return Operation.model_validate(response)
    raise TypeError(f'Expected Operation, got {type(response)}')


class DefineBackgroundModelOptions(BaseModel):
    """Options for defining a background model.

    Matches JS DefineBackgroundModelOptions from js/ai/src/model.ts.

    Attributes:
        name: Unique name for this background model.
        label: Human-readable label (defaults to name).
        versions: Known version names for this model.
        supports: Model capability information.
        config_schema: Custom options schema for this model.
    """

    name: str
    label: str | None = None
    versions: list[str] | None = None
    supports: dict[str, Any] | None = None
    config_schema: type | dict[str, Any] | None = None


def define_background_model(
    registry: Registry,
    name: str,
    start: StartModelOpFn,
    check: CheckModelOpFn,
    cancel: CancelModelOpFn | None = None,
    label: str | None = None,
    info: ModelInfo | None = None,
    config_schema: type | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    description: str | None = None,
) -> BackgroundAction[GenerateResponse]:
    """Define and register a background model.

    This matches the JS defineBackgroundModel function from js/ai/src/model.ts.

    A background model consists of three actions:
    - Start action: /{background-model}/{name}
    - Check action: /check-operation/{name}/check
    - Cancel action: /cancel-operation/{name}/cancel (optional)

    Args:
        registry: The registry to register the actions with.
        name: The unique name for this background model.
        start: Function to start the background operation.
        check: Function to check operation status.
        cancel: Optional function to cancel operations.
        label: Human-readable label (defaults to name).
        info: Model capability information.
        config_schema: Schema for model configuration options.
        metadata: Additional metadata for the model.
        description: Description for the model action.

    Returns:
        A BackgroundAction that can be used to interact with the model.

    Example:
        >>> action = define_background_model(
        ...     registry=registry,
        ...     name='video-gen',
        ...     start=start_fn,
        ...     check=check_fn,
        ... )
        >>> op = await action.start(request)
        >>> while not op.done:
        ...     await asyncio.sleep(5)
        ...     op = await action.check(op)
    """
    label = label or name
    action_key = _make_action_key(ActionKind.BACKGROUND_MODEL, name)

    # Build model metadata matching JS structure
    model_meta: dict[str, Any] = metadata.copy() if metadata else {}
    model_options: dict[str, Any] = {}

    if info:
        info_dict = dump_dict(info)
        if isinstance(info_dict, dict):
            model_options.update(info_dict)  # type: ignore[arg-type]

    model_options['label'] = label
    if config_schema:
        model_options['customOptions'] = to_json_schema(config_schema)

    model_meta['model'] = model_options

    # Build output schema metadata (matching JS)
    output_schema_meta = to_json_schema(GenerateResponse)
    model_meta['outputSchema'] = output_schema_meta

    # Wrap the start function to add the action key and timing (matching JS)
    async def wrapped_start(request: GenerateRequest, ctx: ActionRunContext) -> Operation:
        start_time = time.perf_counter()
        op = await start(request, ctx)
        # Set action key matching JS format: /{actionType}/{name}
        op.action = action_key
        latency_ms = (time.perf_counter() - start_time) * 1000
        if op.metadata is None:
            op.metadata = {}
        op.metadata['latencyMs'] = latency_ms
        return op

    # Wrap the check function (matching JS - no ctx parameter)
    async def wrapped_check(op: Operation, ctx: ActionRunContext) -> Operation:
        updated = await check(op)
        # Preserve action key
        updated.action = action_key
        return updated

    # Register the start action
    # JS: actionType: config.actionType (background-model)
    # JS: name: config.name
    start_action = registry.register_action(
        name=name,
        kind=ActionKind.BACKGROUND_MODEL,
        fn=wrapped_start,
        metadata=model_meta,
        description=description or f'Background model: {label}',
    )

    # Register the check action
    # JS: actionType: 'check-operation'
    # JS: name: `${config.name}/check`
    check_action = registry.register_action(
        name=f'{name}/check',
        kind=ActionKind.CHECK_OPERATION,
        fn=wrapped_check,
        metadata={'outputSchema': output_schema_meta},
        description=f'Check operation status for {label}',
    )

    # Register the cancel action if provided
    # JS: actionType: 'cancel-operation'
    # JS: name: `${config.name}/cancel`
    cancel_action = None
    if cancel is not None:
        # Capture cancel in local scope for the nested function
        cancel_fn = cancel

        async def wrapped_cancel(op: Operation, ctx: ActionRunContext) -> Operation:
            cancelled = await cancel_fn(op)
            cancelled.action = action_key
            return cancelled

        cancel_action = registry.register_action(
            name=f'{name}/cancel',
            kind=ActionKind.CANCEL_OPERATION,
            fn=wrapped_cancel,
            metadata={'outputSchema': output_schema_meta},
            description=f'Cancel operation for {label}',
        )

    return BackgroundAction(
        start_action=start_action,
        check_action=check_action,
        cancel_action=cancel_action,
    )


async def lookup_background_action(
    registry: Registry,
    key: str,
) -> BackgroundAction[GenerateResponse] | None:
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


async def check_operation(
    registry: Registry,
    operation: Operation,
) -> Operation:
    """Check the status of a background operation.

    Matches JS checkOperation from js/ai/src/check-operation.ts.

    Args:
        registry: The registry to look up actions from.
        operation: The operation to check.

    Returns:
        Updated Operation with current status.

    Raises:
        ValueError: If operation is missing action or action not found.
    """
    if not operation.action:
        raise ValueError('Provided operation is missing original request information')

    background_action = await lookup_background_action(registry, operation.action)
    if background_action is None:
        raise ValueError(f'Failed to resolve background action from original request: {operation.action}')

    return await background_action.check(operation)
