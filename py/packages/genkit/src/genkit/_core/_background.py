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

"""Background actions: start, check and cancel over a generic Operation.

Mirrors Go ``core.BackgroundActionDef[In, Out]`` and JS
``BackgroundAction<I, O>``. Knows nothing about models; the AI layer builds
``BackgroundModel`` on top of it (``genkit._ai._background_model``).
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast, get_args, get_origin

from pydantic import BaseModel, ValidationError
from typing_extensions import Self

from genkit._core._action import Action, ActionKind, ActionRunContext, get_current_context, set_action_name
from genkit._core._error import GenkitError, RuntimeErrorReason, StatusName
from genkit._core._operation import Operation
from genkit._core._schema import to_json_schema
from genkit._core._typing import ActionMetadata

if TYPE_CHECKING:
    from genkit._core._registry import Registry

InT = TypeVar('InT')
OutT = TypeVar('OutT')

# Annotation-only aliases. Operation[OutT] is a concrete pydantic class at
# runtime, so CheckOpFn[X] can't be subscripted outside an annotation; spell
# the Callable out when you need a runtime alias.
StartOpFn = Callable[[InT, ActionRunContext], Awaitable[Operation]]
CheckOpFn = Callable[[Operation[OutT], ActionRunContext], Awaitable[Operation]]
CancelOpFn = Callable[[Operation[OutT], ActionRunContext], Awaitable[Operation]]


def _make_action_key(action_type: ActionKind | str, name: str) -> str:
    """Create an action key in format: /{action_type}/{name}."""
    return f'/{action_type}/{name}'


def stamp_operation_action(
    *,
    operation: Operation,
    name: str,
    kind: ActionKind = ActionKind.BACKGROUND_MODEL,
) -> None:
    """A handle needs the start action key so check/cancel can find the job."""
    if operation.action:
        return
    operation.action = _make_action_key(kind, name)


def first_validation_error(error: ValidationError) -> str:
    """One line naming the first bad field, e.g. ``Done: Extra inputs are not permitted``."""
    errors = error.errors()
    first = errors[0]
    loc = '.'.join(str(part) for part in first['loc']) or '<root>'
    more = f' (+{len(errors) - 1} more)' if len(errors) > 1 else ''
    return f'{loc}: {first["msg"]}{more}'


def _operation_class(output_type: type[Any] | None) -> type[Operation[Any]]:
    # A runtime type argument, so the subscript goes through Any for checkers.
    return cast(Any, Operation)[output_type] if output_type is not None else Operation


def validate_operation(
    operation: Operation,
    *,
    output_type: type[Any] | None,
    name: str,
    status: StatusName,
) -> Operation[Any]:
    """The same handle as an ``Operation[output_type]``, with every field validated.

    A fn may return a plain ``Operation`` or assign ``op.output = {...}`` or
    ``op.error = ...``, which skips validation, so the handle is always
    rebuilt. Nested model instances are not revalidated, so this is cheap.
    ``status`` says whose mistake a bad field is: INTERNAL (reason
    INVALID_OUTPUT) for what a fn returned, INVALID_ARGUMENT (reason
    INVALID_INPUT) for a handle the caller passed in.

    Raises:
        GenkitError: ``status`` naming the first field that fails.
    """
    fields = {field: getattr(operation, field) for field in operation.model_fields_set}
    try:
        return _operation_class(output_type).model_validate(fields)
    except ValidationError as e:
        raise GenkitError(
            status=status,
            message=f"'{name}': invalid Operation: {first_validation_error(e)}",
            reason=RuntimeErrorReason.INVALID_INPUT
            if status == 'INVALID_ARGUMENT'
            else RuntimeErrorReason.INVALID_OUTPUT,
        ) from e


def require_operation(*, value: object, output_type: type[Any] | None = None) -> Operation[Any]:
    """A poll handle: a live Operation, or the dict a stored one dumped to.

    Either is validated as an ``Operation[output_type]`` as strictly as the
    check/cancel action input (``extra='forbid'``), so a typo'd key like
    ``Done`` or a bad ``output`` fails here with the field named, before any
    action runs.
    """
    if isinstance(value, Operation):
        fields: dict[str, object] = {field: getattr(value, field) for field in value.model_fields_set}
    elif isinstance(value, Mapping):
        fields = dict(value)
    else:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'got {type(value).__name__}, expected Operation',
            reason=RuntimeErrorReason.INVALID_INPUT,
        )
    try:
        return _operation_class(output_type).model_validate(fields)
    except ValidationError as e:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'invalid Operation: {first_validation_error(e)}',
            reason=RuntimeErrorReason.INVALID_INPUT,
        ) from e


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


class MissingOperationError(GenkitError):
    """A background action lost the operation handle it created."""


def missing_operation_error(*, name: str) -> MissingOperationError:
    """The caller asked for a handle and this action did not return one."""
    return MissingOperationError(
        status='FAILED_PRECONDITION',
        message=f"'{name}' did not return an operation.",
    )


def _concrete_type(arg: object) -> type[Any] | None:
    """A type argument Genkit can validate against; None for TypeVars and Any."""
    return arg if isinstance(arg, type) else None


class BackgroundAction(Generic[InT, OutT]):
    """A handle over a background job's start, check and cancel actions.

    A composite, not an ``Action``: start, check and cancel each have their
    own registry key. ``start`` returns an ``Operation[OutT]``; pass it to
    ``check`` until ``done``. Subclass with concrete type arguments, e.g.
    ``class ReportJob(BackgroundAction[ReportRequest, Report])``, and the
    handle's ``output`` validates as ``Report``.

    Every attribute is either a ClassVar derived from the type arguments or
    derived from the three actions in ``_bind``, so a handle rebuilt from the
    registry (``_from_actions``) behaves like the one that was defined.
    """

    # Derived from the type arguments in __init_subclass__. None means Any.
    _input_type: ClassVar[type[Any] | None] = None
    _output_type: ClassVar[type[Any] | None] = None
    # How error messages name this kind of job.
    _noun: ClassVar[str] = 'Background action'
    _cancel_unsupported_reason: ClassVar[RuntimeErrorReason | None] = None

    start_action: Action
    check_action: Action
    cancel_action: Action | None

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Read InT / OutT off ``class X(BackgroundAction[In, Out])``."""
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, '__orig_bases__', ()):
            origin = get_origin(base)
            if isinstance(origin, type) and issubclass(origin, BackgroundAction):
                args = get_args(base)
                if len(args) == 2:
                    cls._input_type = _concrete_type(args[0])
                    cls._output_type = _concrete_type(args[1])

    def __init__(
        self,
        start_action: Action,
        check_action: Action,
        cancel_action: Action | None = None,
    ) -> None:
        """Wrap already-built start, check and (optional) cancel actions."""
        self._bind(start_action, check_action, cancel_action)

    def _bind(self, start: Action, check: Action, cancel: Action | None) -> None:
        """The only writer of instance state. Subclasses derive extra fields here."""
        self.start_action = start
        self.check_action = check
        self.cancel_action = cancel

    @classmethod
    def _from_actions(cls, start: Action, check: Action, cancel: Action | None = None) -> Self:
        """Rebuild from registered actions without re-running the fn wrapping."""
        obj = cls.__new__(cls)
        obj._bind(start, check, cancel)
        return obj

    @classmethod
    async def lookup(cls, registry: Registry, key: str) -> Self | None:
        """Look up a background action by its start action key.

        Matches JS ``lookupBackgroundAction``. The key format is
        ``/{actionType}/{name}``, e.g. ``/background-model/video-gen``.
        """
        start_action = await registry.resolve_action_by_key(key)
        if start_action is None:
            return None
        # JS: const actionName = key.substring(key.indexOf('/', 1) + 1);
        parts = key.split('/', 2)  # ['', 'background-model', 'name']
        if len(parts) < 3:
            return None
        action_name = parts[2]
        check_action = await registry.resolve_action_by_key(f'/{ActionKind.CHECK_OPERATION}/{action_name}/check')
        if check_action is None:
            return None
        cancel_action = await registry.resolve_action_by_key(f'/{ActionKind.CANCEL_OPERATION}/{action_name}/cancel')
        return cls._from_actions(start_action, check_action, cancel_action)

    @property
    def name(self) -> str:
        """The name of the start action."""
        return self.start_action.name

    @property
    def kind(self) -> ActionKind:
        """The action kind of the start action."""
        return self.start_action.kind

    @property
    def metadata(self) -> dict[str, Any]:
        """The start action's metadata."""
        return self.start_action.metadata

    @property
    def input_schema(self) -> dict[str, Any] | None:
        """The start action's input JSON schema."""
        return self.start_action.input_schema

    @property
    def output_schema(self) -> dict[str, Any] | None:
        """The start action's output JSON schema (the Operation)."""
        return self.start_action.output_schema

    @property
    def supports_cancel(self) -> bool:
        """Whether this background action supports cancellation."""
        return self.cancel_action is not None

    @property
    def actions(self) -> list[Action]:
        """The start, check and (if any) cancel actions, for registering or listing.

        A background action is three registered actions, not one. Register all
        of them, or a poll fails later with the check action not found.
        """
        actions = [self.start_action, self.check_action]
        if self.cancel_action is not None:
            actions.append(self.cancel_action)
        return actions

    def _action_metadata(self) -> ActionMetadata:
        """Listed once, as the start action, for Registry.list_actions()."""
        return self.start_action._action_metadata()

    def _rename(self, name: str) -> None:
        """Name start ``name``, check ``name/check``, cancel ``name/cancel``."""
        set_action_name(self.start_action, name)
        set_action_name(self.check_action, f'{name}/check')
        if self.cancel_action is not None:
            set_action_name(self.cancel_action, f'{name}/cancel')

    def _action_for(self, kind: ActionKind) -> Action | None:
        """The member action a lookup of ``kind`` wants: check, cancel, or start."""
        if kind == ActionKind.CHECK_OPERATION:
            return self.check_action
        if kind == ActionKind.CANCEL_OPERATION:
            return self.cancel_action
        return self.start_action

    def _require_handle(self, value: object) -> Operation[OutT]:
        """A caller's handle (Operation or stored dict) as ``Operation[OutT]``."""
        return require_operation(value=value, output_type=self._output_type)

    def _ensure_operation(self, *, response: object, name: str) -> Operation[OutT]:
        """A start/check/cancel fn returns an Operation, not a dict."""
        if isinstance(response, Operation):
            return validate_operation(response, output_type=self._output_type, name=name, status='INTERNAL')
        raise missing_operation_error(name=name)

    async def start(
        self,
        input: InT | None = None,  # noqa: A002
        *,
        context: dict[str, Any] | None = None,
    ) -> Operation[OutT]:
        """Start a background operation.

        Args:
            input: The start request.
            context: Optional run context. Per-request keys go in
                ``context['secrets']``.

        Returns:
            An Operation with an ID to track the job.
        """
        # Same pocket as check/cancel: a tenant key on start has to reach the
        # plugin, not die on this wrapper.
        result = await self.start_action.run(input, context=context)
        return self._ensure_operation(response=result.response, name=self.start_action.name)

    async def check(
        self,
        operation: Operation | Mapping[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> Operation[OutT]:
        """Check the status of a background operation.

        Args:
            operation: The operation to check, or the dict a stored one
                dumped to.
            context: Optional run context (secrets, folded client config).

        Raises:
            GenkitError: INVALID_ARGUMENT if ``operation`` is not an
                ``Operation`` or a dict that validates as one.
        """
        operation = self._require_handle(operation)
        result = await self.check_action.run(operation, context=context)
        return self._ensure_operation(response=result.response, name=self.check_action.name)

    async def cancel(
        self,
        operation: Operation | Mapping[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> Operation[OutT]:
        """Cancel a background operation.

        Args:
            operation: The operation to cancel, or the dict a stored one
                dumped to.
            context: Optional run context (secrets, folded client config).

        Raises:
            GenkitError: UNIMPLEMENTED if this action does not implement
                cancel, INVALID_ARGUMENT if ``operation`` is not an
                ``Operation`` or a dict that validates as one.
        """
        operation = self._require_handle(operation)
        # Raising here is deliberate: returning the operation unchanged would
        # make "can't cancel" indistinguishable from "cancelled".
        if self.cancel_action is None:
            raise GenkitError(
                status='UNIMPLEMENTED',
                message=f'{self._noun} {operation.action} does not support cancellation.',
                reason=self._cancel_unsupported_reason,
            )
        result = await self.cancel_action.run(operation, context=context)
        return self._ensure_operation(response=result.response, name=self.cancel_action.name)


def _operation_action(
    *,
    cls: type[BackgroundAction[Any, Any]],
    kind: ActionKind,
    start_kind: ActionKind,
    name: str,
    fn: Callable[..., Awaitable[Operation]],
    description: str,
    metadata: dict[str, object],
    suffix: str = '',
    config_schema: type[BaseModel] | dict[str, Any] | None = None,
    check_start_input: Callable[[object, str], None] | None = None,
) -> Action:
    """An Action for start/check/cancel that validates and stamps the Operation.

    ``fn``'s signature is still the Action's (``metadata_fn``), so
    ``(ctx, request)`` and ``(request, ctx)`` both work: the wrapper forwards
    through ``params.call``. The stamp is the start action key, so a caller who
    passes the Operation back reaches the right check/cancel.

    The action is named ``{name}{suffix}``. The registry may add the plugin
    prefix after this is built, so the key is read from the action's live name
    minus ``suffix`` (e.g. ``/check``) when it runs.
    """
    # Action rejects a sync fn, but it only sees the async wrapper below.
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(f"Action handlers must be async functions. Got sync function for '{name}{suffix}'.")
    output_type = cls._output_type

    # wraps keeps fn's annotations on the wrapper, e.g. ModelRequest[VeoConfig].
    @wraps(fn)
    async def run_and_stamp(input: object, ctx: ActionRunContext) -> Operation:  # noqa: A002
        if check_start_input is not None:
            check_start_input(input, action.name)
        if isinstance(input, Operation):
            # check/cancel: the caller's handle, often reloaded from storage.
            input = validate_operation(input, output_type=output_type, name=action.name, status='INVALID_ARGUMENT')  # noqa: A001
        op = await action.params.call(fn, input, ctx)
        if isinstance(op, Operation):
            stamp_operation_action(operation=op, name=action.name.removesuffix(suffix), kind=start_kind)
            op = validate_operation(op, output_type=output_type, name=action.name, status='INTERNAL')
        return op

    action = Action(
        kind=kind,
        name=f'{name}{suffix}',
        fn=run_and_stamp,
        metadata_fn=fn,
        metadata=metadata,
        description=description,
        config_schema=config_schema,
    )
    return action


def operation_actions(
    cls: type[BackgroundAction[Any, Any]],
    name: str,
    *,
    kind: ActionKind,
    start: Callable[..., Awaitable[Operation]],
    check: Callable[..., Awaitable[Operation]],
    cancel: Callable[..., Awaitable[Operation]] | None = None,
    metadata: dict[str, Any] | None = None,
    description: str | None = None,
    label: str | None = None,
    config_schema: type[BaseModel] | dict[str, Any] | None = None,
    check_start_input: Callable[[object, str], None] | None = None,
) -> tuple[Action, Action, Action | None]:
    """Build the start, check and cancel actions for a ``cls`` background job.

    Like JS ``backgroundAction``, check and cancel carry the start metadata
    plus ``outputSchema`` (the ``Operation.output`` type, from ``cls``).
    """
    label = label or name
    base_meta: dict[str, Any] = metadata.copy() if metadata else {}
    start_meta = dict(base_meta)
    sub_meta = dict(base_meta)
    if cls._output_type is not None:
        output_schema_meta = to_json_schema(cls._output_type)
        start_meta['outputSchema'] = output_schema_meta
        sub_meta['outputSchema'] = output_schema_meta

    start_action = _operation_action(
        cls=cls,
        kind=kind,
        start_kind=kind,
        name=name,
        fn=start,
        metadata=start_meta,
        description=description or f'{cls._noun}: {label}',
        config_schema=config_schema,
        check_start_input=check_start_input,
    )
    if not start_action.input_schema and cls._input_type is not None:
        start_action.input_schema = to_json_schema(cls._input_type)
    if not start_action.output_schema:
        start_action.output_schema = to_json_schema(Operation)

    check_action = _operation_action(
        cls=cls,
        kind=ActionKind.CHECK_OPERATION,
        start_kind=kind,
        name=name,
        suffix='/check',
        fn=check,
        metadata=dict(sub_meta),
        description=f'Check operation status for {label}',
    )
    cancel_action = None
    if cancel is not None:
        cancel_action = _operation_action(
            cls=cls,
            kind=ActionKind.CANCEL_OPERATION,
            start_kind=kind,
            name=name,
            suffix='/cancel',
            fn=cancel,
            metadata=dict(sub_meta),
            description=f'Cancel operation for {label}',
        )
    return start_action, check_action, cancel_action


BackgroundActionT = TypeVar('BackgroundActionT', bound=BackgroundAction[Any, Any])


async def resolve_operation_action(
    registry: Registry,
    key: str | None,
    *,
    lookup: Callable[[Registry, str], Awaitable[BackgroundActionT | None]],
    noun: str = 'background action',
) -> BackgroundActionT:
    """The background action that owns a handle's ``action`` key."""
    if not key:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='Provided operation is missing original request information',
            reason=RuntimeErrorReason.INVALID_INPUT,
        )
    try:
        found = await lookup(registry, key)
    except ValueError as e:
        # The key is caller data (often reloaded from storage), so a mangled
        # key is the caller's bad argument, not an internal failure.
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'Failed to resolve {noun} from original request: {key}',
            reason=RuntimeErrorReason.ACTION_NOT_FOUND,
        ) from e
    if found is None:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'Failed to resolve {noun} from original request: {key}',
            reason=RuntimeErrorReason.ACTION_NOT_FOUND,
        )
    return found


def background_base_id(kind: ActionKind, requested_id: str) -> str:
    """The start id behind a resolve request for any of the three keys.

    ``/check-operation/{n}/check`` and ``/cancel-operation/{n}/cancel`` both
    belong to the background action registered as ``{n}``.
    """
    if kind == ActionKind.CHECK_OPERATION:
        return requested_id.removesuffix('/check')
    if kind == ActionKind.CANCEL_OPERATION:
        return requested_id.removesuffix('/cancel')
    return requested_id


def operation_key(value: object) -> str | None:
    """The ``action`` key on a handle or its stored dict, if any."""
    if isinstance(value, Operation):
        return value.action
    if isinstance(value, Mapping):
        key = cast(Mapping[str, object], value).get('action')
        return key if isinstance(key, str) else None
    return None
