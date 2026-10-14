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

"""Error classes and utilities for the Genkit framework."""

import json
import logging
import math
import reprlib
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from enum import IntEnum
from typing import Any, ClassVar, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, model_validator
from pydantic.alias_generators import to_camel

from genkit._core._compat import StrEnum
from genkit._core._typing import Operation, _RuntimeErrorWire


class StatusCodes(IntEnum):
    """gRPC-style status codes. See _STATUS_CODE_MAP for HTTP mappings."""

    OK = 0
    CANCELLED = 1
    UNKNOWN = 2
    INVALID_ARGUMENT = 3
    DEADLINE_EXCEEDED = 4
    NOT_FOUND = 5
    ALREADY_EXISTS = 6
    PERMISSION_DENIED = 7
    RESOURCE_EXHAUSTED = 8
    FAILED_PRECONDITION = 9
    ABORTED = 10
    OUT_OF_RANGE = 11
    UNIMPLEMENTED = 12
    INTERNAL = 13
    UNAVAILABLE = 14
    DATA_LOSS = 15
    UNAUTHENTICATED = 16


# Type alias for status names
StatusName = Literal[
    'OK',
    'CANCELLED',
    'UNKNOWN',
    'INVALID_ARGUMENT',
    'DEADLINE_EXCEEDED',
    'NOT_FOUND',
    'ALREADY_EXISTS',
    'PERMISSION_DENIED',
    'UNAUTHENTICATED',
    'RESOURCE_EXHAUSTED',
    'FAILED_PRECONDITION',
    'ABORTED',
    'OUT_OF_RANGE',
    'UNIMPLEMENTED',
    'INTERNAL',
    'UNAVAILABLE',
    'DATA_LOSS',
]


class ErrorReason(StrEnum):
    """Why the framework classified a failure, beyond its status.

    Read it from ``GenkitError.reason`` or ``Failure.reason``. It travels in
    ``details['reason']``; the other SDKs don't have it.
    """

    INVALID_SCHEMA = 'INVALID_SCHEMA'
    INVALID_INPUT = 'INVALID_INPUT'
    INVALID_OUTPUT = 'INVALID_OUTPUT'
    ACTION_NOT_FOUND = 'ACTION_NOT_FOUND'
    MODEL_NOT_FOUND = 'MODEL_NOT_FOUND'
    TOOL_NOT_FOUND = 'TOOL_NOT_FOUND'
    MAX_TURNS_EXCEEDED = 'MAX_TURNS_EXCEEDED'
    TOOL_FAILED = 'TOOL_FAILED'
    UNSUPPORTED_BY_MODEL = 'UNSUPPORTED_BY_MODEL'
    INVALID_PART = 'INVALID_PART'
    UNRESOLVED_TOOL_REQUEST = 'UNRESOLVED_TOOL_REQUEST'
    INVALID_RESUME = 'INVALID_RESUME'
    SNAPSHOT_NOT_FOUND = 'SNAPSHOT_NOT_FOUND'
    SNAPSHOT_NOT_RESUMABLE = 'SNAPSHOT_NOT_RESUMABLE'
    SESSION_STORE_NOT_CONFIGURED = 'SESSION_STORE_NOT_CONFIGURED'
    SESSION_ID_REQUIRED = 'SESSION_ID_REQUIRED'
    INVALID_SESSION_ID = 'INVALID_SESSION_ID'
    INVALID_SNAPSHOT_ID = 'INVALID_SNAPSHOT_ID'
    CONNECTION_CLOSED = 'CONNECTION_CLOSED'
    # An on_chunk callback (the caller's or one middleware installed) raised; the model isn't retried.
    STREAM_CALLBACK_FAILED = 'STREAM_CALLBACK_FAILED'


def error_reason(details: object) -> ErrorReason | None:
    """Read a known stable reason from error details."""
    if not isinstance(details, Mapping):
        return None
    value = cast(Mapping[str, object], details).get('reason')
    if not isinstance(value, str):
        return None
    try:
        return ErrorReason(value)  # pyrefly: ignore[bad-return]
    except ValueError:
        return None


_STATUS_NAMES: frozenset[str] = frozenset(get_args(StatusName))
_FAILURE_KEYS: frozenset[str] = frozenset({'status', 'message', 'details'})


class Failure(_RuntimeErrorWire):
    """A failure value on ``response.error``, ``Operation.error``, ``AgentOutput.error``, or ``SessionSnapshot.error``.

    Not an exception: generate and agents return failures as values. ``reason``
    is set when the framework classified the failure.

        response = await ai.generate(prompt='Refund order A-17.', tools=['refund_order'])
        response.error.status   # => 'INTERNAL'
        response.error.message  # => 'internal error'
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    # A failure value isn't a set member or dict key, and dict details
    # can't hash anyway.
    __hash__ = None  # type: ignore[assignment]

    status: StatusName | None = None  # pyright: ignore[reportIncompatibleVariableOverride]  # pyrefly: ignore[bad-override]
    details: dict[str, Any] | None = None  # pyright: ignore[reportIncompatibleVariableOverride]  # pyrefly: ignore[bad-override]

    @model_validator(mode='before')
    @classmethod
    def _from_wire(cls, value: object) -> object:
        # A session store built against the generated class still loads.
        if isinstance(value, _RuntimeErrorWire) and not isinstance(value, cls):
            value = value.model_dump(exclude_none=True)
        # A snapshot, operation, or remote agent written by another SDK or a
        # newer version can carry extra keys or a status this one doesn't know.
        # Load it rather than fail the read.
        if isinstance(value, Mapping):
            raw = {k: v for k, v in cast(Mapping[str, object], value).items() if k in _FAILURE_KEYS}
            status = raw.get('status')
            if isinstance(status, str) and status not in _STATUS_NAMES:
                raw['status'] = 'UNKNOWN'
            return raw
        return value

    @property
    def reason(self) -> ErrorReason | None:
        return error_reason(self.details)


Operation.model_rebuild(_types_namespace={'Failure': Failure})


# Mapping of status names to HTTP status codes
_STATUS_CODE_MAP: dict[StatusName, int] = {
    'OK': 200,
    'CANCELLED': 499,
    'UNKNOWN': 500,
    'INVALID_ARGUMENT': 400,
    'DEADLINE_EXCEEDED': 504,
    'NOT_FOUND': 404,
    'ALREADY_EXISTS': 409,
    'PERMISSION_DENIED': 403,
    'UNAUTHENTICATED': 401,
    'RESOURCE_EXHAUSTED': 429,
    'FAILED_PRECONDITION': 400,
    'ABORTED': 409,
    'OUT_OF_RANGE': 400,
    'UNIMPLEMENTED': 501,
    'INTERNAL': 500,
    'UNAVAILABLE': 503,
    'DATA_LOSS': 500,
}

# Reverse of _STATUS_CODE_MAP. A few HTTP codes are shared (400, 409, 500);
# the overlays pick the status retry should treat as the default for that
# code — a bad request, a conflict abort, an internal failure.
_HTTP_CODE_TO_STATUS: dict[int, StatusName] = {code: name for name, code in _STATUS_CODE_MAP.items()}
_HTTP_CODE_TO_STATUS.update({
    400: 'INVALID_ARGUMENT',
    408: 'DEADLINE_EXCEEDED',
    409: 'ABORTED',
    500: 'INTERNAL',
})


def http_status_code(status: StatusName) -> int:
    """Gets the HTTP status code for a given status name.

    Args:
        status: The status name to get the HTTP code for.

    Returns:
        The corresponding HTTP status code.
    """
    return _STATUS_CODE_MAP[status]


def http_code(code: object) -> int | None:
    """A real HTTP status (100-599), or None if this was not a status at all.

    ``-1``, ``0``, ``None``, and ``'nope'`` are missing values, not unmapped
    4xx; provider_error makes either one UNKNOWN.
    """
    if isinstance(code, bool):
        return None
    resolved: int
    if isinstance(code, int):
        resolved = code
    else:
        try:
            resolved = int(code)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    if 100 <= resolved <= 599:
        return resolved
    return None


def from_http_code(code: int) -> StatusName:
    """Canonical status name for an HTTP status code.

    Any 5xx with no explicit entry falls through to ``INTERNAL``; unmapped
    4xx codes return ``UNKNOWN``. A 408 is ``DEADLINE_EXCEEDED`` so retry
    can wait out a transient timeout. provider_error maps with this so
    retry can skip a 400 without also skipping a 503.
    """
    mapped = _HTTP_CODE_TO_STATUS.get(code)
    if mapped is not None:
        return mapped
    if code >= 500:
        return 'INTERNAL'
    return 'UNKNOWN'


def parse_retry_after_ms(value: str) -> float | None:
    """Parse an HTTP Retry-After value into milliseconds.

    Accepts delay-seconds (``60``, ``1.5``) and HTTP-date values. Retry uses
    this as a floor so a provider that said wait 60s is not hit again in 1s.
    """
    value = value.strip()
    if not value:
        return None

    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        # Check the scaled value: a large finite input can overflow to inf.
        retry_after_ms = seconds * 1000
        if seconds >= 0 and math.isfinite(retry_after_ms):
            return retry_after_ms

    try:
        retry_at_ms = parsedate_to_datetime(value).timestamp() * 1000
    except (OSError, OverflowError, TypeError, ValueError):
        return None
    return max(0.0, retry_at_ms - time.time() * 1000)


def retry_after_ms_from_headers(headers: object) -> float | None:
    """Retry-After from response headers in ms, or None when absent or unreadable.

    Header names are matched case-insensitively since SDKs hand back
    'Retry-After', 'retry-after', or a case-insensitive mapping.
    """
    if headers is None:
        return None
    try:
        raw = next(
            (value for key, value in headers.items() if isinstance(key, str) and key.lower() == 'retry-after'),  # type: ignore[attr-defined]
            None,
        )
    except (AttributeError, TypeError):
        return None
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if not isinstance(raw, str):
        raw = str(raw) if raw is not None else None
    if not raw:
        return None
    return parse_retry_after_ms(raw)


# =============================================================================
# Error Classes
# =============================================================================


class ReflectionErrorDetails(BaseModel):
    """Wire format for reflection API error details."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow', populate_by_name=True, alias_generator=to_camel)

    stack: str | None = None
    trace_id: str | None = None


class ReflectionError(BaseModel):
    """Wire format for reflection API errors."""

    details: ReflectionErrorDetails | None = None
    message: str
    code: int = StatusCodes.INTERNAL.value

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra='forbid',
        populate_by_name=True,
    )


class Interrupt(BaseException):
    """Pause a tool so the caller can approve, reply, or restart.

    Raise ``Interrupt(metadata)`` from a tool or from tool middleware.
    It is a control-flow signal like ``asyncio.CancelledError``, so it
    subclasses ``BaseException``: an ``except Exception`` in tool code can't
    swallow it and skip the approval. Catch ``Interrupt`` by name if a tool
    needs to clean up, then re-raise. Raised anywhere else (a flow, an
    agent turn body), the action fails with ``GenkitError`` INTERNAL.
    Tracing treats it as control flow, not a failed span.
    """

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.metadata: dict[str, Any] = {} if metadata is None else metadata


class _StreamingCallbackError(BaseException):
    """Internal carrier for an exception raised by an ``on_chunk`` callback.

    Covers the caller's callback and any callback middleware installs on
    ``GenerateMiddlewareContext.on_chunk``. Subclasses ``BaseException`` (like
    ``Interrupt`` and ``asyncio.CancelledError``) so ``except Exception``
    blocks in model plugins and middleware cannot catch or retry a stream
    callback failure. ``generate()`` unwraps ``__cause__`` and boxes the
    original failure onto ``ModelResponse`` with reason
    ``STREAM_CALLBACK_FAILED``.
    """


# Short previews of the offending value: `'acme'`, `None`, `{'dish': 'pad thai', ...}`.
_value_preview = reprlib.Repr()
_value_preview.maxstring = _value_preview.maxother = 40
_value_preview.maxlist = _value_preview.maxtuple = _value_preview.maxdict = _value_preview.maxset = 3
_value_preview.maxlevel = 2


def format_validation_error(error: ValidationError, *, max_errors: int = 3) -> str:
    """One short clause per Pydantic error: where, what was expected, what came in.

    Pydantic already words each error for every type it validates (str, int,
    models, lists, dicts, unions, Literal, Enum, TypedDict, dataclasses), so
    this only drops the noise around it: the "N validation errors for X"
    header, the ``[type=..., input_value=...]`` bracket, and the docs URL.

    Example:
        ``items[1].qty: Field required; table: Input should be a valid integer, got 'x'``
    """
    # Follow Pydantic's own decision about whether the value is safe to print.
    hide_input = error.error_count() > 0 and 'input_value=' not in str(error)
    problems: list[str] = []
    for err in error.errors(include_url=False)[:max_errors]:
        text = err['msg']
        # For a missing field the input is the whole parent object, which says nothing new.
        if err['type'] != 'missing' and not hide_input:
            text = f'{text}, got {_value_preview.repr(err["input"])}'
        path = ''.join(f'[{p}]' if isinstance(p, int) else f'.{p}' for p in err['loc']).lstrip('.')
        problems.append(f'{path}: {text}' if path else text)
    hidden = error.error_count() - max_errors
    if hidden > 0:
        problems.append(f'and {hidden} more')
    return '; '.join(problems)


class GenkitError(Exception):
    """Base error class for Genkit errors."""

    def __init__(
        self,
        *,
        message: str,
        status: StatusName | None = None,
        cause: BaseException | None = None,
        details: Any = None,  # noqa: ANN401
        reason: ErrorReason | None = None,
        source: str | None = None,
        retry_after_ms: float | None = None,
    ) -> None:
        """Initialize a GenkitError.

        Args:
            message: The error message.
            status: The status name for this error.
            cause: The underlying exception; set as ``__cause__``.
            details: Optional detail information.
            reason: Extra why when we classified the failure.
            source: Optional prefix for ``str(err)``.
            retry_after_ms: How long the provider asked callers to wait before retrying.
        """
        temp_status: StatusName
        if status:
            temp_status = status
        elif isinstance(cause, GenkitError):
            temp_status = cause.status
        else:
            temp_status = 'INTERNAL'
        self.status: StatusName = temp_status
        self._http_code: int = http_status_code(temp_status)

        # When this error wraps another (the common shape — the action runtime
        # catches the underlying failure and re-raises as ``GenkitError(...,
        # cause=original)``), surface the cause in the default string form so
        # downstream consumers (logs, model-facing tool error messages, the Dev
        # UI) see the real reason instead of the bare wrapper text.
        source_prefix = f'{source}: ' if source else ''
        if isinstance(cause, ValidationError):
            formatted = format_validation_error(cause)
            cause_suffix = f': {formatted}' if formatted else ''
        else:
            # Skip a cause whose text is empty or already in the message, so a
            # bare TimeoutError() adds no dangling ': ' and nothing repeats.
            cause_text = str(cause) if cause is not None else ''
            cause_suffix = f': {cause_text}' if cause_text and cause_text not in message else ''
        super().__init__(f'{source_prefix}{self.status}: {message}{cause_suffix}')
        self.message: str = message

        if not details:
            details = {}
        if reason is not None:
            details = dict(details)
            details['reason'] = reason.value
        if isinstance(cause, ValidationError) and 'errors' not in details:
            details = dict(details)
            details['errors'] = [
                {'loc': list(err['loc']), 'message': err['msg'], 'type': err['type']} for err in cause.errors()
            ]

        self.details: Any = details
        self._source: str | None = source
        if cause is not None:
            self.__cause__ = cause
        # Only the cause passed as ``cause=``. ``raise ... from e`` also sets
        # ``__cause__``, but finish-message redaction and the Dev UI message
        # key off what the raiser chose to attach.
        self._cause: BaseException | None = cause
        self.retry_after_ms: float | None = retry_after_ms
        # Plugin errors built from a provider response keep their status
        # in-process (Retry, Fallback) but serve as a crash at the HTTP
        # boundary so a dead server key is not a 401 to the end caller.
        self._provider_sourced: bool = False
        # The served action's own input/init check is the caller's request,
        # so it keeps a 4xx. A plugin "bad role" inside the flow does not.
        self._request_sourced: bool = False

    @property
    def reason(self) -> ErrorReason | None:
        return error_reason(self.details)

    def _to_serializable(self) -> ReflectionError:
        """Dev UI (reflection) error body; ``get_reflection_json`` adds the trace id."""
        details = dict(self.details) if isinstance(self.details, Mapping) else {}
        details.setdefault('stack', get_error_stack(self._cause or self))
        return ReflectionError(
            details=ReflectionErrorDetails(**details),
            code=StatusCodes[self.status].value,
            message=f'{self.message}: {repr(self._cause)}' if self._cause else self.message,
        )


def mark_request_error(*, error: GenkitError) -> GenkitError:
    """Mark an error from the served action's own input or init check.

    The HTTP caller sent a body the action cannot accept, so they get a 4xx
    with a generic sentence. The validation dump stays off the wire.
    """
    error._request_sourced = True
    return error


def mark_provider_error(*, error: GenkitError) -> GenkitError:
    """Mark an error built from a provider response.

    Served flows treat this like a crash (500 Internal Error, traceback in
    the logs). In-process callers still see the real status so Retry and
    Fallback can act on it. An app that wants the caller to see "busy, try
    later" raises PublicError itself.
    """
    error._provider_sourced = True
    return error


def provider_error(
    error: BaseException,
    *,
    http_status: int | None = None,
    status: StatusName | None = None,
    headers: Mapping[str, str] | None = None,
    retry_after_ms: float | None = None,
    message: str | None = None,
) -> GenkitError:
    """Turn a model provider's failure into a GenkitError that Retry and Fallback can act on.

    Raise the result from a model or embedder when the provider call fails:

        except httpx.HTTPStatusError as e:
            raise provider_error(e, http_status=e.response.status_code, headers=e.response.headers) from e

    Status: ``status`` if given, else mapped from ``http_status`` (400 is
    INVALID_ARGUMENT, 429 RESOURCE_EXHAUSTED, 503 UNAVAILABLE, any other 5xx
    INTERNAL). A 4xx with no matching status (413, 422) or no usable code at
    all is UNKNOWN, which default Retry and Fallback don't act on.

    Retry delay: ``retry_after_ms`` if given, else the ``Retry-After`` header
    (seconds or an HTTP date). Retry waits at least that long before the next
    attempt. A missing or malformed header just means no floor.

    The message is ``message``, else ``str(error)``, else the error's type
    name, and ``error`` is kept as the cause; ``str()`` appends the cause's
    text unless the message already contains it. A served flow still answers
    500 Internal Error: a dead API key on the server isn't the caller's 401.
    Raise PublicError to show the caller a status.

    Args:
        error: What the provider SDK raised.
        http_status: The provider's HTTP status code, if there was one.
        status: A Genkit status to use instead of mapping ``http_status``.
        headers: The provider's response headers.
        retry_after_ms: How long to wait before retrying, instead of reading headers.
        message: Message to use instead of ``str(error)``.

    Returns:
        A GenkitError to raise.
    """
    if status is None:
        code = http_code(http_status)
        # A 2xx/3xx on an exception isn't a failure status we can trust.
        status = from_http_code(code) if code is not None and code >= 400 else 'UNKNOWN'
    if retry_after_ms is None:
        retry_after_ms = retry_after_ms_from_headers(headers)
    # A bare TimeoutError() or httpx.ReadTimeout often has empty str(); the type
    # name at least says what failed.
    text = message if message is not None else (str(error) or type(error).__name__)
    result = GenkitError(status=status, message=text, cause=error, retry_after_ms=retry_after_ms)
    return mark_provider_error(error=result)


def interrupt_outside_tool_error(interrupt: Interrupt) -> GenkitError:
    """The error an Interrupt becomes when it escapes something that isn't a tool.

    Only generate's tool loop can turn an Interrupt into a pause. Anywhere
    else it would escape ``except Exception`` handlers as a bare
    BaseException, so the action boundary and the agent turn loop report it
    as a failed run instead. ``__cause__`` keeps the metadata for debugging.
    """
    return GenkitError(status='INTERNAL', message='Interrupt raised outside a tool', cause=interrupt)


class PublicError(GenkitError):
    """Error class for issues to be returned to users.

    Using this error allows a web framework handler (e.g. FastAPI, Flask) to know it
    is safe to return the message, details, and HTTP status in a request. Any
    other GenkitError is 500 Internal Error on the wire.
    """

    def __init__(self, status: StatusName, message: str, details: Any = None) -> None:  # noqa: ANN401
        """Initialize a PublicError.

        Args:
            status: The status name for this error.
            message: The error message.
            details: Optional details to include.
        """
        super().__init__(status=status, message=message, details=details)


_INTERNAL_CLIENT_BODY: dict[str, Any] = {'message': 'Internal Error', 'status': 'INTERNAL'}


def _client_facing_error(error: object) -> GenkitError | None:
    """The error whose status and sentence a served flow may show, or None to redact.

    A PublicError is the app saying this status and sentence are for the
    caller. The served action's own input/init check is the caller's
    request, so it keeps a 4xx and a generic sentence. A missing model or
    a plugin "bad role" is the server failing to run the flow.
    """
    if isinstance(error, PublicError):
        return error
    if isinstance(error, GenkitError) and error._request_sourced and not error._provider_sourced:
        return error
    return None


def _generic_client_message(status: StatusName) -> str:
    """'INVALID_ARGUMENT' -> 'Invalid argument'; INTERNAL uses 'Internal Error'."""
    if status == 'INTERNAL':
        return 'Internal Error'
    return status.replace('_', ' ').capitalize()


_ANY_ADAPTER: TypeAdapter[Any] = TypeAdapter(Any)


def _client_details(details: Any) -> Any:  # noqa: ANN401
    """Details safe to put on the wire: dump nested models, drop stack, omit empty."""
    if not details:
        return None
    dumped = _ANY_ADAPTER.dump_python(details, mode='json', by_alias=True, exclude_none=True)
    if isinstance(dumped, dict):
        dumped.pop('stack', None)
        return dumped or None
    return dumped


def get_http_status(error: object) -> int:
    """HTTP status for a served-flow error.

    A PublicError keeps its own status (NOT_FOUND is a 404). Any other
    GenkitError, a provider-sourced error, a plain exception, or anything
    else is a 500.
    """
    facing = _client_facing_error(error)
    if facing is not None:
        return facing._http_code  # noqa: SLF001
    return 500


def get_reflection_json(error: object, *, trace_id: str | None = None) -> ReflectionError:
    """Get the JSON representation of an error for reflection API responses.

    Args:
        error: The error to convert to JSON.
        trace_id: The run's trace id, used when the error doesn't carry one,
            so the Dev UI can link a failed run to its trace.

    Returns:
        A ReflectionError model instance.
    """
    if isinstance(error, GenkitError):
        ref = error._to_serializable()  # noqa: SLF001
    else:
        ref = ReflectionError(
            message=str(error),
            code=StatusCodes.INTERNAL.value,
            details=ReflectionErrorDetails(stack=get_error_stack(error)),
        )
    if not trace_id or (ref.details is not None and ref.details.trace_id):
        return ref
    details = (
        ref.details.model_copy(update={'trace_id': trace_id})
        if ref.details is not None
        else ReflectionErrorDetails(trace_id=trace_id)
    )
    return ref.model_copy(update={'details': details})


def get_callable_json(error: object) -> dict[str, Any]:
    """JSON body for a served-flow HTTP or SSE error.

    Only a PublicError's message, details, and status go on the wire; it's
    the one error whose author said the text is safe for callers. Any other
    GenkitError, a provider-sourced error, and anything else become
    ``{"message": "Internal Error", "status": "INTERNAL"}``.
    """
    facing = _client_facing_error(error)
    if facing is None:
        return dict(_INTERNAL_CLIENT_BODY)
    message = facing.message if isinstance(facing, PublicError) else _generic_client_message(facing.status)
    body: dict[str, Any] = {
        'message': message,
        'status': facing.status,
    }
    if isinstance(facing, PublicError):
        details = _client_details(facing.details)
        if details is not None:
            body['details'] = details
    return body


_JSON_SEPARATORS = (',', ':')


def served_error_json(*, error: object) -> tuple[int, str]:
    """HTTP status and compact JSON body for a served-flow failure."""
    return get_http_status(error), json.dumps(get_callable_json(error), separators=_JSON_SEPARATORS)


def served_stream_error_event(*, error: object) -> str:
    """SSE ``data: {"error": ...}`` event for a served-flow failure."""
    return f'data: {json.dumps({"error": get_callable_json(error)}, separators=_JSON_SEPARATORS)}\n\n'


def log_served_failure(*, adapter_logger: logging.Logger, error: Exception, where: str) -> None:
    """Log a served-flow failure; 5xx includes the traceback."""
    if get_http_status(error) >= 500:
        adapter_logger.exception('served flow %s failed', where)
    else:
        adapter_logger.warning('served flow %s failed: %s', where, error)


def get_error_stack(error: object) -> str | None:
    """Extract stack trace from an error object.

    Args:
        error: The error to get the stack trace from.

    Returns:
        The stack trace string if available, None otherwise.
    """
    if isinstance(error, BaseException):
        # Stack traces are valuable for debugging; consider making this configurable
        # to enable them in development/staging and suppress in production.
        # For now, return an empty string to keep Dev UI clean as per requirements.
        return ''
    return None
