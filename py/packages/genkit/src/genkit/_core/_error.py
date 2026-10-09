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

import math
import reprlib
import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from enum import IntEnum
from typing import Any, ClassVar, Literal, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from pydantic.alias_generators import to_camel

from genkit._core._compat import StrEnum
from genkit._core._typing import GenkitRuntimeError as GenkitRuntimeErrorData


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


class RuntimeErrorReason(StrEnum):
    """Extra why on a classified generate failure or a helper raise.

    The message stays human. The helper that fails sets this so it
    bubbles on the exception the caller actually catches.
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


def runtime_error_reason(details: object) -> RuntimeErrorReason | None:
    """Read a known stable reason from runtime error details."""
    if not isinstance(details, Mapping):
        return None
    value = cast(Mapping[str, object], details).get('reason')
    if not isinstance(value, str):
        return None
    try:
        return RuntimeErrorReason(value)  # pyrefly: ignore[bad-return]
    except ValueError:
        return None


class GenkitRuntimeError(GenkitRuntimeErrorData):
    """Classified failure carried as data: ``response.error``, ``AgentOutput.error``, ``SessionSnapshot.error``.

    Wire shape is the shared ``RuntimeError`` schema (status, message, details).

    Plain data, not an exception: generate returns failures as values, so
    ``raise res.error`` would make a returning call look like a throwing one.
    ``reason`` is set when the framework classified the failure, so callers
    can branch without parsing the message.

    Fields can't be reassigned. ``details`` is the dict as received.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    # A failure value isn't a set member or dict key, and dict details
    # can't hash anyway.
    __hash__ = None  # type: ignore[assignment]

    @model_validator(mode='before')
    @classmethod
    def _from_wire(cls, value: object) -> object:
        # A session store built against the generated class still loads.
        if isinstance(value, GenkitRuntimeErrorData) and not isinstance(value, cls):
            return value.model_dump(exclude_none=True)
        return value

    @property
    def reason(self) -> RuntimeErrorReason | None:
        return runtime_error_reason(self.details)


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
    4xx. Callers that wrap should leave those unclassified so retry can still
    try again.
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
    can wait out a transient timeout. Plugins wrap provider HTTP errors
    with this so retry can skip a 400 without also skipping a 503.
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


def retry_after_ms_from_error(error: Exception) -> float | None:
    """Read Retry-After off a provider SDK error, if it carried one."""
    headers = None
    response = getattr(error, 'response', None)
    if response is not None:
        headers = getattr(response, 'headers', None)
    if headers is None:
        headers = getattr(error, 'headers', None)
    if headers is None:
        return None
    try:
        raw = headers.get('retry-after')
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


class Status(BaseModel):
    """Represents a status with a name and optional message."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra='forbid',
        populate_by_name=True,
    )

    name: StatusName
    message: str = Field(default='')


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


class HttpErrorWireFormat(BaseModel):
    """Wire format for HTTP error details."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow', populate_by_name=True)

    details: Any
    message: str
    status: str = StatusCodes.INTERNAL.name


class ErrorResponseMetadata(TypedDict, total=False):
    """Metadata from the HTTP response that triggered an error.

    This metadata is available only in-process and is not serialized into
    callable or reflection error wire formats.
    """

    retry_after_ms: float
    headers: dict[str, str]


class Interrupt(Exception):  # noqa: N818 - public Genkit name; not renamed *Error for style
    """Pause a tool or generate so the caller can approve, reply, or restart.

    Raise ``Interrupt(metadata)`` from a tool or from tool middleware.
    Tracing treats this as control flow, not a failed span.
    """

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.metadata: dict[str, Any] = {} if metadata is None else metadata


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
        cause: Exception | None = None,
        details: Any = None,  # noqa: ANN401
        reason: RuntimeErrorReason | None = None,
        trace_id: str | None = None,
        source: str | None = None,
        response_metadata: ErrorResponseMetadata | None = None,
    ) -> None:
        """Initialize a GenkitError.

        Args:
            message: The error message.
            status: The status name for this error.
            cause: The underlying exception that caused this error.
            details: Optional detail information.
            reason: Extra why when we classified the failure.
            trace_id: A unique identifier for tracing the action execution.
            source: Optional source of the error.
            response_metadata: Optional HTTP response metadata for in-process use.
        """
        temp_status: StatusName
        if status:
            temp_status = status
        elif isinstance(cause, GenkitError):
            temp_status = cause.status
        else:
            temp_status = 'INTERNAL'
        self.status: StatusName = temp_status
        self.http_code: int = http_status_code(temp_status)

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
            cause_suffix = f': {cause}' if cause else ''
        super().__init__(f'{source_prefix}{self.status}: {message}{cause_suffix}')
        self.original_message: str = message

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
        if 'stack' not in details:
            details['stack'] = get_error_stack(cause if cause else self)
        if 'trace_id' not in details and trace_id:
            details['trace_id'] = trace_id

        self.details: Any = details
        self.source: str | None = source
        self.trace_id: str | None = trace_id
        self.cause: Exception | None = cause
        self.response_metadata: ErrorResponseMetadata | None = response_metadata
        # Plugin errors built from a provider response keep their status
        # in-process (Retry, Fallback) but serve as a crash at the HTTP
        # boundary so a dead server key is not a 401 to the end caller.
        self._provider_sourced: bool = False
        # The served action's own input/init check is the caller's request,
        # so it keeps a 4xx. A plugin "bad role" inside the flow does not.
        self._request_sourced: bool = False

    @property
    def reason(self) -> RuntimeErrorReason | None:
        return runtime_error_reason(self.details)

    def to_callable_serializable(self) -> HttpErrorWireFormat:
        """Served-flow wire body; same redaction as ``error_body``.

        Only a PublicError keeps its message and details. In-process code
        that needs the real error reads ``original_message`` and ``details``.
        """
        body = error_body(self)
        return HttpErrorWireFormat(
            details=body.get('details'),
            status=body['status'],
            message=body['message'],
        )

    def to_serializable(self) -> ReflectionError:
        """Returns a JSON-serializable representation of this object.

        Returns:
            A ReflectionError model instance.
        """
        return ReflectionError(
            details=ReflectionErrorDetails(**self.details) if self.details else None,
            code=StatusCodes[self.status].value,
            message=f'{self.original_message}: {repr(self.cause)}' if self.cause else self.original_message,
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


def wrap_http_error(error: Exception, *, status_code: object, message: str | None = None) -> GenkitError:
    """Classify a provider HTTP error so retry can skip a 400 without retrying a 503.

    The result keeps that status in-process. A served flow treats it as the
    server's failure: callers see 500 Internal Error, and the process logs
    the traceback. A plugin that builds ``GenkitError(status=...)`` by hand
    is the same 500 on the wire; raise PublicError to give the caller a 4xx.

    A missing or non-HTTP ``status_code``, or a 4xx with no canonical status
    (413, 418), is left unclassified — raise the original error so retry
    still sees a raw failure instead of UNKNOWN. Also reads Retry-After
    when the SDK left it on the error, so retry waits what the provider asked
    instead of coming back in a second.
    """
    resolved = http_code(status_code)
    # A 2xx/3xx on an exception is not a failure status. Leave it
    # unclassified so retry still sees the raw error, instead of a
    # GenkitError that claims OK.
    if resolved is None or resolved < 400:
        raise error
    status = from_http_code(resolved)
    if status == 'UNKNOWN':
        raise error
    retry_after_ms = retry_after_ms_from_error(error)
    response_metadata: ErrorResponseMetadata | None = None
    if retry_after_ms is not None:
        response_metadata = {'retry_after_ms': retry_after_ms}
    return mark_provider_error(
        error=GenkitError(
            status=status,
            message=message if message is not None else str(error),
            cause=error,
            response_metadata=response_metadata,
        )
    )


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


def error_status(error: object) -> int:
    """HTTP status to send back when a served flow fails; pairs with ``error_body``.

    A PublicError keeps its own status (NOT_FOUND is a 404), and a bad
    request to the flow itself keeps its 4xx. Any other GenkitError, a
    provider error, a plain exception, or anything else is a 500.
    """
    facing = _client_facing_error(error)
    if facing is not None:
        return facing.http_code
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
        ref = error.to_serializable()
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


def error_body(error: object) -> dict[str, Any]:
    """JSON body to send back when a served flow fails.

    Only a PublicError's message, details, and status go on the wire; it's
    the one error whose author said the text is safe for callers. A bad
    request to the flow itself keeps its 4xx status with a generic sentence
    such as ``"Invalid argument"``. Any other GenkitError, a provider error,
    and anything else become ``{"message": "Internal Error", "status": "INTERNAL"}``.

    Example:
        ```python
        from genkit.web import error_body, error_status

        try:
            response = await my_flow.run(input=data)
        except Exception as e:
            return JSONResponse(error_body(e), status_code=error_status(e))
        ```
    """
    facing = _client_facing_error(error)
    if facing is None:
        return dict(_INTERNAL_CLIENT_BODY)
    message = facing.original_message if isinstance(facing, PublicError) else _generic_client_message(facing.status)
    body: dict[str, Any] = {
        'message': message,
        'status': facing.status,
    }
    if isinstance(facing, PublicError):
        details = _client_details(facing.details)
        if details is not None:
            body['details'] = details
    return body


def get_error_stack(error: object) -> str | None:
    """Extract stack trace from an error object.

    Args:
        error: The error to get the stack trace from.

    Returns:
        The stack trace string if available, None otherwise.
    """
    if isinstance(error, Exception):
        # Stack traces are valuable for debugging; consider making this configurable
        # to enable them in development/staging and suppress in production.
        # For now, return an empty string to keep Dev UI clean as per requirements.
        return ''
    return None
