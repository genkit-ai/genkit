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
import time
from email.utils import parsedate_to_datetime
from enum import IntEnum
from typing import Any, ClassVar, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


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


class GenkitInterrupt(Exception):  # noqa: N818 - marker base class; intentionally not suffixed *Error
    """Marker base class for tool interrupts.

    Raised by tools to pause execution and hand control back to the caller.
    The tracing wrapper uses this to distinguish control-flow interrupts from
    real errors so they don't appear as red failures in the Dev UI.
    """


class GenkitError(Exception):
    """Base error class for Genkit errors."""

    def __init__(
        self,
        *,
        message: str,
        status: StatusName | None = None,
        cause: Exception | None = None,
        details: Any = None,  # noqa: ANN401
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
        cause_suffix = f': {cause}' if cause else ''
        super().__init__(f'{source_prefix}{self.status}: {message}{cause_suffix}')
        self.original_message: str = message

        if not details:
            details = {}
        if 'stack' not in details:
            details['stack'] = get_error_stack(cause if cause else self)
        if 'trace_id' not in details and trace_id:
            details['trace_id'] = trace_id

        self.details: Any = details
        self.source: str | None = source
        self.trace_id: str | None = trace_id
        self.cause: Exception | None = cause
        self.response_metadata: ErrorResponseMetadata | None = response_metadata

    def to_callable_serializable(self) -> HttpErrorWireFormat:
        """Returns a JSON-serializable representation of this object.

        Returns:
            An HttpErrorWireFormat model instance.
        """
        # This error type is used by 3P authors with the field "details",
        # but the actual Callable protocol value is "details"
        return HttpErrorWireFormat(
            details=self.details,
            status=StatusCodes[self.status].name,
            message=self.original_message,
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


def wrap_http_error(error: Exception, *, status_code: object, message: str | None = None) -> GenkitError:
    """Classify a provider HTTP error so retry can skip a 400 without retrying a 503.

    A missing or non-HTTP ``status_code`` is left unclassified — raise the
    original error so retry still sees a raw failure. Also reads Retry-After
    when the SDK left it on the error, so retry waits what the provider asked
    instead of coming back in a second.
    """
    resolved = http_code(status_code)
    # A 2xx/3xx on an exception is not a failure status. Leave it
    # unclassified so retry still sees the raw error, instead of a
    # GenkitError that claims OK.
    if resolved is None or resolved < 400:
        raise error
    retry_after_ms = retry_after_ms_from_error(error)
    response_metadata: ErrorResponseMetadata | None = None
    if retry_after_ms is not None:
        response_metadata = {'retry_after_ms': retry_after_ms}
    return GenkitError(
        status=from_http_code(resolved),
        message=message if message is not None else str(error),
        cause=error,
        response_metadata=response_metadata,
    )


class PublicError(GenkitError):
    """Error class for issues to be returned to users.

    Using this error allows a web framework handler (e.g. FastAPI, Flask) to know it
    is safe to return the message in a request. Other kinds of errors will
    result in a generic 500 message to avoid the possibility of internal
    exceptions being leaked to attackers.
    """

    def __init__(self, status: StatusName, message: str, details: Any = None) -> None:  # noqa: ANN401
        """Initialize a PublicError.

        Args:
            status: The status name for this error.
            message: The error message.
            details: Optional details to include.
        """
        super().__init__(status=status, message=message, details=details)


def get_http_status(error: object) -> int:
    """Get the HTTP status code for an error.

    Args:
        error: The error to get the status code for.

    Returns:
        The HTTP status code (500 for non-Genkit errors).
    """
    if isinstance(error, GenkitError):
        return error.http_code
    return 500


def get_reflection_json(error: object) -> ReflectionError:
    """Get the JSON representation of an error for reflection API responses.

    Args:
        error: The error to convert to JSON.

    Returns:
        A ReflectionError model instance.
    """
    if isinstance(error, GenkitError):
        return error.to_serializable()
    return ReflectionError(
        message=str(error),
        code=StatusCodes.INTERNAL.value,
        details=ReflectionErrorDetails(stack=get_error_stack(error)),
    )


def get_callable_json(error: object) -> dict[str, Any]:
    """Get the JSON-serializable representation of an error for callable responses.

    Args:
        error: The error to convert to JSON.

    Returns:
        A dict ready for json.dumps (message, status, details keys).
    """
    if isinstance(error, GenkitError):
        wire = error.to_callable_serializable()
    else:
        wire = HttpErrorWireFormat(
            message=str(error),
            status=StatusCodes.INTERNAL.name,
            details={'stack': get_error_stack(error)},
        )
    return wire.model_dump()


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
