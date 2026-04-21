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

from enum import IntEnum
from typing import Any, ClassVar, Literal

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


def http_status_code(status: StatusName) -> int:
    """Gets the HTTP status code for a given status name.

    Args:
        status: The status name to get the HTTP code for.

    Returns:
        The corresponding HTTP status code.
    """
    return _STATUS_CODE_MAP[status]


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
    ) -> None:
        """Initialize a GenkitError.

        Args:
            message: The error message.
            status: The status name for this error.
            cause: The underlying exception that caused this error.
            details: Optional detail information.
            trace_id: A unique identifier for tracing the action execution.
            source: Optional source of the error.
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

        source_prefix = f'{source}: ' if source else ''
        super().__init__(f'{source_prefix}{self.status}: {message}')
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
            message=repr(self.cause) if self.cause else self.original_message,
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
