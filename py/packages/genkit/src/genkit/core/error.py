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

"""Error classes and utilities for the Genkit framework.

This module defines the error hierarchy and utilities for handling errors
in Genkit applications. It provides structured error types with status codes,
trace IDs, and serialization for HTTP responses.

Overview:
    Genkit uses a structured error system based on gRPC-style status codes.
    The base ``GenkitError`` class provides rich error context including
    status codes, trace IDs, and stack traces for debugging.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        Error Class Hierarchy                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Exception                                                              │
    │      │                                                                  │
    │      └── GenkitError                                                    │
    │              │                                                          │
    │              ├── UserFacingError  (safe to return to users)             │
    │              │                                                          │
    │              └── UnstableApiError (beta/alpha API misuse)               │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Terminology:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Term              │ Description                                         │
    ├───────────────────┼─────────────────────────────────────────────────────┤
    │ GenkitError       │ Base error with status, trace_id, and details       │
    │ UserFacingError   │ Error safe to return in HTTP responses              │
    │ StatusName        │ gRPC status name (e.g., 'NOT_FOUND', 'INTERNAL')    │
    │ StatusCodes       │ Enum mapping status names to numeric codes          │
    │ http_code         │ HTTP status code derived from StatusName            │
    │ trace_id          │ Unique ID linking error to trace spans              │
    └───────────────────┴─────────────────────────────────────────────────────┘

Key Functions:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Function            │ Purpose                                           │
    ├─────────────────────┼───────────────────────────────────────────────────┤
    │ get_http_status()   │ Get HTTP status code from any error               │
    │ get_callable_json() │ Serialize error for callable HTTP responses       │
    │ get_error_message() │ Extract message string from any error             │
    │ get_error_stack()   │ Extract stack trace from an exception             │
    └─────────────────────┴───────────────────────────────────────────────────┘

Example:
    Raising and handling errors:

    ```python
    from genkit.core.error import GenkitError, UserFacingError, get_http_status

    # Raise a structured error
    raise GenkitError(
        message='Model not found',
        status='NOT_FOUND',
        trace_id='abc123',
    )

    # User-facing error (safe to return in HTTP response)
    raise UserFacingError(
        status='INVALID_ARGUMENT',
        message='Invalid prompt: too long',
    )

    # Get HTTP status for any error
    try:
        await ai.generate(...)
    except Exception as e:
        status_code = get_http_status(e)  # 404 for NOT_FOUND, 500 otherwise
    ```

Caveats:
    - Only ``UserFacingError`` messages are safe to return to end users
    - Other ``GenkitError`` messages may contain internal details
    - Use ``get_callable_json()`` for Genkit callable serialization format

See Also:
    - gRPC status codes: https://grpc.io/docs/guides/status-codes/
    - genkit.core.status_types: Status code definitions
"""

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from genkit.core.status_types import StatusCodes, StatusName, http_status_code


class GenkitReflectionApiDetailsWireFormat(BaseModel):
    """Wire format for HTTP error details."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='allow', populate_by_name=True, alias_generator=to_camel)

    stack: str | None = None
    trace_id: str | None = None


class GenkitReflectionApiErrorWireFormat(BaseModel):
    """Wire format for HTTP errors."""

    details: GenkitReflectionApiDetailsWireFormat | None = None
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

    def to_serializable(self) -> GenkitReflectionApiErrorWireFormat:
        """Returns a JSON-serializable representation of this object.

        Returns:
            An HttpErrorWireFormat model instance.
        """
        # This error type is used by 3P authors with the field "details",
        # but the actual Callable protocol value is "details"
        return GenkitReflectionApiErrorWireFormat(
            details=GenkitReflectionApiDetailsWireFormat(**self.details) if self.details else None,
            code=StatusCodes[self.status].value,
            message=f'{self.original_message}: {repr(self.cause)}' if self.cause else self.original_message,
        )


class UnstableApiError(GenkitError):
    """Error raised when using unstable APIs from a more stable instance."""

    def __init__(self, level: str = 'beta', message: str | None = None) -> None:
        """Initialize an UnstableApiError.

        Args:
            level: The stability level required.
            message: Optional message describing which feature is not allowed.
        """
        msg_prefix = f'{message} ' if message else ''
        super().__init__(
            status='FAILED_PRECONDITION',
            message=f"{msg_prefix}This API requires '{level}' stability level.\n\n"
            + f'To use this feature, initialize Genkit using `from genkit.{level} import genkit`.',
        )


class UserFacingError(GenkitError):
    """Error class for issues to be returned to users.

    Using this error allows a web framework handler (e.g. FastAPI, Flask) to know it
    is safe to return the message in a request. Other kinds of errors will
    result in a generic 500 message to avoid the possibility of internal
    exceptions being leaked to attackers.
    """

    def __init__(self, status: StatusName, message: str, details: Any = None) -> None:  # noqa: ANN401
        """Initialize a UserFacingError.

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


def get_reflection_json(error: object) -> GenkitReflectionApiErrorWireFormat:
    """Get the JSON representation of an error for callable responses.

    Args:
        error: The error to convert to JSON.

    Returns:
        An HttpErrorWireFormat model instance.
    """
    if isinstance(error, GenkitError):
        return error.to_serializable()
    return GenkitReflectionApiErrorWireFormat(
        message=str(error),
        code=StatusCodes.INTERNAL.value,
        details=GenkitReflectionApiDetailsWireFormat(stack=get_error_stack(error)),
    )


def get_callable_json(error: object) -> HttpErrorWireFormat:
    """Get the JSON representation of an error for callable responses.

    Args:
        error: The error to convert to JSON.

    Returns:
        An HttpErrorWireFormat model instance.
    """
    if isinstance(error, GenkitError):
        return error.to_callable_serializable()
    return HttpErrorWireFormat(
        message=str(error),
        status=StatusCodes.INTERNAL.name,
        details={'stack': get_error_stack(error)},
    )


def get_error_message(error: object) -> str:
    """Extract error message from an error object.

    Args:
        error: The error to get the message from.

    Returns:
        The error message string.
    """
    if isinstance(error, Exception):
        return str(error)
    return str(error)


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
