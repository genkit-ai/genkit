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

"""Bedrock model action implementation (Converse and ConverseStream APIs)."""

import math
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import structlog
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    NoRegionError,
    ParamValidationError,
    PartialCredentialsError,
    ReadTimeoutError,
)

from genkit import ErrorResponseMetadata, ModelRequest, ModelResponse
from genkit.plugin_api import ActionRunContext, GenkitError, StatusName
from genkit_amazon_bedrock.converters import build_converse_request, to_model_response, usage_log_fields
from genkit_amazon_bedrock.stream import consume_converse_stream

logger = structlog.get_logger(__name__)


class ConverseTransport(Protocol):
    """Structural contract for the transport seam (see ``transport.py``)."""

    async def converse(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Calls the Converse API and returns the raw response dict."""
        ...

    def converse_stream(self, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:  # noqa: ANN401
        """Calls the ConverseStream API and yields raw event dicts."""
        ...


# AWS error codes → Genkit statuses; anything unlisted maps to UNKNOWN.
_ERROR_CODE_STATUS: dict[str, StatusName] = {
    'ThrottlingException': 'RESOURCE_EXHAUSTED',
    'TooManyRequestsException': 'RESOURCE_EXHAUSTED',
    'ServiceQuotaExceededException': 'RESOURCE_EXHAUSTED',
    'ValidationException': 'INVALID_ARGUMENT',
    'AccessDeniedException': 'PERMISSION_DENIED',
    'UnrecognizedClientException': 'UNAUTHENTICATED',
    'ExpiredTokenException': 'UNAUTHENTICATED',
    'ResourceNotFoundException': 'NOT_FOUND',
    'ModelTimeoutException': 'DEADLINE_EXCEEDED',
    'ModelNotReadyException': 'UNAVAILABLE',
    'ServiceUnavailableException': 'UNAVAILABLE',
    'ModelErrorException': 'INTERNAL',
    'InternalServerException': 'INTERNAL',
    # Stream-only; botocore surfaces it as an EventStreamError mid-stream.
    'ModelStreamErrorException': 'INTERNAL',
}


# Client-side botocore failures never reach the service, so they carry no error
# code; map the exception type instead. Anything unlisted stays UNKNOWN.
_BOTOCORE_ERROR_STATUS: tuple[tuple[type[BotoCoreError], StatusName], ...] = (
    (ParamValidationError, 'INVALID_ARGUMENT'),
    (NoCredentialsError, 'UNAUTHENTICATED'),
    (PartialCredentialsError, 'UNAUTHENTICATED'),
    (NoRegionError, 'FAILED_PRECONDITION'),
    (ReadTimeoutError, 'DEADLINE_EXCEEDED'),
    (ConnectTimeoutError, 'DEADLINE_EXCEEDED'),
    (EndpointConnectionError, 'UNAVAILABLE'),
)


def _parse_retry_after_ms(value: str) -> float | None:
    """Parses an HTTP Retry-After value into milliseconds.

    Accepts both forms the header allows, delay-seconds and an HTTP-date.
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


def _retry_after_ms(error: ClientError) -> float | None:
    """Pulls Retry-After out of the response headers Bedrock throttling sends."""
    metadata = error.response.get('ResponseMetadata') or {}
    headers = metadata.get('HTTPHeaders') or {}
    if not isinstance(headers, dict):
        return None
    # Header names are case-insensitive; botocore's lowercasing is not promised.
    for name, value in headers.items():
        if isinstance(name, str) and name.lower() == 'retry-after' and isinstance(value, str):
            return _parse_retry_after_ms(value)
    return None


def _from_client_error(error: ClientError, operation: str = 'converse') -> GenkitError:
    error_info: dict[str, Any] = error.response.get('Error') or {}
    code = error_info.get('Code') or ''
    message = error_info.get('Message') or str(error)
    prefix = f'bedrock {operation} failed'
    retry_after_ms = _retry_after_ms(error)
    response_metadata: ErrorResponseMetadata | None = None
    if retry_after_ms is not None:
        response_metadata = {'retry_after_ms': retry_after_ms}
    return GenkitError(
        message=f'{prefix}: {code}: {message}' if code else f'{prefix}: {message}',
        status=_ERROR_CODE_STATUS.get(_normalize_error_code(code), 'UNKNOWN'),
        response_metadata=response_metadata,
    )


def _normalize_error_code(code: str) -> str:
    """Upper-cases the leading letter of an AWS error code.

    Mid-stream failures are named by the event stream's ``:exception-type``
    header, which is lowerCamelCase (``throttlingException``) where the
    modelled exception names are UpperCamel.
    """
    return code[:1].upper() + code[1:] if code else code


def _from_botocore_error(error: BotoCoreError, operation: str = 'converse') -> GenkitError:
    status: StatusName = 'UNKNOWN'
    for error_type, mapped in _BOTOCORE_ERROR_STATUS:
        if isinstance(error, error_type):
            status = mapped
            break
    return GenkitError(message=f'bedrock {operation} failed: {error}', status=status)


class BedrockModel:
    """Handles a generate call for one Bedrock chat/text model."""

    def __init__(self, model_id: str, transport: ConverseTransport) -> None:
        """Initializes the model handler.

        Args:
            model_id: Bedrock model ID, inference-profile ID, or ARN, sent to
                the Converse API verbatim.
            transport: The shared transport seam owning the boto3 client.
        """
        self._model_id = model_id
        self._transport = transport

    async def generate(self, request: ModelRequest[Any], ctx: ActionRunContext | None = None) -> ModelResponse:
        """Runs a Converse or ConverseStream call.

        Args:
            request: The Genkit model request.
            ctx: Action run context; a streaming callback routes the call to
                ConverseStream. Both paths build the same request.

        Returns:
            The converted model response.
        """
        streaming = ctx is not None and ctx.is_streaming
        converse_kwargs = build_converse_request(self._model_id, request)
        logger.debug(
            'Bedrock generate request',
            model=self._model_id,
            streaming=streaming,
            messages=len(converse_kwargs.get('messages') or []),
            tools=len((converse_kwargs.get('toolConfig') or {}).get('tools') or []),
        )
        if streaming and ctx is not None:
            return await self._generate_stream(converse_kwargs, request, ctx)
        try:
            response = await self._transport.converse(**converse_kwargs)
        except ClientError as e:
            raise _from_client_error(e) from e
        except BotoCoreError as e:
            raise _from_botocore_error(e) from e
        # Guarded so a transport returning None still reaches to_model_response,
        # which reports it as INTERNAL rather than dying here on an attribute.
        logged = response or {}
        logger.debug(
            'Bedrock generate response',
            model=self._model_id,
            stop_reason=logged.get('stopReason'),
            **usage_log_fields(logged.get('usage')),
        )
        return to_model_response(response, request)

    async def _generate_stream(
        self,
        converse_kwargs: dict[str, Any],
        request: ModelRequest[Any],
        ctx: ActionRunContext,
    ) -> ModelResponse:
        """Streams a ConverseStream call, accumulating the final response.

        The pump runs inside the try so mid-stream failures map like any other
        AWS error: botocore raises them as EventStreamError, a ClientError
        subclass. ``aclosing`` guarantees the event stream is closed even when
        a chunk callback raises.
        """
        try:
            async with aclosing(self._transport.converse_stream(**converse_kwargs)) as events:
                return await consume_converse_stream(events, request, ctx, self._model_id)
        except ClientError as e:
            raise _from_client_error(e, 'converse stream') from e
        except BotoCoreError as e:
            raise _from_botocore_error(e, 'converse stream') from e
