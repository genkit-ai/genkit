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

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any, Protocol

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

from genkit import ModelRequest, ModelResponse
from genkit.plugin_api import ActionRunContext, GenkitError, StatusName
from genkit_amazon_bedrock.converters import build_converse_request, to_model_response
from genkit_amazon_bedrock.stream import consume_converse_stream


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


def _from_client_error(error: ClientError, operation: str = 'converse') -> GenkitError:
    error_info: dict[str, Any] = error.response.get('Error') or {}
    code = error_info.get('Code') or ''
    message = error_info.get('Message') or str(error)
    prefix = f'bedrock {operation} failed'
    return GenkitError(
        message=f'{prefix}: {code}: {message}' if code else f'{prefix}: {message}',
        status=_ERROR_CODE_STATUS.get(_normalize_error_code(code), 'UNKNOWN'),
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
        converse_kwargs = build_converse_request(self._model_id, request)
        if ctx is not None and ctx.is_streaming:
            return await self._generate_stream(converse_kwargs, request, ctx)
        try:
            response = await self._transport.converse(**converse_kwargs)
        except ClientError as e:
            raise _from_client_error(e) from e
        except BotoCoreError as e:
            raise _from_botocore_error(e) from e
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
                return await consume_converse_stream(events, request, ctx)
        except ClientError as e:
            raise _from_client_error(e, 'converse stream') from e
        except BotoCoreError as e:
            raise _from_botocore_error(e, 'converse stream') from e
