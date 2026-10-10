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

import structlog
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConfigNotFound,
    ConnectionError as BotoConnectionError,
    ConnectTimeoutError,
    CredentialRetrievalError,
    HTTPClientError,
    IncompleteReadError,
    MetadataRetrievalError,
    NoAuthTokenError,
    NoCredentialsError,
    NoRegionError,
    ParamValidationError,
    PartialCredentialsError,
    ProfileNotFound,
    ReadTimeoutError,
    SSOTokenLoadError,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)

from genkit import ActionRunContext, GenkitError, ModelResponse
from genkit.model import ModelRequest
from genkit.plugin_api import StatusName, provider_error
from genkit_amazon_bedrock._converters import build_converse_request, to_model_response, usage_log_fields
from genkit_amazon_bedrock._stream import consume_converse_stream

logger = structlog.get_logger(__name__)


class ConverseTransport(Protocol):
    """Structural contract for the transport seam (see ``_transport.py``)."""

    async def converse(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Calls the Converse API and returns the raw response dict."""
        ...

    def converse_stream(self, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:  # noqa: ANN401
        """Calls the ConverseStream API and yields raw event dicts."""
        ...


# AWS error codes → Genkit statuses. An unlisted code falls back to the
# response's HTTP status (see ``_from_client_error``).
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
# code; map the exception type instead, first match wins. Timeouts sit above
# the connection families they subclass. Anything unlisted is UNKNOWN.
_BOTOCORE_ERROR_STATUS: tuple[tuple[type[BotoCoreError], StatusName], ...] = (
    (ParamValidationError, 'INVALID_ARGUMENT'),
    (NoCredentialsError, 'UNAUTHENTICATED'),
    (PartialCredentialsError, 'UNAUTHENTICATED'),
    (CredentialRetrievalError, 'UNAUTHENTICATED'),
    (TokenRetrievalError, 'UNAUTHENTICATED'),
    (NoAuthTokenError, 'UNAUTHENTICATED'),
    (SSOTokenLoadError, 'UNAUTHENTICATED'),
    (UnauthorizedSSOTokenError, 'UNAUTHENTICATED'),
    (NoRegionError, 'FAILED_PRECONDITION'),
    (ProfileNotFound, 'FAILED_PRECONDITION'),
    (ConfigNotFound, 'FAILED_PRECONDITION'),
    (ReadTimeoutError, 'DEADLINE_EXCEEDED'),
    (ConnectTimeoutError, 'DEADLINE_EXCEEDED'),
    # Refused, reset, SSL, proxy, or dropped mid-response: the network let us
    # down rather than Bedrock saying no, so another try can work.
    (BotoConnectionError, 'UNAVAILABLE'),
    (HTTPClientError, 'UNAVAILABLE'),
    (IncompleteReadError, 'UNAVAILABLE'),
)


def _from_client_error(error: ClientError, operation: str = 'converse') -> GenkitError:
    """Classifies an AWS service error by its error code, then its HTTP status.

    A code missing from ``_ERROR_CODE_STATUS`` with no usable failure status
    is UNKNOWN: an unmapped 4xx like 418, or a mid-stream EventStreamError of
    a new exception type, since the event carries no HTTP status of its own.
    """
    error_info: dict[str, Any] = error.response.get('Error') or {}
    code = error_info.get('Code') or ''
    metadata = error.response.get('ResponseMetadata')
    if not isinstance(metadata, dict):
        metadata = {}
    # str(ClientError) already names the code, the operation, and AWS's message
    # ("An error occurred (ThrottlingException) when calling the Converse
    # operation: Rate exceeded"). Embedding it once keeps str() from repeating it.
    return provider_error(
        error,
        status=_ERROR_CODE_STATUS.get(_normalize_error_code(code)),
        http_status=metadata.get('HTTPStatusCode'),
        headers=metadata.get('HTTPHeaders'),
        message=f'bedrock {operation} failed: {error}',
    )


def _normalize_error_code(code: str) -> str:
    """Upper-cases the leading letter of an AWS error code.

    Mid-stream failures are named by the event stream's ``:exception-type``
    header, which is lowerCamelCase (``throttlingException``) where the
    modelled exception names are UpperCamel.
    """
    return code[:1].upper() + code[1:] if code else code


def _is_transient_credential_error(error: BotoCoreError) -> bool:
    """A credential refresh that failed to reach the ECS/EKS credential endpoint.

    botocore's ContainerProvider catches the endpoint's MetadataRetrievalError
    (timeout, refused connection) inside an ``except`` and raises
    CredentialRetrievalError without ``from``, so the original is only on
    ``__context__``. That is a transport failure, not a rejected credential.
    """
    if not isinstance(error, CredentialRetrievalError):
        return False
    return isinstance(error.__cause__ or error.__context__, MetadataRetrievalError)


def _botocore_status(error: BotoCoreError) -> StatusName:
    if _is_transient_credential_error(error):
        return 'UNAVAILABLE'
    for error_type, status in _BOTOCORE_ERROR_STATUS:
        if isinstance(error, error_type):
            return status
    return 'UNKNOWN'


def _from_botocore_error(error: BotoCoreError, operation: str = 'converse') -> GenkitError:
    """Classifies a client-side botocore failure by exception type."""
    # Some botocore errors (a total-timeout cancel) have an empty str().
    detail = str(error) or type(error).__name__
    return provider_error(error, status=_botocore_status(error), message=f'bedrock {operation} failed: {detail}')


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
