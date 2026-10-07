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

"""Tests for the BedrockModel generate orchestration (no AWS involved)."""

from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConfigNotFound,
    ConnectionClosedError,
    ConnectTimeoutError,
    CredentialRetrievalError,
    EndpointConnectionError,
    EventStreamError,
    IncompleteReadError,
    MetadataRetrievalError,
    NoAuthTokenError,
    NoCredentialsError,
    NoRegionError,
    ParamValidationError,
    PartialCredentialsError,
    ProfileNotFound,
    ProxyConnectionError,
    ReadTimeoutError,
    SSLError,
    SSOTokenLoadError,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)
from genkit_amazon_bedrock.models import BedrockModel

from genkit import ActionRunContext, FinishReason, GenkitError, Message, Part, Role
from genkit._core._error import get_callable_json, get_http_status
from genkit.model import ModelRequest


class FakeTransport:
    """Stands in for BedrockTransport; records the Converse kwargs."""

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
        stream_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.stream_events = stream_events or []
        self.kwargs: dict[str, Any] | None = None
        self.stream_kwargs: dict[str, Any] | None = None
        self.stream_closed = False

    async def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response or {}

    async def converse_stream(self, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        self.stream_kwargs = kwargs
        try:
            for event in self.stream_events:
                yield event
            if self.error is not None:
                raise self.error
        finally:
            self.stream_closed = True


def text_request(text: str = 'hello') -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text(text)])])


def text_response(text: str = 'world') -> dict[str, Any]:
    return {
        'output': {'message': {'role': 'assistant', 'content': [{'text': text}]}},
        'stopReason': 'end_turn',
        'usage': {'inputTokens': 1, 'outputTokens': 2, 'totalTokens': 3},
    }


@pytest.mark.asyncio
async def test_generate_round_trip() -> None:
    transport = FakeTransport(response=text_response('hi there'))
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)

    response = await model.generate(text_request())

    assert transport.kwargs is not None
    assert transport.kwargs['modelId'] == 'amazon.nova-lite-v1:0'
    assert transport.kwargs['messages'] == [{'role': 'user', 'content': [{'text': 'hello'}]}]
    assert response.message is not None
    assert response.message.content[0].text == 'hi there'
    assert response.finish_reason == FinishReason.STOP
    assert response.usage is not None
    assert response.usage.total_tokens == 3


@pytest.mark.asyncio
async def test_streaming_context_routes_to_converse_stream() -> None:
    transport = FakeTransport(
        stream_events=[
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'strea'}}},
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'med'}}},
            {'messageStop': {'stopReason': 'end_turn'}},
        ]
    )
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)
    chunks = []

    response = await model.generate(text_request(), ActionRunContext(streaming_callback=chunks.append))

    # The streaming path builds the same request as the sync one.
    assert transport.stream_kwargs is not None
    assert transport.stream_kwargs['modelId'] == 'amazon.nova-lite-v1:0'
    assert transport.stream_kwargs['messages'] == [{'role': 'user', 'content': [{'text': 'hello'}]}]
    assert transport.kwargs is None
    assert [chunk.content[0].text for chunk in chunks] == ['strea', 'med']
    assert response.message is not None
    assert response.message.content[0].text == 'streamed'
    assert transport.stream_closed


@pytest.mark.asyncio
async def test_non_streaming_context_uses_converse_and_sends_no_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeTransport(response=text_response())
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)
    ctx = ActionRunContext()
    # Recorded directly: a context with no callback reports is_streaming False,
    # so only spying on send_chunk proves the guard is what suppresses chunks.
    sent: list[Any] = []
    monkeypatch.setattr(ctx, 'send_chunk', sent.append)

    await model.generate(text_request(), ctx)

    assert sent == []
    assert transport.kwargs is not None
    assert transport.stream_kwargs is None


ENDPOINT = 'https://bedrock-runtime.us-east-1.amazonaws.com'


@pytest.mark.parametrize(
    'error,expected_status',
    [
        (ParamValidationError(report='bad param'), 'INVALID_ARGUMENT'),
        (NoCredentialsError(), 'UNAUTHENTICATED'),
        (PartialCredentialsError(provider='env', cred_var='aws_secret_access_key'), 'UNAUTHENTICATED'),
        (CredentialRetrievalError(provider='assume-role', error_msg='role denied'), 'UNAUTHENTICATED'),
        (TokenRetrievalError(provider='sso', error_msg='token refresh failed'), 'UNAUTHENTICATED'),
        (NoAuthTokenError(), 'UNAUTHENTICATED'),
        (SSOTokenLoadError(error_msg='run aws sso login'), 'UNAUTHENTICATED'),
        (UnauthorizedSSOTokenError(), 'UNAUTHENTICATED'),
        (NoRegionError(), 'FAILED_PRECONDITION'),
        (ProfileNotFound(profile='kitchen-prod'), 'FAILED_PRECONDITION'),
        (ConfigNotFound(path='~/.aws/kitchen-config'), 'FAILED_PRECONDITION'),
        (ReadTimeoutError(endpoint_url=ENDPOINT), 'DEADLINE_EXCEEDED'),
        (ConnectTimeoutError(endpoint_url=ENDPOINT), 'DEADLINE_EXCEEDED'),
        (EndpointConnectionError(endpoint_url=ENDPOINT), 'UNAVAILABLE'),
    ],
    ids=[
        'param_validation',
        'no_credentials',
        'partial_credentials',
        'credential_retrieval',
        'token_retrieval',
        'no_auth_token',
        'sso_token_load',
        'unauthorized_sso_token',
        'no_region',
        'profile_not_found',
        'config_not_found',
        'read_timeout',
        'connect_timeout',
        'endpoint_connection',
    ],
)
@pytest.mark.asyncio
async def test_botocore_errors_map_to_genkit_statuses(error: BotoCoreError, expected_status: str) -> None:
    transport = FakeTransport(error=error)
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)

    with pytest.raises(GenkitError) as excinfo:
        await model.generate(text_request())

    assert excinfo.value.status == expected_status
    assert 'bedrock converse failed' in excinfo.value.original_message
    assert excinfo.value.__cause__ is error


def _container_endpoint_timeout() -> CredentialRetrievalError:
    """What botocore's ContainerProvider raises when the ECS/EKS credential endpoint times out."""
    try:
        try:
            raise MetadataRetrievalError(error_msg='Read timeout on endpoint URL')
        except MetadataRetrievalError as e:
            raise CredentialRetrievalError(provider='container-role', error_msg=str(e))  # noqa: B904
    except CredentialRetrievalError as error:
        return error


@pytest.mark.asyncio
async def test_container_credential_endpoint_timeout_stays_raw() -> None:
    """A credential-endpoint blip is a transport failure, so retry still sees it unclassified."""
    error = _container_endpoint_timeout()
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=FakeTransport(error=error))

    with pytest.raises(CredentialRetrievalError) as excinfo:
        await model.generate(text_request())

    assert excinfo.value is error


@pytest.mark.asyncio
async def test_bedrock_credentials_error_is_served_as_internal_error() -> None:
    """Missing Bedrock credentials stay UNAUTHENTICATED in-process and serve as 500 Internal Error."""
    transport = FakeTransport(error=NoCredentialsError())
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)

    with pytest.raises(GenkitError) as excinfo:
        await model.generate(text_request())

    error = excinfo.value
    assert error.status == 'UNAUTHENTICATED'
    assert get_callable_json(error) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert get_http_status(error) == 500


@pytest.mark.parametrize(
    'error',
    [
        BotoCoreError(),
        ConnectionClosedError(endpoint_url=ENDPOINT),
        SSLError(endpoint_url=ENDPOINT, error=ConnectionResetError('handshake reset')),
        ProxyConnectionError(proxy_url='http://proxy.internal:3128', error='refused'),
        IncompleteReadError(actual_bytes=512, expected_bytes=2048),
    ],
    ids=['bare', 'connection_closed', 'ssl', 'proxy', 'incomplete_read'],
)
@pytest.mark.asyncio
async def test_unlisted_botocore_errors_are_reraised_unclassified(error: BotoCoreError) -> None:
    # No status is known for these, so the raw error reaches the caller and
    # retry treats it as unclassified instead of skipping an UNKNOWN.
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=FakeTransport(error=error))

    with pytest.raises(BotoCoreError) as excinfo:
        await model.generate(text_request())

    assert excinfo.value is error


@pytest.mark.parametrize(
    'code,expected_status',
    [
        ('ThrottlingException', 'RESOURCE_EXHAUSTED'),
        ('TooManyRequestsException', 'RESOURCE_EXHAUSTED'),
        ('ServiceQuotaExceededException', 'RESOURCE_EXHAUSTED'),
        ('ValidationException', 'INVALID_ARGUMENT'),
        ('AccessDeniedException', 'PERMISSION_DENIED'),
        ('UnrecognizedClientException', 'UNAUTHENTICATED'),
        ('ExpiredTokenException', 'UNAUTHENTICATED'),
        ('ResourceNotFoundException', 'NOT_FOUND'),
        ('ModelTimeoutException', 'DEADLINE_EXCEEDED'),
        ('ModelNotReadyException', 'UNAVAILABLE'),
        ('ServiceUnavailableException', 'UNAVAILABLE'),
        ('ModelErrorException', 'INTERNAL'),
    ],
)
@pytest.mark.asyncio
async def test_client_errors_map_to_genkit_statuses(code: str, expected_status: str) -> None:
    error = ClientError({'Error': {'Code': code, 'Message': 'nope'}}, 'Converse')
    transport = FakeTransport(error=error)
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)

    with pytest.raises(GenkitError) as excinfo:
        await model.generate(text_request())

    assert excinfo.value.status == expected_status
    assert 'bedrock converse failed' in excinfo.value.original_message
    assert excinfo.value.__cause__ is error


@pytest.mark.parametrize(
    'code,http_status,expected_status',
    [
        ('ConflictException', 409, 'ABORTED'),
        ('ServiceQuotaWarmupException', 429, 'RESOURCE_EXHAUSTED'),
        ('SomeFutureException', 400, 'INVALID_ARGUMENT'),
        ('SomeFutureException', 503, 'UNAVAILABLE'),
        ('SomeFutureException', 599, 'INTERNAL'),
        ('', 404, 'NOT_FOUND'),
    ],
)
@pytest.mark.asyncio
async def test_unlisted_client_error_codes_fall_back_to_http_status(
    code: str, http_status: int, expected_status: str
) -> None:
    error = ClientError(
        cast(Any, {'Error': {'Code': code, 'Message': 'nope'}, 'ResponseMetadata': {'HTTPStatusCode': http_status}}),
        'Converse',
    )

    genkit_error = await generate_error(error)

    assert genkit_error.status == expected_status
    assert genkit_error.__cause__ is error


@pytest.mark.parametrize(
    'response',
    [
        {'Error': {'Code': 'SomeFutureException', 'Message': 'nope'}},
        {'Error': {'Code': '', 'Message': 'nope'}},
        {'Error': {'Code': 'SomeFutureException'}, 'ResponseMetadata': {'HTTPStatusCode': 200}},
        # 418 has no canonical status; UNKNOWN would make retry skip it.
        {'Error': {'Code': 'SomeFutureException'}, 'ResponseMetadata': {'HTTPStatusCode': 418}},
        {'Error': {'Code': 'SomeFutureException'}, 'ResponseMetadata': {'HTTPStatusCode': '503'}},
    ],
    ids=['no_metadata', 'no_code', 'ok_status', 'unmapped_4xx', 'non_int_status'],
)
@pytest.mark.asyncio
async def test_client_errors_without_a_usable_status_are_reraised_unclassified(response: dict[str, Any]) -> None:
    error = ClientError(cast(Any, response), 'Converse')
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=FakeTransport(error=error))

    with pytest.raises(ClientError) as excinfo:
        await model.generate(text_request())

    assert excinfo.value is error


def throttling_error(headers: dict[str, str] | None = None) -> ClientError:
    response: dict[str, Any] = {'Error': {'Code': 'ThrottlingException', 'Message': 'slow down'}}
    if headers is not None:
        response['ResponseMetadata'] = {'HTTPHeaders': headers}
    return ClientError(response, 'Converse')


async def generate_error(error: Exception) -> GenkitError:
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=FakeTransport(error=error))
    with pytest.raises(GenkitError) as excinfo:
        await model.generate(text_request())
    return excinfo.value


@pytest.mark.asyncio
async def test_throttling_surfaces_retry_after_seconds() -> None:
    genkit_error = await generate_error(throttling_error({'retry-after': '2'}))

    assert genkit_error.status == 'RESOURCE_EXHAUSTED'
    assert genkit_error.response_metadata == {'retry_after_ms': 2000.0}


@pytest.mark.asyncio
async def test_retry_after_accepts_an_http_date() -> None:
    genkit_error = await generate_error(throttling_error({'Retry-After': 'Wed, 21 Oct 2015 07:28:00 GMT'}))

    assert genkit_error.response_metadata is not None
    # The date is long past, so the wait clamps to zero rather than going negative.
    assert genkit_error.response_metadata['retry_after_ms'] == 0.0


@pytest.mark.parametrize(
    'headers',
    [None, {}, {'retry-after': ''}, {'retry-after': 'soon'}, {'content-type': 'application/json'}],
    ids=['no_metadata', 'no_headers', 'empty', 'unparseable', 'absent'],
)
@pytest.mark.asyncio
async def test_missing_or_unparseable_retry_after_is_omitted(headers: dict[str, str] | None) -> None:
    genkit_error = await generate_error(throttling_error(headers))

    assert genkit_error.status == 'RESOURCE_EXHAUSTED'
    assert genkit_error.response_metadata is None


# Mid-stream failures are named by the event stream's ``:exception-type``
# header, which is lowerCamelCase where the modelled exception names are not.
@pytest.mark.parametrize(
    'code,expected_status',
    [
        ('throttlingException', 'RESOURCE_EXHAUSTED'),
        ('validationException', 'INVALID_ARGUMENT'),
        ('internalServerException', 'INTERNAL'),
        ('serviceUnavailableException', 'UNAVAILABLE'),
        ('modelStreamErrorException', 'INTERNAL'),
    ],
)
@pytest.mark.asyncio
async def test_mid_stream_errors_map_to_genkit_statuses(code: str, expected_status: str) -> None:
    error = EventStreamError({'Error': {'Code': code, 'Message': 'nope'}}, 'ConverseStream')
    transport = FakeTransport(
        error=error,
        stream_events=[{'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'partial'}}}],
    )
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)
    chunks = []

    with pytest.raises(GenkitError) as excinfo:
        await model.generate(text_request(), ActionRunContext(streaming_callback=chunks.append))

    assert excinfo.value.status == expected_status
    assert 'bedrock converse stream failed' in excinfo.value.original_message
    assert excinfo.value.__cause__ is error
    # Chunks already delivered stand; the stream is closed on the way out.
    assert len(chunks) == 1
    assert transport.stream_closed


@pytest.mark.parametrize(
    'error',
    [
        # The event stream carries no HTTP status, so an unknown type has none.
        EventStreamError({'Error': {'Code': 'someFutureException', 'Message': 'nope'}}, 'ConverseStream'),
        ConnectionClosedError(endpoint_url=ENDPOINT),
    ],
    ids=['unknown_exception_type', 'connection_closed'],
)
@pytest.mark.asyncio
async def test_unclassifiable_mid_stream_errors_are_reraised(error: Exception) -> None:
    transport = FakeTransport(
        error=error,
        stream_events=[{'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'partial'}}}],
    )
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)

    with pytest.raises(type(error)) as excinfo:
        await model.generate(text_request(), ActionRunContext(streaming_callback=lambda _chunk: None))

    assert excinfo.value is error
    assert transport.stream_closed


@pytest.mark.asyncio
async def test_stream_botocore_errors_map_to_genkit_statuses() -> None:
    error = ReadTimeoutError(endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com')
    transport = FakeTransport(error=error)
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)

    with pytest.raises(GenkitError) as excinfo:
        await model.generate(text_request(), ActionRunContext(streaming_callback=lambda _chunk: None))

    assert excinfo.value.status == 'DEADLINE_EXCEEDED'
    assert 'bedrock converse stream failed' in excinfo.value.original_message
    assert transport.stream_closed


@pytest.mark.asyncio
async def test_stream_is_closed_when_a_chunk_callback_raises() -> None:
    def explode(_chunk: Any) -> None:  # noqa: ANN401
        raise RuntimeError('callback failed')

    transport = FakeTransport(
        stream_events=[
            {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'hi'}}},
            {'messageStop': {'stopReason': 'end_turn'}},
        ]
    )
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)

    with pytest.raises(RuntimeError, match='callback failed'):
        await model.generate(text_request(), ActionRunContext(streaming_callback=explode))

    assert transport.stream_closed
