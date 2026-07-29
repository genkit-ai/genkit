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

from typing import Any

import pytest
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    ParamValidationError,
    ReadTimeoutError,
)
from genkit_amazon_bedrock.models import BedrockModel

from genkit import FinishReason, Message, ModelRequest, Part, Role, TextPart
from genkit.plugin_api import ActionRunContext, GenkitError


class FakeTransport:
    """Stands in for BedrockTransport; records the Converse kwargs."""

    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict[str, Any] | None = None

    async def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response or {}


def text_request(text: str = 'hello') -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text=text))])])


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
    assert response.message.content[0].root.text == 'hi there'
    assert response.finish_reason == FinishReason.STOP
    assert response.usage is not None
    assert response.usage.total_tokens == 3


@pytest.mark.asyncio
async def test_streaming_context_receives_single_bridge_chunk() -> None:
    transport = FakeTransport(response=text_response('streamed'))
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)
    chunks = []
    ctx = ActionRunContext(streaming_callback=chunks.append)

    response = await model.generate(text_request(), ctx)

    assert len(chunks) == 1
    assert chunks[0].content[0].root.text == 'streamed'
    assert response.message is not None
    assert response.message.content[0].root.text == 'streamed'


@pytest.mark.asyncio
async def test_non_streaming_context_sends_no_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeTransport(response=text_response())
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)
    ctx = ActionRunContext()
    # Recorded directly: a context with no callback reports is_streaming False,
    # so only spying on send_chunk proves the guard is what suppresses chunks.
    sent: list[Any] = []
    monkeypatch.setattr(ctx, 'send_chunk', sent.append)

    await model.generate(text_request(), ctx)

    assert sent == []


@pytest.mark.parametrize(
    'error,expected_status',
    [
        (ParamValidationError(report='bad param'), 'INVALID_ARGUMENT'),
        (NoCredentialsError(), 'UNAUTHENTICATED'),
        (ReadTimeoutError(endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com'), 'DEADLINE_EXCEEDED'),
        (EndpointConnectionError(endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com'), 'UNAVAILABLE'),
        (BotoCoreError(), 'UNKNOWN'),
    ],
    ids=['param_validation', 'no_credentials', 'read_timeout', 'endpoint_connection', 'unlisted'],
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


@pytest.mark.parametrize(
    'code,expected_status',
    [
        ('ThrottlingException', 'RESOURCE_EXHAUSTED'),
        ('ValidationException', 'INVALID_ARGUMENT'),
        ('AccessDeniedException', 'PERMISSION_DENIED'),
        ('ResourceNotFoundException', 'NOT_FOUND'),
        ('ModelTimeoutException', 'DEADLINE_EXCEEDED'),
        ('ServiceUnavailableException', 'UNAVAILABLE'),
        ('SomeFutureException', 'UNKNOWN'),
    ],
)
@pytest.mark.asyncio
async def test_client_errors_map_to_genkit_statuses(code, expected_status) -> None:
    error = ClientError({'Error': {'Code': code, 'Message': 'nope'}}, 'Converse')
    transport = FakeTransport(error=error)
    model = BedrockModel(model_id='amazon.nova-lite-v1:0', transport=transport)

    with pytest.raises(GenkitError) as excinfo:
        await model.generate(text_request())

    assert excinfo.value.status == expected_status
    assert 'bedrock converse failed' in excinfo.value.original_message
    assert excinfo.value.__cause__ is error
