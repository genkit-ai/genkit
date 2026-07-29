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

"""Live Bedrock tests, mirroring the Go plugin's live matrix.

Opt-in: set ``BEDROCK_LIVE_TESTS=1`` plus working AWS credentials and a
region (``AWS_REGION`` or ``~/.aws/config``). These call real models and
incur cost. The models used need only default model access.
"""

import os

import pytest
from genkit_amazon_bedrock.config import BedrockConfig
from genkit_amazon_bedrock.converters import (
    REASONING_SIGNATURE_METADATA_KEY,
)
from genkit_amazon_bedrock.models import BedrockModel
from genkit_amazon_bedrock.transport import BedrockTransport

from genkit import FinishReason, Message, ModelRequest, Part, Role, TextPart, ToolDefinition

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get('BEDROCK_LIVE_TESTS'),
        reason='BEDROCK_LIVE_TESTS not set; live Bedrock tests are opt-in',
    ),
]

NOVA = 'us.amazon.nova-lite-v1:0'
DEEPSEEK = 'us.deepseek.r1-v1:0'


def make_model(model_id: str) -> BedrockModel:
    transport = BedrockTransport(
        region=os.environ.get('AWS_REGION'),
        max_retries=3,
        read_timeout=300.0,
        connect_timeout=60.0,
        max_pool_connections=10,
    )
    return BedrockModel(model_id=model_id, transport=transport)


def text_request(text: str, **kwargs) -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text=text))])],
        **kwargs,
    )


def undocumented_weather_tool() -> ToolDefinition:
    # No description on purpose: Bedrock rejects an empty one, and a Genkit
    # tool declared without a docstring arrives that way.
    return ToolDefinition(
        name='get_weather',
        description='',
        input_schema={
            'type': 'object',
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
        },
    )


async def test_nova_sync() -> None:
    response = await make_model(NOVA).generate(text_request("Reply with the single word 'pong'."))
    assert response.finish_reason == FinishReason.STOP
    assert response.message is not None
    assert response.message.content[0].root.text
    assert response.usage is not None
    assert response.usage.input_tokens is not None and response.usage.input_tokens > 0


async def test_undocumented_tool_round_trip() -> None:
    weather = undocumented_weather_tool()
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='What is the weather in Lagos?'))])],
        tools=[weather],
        config=BedrockConfig(tool_choice='get_weather'),
    )
    response = await make_model(NOVA).generate(request)

    assert response.message is not None
    tool_requests = [part.root.tool_request for part in response.message.content if part.root.tool_request is not None]
    assert tool_requests, 'expected the model to call the tool'
    assert tool_requests[0].name == 'get_weather'
    assert tool_requests[0].ref

    # Feeding the result back must also be accepted.
    follow_up = ModelRequest(
        messages=[
            *request.messages,
            response.message,
            Message(
                role=Role.TOOL,
                content=[
                    Part.model_validate({
                        'toolResponse': {
                            'ref': tool_requests[0].ref,
                            'name': 'get_weather',
                            'output': {'celsius': 31},
                        }
                    })
                ],
            ),
        ],
        tools=[weather],
    )
    assert (await make_model(NOVA).generate(follow_up)).message is not None


async def test_deepseek_reasoning_sync_and_round_trip() -> None:
    model = make_model(DEEPSEEK)
    config = BedrockConfig(max_tokens=2048)
    request = text_request('What is 17 * 23? Think it through.', config=config)
    response = await model.generate(request)

    assert response.message is not None
    reasoning_parts = [
        part.root for part in response.message.content if getattr(part.root, 'reasoning', None) is not None
    ]
    assert reasoning_parts, 'expected a reasoning part from a reasoning model'
    # Signatures are Anthropic-specific, so replay stays gated off here.
    metadata = reasoning_parts[0].metadata
    assert metadata is None or not metadata.get(REASONING_SIGNATURE_METADATA_KEY)

    follow_up = ModelRequest(
        messages=[
            *request.messages,
            response.message,
            Message(role=Role.USER, content=[Part(root=TextPart(text='Now add 100 to that.'))]),
        ],
        config=config,
    )
    assert (await model.generate(follow_up)).finish_reason == FinishReason.STOP
