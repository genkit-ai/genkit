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

"""Live Bedrock tests.

Opt-in: set ``BEDROCK_LIVE_TESTS=1`` plus working AWS credentials and a
region (``AWS_REGION`` or ``~/.aws/config``). These call real models and
incur cost. The Anthropic models additionally need the account's one-time
use-case agreement (Bedrock console -> Model access); the embedding models
each need their own model-access grant, separate from the chat models.
"""

import os

import pytest
from genkit_amazon_bedrock.config import BedrockConfig
from genkit_amazon_bedrock.converters import (
    REASONING_SIGNATURE_METADATA_KEY,
)
from genkit_amazon_bedrock.embedders import BedrockEmbedder
from genkit_amazon_bedrock.models import BedrockModel
from genkit_amazon_bedrock.transport import BedrockTransport

from genkit import (
    DocumentPart,
    FinishReason,
    Media,
    MediaPart,
    Message,
    ModelRequest,
    Part,
    Role,
    TextPart,
    ToolDefinition,
)
from genkit._core._typing import DocumentData
from genkit.embedder import EmbedRequest
from genkit.plugin_api import ActionRunContext, GenkitError

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get('BEDROCK_LIVE_TESTS'),
        reason='BEDROCK_LIVE_TESTS not set; live Bedrock tests are opt-in',
    ),
]

CLAUDE = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
NOVA = 'us.amazon.nova-lite-v1:0'
DEEPSEEK = 'us.deepseek.r1-v1:0'

TITAN_EMBED_V1 = 'amazon.titan-embed-text-v1'
TITAN_EMBED_V2 = 'amazon.titan-embed-text-v2:0'
TITAN_EMBED_IMAGE = 'amazon.titan-embed-image-v1'
COHERE_EMBED = 'cohere.embed-english-v3'
COHERE_EMBED_MULTI = 'cohere.embed-multilingual-v3'
NOVA_EMBED = 'amazon.nova-2-multimodal-embeddings-v1:0'

# Smallest PNG Titan accepts, ported from the Go plugin's live tests.
PNG_1X1 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII='
PNG_1X1_DATA_URL = f'data:image/png;base64,{PNG_1X1}'


def make_transport() -> BedrockTransport:
    return BedrockTransport(
        region=os.environ.get('AWS_REGION'),
        max_retries=3,
        read_timeout=300.0,
        connect_timeout=60.0,
        max_pool_connections=10,
    )


def make_model(model_id: str) -> BedrockModel:
    return BedrockModel(model_id=model_id, transport=make_transport())


async def embed(model_id: str, documents: list[DocumentData]) -> list[list[float]]:
    embedder = BedrockEmbedder(model_id=model_id, transport=make_transport())
    response = await embedder.embed(EmbedRequest(input=documents))
    return [embedding.embedding for embedding in response.embeddings]


def text_doc(text: str) -> DocumentData:
    return DocumentData(content=[DocumentPart(root=TextPart(text=text))])


def image_doc(data_url: str = PNG_1X1_DATA_URL) -> DocumentData:
    return DocumentData(content=[DocumentPart(root=MediaPart(media=Media(url=data_url)))])


def text_request(text: str, **kwargs) -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text=text))])],
        **kwargs,
    )


def streaming_ctx() -> tuple[ActionRunContext, list]:
    chunks: list = []
    return ActionRunContext(streaming_callback=chunks.append), chunks


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


async def test_nova_stream() -> None:
    ctx, chunks = streaming_ctx()
    response = await make_model(NOVA).generate(text_request('Count from 1 to 5, one number per line.'), ctx)

    assert len(chunks) > 1, 'expected the response to arrive as multiple deltas'
    streamed = ''.join(chunk.content[0].root.text or '' for chunk in chunks)
    assert response.message is not None
    # Deltas, not snapshots: concatenating them must reproduce the final text.
    assert streamed == response.message.content[0].root.text
    assert response.finish_reason == FinishReason.STOP
    assert response.usage is not None
    assert response.usage.output_tokens is not None and response.usage.output_tokens > 0


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


async def test_undocumented_tool_stream() -> None:
    weather = undocumented_weather_tool()
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='What is the weather in Lagos?'))])],
        tools=[weather],
        config=BedrockConfig(tool_choice='get_weather'),
    )
    ctx, chunks = streaming_ctx()
    response = await make_model(NOVA).generate(request, ctx)

    # Tool input arrives as JSON fragments, so it is held back and emitted
    # once, whole, when the content block closes.
    tool_chunks = [
        chunk.content[0].root.tool_request for chunk in chunks if chunk.content[0].root.tool_request is not None
    ]
    assert len(tool_chunks) == 1
    assert tool_chunks[0].name == 'get_weather'
    assert tool_chunks[0].ref
    assert isinstance(tool_chunks[0].input, dict) and tool_chunks[0].input.get('city')

    assert response.message is not None
    final = [part.root.tool_request for part in response.message.content if part.root.tool_request is not None]
    assert len(final) == 1
    assert final[0].input == tool_chunks[0].input


async def test_claude_sync_without_config() -> None:
    # No config on purpose: Converse accepts Claude requests without maxTokens
    # and applies a service default, so the plugin injects nothing.
    response = await make_model(CLAUDE).generate(text_request("Reply with the single word 'pong'."))
    assert response.finish_reason == FinishReason.STOP
    assert response.message is not None
    text = response.message.content[0].root.text
    assert text is not None and 'pong' in text.lower()


async def test_claude_reasoning_signature_round_trip() -> None:
    model = make_model(CLAUDE)
    # Bedrock requires budget_tokens >= 1024 and maxTokens above it; thinking
    # requests reject custom temperature, so none is set.
    config = BedrockConfig(
        max_output_tokens=4096,
        additional_model_request_fields={'thinking': {'type': 'enabled', 'budget_tokens': 1024}},
    )
    request = text_request('What is 17 * 23? Think it through.', config=config)
    response = await model.generate(request)

    assert response.message is not None
    reasoning_parts = [
        part.root for part in response.message.content if getattr(part.root, 'reasoning', None) is not None
    ]
    assert reasoning_parts, 'expected a reasoning part on a thinking-enabled sync call'
    assert reasoning_parts[0].metadata is not None
    assert reasoning_parts[0].metadata.get(REASONING_SIGNATURE_METADATA_KEY)

    # Replaying the signed reasoning verbatim must be accepted by Bedrock.
    follow_up = ModelRequest(
        messages=[
            *request.messages,
            response.message,
            Message(role=Role.USER, content=[Part(root=TextPart(text='Now add 100 to that.'))]),
        ],
        config=config,
    )
    follow_up_response = await model.generate(follow_up)
    assert follow_up_response.finish_reason == FinishReason.STOP


async def test_claude_thinking_stream() -> None:
    config = BedrockConfig(
        max_output_tokens=4096,
        additional_model_request_fields={'thinking': {'type': 'enabled', 'budget_tokens': 1024}},
    )
    ctx, chunks = streaming_ctx()
    response = await make_model(CLAUDE).generate(text_request('What is 17 * 23? Think it through.', config=config), ctx)

    assert any(getattr(chunk.content[0].root, 'reasoning', None) for chunk in chunks), (
        'expected reasoning deltas on a thinking-enabled stream'
    )
    assert response.message is not None
    reasoning_parts = [
        part.root for part in response.message.content if getattr(part.root, 'reasoning', None) is not None
    ]
    assert reasoning_parts
    assert reasoning_parts[0].metadata is not None
    # The signature arrives in its own delta, which streams nothing, so this
    # only passes if reassembly kept it.
    assert reasoning_parts[0].metadata.get(REASONING_SIGNATURE_METADATA_KEY)


async def test_deepseek_reasoning_stream() -> None:
    ctx, chunks = streaming_ctx()
    response = await make_model(DEEPSEEK).generate(
        text_request('What is 17 * 23? Think it through.', config=BedrockConfig(max_output_tokens=2048)), ctx
    )

    assert any(getattr(chunk.content[0].root, 'reasoning', None) for chunk in chunks)
    assert response.message is not None
    reasoning_parts = [
        part.root for part in response.message.content if getattr(part.root, 'reasoning', None) is not None
    ]
    assert reasoning_parts
    # Unsigned reasoning: this model sends no signature delta at all.
    metadata = reasoning_parts[0].metadata
    assert metadata is None or not metadata.get(REASONING_SIGNATURE_METADATA_KEY)


async def test_deepseek_reasoning_sync_and_round_trip() -> None:
    model = make_model(DEEPSEEK)
    config = BedrockConfig(max_output_tokens=2048)
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


async def test_embed_titan_text_v1() -> None:
    vectors = await embed(TITAN_EMBED_V1, [text_doc('a red apple'), text_doc('a green pear')])
    assert len(vectors) == 2
    assert all(len(vector) == 1536 for vector in vectors)


async def test_embed_titan_text_v2() -> None:
    vectors = await embed(TITAN_EMBED_V2, [text_doc('a red apple')])
    assert len(vectors[0]) == 1024


async def test_embed_titan_multimodal_text() -> None:
    vectors = await embed(TITAN_EMBED_IMAGE, [text_doc('a red apple')])
    assert len(vectors[0]) == 1024


async def test_embed_titan_multimodal_image() -> None:
    vectors = await embed(TITAN_EMBED_IMAGE, [image_doc()])
    assert len(vectors[0]) == 1024


async def test_embed_titan_multimodal_text_and_image() -> None:
    document = DocumentData(
        content=[
            DocumentPart(root=TextPart(text='a white square')),
            DocumentPart(root=MediaPart(media=Media(url=PNG_1X1_DATA_URL))),
        ]
    )
    vectors = await embed(TITAN_EMBED_IMAGE, [document])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1024


async def test_embed_cohere_text_batch() -> None:
    vectors = await embed(COHERE_EMBED, [text_doc('one'), text_doc('two'), text_doc('three')])
    assert len(vectors) == 3
    assert len({len(vector) for vector in vectors}) == 1
    assert len(vectors[0]) > 0


async def test_embed_cohere_multilingual() -> None:
    vectors = await embed(COHERE_EMBED_MULTI, [text_doc('bonjour le monde')])
    assert len(vectors[0]) > 0


async def test_embed_cohere_rejects_an_image() -> None:
    # Bedrock reports inputModalities [TEXT] for the Cohere v3 models, so this
    # is refused locally rather than sent and rejected as a bad request.
    with pytest.raises(GenkitError, match='text-only'):
        await embed(COHERE_EMBED, [image_doc()])


@pytest.mark.skipif(
    os.environ.get('AWS_REGION') != 'us-east-1',
    reason='amazon.nova-2-multimodal-embeddings-v1:0 is only offered in us-east-1',
)
async def test_embed_nova_2() -> None:
    vectors = await embed(NOVA_EMBED, [text_doc('a red apple')])
    assert len(vectors[0]) == 3072
