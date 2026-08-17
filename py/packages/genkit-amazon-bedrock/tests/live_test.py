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
and the image models each need their own model-access grant, separate from
the chat models and from each other.
"""

import contextlib
import os
import re
from collections.abc import Iterator

import pytest
from genkit_amazon_bedrock.config import BedrockConfig
from genkit_amazon_bedrock.converters import (
    REASONING_SIGNATURE_METADATA_KEY,
    cache_point_part,
)
from genkit_amazon_bedrock.embedders import BedrockEmbedder
from genkit_amazon_bedrock.image import BedrockImageModel
from genkit_amazon_bedrock.models import BedrockModel
from genkit_amazon_bedrock.rerank import BedrockReranker, BedrockRerankOptions, RerankerRequest
from genkit_amazon_bedrock.transport import BedrockTransport

from genkit import (
    DocumentPart,
    FinishReason,
    Media,
    MediaPart,
    Message,
    ModelRequest,
    ModelResponse,
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

NOVA_CANVAS = 'amazon.nova-canvas-v1:0'
SD3 = 'stability.sd3-5-large-v1:0'

COHERE_RERANK = 'cohere.rerank-v3-5:0'
AMAZON_RERANK = 'amazon.rerank-v1:0'
AMAZON_RERANK_REGIONS = ('us-west-2', 'eu-central-1', 'ap-northeast-1', 'ca-central-1')

# Smallest PNG Titan accepts.
PNG_1X1 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII='
PNG_1X1_DATA_URL = f'data:image/png;base64,{PNG_1X1}'

# A 64x48 PNG of three horizontal bands (red, white, blue). The 1x1 pixel above
# is too small to ask a model about; these bands give a checkable answer.
STRIPES_PNG_DATA_URL = (
    'data:image/png;base64,'
    'iVBORw0KGgoAAAANSUhEUgAAAEAAAAAwCAIAAAAuKetIAAAATklEQVR42u3PMQ0AMAwEsWAKksIOiNLoXg7ZXvLpCLhud/QFAAAAAAAA'
    'AAAAALAGvPAAAAAAAAAAAAAAAPaAPhM9AAAAAAAAAAAAAMD6DyqspTygWsHeAAAAAElFTkSuQmCC'
)

# A one-page PDF reading "Genkit Bedrock sample memo / The office kettle is
# broken. / Tea service resumes on Friday."
MEMO_PDF_DATA_URL = (
    'data:application/pdf;base64,'
    'JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2JqCjIgMCBvYmoKPDwgL1R5cGUgL1Bh'
    'Z2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2JqCjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVk'
    'aWFCb3ggWzAgMCA2MTIgNzkyXSAvUmVzb3VyY2VzIDw8IC9Gb250IDw8IC9GMSA1IDAgUiA+PiA+PiAvQ29udGVudHMgNCAwIFIgPj4K'
    'ZW5kb2JqCjQgMCBvYmoKPDwgL0xlbmd0aCAxNTAgPj4Kc3RyZWFtCkJUCi9GMSAxNCBUZgoxIDAgMCAxIDcyIDcyMCBUbQoxOCBUTAoo'
    'R2Vua2l0IEJlZHJvY2sgc2FtcGxlIG1lbW8pIFRqClQqCihUaGUgb2ZmaWNlIGtldHRsZSBpcyBicm9rZW4uKSBUagpUKgooVGVhIHNl'
    'cnZpY2UgcmVzdW1lcyBvbiBGcmlkYXkuKSBUagpUKgpFVAplbmRzdHJlYW0KZW5kb2JqCjUgMCBvYmoKPDwgL1R5cGUgL0ZvbnQgL1N1'
    'YnR5cGUgL1R5cGUxIC9CYXNlRm9udCAvSGVsdmV0aWNhIC9FbmNvZGluZyAvV2luQW5zaUVuY29kaW5nID4+CmVuZG9iagp4cmVmCjAg'
    'NgowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwMDkgMDAwMDAgbiAKMDAwMDAwMDA1OCAwMDAwMCBuIAowMDAwMDAwMTE1IDAwMDAw'
    'IG4gCjAwMDAwMDAyNDEgMDAwMDAgbiAKMDAwMDAwMDQ0MiAwMDAwMCBuIAp0cmFpbGVyCjw8IC9TaXplIDYgL1Jvb3QgMSAwIFIgPj4K'
    'c3RhcnR4cmVmCjUzOQolJUVPRgo='
)


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


def make_image_model(model_id: str) -> BedrockImageModel:
    return BedrockImageModel(model_id=model_id, transport=make_transport())


# Bedrock refusing the model itself, rather than the request. Model access is
# per account: a Legacy model also drops out after 30 days of disuse, which no
# test can control.
_UNAVAILABLE_MARKERS = (
    'AccessDeniedException',
    'ResourceNotFoundException',
    'marked by provider as Legacy',
)


@contextlib.contextmanager
def skip_if_model_unavailable(model_id: str) -> Iterator[None]:
    """Turns a model-access failure into a skip, keeping real bugs failing."""
    try:
        yield
    except GenkitError as error:
        message = str(error)
        invalid_id = 'ValidationException' in message and 'model identifier is invalid' in message
        if invalid_id or any(marker in message for marker in _UNAVAILABLE_MARKERS):
            pytest.skip(f'{model_id} unavailable to this account or region: {message}')
        raise


def assert_image_response(response: ModelResponse, mime: str = 'image/png') -> None:
    assert response.finish_reason == FinishReason.STOP
    assert response.message is not None
    prefix = f'data:{mime};base64,'
    media_parts = [part.root for part in response.message.content if isinstance(part.root, MediaPart)]
    assert media_parts, 'expected at least one media part'
    for part in media_parts:
        assert part.media.content_type == mime
        assert part.media.url.startswith(prefix)
        assert part.media.url.removeprefix(prefix)


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


def media_request(data_url: str, text: str, **kwargs) -> ModelRequest:
    """A user message carrying media plus the text Converse requires with it."""
    return ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=MediaPart(media=Media(url=data_url))), Part(root=TextPart(text=text))],
            )
        ],
        **kwargs,
    )


_HANDBOOK_TOPICS = [
    ('Returns', 'fulfilment', 'three business days'),
    ('Shipping', 'logistics', 'one business day'),
    ('Warranty', 'aftercare', 'five business days'),
    ('Billing', 'finance', 'two business days'),
    ('Accounts', 'identity', 'one business day'),
    ('Privacy', 'legal', 'ten business days'),
    ('Accessibility', 'design', 'four business days'),
    ('Recycling', 'sustainability', 'six business days'),
]

_HANDBOOK_REGIONS = ['North America', 'Ireland', 'Germany', 'Japan']


def mentions_any(text: str | None, *words: str) -> bool:
    """Whether the text contains any of the words as whole words.

    Substring matching would pass on 'coloured' for 'red' and 'instead' for
    'tea', which a refusal could satisfy without ever reading the attachment.
    """
    haystack = (text or '').lower()
    return any(re.search(rf'\b{re.escape(word)}\b', haystack) for word in words)


def cacheable_prefix() -> str:
    """A system prompt long enough to be cacheable, at roughly 3,500 tokens.

    Claude will not cache a prefix below about 1,000 tokens, and reports no
    error when it declines.
    """
    articles = []
    for number, (topic, team, window) in enumerate(_HANDBOOK_TOPICS, start=1):
        articles.append(
            f'Article {number}. {topic}\n'
            f'{topic} requests are handled by the {team} team in {window}. Escalations go to the duty manager, '
            f'who answers within one business day. Records are retained for seven years and can be exported on '
            f'request. This article is reviewed every quarter and supersedes any earlier guidance on {topic}.'
        )
        for section, region in enumerate(_HANDBOOK_REGIONS, start=1):
            articles.append(
                f'Article {number}.{section} {topic} in {region}\n'
                f'In {region}, {topic.lower()} follows the same {window} target, measured in local business '
                f'hours and excluding public holidays. The {team} team keeps a regional queue, and a request '
                f'that misses its target is reported in the weekly service review. Customers are notified by '
                f'email at intake and again at resolution, and may ask for a written summary of the decision '
                f'at any point without charge.'
            )
    return 'Answer only from this handbook.\n\n' + '\n\n'.join(articles)


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


async def test_nova_image_input() -> None:
    # An image block Bedrock cannot parse fails the call outright, so reaching a
    # normal stop is itself the assertion that the media round-tripped; the
    # colour check is the evidence the pixels reached the model.
    request = media_request(STRIPES_PNG_DATA_URL, 'What colours are the bands in this image?')
    response = await make_model(NOVA).generate(request)
    assert response.finish_reason == FinishReason.STOP
    assert mentions_any(response.text, 'red', 'white', 'blue'), response.text


async def test_claude_document_input() -> None:
    # Bedrock parses the PDF server-side, so a malformed document block or a
    # double-encoded payload fails the call rather than degrading quietly.
    # Claude, not Nova: document support is per model.
    request = media_request(
        MEMO_PDF_DATA_URL, 'What does this document say?', config=BedrockConfig(max_output_tokens=512)
    )
    response = await make_model(CLAUDE).generate(request)
    assert response.finish_reason == FinishReason.STOP
    assert mentions_any(response.text, 'kettle', 'tea', 'friday'), response.text


async def test_claude_prompt_cache_read() -> None:
    # Two calls sharing a byte-identical cached prefix: the first writes the
    # cache, the second reads it. Only reads surface, since the plugin drops
    # cacheWriteInputTokens, so the second call carries the evidence.
    model = make_model(CLAUDE)
    system = Message(
        role=Role.SYSTEM,
        content=[Part(root=TextPart(text=cacheable_prefix())), cache_point_part()],
    )
    config = BedrockConfig(max_output_tokens=512)

    def ask(question: str) -> ModelRequest:
        return ModelRequest(
            messages=[system, Message(role=Role.USER, content=[Part(root=TextPart(text=question))])],
            config=config,
        )

    first = await model.generate(ask('Which article covers returns? Answer with its number.'))
    assert first.finish_reason == FinishReason.STOP

    second = await model.generate(ask('Which article covers shipping? Answer with its number.'))
    assert second.finish_reason == FinishReason.STOP
    assert second.usage is not None
    cached = second.usage.cached_content_tokens
    # Above the minimum, not merely above zero: that also catches the prefix
    # being shrunk under it, which Bedrock answers by quietly not caching.
    # Do not assert on input_tokens; it excludes cached tokens, so it stays
    # small on both calls however well the cache works.
    assert cached is not None and cached > 1024, f'expected a cache read over the minimum, got {cached!r}'


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


async def test_rerank_cohere() -> None:
    reranker = BedrockReranker(model_id=COHERE_RERANK, transport=make_transport())
    response = await reranker.rerank(
        RerankerRequest(
            query=text_doc('What is the capital of France?'),
            documents=[
                text_doc('The capital of France is Paris.'),
                text_doc('The tallest mountain is Everest.'),
                text_doc('Bananas are yellow.'),
            ],
            options=BedrockRerankOptions(top_n=2),
        )
    )

    assert len(response.documents) == 2
    top = [part.root for part in response.documents[0].content if isinstance(part.root, TextPart)]
    assert top and 'Paris' in (top[0].text or '')
    assert response.documents[0].metadata.score >= response.documents[1].metadata.score


@pytest.mark.skipif(
    os.environ.get('AWS_REGION') not in AMAZON_RERANK_REGIONS,
    reason='amazon.rerank-v1:0 is offered in us-west-2, eu-central-1, ap-northeast-1 and ca-central-1 only',
)
async def test_rerank_amazon() -> None:
    # The Amazon family takes the same body minus api_version, which its schema
    # rejects outright, so this is the only wire check of that difference.
    reranker = BedrockReranker(model_id=AMAZON_RERANK, transport=make_transport())
    response = await reranker.rerank(
        RerankerRequest(
            query=text_doc('What is the capital of France?'),
            documents=[
                text_doc('The capital of France is Paris.'),
                text_doc('The tallest mountain is Everest.'),
                text_doc('Bananas are yellow.'),
            ],
            options=BedrockRerankOptions(top_n=2),
        )
    )

    assert len(response.documents) == 2
    top = [part.root for part in response.documents[0].content if isinstance(part.root, TextPart)]
    assert top and 'Paris' in (top[0].text or '')
    assert response.documents[0].metadata.score >= response.documents[1].metadata.score


@pytest.mark.skipif(
    os.environ.get('AWS_REGION') != 'us-east-1',
    reason='amazon.nova-2-multimodal-embeddings-v1:0 is only offered in us-east-1',
)
async def test_embed_nova_2() -> None:
    vectors = await embed(NOVA_EMBED, [text_doc('a red apple')])
    assert len(vectors[0]) == 3072


@pytest.mark.skipif(
    os.environ.get('AWS_REGION') != 'us-east-1',
    reason=(
        'amazon.nova-canvas-v1:0 is offered in us-east-1, eu-west-1 and ap-northeast-1 only, '
        'and reaches end of life on 2026-09-30'
    ),
)
async def test_image_nova_canvas() -> None:
    # Legacy: new accounts cannot enable it at all, and an enabled one loses
    # access after 30 days of disuse, so unavailability here is not a failure.
    with skip_if_model_unavailable(NOVA_CANVAS):
        response = await make_image_model(NOVA_CANVAS).generate(
            text_request('A futuristic city skyline at dawn, digital art')
        )
        assert_image_response(response)


@pytest.mark.skipif(
    os.environ.get('AWS_REGION') != 'us-west-2',
    reason='stability.sd3-5-large-v1:0 is only offered in us-west-2',
)
async def test_image_stability_sd3() -> None:
    with skip_if_model_unavailable(SD3):
        response = await make_image_model(SD3).generate(
            text_request('A vibrant coral reef teeming with fish, photorealistic')
        )
        assert_image_response(response)


@pytest.mark.skipif(
    os.environ.get('AWS_REGION') != 'us-west-2',
    reason='stability.sd3-5-large-v1:0 is only offered in us-west-2',
)
async def test_image_stability_sd3_with_config() -> None:
    # Flat per-call overrides only reach the wire if nothing coerces them into
    # the Converse config shape on the way in.
    with skip_if_model_unavailable(SD3):
        response = await make_image_model(SD3).generate(
            text_request(
                'A minimalist geometric pattern in blue and gold',
                config={'aspect_ratio': '16:9', 'output_format': 'jpeg'},
            )
        )
        assert_image_response(response, mime='image/jpeg')
