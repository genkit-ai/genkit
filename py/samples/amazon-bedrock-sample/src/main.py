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

"""Amazon Bedrock samples for the Converse, ConverseStream, and embedding paths.

Needs AWS credentials and a region (``AWS_REGION`` or ``~/.aws/config``) with
model access granted for the models below. The flows named ``*_stream`` go
through ConverseStream; the ``embed_*`` flows go through InvokeModel; the rest
are a single Converse call.
"""

import math

from genkit_amazon_bedrock import Bedrock, ModelDefinition
from pydantic import BaseModel, Field

from genkit import (
    ActionRunContext,
    Document,
    DocumentPart,
    Genkit,
    Media,
    MediaPart,
    ModelResponse,
    ReasoningPart,
    TextPart,
)

NOVA = 'bedrock/us.amazon.nova-lite-v1:0'
LLAMA = 'bedrock/us.meta.llama3-3-70b-instruct-v1:0'
DEEPSEEK = 'bedrock/us.deepseek.r1-v1:0'
CLAUDE = 'bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0'
TITAN_EMBED = 'amazon.titan-embed-text-v2:0'
TITAN_EMBED_IMAGE = 'amazon.titan-embed-image-v1'
COHERE_EMBED = 'cohere.embed-english-v3'
NOVA_EMBED = 'amazon.nova-2-multimodal-embeddings-v1:0'

# The smallest PNG Titan accepts (1x1 pixel), inline so the multimodal flow
# needs no asset file. Titan wants raw base64 in the request body; the plugin
# strips this data URL's prefix before the call.
PNG_1X1_DATA_URL = (
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII='
)

# Declaring an embedder costs nothing at startup: no AWS call happens until a flow runs one,
# so a missing model-access grant only surfaces when you use it.
ai = Genkit(
    plugins=[
        Bedrock(
            models=[
                ModelDefinition(name='us.amazon.nova-lite-v1:0'),
                ModelDefinition(name='us.meta.llama3-3-70b-instruct-v1:0'),
                ModelDefinition(name='us.deepseek.r1-v1:0'),
                ModelDefinition(name='us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
            ],
            embedders=[TITAN_EMBED, TITAN_EMBED_IMAGE, COHERE_EMBED, NOVA_EMBED],
        )
    ],
    model=NOVA,
)


class TopicInput(BaseModel):
    """Input for a plain-text generation."""

    topic: str = Field(default='coding', description='Topic for the haiku')


class CatInput(BaseModel):
    """Input for a structured generation."""

    name: str = Field(default='Mittens', description='Name of the cat to invent')


class Cat(BaseModel):
    """Structured cat profile."""

    name: str
    breed: str
    age: int
    personality: str


class CityInput(BaseModel):
    """Input for the weather tool."""

    city: str = Field(default='Lagos', description='City to look up')


class EmbedInput(BaseModel):
    """Input for the embedding flow."""

    text: str = Field(default='Bedrock hosts models from several providers.', description='Text to embed')


class EmbedBatchInput(BaseModel):
    """Input for the batch embedding flow."""

    embedder: str = Field(default=TITAN_EMBED, description='Bedrock embedding model ID to use')
    texts: list[str] = Field(
        default=[
            'Bedrock hosts models from several providers.',
            'A tabby cat naps on the windowsill.',
            'Rust and Go both compile to native binaries.',
            'Bedrock hosts models from several providers.',
        ],
        description='Documents to embed in one call; the last repeats the first on purpose',
    )


class SimilarityInput(BaseModel):
    """Input for the semantic-similarity flow."""

    query: str = Field(default='How do I deploy a machine learning model?', description='Query text')
    candidates: list[str] = Field(
        default=[
            'Serving a trained model behind an HTTP endpoint in production.',
            'Sourdough needs a starter fed for several days before it can leaven bread.',
            'The tabby cat sleeps in the afternoon sun.',
        ],
        description='Candidate texts to rank against the query',
    )


class MultimodalEmbedInput(BaseModel):
    """Input for the Titan multimodal embedding flow."""

    text: str = Field(default='a white square', description='Caption embedded alongside the image')
    image_data_url: str = Field(default=PNG_1X1_DATA_URL, description='PNG or JPEG image as a data URL')


@ai.tool()
async def current_weather(city_input: CityInput) -> str:
    """Return mocked weather data for tool-calling demos."""
    return f'The weather in {city_input.city} is 31C and humid.'


@ai.flow()
async def haiku(data: TopicInput) -> str:
    """Plain-text generate through Converse."""
    response = await ai.generate(prompt=f'Write a haiku about {data.topic}.')
    return response.text


@ai.flow()
async def haiku_stream(data: TopicInput, ctx: ActionRunContext) -> str:
    """Plain-text generate through ConverseStream.

    Chunks are deltas, not snapshots, so the full text is the concatenation.
    """
    stream_response = ai.generate_stream(prompt=f'Write a haiku about {data.topic}.')
    chunks: list[str] = []
    async for chunk in stream_response.stream:
        if chunk.text:
            ctx.send_chunk(chunk.text)
            chunks.append(chunk.text)

    await stream_response.response
    return ''.join(chunks)


@ai.flow()
async def weather_report_stream(data: CityInput, ctx: ActionRunContext) -> str:
    """Tool calling over ConverseStream.

    A tool call's arguments arrive as JSON fragments, so unlike text it cannot
    be forwarded per delta: the whole tool request lands in one chunk when its
    content block closes.
    """
    stream_response = ai.generate_stream(
        prompt=f'What is the weather in {data.city}? Use the tool, then answer in one sentence.',
        tools=['current_weather'],
    )
    tool_calls: list[str] = []
    text: list[str] = []
    async for chunk in stream_response.stream:
        for part in chunk.content:
            if part.root.tool_request is not None:
                tool_calls.append(part.root.tool_request.name)
        if chunk.text:
            ctx.send_chunk(chunk.text)
            text.append(chunk.text)

    await stream_response.response
    return f'tools called: {tool_calls}\n{"".join(text)}'


@ai.flow()
async def thinking_stream(data: TopicInput, ctx: ActionRunContext) -> dict[str, object]:
    """Claude extended thinking over ConverseStream.

    Reasoning text streams delta by delta; the signature arrives in its own
    delta that streams nothing, and is attached to the final reasoning part so
    it can be replayed on the next turn.
    """
    stream_response = ai.generate_stream(
        model=CLAUDE,
        prompt=f'What is 17 * 23? Think it through, then state the answer. Mention {data.topic} once.',
        config={
            'maxTokens': 4096,
            'additionalModelRequestFields': {'thinking': {'type': 'enabled', 'budget_tokens': 1024}},
        },
    )
    reasoning_chunks = 0
    async for chunk in stream_response.stream:
        for part in chunk.content:
            if isinstance(part.root, ReasoningPart):
                reasoning_chunks += 1
        if chunk.text:
            ctx.send_chunk(chunk.text)

    summary = _reasoning_summary(await stream_response.response)
    return {**summary, 'reasoning_chunks': reasoning_chunks}


@ai.flow()
async def cat_profile(data: CatInput) -> Cat:
    """Structured output, carried by prompt instructions.

    Bedrock has no constrained-decoding mode, and the core's json format only
    injects the schema when ``output_instructions`` is set, so it is required
    here. Model choice matters too: the Nova models answer in prose often
    enough to fail extraction, so this uses Llama 3.3.
    """
    response = await ai.generate(
        model=LLAMA,
        prompt=f'Invent a cat named {data.name}.',
        output_format='json',
        output_schema=Cat,
        output_instructions=True,
        config={'maxOutputTokens': 1024},
    )
    return response.output


@ai.flow()
async def weather_report(data: CityInput) -> str:
    """Tool calling: the model calls the tool, then answers from its output."""
    response = await ai.generate(
        prompt=f'What is the weather in {data.city}? Use the tool, then answer in one sentence.',
        tools=['current_weather'],
    )
    return response.text


@ai.flow()
async def reasoning(data: TopicInput) -> dict[str, object]:
    """Reasoning parts parsed off a Converse response.

    DeepSeek R1 reasons on every turn, so no thinking config is needed. Its
    reasoning carries no signature, which is why ``signatures_present`` is
    false here: signatures are Anthropic-specific and gate replay.
    """
    response = await ai.generate(
        model=DEEPSEEK,
        prompt=f'What is 17 * 23? Think it through, then state the answer. Mention {data.topic} once.',
        config={'maxOutputTokens': 2048},
    )
    return _reasoning_summary(response)


@ai.flow()
async def thinking(data: TopicInput) -> dict[str, object]:
    """Claude extended thinking: signed reasoning that survives replay.

    Unlike DeepSeek, Claude signs its reasoning, so ``signatures_present`` is
    true here and the parts are replayed verbatim on multi-turn follow-ups.
    """
    response = await ai.generate(
        model=CLAUDE,
        prompt=f'What is 17 * 23? Think it through, then state the answer. Mention {data.topic} once.',
        config={
            'maxOutputTokens': 4096,
            # Bedrock requires budget_tokens >= 1024, below maxOutputTokens.
            'additionalModelRequestFields': {'thinking': {'type': 'enabled', 'budget_tokens': 1024}},
        },
    )
    return _reasoning_summary(response)


@ai.flow()
async def embed_text(data: EmbedInput) -> dict[str, object]:
    """Embed one document with Titan Text v2 through InvokeModel.

    Titan text embeds one document per call, so this is the plain single-call
    path. Only the head of the vector is returned: 1024 floats are unreadable
    in the Dev UI. Titan v2 normalizes its output, so the magnitude is ~1.0.
    """
    embeddings = await ai.embed(embedder=f'bedrock/{TITAN_EMBED}', content=data.text)
    if not embeddings:
        raise RuntimeError('Bedrock embedder returned no embeddings for a non-empty input.')
    vector = embeddings[0].embedding
    return {
        'model': TITAN_EMBED,
        'dimensions': len(vector),
        'first_values': [round(value, 5) for value in vector[:5]],
        'magnitude': round(_magnitude(vector), 5),
    }


@ai.flow()
async def embed_batch(data: EmbedBatchInput) -> dict[str, object]:
    """Embed several documents in one ``embed_many`` call.

    Titan text has no batch API, so the plugin fans the documents out into
    concurrent InvokeModel calls capped by a semaphore and reassembles them by
    index; Cohere instead sends up to 96 texts in a single call. Either way the
    results come back aligned to their inputs, which is what the repeated text
    shows: its similarity to its own copy is 1.0, at both ends of the list.

    Switch ``embedder`` to ``cohere.embed-english-v3`` for the single-call batch
    path, or to ``amazon.nova-2-multimodal-embeddings-v1:0`` (us-east-1 only)
    for 3072 dimensions.
    """
    embeddings = await ai.embed_many(embedder=f'bedrock/{data.embedder}', content=data.texts)
    if len(embeddings) != len(data.texts):
        raise RuntimeError(f'Expected {len(data.texts)} embeddings, got {len(embeddings)}.')
    vectors = [embedding.embedding for embedding in embeddings]
    return {
        'model': data.embedder,
        'count': len(vectors),
        'documents': [
            {'index': index, 'text': text, 'dimensions': len(vector), 'first_value': round(vector[0], 5)}
            for index, (text, vector) in enumerate(zip(data.texts, vectors, strict=True))
        ],
        'repeated_text_pairs': [
            {'a': first, 'b': second, 'similarity': round(_cosine_similarity(vectors[first], vectors[second]), 5)}
            for first, second in _repeated_pairs(data.texts)
        ],
    }


@ai.flow()
async def embed_similarity(data: SimilarityInput) -> dict[str, object]:
    """Rank candidate texts against a query by cosine similarity.

    A dimension count only proves the vector is well shaped. Ranking proves the
    numbers carry meaning: the candidate on the query's subject scores well
    above the unrelated ones. Titan is used rather than Cohere because Cohere
    requests are pinned to ``input_type: search_document`` for now, which is the
    wrong side of the asymmetry for a query.
    """
    query_embeddings = await ai.embed(embedder=f'bedrock/{TITAN_EMBED}', content=data.query)
    if not query_embeddings:
        raise RuntimeError('Bedrock embedder returned no embeddings for the query.')
    candidate_embeddings = await ai.embed_many(embedder=f'bedrock/{TITAN_EMBED}', content=data.candidates)
    query_vector = query_embeddings[0].embedding
    scored = [
        (text, _cosine_similarity(query_vector, embedding.embedding))
        for text, embedding in zip(data.candidates, candidate_embeddings, strict=True)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return {
        'model': TITAN_EMBED,
        'query': data.query,
        'dimensions': len(query_vector),
        'best_match': scored[0][0] if scored else None,
        'ranked': [{'text': text, 'similarity': round(similarity, 5)} for text, similarity in scored],
    }


@ai.flow()
async def embed_image(data: MultimodalEmbedInput) -> dict[str, object]:
    """Embed a caption and an image together with Titan Multimodal.

    One document carrying both a text part and a media part becomes one joint
    1024-float vector; this is the only flow that exercises the media path. The
    caption is also embedded on its own, so the output shows the image really
    reached the model: the two vectors are not the same.
    """
    joint_document = Document(
        content=[
            DocumentPart(root=TextPart(text=data.text)),
            DocumentPart(root=MediaPart(media=Media(url=data.image_data_url, content_type='image/png'))),
        ]
    )
    embeddings = await ai.embed_many(
        embedder=f'bedrock/{TITAN_EMBED_IMAGE}',
        content=[joint_document, Document.from_text(data.text)],
    )
    if len(embeddings) != 2:
        raise RuntimeError(f'Expected 2 embeddings, got {len(embeddings)}.')
    joint = embeddings[0].embedding
    text_only = embeddings[1].embedding
    return {
        'model': TITAN_EMBED_IMAGE,
        'dimensions': len(joint),
        'first_values': [round(value, 5) for value in joint[:5]],
        'text_only_dimensions': len(text_only),
        'similarity_to_caption_alone': round(_cosine_similarity(joint, text_only), 5),
    }


def _magnitude(vector: list[float]) -> float:
    """Euclidean length of a vector; stdlib math only, the sample has no numpy."""
    return math.sqrt(math.fsum(value * value for value in vector))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity of two vectors, in [-1.0, 1.0] for non-zero inputs."""
    if len(left) != len(right):
        raise ValueError(f'Vectors must be the same length, got {len(left)} and {len(right)}.')
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    scale = _magnitude(left) * _magnitude(right)
    return dot / scale if scale else 0.0


def _repeated_pairs(texts: list[str]) -> list[tuple[int, int]]:
    """Index pairs of identical texts, so per-document alignment can be checked."""
    seen: dict[str, int] = {}
    pairs: list[tuple[int, int]] = []
    for index, text in enumerate(texts):
        first = seen.setdefault(text, index)
        if first != index:
            pairs.append((first, index))
    return pairs


def _reasoning_summary(response: ModelResponse) -> dict[str, object]:
    """Summarize the reasoning parts on a response."""
    reasoning_text: list[str] = []
    signed: list[bool] = []
    for message in response.messages:
        for part in message.content:
            root = part.root
            if isinstance(root, ReasoningPart):
                reasoning_text.append(root.reasoning)
                signed.append(bool(root.metadata and root.metadata.get('bedrockReasoningSignature')))
    return {
        'answer': response.text,
        'reasoning_parts': len(reasoning_text),
        'reasoning_preview': ''.join(reasoning_text)[:500],
        'signatures_present': signed,
    }


async def main() -> None:
    """Run the lightweight flows once from the CLI."""
    try:
        print(await haiku(TopicInput()))  # noqa: T201
        print(await haiku_stream(TopicInput()))  # noqa: T201
        print(await weather_report(CityInput()))  # noqa: T201
        print(await embed_text(EmbedInput()))  # noqa: T201
        print(await embed_similarity(SimilarityInput()))  # noqa: T201
    except Exception as error:
        # Printed, not raised: in dev mode the Dev UI stays up either way.
        print(  # noqa: T201
            f'Set AWS credentials and a region, and grant model access for {NOVA}, {LLAMA}, {DEEPSEEK}, and '
            f'{TITAN_EMBED}, before running this sample.\n{error}'
        )


if __name__ == '__main__':
    ai.run_main(main())
