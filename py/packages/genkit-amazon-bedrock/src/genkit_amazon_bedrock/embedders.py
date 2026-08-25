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

"""Bedrock embedder implementation (InvokeModel API).

Embedding models predate Converse and have no unified request shape: each
family takes its own raw JSON body and returns its own response shape, so
routing by model ID is unavoidable.

Bedrock behaviours that are easy to get wrong: Nova uses the
``amazon.nova-2-multimodal-embeddings-v1:0`` schema rather than a Titan-shaped
body; Cohere Embed v3 is text-only; Cohere text is chunked at Bedrock's
96-document per-request limit; routing matches ``cohere.embed`` rather than
bare ``cohere``, so rerank model IDs are not swallowed; and Titan multimodal
reports input errors in-band, in a ``message`` field on an HTTP 200.

On Cohere: the AWS parameters page documents an ``images`` field and an
``image`` input type for Embed v3, but ``get-foundation-model`` reports
``inputModalities: [TEXT]`` for both v3 model IDs and a live image request is
rejected with ``Invalid parameter combination``. Image embedding needs
``amazon.titan-embed-image-v1``, or Cohere Embed v4 once that lands.
"""

import asyncio
import json
from collections.abc import Coroutine
from typing import Any, Literal, NamedTuple, Protocol, TypeVar, cast

import structlog
from botocore.exceptions import BotoCoreError, ClientError

from genkit import MediaPart, TextPart

# DocumentData has no public re-export yet; the embedder protocol is built on it.
from genkit._core._typing import DocumentData
from genkit.embedder import (
    EmbedderOptions,
    EmbedderSupports,
    Embedding,
    EmbedRequest,
    EmbedResponse,
)
from genkit.plugin_api import GenkitError
from genkit_amazon_bedrock.model_info import model_label, strip_inference_profile_prefix
from genkit_amazon_bedrock.models import _from_botocore_error, _from_client_error

logger = structlog.get_logger(__name__)

# One call's result: a single vector per document, or a chunk of them for Cohere.
_T = TypeVar('_T')

# Caps simultaneous InvokeModel calls; large batches otherwise earn a ThrottlingException.
EMBED_CONCURRENCY_LIMIT = 10

# Bedrock's per-request document limit for Cohere text embeddings.
COHERE_TEXT_BATCH_SIZE = 96

TITAN_SUPPORTED_IMAGE_MIME = frozenset({'image/jpeg', 'image/jpg', 'image/png'})

# InvokeModel carries the image inline, so it cannot fetch any of these.
REMOTE_URL_SCHEMES = ('http://', 'https://', 's3://')

EmbeddingFamily = Literal['titan_text', 'titan_multimodal', 'cohere_v3', 'cohere_v4', 'nova']

# Substring matches over the base model ID, most specific first.
_FAMILY_PATTERNS: tuple[tuple[str, EmbeddingFamily], ...] = (
    ('titan-embed-image', 'titan_multimodal'),
    ('titan-embed', 'titan_text'),
    ('cohere.embed-v4', 'cohere_v4'),
    # Not bare 'cohere': that would swallow cohere.rerank-* and cohere.command-*.
    ('cohere.embed', 'cohere_v3'),
    ('nova-2-multimodal-embeddings', 'nova'),
)

TEXT_ONLY = ('text',)
TEXT_AND_IMAGE = ('text', 'image')

_FAMILY_INPUTS: dict[EmbeddingFamily, tuple[str, ...]] = {
    'titan_text': TEXT_ONLY,
    'titan_multimodal': TEXT_AND_IMAGE,
    'cohere_v3': TEXT_ONLY,
    'cohere_v4': TEXT_AND_IMAGE,
    'nova': TEXT_ONLY,
}


class EmbedderInfo(NamedTuple):
    """Published capabilities of a Bedrock embedding model."""

    dimensions: int
    input: tuple[str, ...]


EMBEDDER_INFO: dict[str, EmbedderInfo] = {
    'amazon.titan-embed-text-v1': EmbedderInfo(dimensions=1536, input=TEXT_ONLY),
    'amazon.titan-embed-text-v2:0': EmbedderInfo(dimensions=1024, input=TEXT_ONLY),
    'amazon.titan-embed-image-v1': EmbedderInfo(dimensions=1024, input=TEXT_AND_IMAGE),
    'cohere.embed-english-v3': EmbedderInfo(dimensions=1024, input=TEXT_ONLY),
    'cohere.embed-multilingual-v3': EmbedderInfo(dimensions=1024, input=TEXT_ONLY),
    'amazon.nova-2-multimodal-embeddings-v1:0': EmbedderInfo(dimensions=3072, input=TEXT_ONLY),
}


class InvokeModelTransport(Protocol):
    """Structural contract for the transport seam (see ``transport.py``)."""

    async def invoke_model(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Calls the InvokeModel API and returns the parsed response body."""
        ...


def _embedding_family(model_id: str) -> EmbeddingFamily | None:
    """Routes a model ID to its embedding family, or None if it is not one."""
    base_id = strip_inference_profile_prefix(model_id)
    for pattern, family in _FAMILY_PATTERNS:
        if pattern in base_id:
            return family
    return None


def is_embedding_model(model_id: str) -> bool:
    """Reports whether a Bedrock model ID names an embedding model.

    Args:
        model_id: Bedrock model ID, inference-profile ID, or ARN.

    Returns:
        True when the ID routes to a known embedding family.
    """
    return _embedding_family(model_id) is not None


def looks_like_embedding_model(model_id: str) -> bool:
    """Reports whether a model ID names an embedding model, routable or not.

    Recognition only, never routing: an embedding ID from a family this plugin
    has no request shape for still answers as an unsupported embedder rather
    than as a Converse chat model.

    Args:
        model_id: Bedrock model ID, inference-profile ID, or ARN.

    Returns:
        True when the ID carries the token every Bedrock embedding family uses.
    """
    return 'embed' in strip_inference_profile_prefix(model_id)


def get_embedder_options(model_id: str) -> EmbedderOptions:
    """Builds the Genkit embedder metadata for a Bedrock embedding model.

    Single source for both ``resolve`` and ``list_actions`` so the two can
    never disagree.

    Args:
        model_id: Bedrock model ID, inference-profile ID, or ARN.

    Returns:
        EmbedderOptions with registry dimensions, or unset dimensions and
        family-default modalities for a routable ID that is not registered.
    """
    info = EMBEDDER_INFO.get(strip_inference_profile_prefix(model_id))
    family = _embedding_family(model_id)
    if info is not None:
        inputs = info.input
    elif family is not None:
        inputs = _FAMILY_INPUTS[family]
    else:
        inputs = TEXT_ONLY
    return EmbedderOptions(
        # The label keeps the prefix: it names what the caller asked for.
        label=model_label(model_id),
        dimensions=info.dimensions if info is not None else None,
        supports=EmbedderSupports(input=list(inputs)),
    )


def document_text(document: DocumentData) -> str:
    """Joins a document's text parts.

    Whitespace-only parts are skipped, surviving parts are joined untrimmed,
    and only the result is stripped. Text is never flattened across documents:
    that would break the document-to-embedding alignment.

    Args:
        document: The document to read.

    Returns:
        The joined text, or an empty string when there is none.
    """
    texts = [part.root.text for part in document.content if isinstance(part.root, TextPart) and part.root.text.strip()]
    return '\n'.join(texts).strip()


def image_from_document(document: DocumentData) -> tuple[str, str]:
    """Returns the MIME type and raw base64 of the first image media part.

    Deliberately not ``converters._decode_media_payload``: Converse wants raw
    bytes, InvokeModel bodies are JSON and want base64 text (Titan multimodal
    takes it raw, Cohere wants it wrapped in a data URI).

    Args:
        document: The document to read.

    Returns:
        ``(mime, base64)``, or two empty strings when there is no image.

    Raises:
        GenkitError: INVALID_ARGUMENT when an image part holds a remote URL.
    """
    for part in document.content:
        if not isinstance(part.root, MediaPart):
            continue
        data_url = part.root.media.url
        mime = (part.root.media.content_type or '').split(';', 1)[0].strip().lower()
        # Fall back to the MIME type inside the data URL when contentType is absent.
        if not mime and data_url.startswith('data:'):
            header, found, _ = data_url.partition(',')
            if found:
                mime = header.split(';', 1)[0].removeprefix('data:').strip().lower()
        if not mime.startswith('image/'):
            continue
        if data_url.startswith(REMOTE_URL_SCHEMES):
            raise GenkitError(
                message='bedrock embed: remote URLs are not supported; use a data URL or base64-encoded data',
                status='INVALID_ARGUMENT',
            )
        if data_url.startswith('data:'):
            # No comma means the data URL carries no payload to send.
            _, _, payload = data_url.partition(',')
        else:
            # Bare base64, the other shape converters accepts.
            payload = data_url
        if payload:
            return mime, payload
    return '', ''


def decode_cohere_embeddings(payload: dict[str, Any]) -> list[list[float]]:
    """Decodes a Cohere embedding response into float vectors.

    Handles both the flat shape (``{"embeddings": [[...]]}``) and the typed
    shape that ``embedding_types`` returns (``{"embeddings": {"float": [[...]]}}``).

    Args:
        payload: The parsed InvokeModel response body.

    Returns:
        One vector per input, in order.

    Raises:
        GenkitError: INTERNAL when the response carries no usable embeddings.
    """
    embeddings = payload.get('embeddings')
    if isinstance(embeddings, dict):
        vectors = embeddings.get('float')
        if not vectors:
            raise GenkitError(
                message='bedrock embed: cohere typed response has no float embeddings',
                status='INTERNAL',
            )
    elif isinstance(embeddings, list):
        vectors = embeddings
        if not vectors:
            raise GenkitError(message='bedrock embed: cohere returned no embeddings', status='INTERNAL')
    else:
        raise GenkitError(message='bedrock embed: cohere response missing embeddings field', status='INTERNAL')
    return [[float(value) for value in vector] for vector in vectors]


def _single_embedding(payload: dict[str, Any]) -> list[float]:
    """Reads the ``embedding`` vector off a Titan InvokeModel response."""
    vector = payload.get('embedding')
    return [float(value) for value in vector] if isinstance(vector, list) else []


def _nova_embedding(payload: dict[str, Any]) -> list[float]:
    """Reads the first vector off a Nova-2 response.

    Nova-2 returns ``{"embeddings": [{"embedding": [...]}]}`` — a list of
    objects, not the bare vector the Titan models return.
    """
    embeddings = payload.get('embeddings')
    if not isinstance(embeddings, list) or not embeddings:
        return []
    first = embeddings[0]
    vector = first.get('embedding') if isinstance(first, dict) else None
    return [float(value) for value in vector] if isinstance(vector, list) else []


def _require_vector(vector: list[float]) -> list[float]:
    if not vector:
        raise GenkitError(message='bedrock embed: model returned an empty embedding vector', status='INTERNAL')
    return vector


def _require_text(document: DocumentData, index: int) -> str:
    text = document_text(document)
    if not text:
        raise GenkitError(message=f'bedrock embed: document {index} has no text content', status='INVALID_ARGUMENT')
    return text


def _require_cohere_text(document: DocumentData, index: int) -> str:
    text = document_text(document)
    if not text:
        raise GenkitError(
            message=(
                f'bedrock embed: document {index} has no text content; Bedrock offers the Cohere Embed v3 '
                'models as text-only (use amazon.titan-embed-image-v1 to embed images)'
            ),
            status='INVALID_ARGUMENT',
        )
    return text


# This plugin's own message prefixes, longest first, so wrapping a message in
# document context does not stack a second copy of the prefix.
_OWN_PREFIXES = ('bedrock embed: ', 'bedrock: ', 'bedrock ')


def _without_own_prefix(message: str) -> str:
    for prefix in _OWN_PREFIXES:
        if message.startswith(prefix):
            return message.removeprefix(prefix)
    return message


class BedrockEmbedder:
    """Handles an embed call for one Bedrock embedding model."""

    def __init__(self, model_id: str, transport: InvokeModelTransport) -> None:
        """Initializes the embedder.

        Args:
            model_id: Bedrock model ID, inference-profile ID, or ARN, sent to
                the InvokeModel API verbatim.
            transport: The shared transport seam owning the boto3 client.
        """
        self._model_id = model_id
        self._transport = transport

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Embeds every document in the request.

        ``request.options`` is ignored: the per-request config surface is not
        ported yet, so Cohere stays pinned to ``search_document``.

        Args:
            request: The Genkit embed request.

        Returns:
            One embedding per input document, in input order.

        Raises:
            GenkitError: INVALID_ARGUMENT for an empty or malformed request,
                UNIMPLEMENTED for a model this plugin cannot embed with, and
                the mapped AWS status for a failed call.
        """
        if not request.input:
            raise GenkitError(message='bedrock embed: request contains no documents', status='INVALID_ARGUMENT')
        family = _embedding_family(self._model_id)
        if family is None:
            raise GenkitError(
                message=f'bedrock embed: unsupported embedding model {self._model_id!r}',
                status='UNIMPLEMENTED',
            )
        if family == 'cohere_v4':
            raise GenkitError(
                message=(
                    f'bedrock embed: {self._model_id!r} uses the Cohere Embed v4 request schema, '
                    'which is not supported yet; use cohere.embed-english-v3 or cohere.embed-multilingual-v3'
                ),
                status='UNIMPLEMENTED',
            )

        logger.debug(
            'Bedrock embed request',
            model=self._model_id,
            family=family,
            documents=len(request.input),
        )
        if family == 'titan_text':
            vectors = await self._embed_titan_text(request.input)
        elif family == 'titan_multimodal':
            vectors = await self._embed_titan_multimodal(request.input)
        elif family == 'cohere_v3':
            vectors = await self._embed_cohere(request.input)
        else:
            vectors = await self._embed_nova(request.input)
        logger.debug(
            'Bedrock embed response',
            model=self._model_id,
            vectors=len(vectors),
            dimensions=len(vectors[0]) if vectors else 0,
        )
        return EmbedResponse(embeddings=[Embedding(embedding=vector) for vector in vectors])

    async def _embed_titan_text(self, documents: list[DocumentData]) -> list[list[float]]:
        # Every document is validated before the first call goes out, so a bad
        # batch costs nothing.
        texts = [_require_text(document, index) for index, document in enumerate(documents)]
        return await self._run_bounded([(index, self._titan_text_vector(text)) for index, text in enumerate(texts)])

    async def _titan_text_vector(self, text: str) -> list[float]:
        return _require_vector(_single_embedding(await self._invoke({'inputText': text})))

    async def _embed_titan_multimodal(self, documents: list[DocumentData]) -> list[list[float]]:
        bodies: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            text = document_text(document)
            try:
                mime, image = image_from_document(document)
            except GenkitError as error:
                raise GenkitError(
                    message=f'bedrock embed: document {index}: {_without_own_prefix(error.original_message)}',
                    status=error.status,
                ) from error
            if not text and not image:
                raise GenkitError(
                    message=f'bedrock embed: document {index} has no text or image content',
                    status='INVALID_ARGUMENT',
                )
            if image and mime not in TITAN_SUPPORTED_IMAGE_MIME:
                raise GenkitError(
                    message=(
                        f'bedrock embed: document {index} image format {mime!r} is not supported '
                        'by Titan (use JPEG or PNG)'
                    ),
                    status='INVALID_ARGUMENT',
                )
            # Absent keys, not empty ones: Titan rejects an empty inputImage.
            body: dict[str, Any] = {}
            if text:
                body['inputText'] = text
            if image:
                body['inputImage'] = image
            bodies.append(body)
        return await self._run_bounded([
            (index, self._titan_multimodal_vector(body)) for index, body in enumerate(bodies)
        ])

    async def _titan_multimodal_vector(self, body: dict[str, Any]) -> list[float]:
        payload = await self._invoke(body)
        # Titan multimodal reports input problems in-band, on an HTTP 200.
        message = payload.get('message')
        if message is not None:
            raise GenkitError(message=f'bedrock embed: titan multimodal: {message}', status='INTERNAL')
        return _require_vector(_single_embedding(payload))

    async def _embed_cohere(self, documents: list[DocumentData]) -> list[list[float]]:
        # Any media part is ignored: these models take text only.
        texts = [_require_cohere_text(document, index) for index, document in enumerate(documents)]
        # The one family with a batch API, so chunks replace the per-document
        # fan-out. Each index is its chunk's first document, so a failure names it.
        batches = await self._run_bounded([
            (start, self._cohere_chunk_vectors(texts[start : start + COHERE_TEXT_BATCH_SIZE]))
            for start in range(0, len(texts), COHERE_TEXT_BATCH_SIZE)
        ])
        return [vector for batch in batches for vector in batch]

    async def _cohere_chunk_vectors(self, chunk: list[str]) -> list[list[float]]:
        payload = await self._invoke({
            'texts': chunk,
            'input_type': 'search_document',
            'truncate': 'END',
            'embedding_types': ['float'],
        })
        batch = decode_cohere_embeddings(payload)
        if len(batch) != len(chunk):
            raise GenkitError(
                message=f'bedrock embed: cohere returned {len(batch)} text embeddings for {len(chunk)} inputs',
                status='INTERNAL',
            )
        return batch

    async def _embed_nova(self, documents: list[DocumentData]) -> list[list[float]]:
        texts = [_require_text(document, index) for index, document in enumerate(documents)]
        return await self._run_bounded([(index, self._nova_vector(text)) for index, text in enumerate(texts)])

    async def _nova_vector(self, text: str) -> list[float]:
        payload = await self._invoke({
            'taskType': 'SINGLE_EMBEDDING',
            'singleEmbeddingParams': {
                'embeddingPurpose': 'GENERIC_INDEX',
                'text': {'truncationMode': 'END', 'value': text},
            },
        })
        return _require_vector(_nova_embedding(payload))

    async def _run_bounded(self, calls: list[tuple[int, Coroutine[Any, Any, _T]]]) -> list[_T]:
        """Runs the calls concurrently, cancelling the rest on the first failure.

        The semaphore is built per call rather than once per module: it binds
        to the running loop, and the Dev UI reflection server runs a second one.
        """
        semaphore = asyncio.Semaphore(EMBED_CONCURRENCY_LIMIT)

        async def _bounded(index: int, call: Coroutine[Any, Any, _T]) -> _T:
            try:
                async with semaphore:
                    try:
                        return await call
                    except GenkitError as error:
                        # A batch failure is opaque without the failing document,
                        # but the inner message already names the plugin.
                        raise GenkitError(
                            message=f'bedrock embed: document {index}: {_without_own_prefix(error.original_message)}',
                            status=error.status,
                        ) from error
            except asyncio.CancelledError:
                # Cancelled while queued on the semaphore, so nothing awaited it.
                call.close()
                raise

        tasks = [asyncio.create_task(_bounded(index, call)) for index, call in calls]
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            # Without this the rest of the batch still bills one call each.
            task.cancel()
        # Also retrieves the cancelled outcomes, so none resurfaces later as
        # "Task exception was never retrieved".
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            # Index order, not completion order, so the reported failure is
            # deterministic across the calls that were left to run.
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                raise result
        return cast(list[_T], results)

    async def _invoke(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._transport.invoke_model(
                modelId=self._model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json',
            )
        except ClientError as error:
            raise _from_client_error(error, 'invoke model') from error
        except BotoCoreError as error:
            raise _from_botocore_error(error, 'invoke model') from error
