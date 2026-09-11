# Copyright 2025 Google LLC
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

"""Google-Genai embedder model."""

import asyncio
import json
import sys
from collections.abc import Coroutine
from typing import Any, Protocol, TypeVar, cast

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel

from genkit import DocumentPart, Embedding, EmbedRequest, EmbedResponse, GenkitError
from genkit._core._typing import DocumentData, MediaPart, TextPart
from genkit.embedder import EmbedderInfo, EmbedderSupports
from genkit.plugin_api import to_json_schema
from genkit_google_genai.models._routing import strip_ref_prefixes
from genkit_google_genai.models._sdk_config import sdk_config_error
from genkit_google_genai.models.utils import PartConverter


class VertexEmbeddingModels(StrEnum):
    """Embedding models supported by Google-Genai vertex."""

    GECKO_003_ENG = 'textembedding-gecko@003'
    TEXT_EMBEDDING_004_ENG = 'text-embedding-004'
    TEXT_EMBEDDING_005_ENG = 'text-embedding-005'
    GECKO_MULTILINGUAL = 'textembedding-gecko-multilingual@001'
    TEXT_EMBEDDING_002_MULTILINGUAL = 'text-multilingual-embedding-002'
    MULTIMODAL_EMBEDDING_001 = 'multimodalembedding@001'
    GEMINI_EMBEDDING_001 = 'gemini-embedding-001'


class GeminiEmbeddingModels(StrEnum):
    """Embedding models supported by Google-Genai gemini."""

    GEMINI_EMBEDDING_2_PREVIEW = 'gemini-embedding-2-preview'
    GEMINI_EMBEDDING_2 = 'gemini-embedding-2'
    GEMINI_EMBEDDING_EXP_03_07 = 'gemini-embedding-exp-03-07'
    TEXT_EMBEDDING_004 = 'text-embedding-004'
    GEMINI_EMBEDDING_001 = 'gemini-embedding-001'


class EmbeddingTaskType(StrEnum):
    """Embedding task types supported by Google-Genai."""

    RETRIEVAL_QUERY = 'RETRIEVAL_QUERY'
    RETRIEVAL_DOCUMENT = 'RETRIEVAL_DOCUMENT'
    SEMANTIC_SIMILARITY = 'SEMANTIC_SIMILARITY'
    CLASSIFICATION = 'CLASSIFICATION'
    CLUSTERING = 'CLUSTERING'
    CODE_RETRIEVAL_QUERY = 'CODE_RETRIEVAL_QUERY'
    QUESTION_ANSWERING = 'QUESTION_ANSWERING'
    FACT_VERIFICATION = 'FACT_VERIFICATION'


class EmbeddingConfigSchema(BaseModel):
    """Options accepted by Google GenAI embedders on both backends.

    Keys may be written in snake_case or camelCase (``task_type`` or
    ``taskType``). Unknown keys are kept and ignored.
    """

    model_config = ConfigDict(extra='allow', populate_by_name=True, alias_generator=to_camel)

    task_type: EmbeddingTaskType | None = Field(
        default=None,
        description='Intended downstream use of the embedding; helps the model produce better embeddings.',
    )
    title: str | None = Field(
        default=None,
        description='Title of the text. Only used when task_type is RETRIEVAL_DOCUMENT.',
    )
    output_dimensionality: int | None = Field(
        default=None,
        ge=1,
        description='Number of dimensions in the returned embedding; trailing values are truncated.',
    )
    version: str | None = Field(
        default=None,
        description='API model id to call instead of the registered one.',
    )


class VertexEmbeddingConfigSchema(EmbeddingConfigSchema):
    """Options accepted by Vertex AI embedders.

    Extends the common options with the fields only the Vertex AI embedding
    endpoint accepts.
    """

    mime_type: str | None = Field(
        default=None,
        description='MIME type of the input.',
    )
    auto_truncate: bool | None = Field(
        default=None,
        description='Truncate inputs longer than the model maximum instead of failing.',
    )


# Per-request input limits of the embedding endpoints. The Gemini API's
# batchEmbedContents endpoint accepts up to 100 contents per call; the Vertex AI
# :predict endpoint accepts up to 250 input texts per call for the
# text-embedding models and a single input text for gemini-* models.
GOOGLEAI_EMBED_BATCH_SIZE = 100
VERTEXAI_EMBED_BATCH_SIZE = 250

# Caps simultaneous embedding calls; large batches otherwise trip rate limits or
# exhaust the client's connection pool.
EMBED_CONCURRENCY_LIMIT = 10


# Static dimensions for known embedders. Keys are version-suffix free
# (e.g. 'multimodalembedding', not 'multimodalembedding@001') because model
# discovery returns the bare name on some accounts/regions; lookups strip the
# '@version' suffix before matching (see get_embedder_info).
EMBEDDER_DIMENSIONS: dict[str, int] = {
    # Google AI
    'gemini-embedding-2-preview': 3072,
    'gemini-embedding-2': 3072,
    'gemini-embedding-001': 3072,
    'text-embedding-004': 768,
    # Vertex AI
    'text-embedding-005': 768,
    'text-multilingual-embedding-002': 768,
    'multimodalembedding': 1408,  # default; valid dims 128/256/512/1408 (not 768)
}


# Curated set of Vertex AI embedders that are verified to be callable.
# The Vertex catalog over-lists embedders (and returns supported_actions=None),
# so embedders are advertised from this list rather than discovered. Multimodal
# embedders route through the :predict endpoint (see Embedder._is_multimodal).
VERTEX_KNOWN_EMBEDDERS: tuple[str, ...] = (
    'text-embedding-005',
    'text-multilingual-embedding-002',
    'gemini-embedding-001',
    'multimodalembedding@001',
)

# Advertised input modalities, per backend. Unknown names default to text-only.
GOOGLEAI_EMBEDDER_INPUT_SUPPORTS: dict[str, list[str]] = {
    'gemini-embedding-2-preview': ['text', 'image', 'video'],
    'gemini-embedding-2': ['text', 'image', 'video'],
}

VERTEX_EMBEDDER_INPUT_SUPPORTS: dict[str, list[str]] = {
    'multimodalembedding': ['text', 'image', 'video'],
}


def _base_name(name: str) -> str:
    """Strip a trailing '@version' suffix from a model name (e.g. '@001')."""
    return name.split('@', 1)[0]


def _options_schema(is_vertex: bool) -> type[EmbeddingConfigSchema]:
    """Schema of the options accepted by a backend's embedders."""
    return VertexEmbeddingConfigSchema if is_vertex else EmbeddingConfigSchema


def get_embedder_info(name: str, label: str, is_vertex: bool = False) -> EmbedderInfo:
    """Return catalog info for a discovered embedder model.

    Args:
        name: The bare (unprefixed) model name, e.g. 'gemini-embedding-2'.
        label: Human-readable label for the embedder.
        is_vertex: True when resolving for the Vertex backend.

    Returns:
        EmbedderInfo describing the model's label, supported inputs,
        static dimensions and the JSON schema of its options.
    """
    base = _base_name(name)
    supports_map = VERTEX_EMBEDDER_INPUT_SUPPORTS if is_vertex else GOOGLEAI_EMBEDDER_INPUT_SUPPORTS
    supports = supports_map.get(name) or supports_map.get(base) or ['text']
    dimensions = EMBEDDER_DIMENSIONS.get(name) or EMBEDDER_DIMENSIONS.get(base)
    return EmbedderInfo(
        label=label,
        supports=EmbedderSupports(input=supports),
        dimensions=dimensions,
        config_schema=to_json_schema(_options_schema(is_vertex)),
    )


# One call's result: the embeddings decoded from a single API response.
_T = TypeVar('_T')


async def _run_bounded(calls: list[Coroutine[Any, Any, _T]]) -> list[_T]:
    """Run the calls concurrently, cancelling the rest on the first failure.

    The semaphore is built per call rather than once per module: it binds to the
    running loop, and the Dev UI reflection server runs a second one.

    Args:
        calls: Coroutines to run, in the order their results are wanted.

    Returns:
        The results, in the order the calls were given.

    Raises:
        BaseException: The failure of the earliest failing call in that order,
            so the reported error does not depend on completion order.
    """
    if not calls:
        return []
    semaphore = asyncio.Semaphore(EMBED_CONCURRENCY_LIMIT)

    async def _bounded(call: Coroutine[Any, Any, _T]) -> _T:
        try:
            async with semaphore:
                return await call
        except asyncio.CancelledError:
            # Cancelled while queued on the semaphore, so nothing awaited it.
            call.close()
            raise

    tasks = [asyncio.create_task(_bounded(call)) for call in calls]
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    for task in pending:
        # Without this the rest of the batch still bills one call each.
        task.cancel()
    # Also retrieves the cancelled outcomes, so none resurfaces later as
    # "Task exception was never retrieved".
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        # Call order, not completion order, so the reported failure is
        # deterministic across the calls that were left to run.
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            raise result
    return cast(list[_T], results)


class _HttpResponse(Protocol):
    """The google-genai HTTP response surface used to read a JSON body."""

    body: str | bytes | None


class _AsyncRequester(Protocol):
    """The google-genai transport surface used to POST to ``:predict``."""

    async def async_request(self, *, http_method: str, path: str, request_dict: dict[str, Any]) -> _HttpResponse:
        """Send an authenticated request and return the HTTP response."""
        ...


class Embedder:
    """Embedder for Google-Genai."""

    def __init__(
        self,
        version: VertexEmbeddingModels | GeminiEmbeddingModels | str,
        client: genai.Client,
        is_vertex: bool = False,
    ) -> None:
        """Initialize the embedder.

        Args:
            version: Embedding model version.
            client: Google-Genai client.
            is_vertex: Whether the client targets Vertex AI (as opposed to the
                Gemini Developer API). Multimodal embedding requires Vertex.
        """
        self._client = client
        self._version = version
        self._is_vertex = is_vertex

    async def generate(self, request: EmbedRequest) -> EmbedResponse:
        """Generate embeddings for a given request.

        Requests with more documents than the endpoint accepts per call are
        split into batches, sent with bounded concurrency; the response carries
        one embedding per document, in input order.

        Args:
            request: Genkit embed request.

        Returns:
            EmbedResponse

        Raises:
            GenkitError: INVALID_ARGUMENT when the request options fail
                validation; INTERNAL when the endpoint returns a different
                number of embeddings than documents were sent.
        """
        request = EmbedRequest.model_validate(request)
        if not request.input:
            raise ValueError(
                'Embed request input is empty: provide at least one document with content '
                '(for example input: [{"content": [{"text": "your text here"}]}]).'
            )
        options = self._parse_options(request.options)
        model = self._embed_model(options)
        if self._is_multimodal(model):
            return await self._generate_multimodal(request, model, options)
        contents = await self._build_contents(request)
        config = self._genkit_to_googleai_cfg(options)
        batch_size = self._embed_batch_size(model)

        batches = [contents[start : start + batch_size] for start in range(0, len(contents), batch_size)]
        results = await _run_bounded([self._embed_batch(model, batch, config) for batch in batches])

        embeddings: list[Embedding] = []
        for batch_embeddings in results:
            embeddings.extend(batch_embeddings)
        return EmbedResponse(embeddings=embeddings)

    async def _embed_batch(
        self,
        model: str,
        batch: list[genai.types.Content],
        config: genai.types.EmbedContentConfig | None,
    ) -> list[Embedding]:
        """Embed one batch of contents in a single ``embed_content`` call.

        Args:
            model: API model id.
            batch: Contents to send in this call.
            config: Google-genai embed config, or None.

        Returns:
            One embedding per content in the batch, in batch order.

        Raises:
            GenkitError: INTERNAL when the endpoint returns a different number
                of embeddings than documents were sent.
        """
        response = await self._client.aio.models.embed_content(
            model=model,
            contents=cast(genai_types.ContentListUnion, batch),
            config=config,
        )
        returned = response.embeddings or []
        if len(returned) != len(batch):
            raise GenkitError(
                status='INTERNAL',
                message=(
                    f'{model} returned {len(returned)} embeddings for {len(batch)} documents; '
                    'embeddings cannot be aligned with the input.'
                ),
            )
        return [Embedding(embedding=em.values or []) for em in returned]

    def _parse_options(self, options: object) -> EmbeddingConfigSchema:
        """Validate raw request options against this backend's option schema.

        Args:
            options: Request options as a dict, a schema instance, or None.

        Returns:
            The validated options. An instance of a different schema is
            re-validated so only the fields this backend accepts are typed.

        Raises:
            GenkitError: INVALID_ARGUMENT naming the field that failed validation.
        """
        schema = _options_schema(self._is_vertex)
        if options is None:
            return schema()
        if isinstance(options, EmbeddingConfigSchema) and type(options) is schema:
            return options
        if isinstance(options, BaseModel):
            options = options.model_dump(exclude_none=True)
        try:
            return schema.model_validate(options)
        except ValidationError as e:
            raise sdk_config_error(action_name=str(self._version), error=e) from e

    def _embed_model(self, options: EmbeddingConfigSchema) -> str:
        """API model id: options.version overlays the action's registered id."""
        if options.version:
            return strip_ref_prefixes(options.version)
        return str(self._version)

    def _embed_batch_size(self, model: str) -> int:
        """Maximum number of documents sent in one ``embed_content`` call.

        Vertex AI serves gemini-* and MaaS embedding models one input per
        request; other Vertex models accept 250 and the Gemini API 100.
        """
        if not self._is_vertex:
            return GOOGLEAI_EMBED_BATCH_SIZE
        lowered = model.lower()
        if 'gemini' in lowered or 'maas' in lowered:
            return 1
        return VERTEXAI_EMBED_BATCH_SIZE

    def _is_multimodal(self, model: str) -> bool:
        """Whether this embedder uses the Vertex multimodal ``:predict`` API.

        The google-genai ``embed_content`` API only accepts text on Vertex (it
        silently drops image/video parts), so multimodal embedders must call the
        ``predict`` endpoint with structured ``{text, image, video}`` instances
        instead.
        """
        return 'multimodalembedding' in model.lower()

    async def _generate_multimodal(
        self, request: EmbedRequest, model: str, options: EmbeddingConfigSchema
    ) -> EmbedResponse:
        """Embed text/image/video via the Vertex multimodal ``:predict`` endpoint.

        ``multimodalembedding@001`` accepts one instance per ``:predict`` call,
        so every document is sent as its own request, with bounded concurrency,
        and the resulting embeddings are concatenated in document order. All
        documents are validated before the first request is made.

        Args:
            request: Genkit embed request.
            model: API model id (action id, or options.version overlay).
            options: Validated embedding options.

        Returns:
            EmbedResponse
        """
        if not self._is_vertex:
            raise ValueError(
                f'{model} embedding is only available on Vertex AI; '
                'it is not supported by the Gemini Developer API. Use the VertexAI plugin instead.'
            )
        instances = [self._build_multimodal_instance(doc) for doc in request.input]

        # google-genai exposes no typed multimodal-embedding method, so reuse the
        # client's authenticated low-level transport to POST to :predict. For
        # Vertex, the project/location prefix is added by the SDK automatically.
        # These are private SDK internals, so guard against them drifting.
        api_client = getattr(self._client, '_api_client', None)
        if api_client is None or not hasattr(api_client, 'async_request'):
            raise RuntimeError(
                'Multimodal embedding relies on google-genai client internals that are '
                'unavailable in the installed google-genai version; install google-genai>=1.63.0.'
            )

        results = await _run_bounded([
            self._predict_multimodal(api_client, model, instance, options) for instance in instances
        ])

        embeddings: list[Embedding] = []
        for instance_embeddings in results:
            embeddings.extend(instance_embeddings)
        return EmbedResponse(embeddings=embeddings)

    async def _predict_multimodal(
        self,
        api_client: _AsyncRequester,
        model: str,
        instance: dict[str, Any],
        options: EmbeddingConfigSchema,
    ) -> list[Embedding]:
        """Send one multimodal instance to ``:predict`` and decode its embeddings.

        Args:
            api_client: The google-genai client's low-level transport.
            model: API model id.
            instance: A single ``{text, image, video}`` instance.
            options: Validated embedding options.

        Returns:
            The embeddings decoded from this instance's predictions.
        """
        payload: dict[str, Any] = {'instances': [instance]}
        if options.output_dimensionality is not None:
            payload['parameters'] = {'dimension': options.output_dimensionality}
        http_response = await api_client.async_request(
            http_method='post',
            path=f'publishers/google/models/{model}:predict',
            request_dict=payload,
        )
        body = json.loads(http_response.body) if http_response.body else {}
        predictions = body.get('predictions', []) if isinstance(body, dict) else []
        embeddings: list[Embedding] = []
        for prediction in predictions:
            embeddings.extend(self._prediction_to_embeddings(prediction))
        return embeddings

    def _build_multimodal_instance(self, doc: DocumentData) -> dict[str, Any]:
        """Build a Vertex multimodal embedding instance from a Genkit document.

        A Vertex instance accepts at most one text, one image and one video
        field (the three may be combined in a single instance). Multiple text
        parts are concatenated, matching ``Document.text``; multiple images or
        multiple videos raise, since the API would otherwise silently keep only
        the last of each.
        """
        if not isinstance(doc, DocumentData):
            doc = DocumentData.model_validate(doc)

        instance: dict[str, Any] = {}
        text_parts: list[str] = []
        for p in doc.content:
            part = p if isinstance(p, DocumentPart) else DocumentPart.model_validate(p)
            root = part.root
            if isinstance(root, TextPart):
                if root.text:
                    text_parts.append(root.text)
            elif isinstance(root, MediaPart):
                content_type = root.media.content_type or ''
                if content_type.startswith('image/'):
                    if 'image' in instance:
                        raise ValueError('Multimodal embed document cannot contain more than one image.')
                    instance['image'] = self._media_reference(root.media.url, content_type)
                elif content_type.startswith('video/'):
                    if 'video' in instance:
                        raise ValueError('Multimodal embed document cannot contain more than one video.')
                    video = self._media_reference(root.media.url, content_type, include_mime_type=False)
                    segment_config = (doc.metadata or {}).get('video_segment_config') or (doc.metadata or {}).get(
                        'videoSegmentConfig'
                    )
                    if segment_config:
                        video['videoSegmentConfig'] = segment_config
                    instance['video'] = video
                else:
                    raise ValueError(f'Unsupported contentType for multimodal embedding: {content_type!r}')

        if text_parts:
            instance['text'] = ''.join(text_parts)

        if not instance:
            raise ValueError('Multimodal embed document has no text, image, or video content.')
        return instance

    @staticmethod
    def _media_reference(url: str, content_type: str, include_mime_type: bool = True) -> dict[str, Any]:
        """Map a media URL to a Vertex image/video reference (gcsUri or base64).

        http(s) URLs raise instead of being forwarded as a ``gcsUri``: Vertex
        only accepts ``gs://`` URIs there, so passing an http(s) URL produces an
        opaque API error. Failing fast is clearer.
        """
        if url.startswith('gs://'):
            ref: dict[str, Any] = {'gcsUri': url}
        elif url.startswith('http'):
            raise ValueError(
                'Vertex multimodal embedding does not accept http(s) media URLs. '
                'Upload the file to Cloud Storage and pass a gs:// URI, or inline it as a data: URL.'
            )
        elif url.startswith('data:'):
            marker = ';base64,'
            marker_index = url.find(marker)
            if marker_index == -1:
                raise ValueError(
                    'Vertex multimodal embedding requires base64-encoded data: URLs (data:<mimeType>;base64,<data>).'
                )
            ref = {'bytesBase64Encoded': url[marker_index + len(marker) :]}
        else:
            ref = {'bytesBase64Encoded': url}
        if include_mime_type and content_type:
            ref['mimeType'] = content_type
        return ref

    @staticmethod
    def _prediction_to_embeddings(prediction: dict[str, Any]) -> list[Embedding]:
        """Convert one multimodal prediction into Genkit embeddings.

        A prediction can carry image, text and/or video embeddings, so one
        document may fan out to several embeddings (a text+image document yields
        two; a video yields one embedding per chunk). Embeddings are told apart
        by their ``embedType`` metadata rather than by position, so consumers
        must correlate via metadata instead of zipping positionally against the
        input documents. Video chunk offsets are preserved in each embedding's
        metadata.
        """
        embeddings: list[Embedding] = []
        if prediction.get('imageEmbedding'):
            embeddings.append(
                Embedding(embedding=prediction['imageEmbedding'], metadata={'embedType': 'imageEmbedding'})
            )
        if prediction.get('textEmbedding'):
            embeddings.append(Embedding(embedding=prediction['textEmbedding'], metadata={'embedType': 'textEmbedding'}))
        for video_embedding in prediction.get('videoEmbeddings', []) or []:
            values = video_embedding.get('embedding')
            if values:
                metadata = {k: v for k, v in video_embedding.items() if k != 'embedding'}
                metadata['embedType'] = 'videoEmbedding'
                embeddings.append(Embedding(embedding=values, metadata=metadata))
        return embeddings

    async def _build_contents(self, request: EmbedRequest) -> list[genai.types.Content]:
        """Build google-genai request contents from Genkit request.

        Args:
            request: Genkit request.

        Returns:
            list of google-genai contents.
        """
        request_contents: list[genai.types.Content] = []
        for doc in request.input:
            if not isinstance(doc, DocumentData):
                doc = DocumentData.model_validate(doc)
            content_parts: list[genai.types.Part] = []
            for p in doc.content:
                part = p if isinstance(p, DocumentPart) else DocumentPart.model_validate(p)
                converted = await PartConverter.to_gemini(part)
                if isinstance(converted, list):
                    content_parts.extend(converted)
                else:
                    content_parts.append(converted)
            request_contents.append(genai.types.Content(parts=content_parts))

        return request_contents

    def _genkit_to_googleai_cfg(self, options: EmbeddingConfigSchema) -> genai.types.EmbedContentConfig | None:
        """Translate embedding options into a google-genai EmbedContentConfig.

        ``mime_type`` and ``auto_truncate`` are forwarded only from a
        VertexEmbeddingConfigSchema; the Gemini API rejects them.

        Args:
            options: Validated embedding options.

        Returns:
            Google-genai embed config, or None when no config field is set.
        """
        fields: dict[str, Any] = {}
        if options.task_type is not None:
            fields['task_type'] = options.task_type.value
        if options.title is not None:
            fields['title'] = options.title
        if options.output_dimensionality is not None:
            fields['output_dimensionality'] = options.output_dimensionality
        if isinstance(options, VertexEmbeddingConfigSchema):
            if options.mime_type is not None:
                fields['mime_type'] = options.mime_type
            if options.auto_truncate is not None:
                fields['auto_truncate'] = options.auto_truncate
        if not fields:
            return None
        return genai.types.EmbedContentConfig(**fields)
