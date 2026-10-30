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
from collections.abc import Coroutine
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, cast

from genkit_google_genai._auth import GOOGLE_AUTH_ERRORS, raise_auth_error
from genkit_google_genai._models._routing import strip_ref_prefixes
from genkit_google_genai._models._sdk_config import sdk_config_error
from genkit_google_genai._models._utils import PartConverter
from genkit_google_genai._provider_errors import TRANSPORT_ERRORS, api_error, transport_error
from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.alias_generators import to_camel

from genkit import Document, Embedding, GenkitError, Part, get_logger
from genkit._core._schema import to_json_schema
from genkit.embedder import EmbedderInfo, EmbedderSupports, EmbedRequest, EmbedResponse

logger = get_logger(__name__)

# Known task_type values, offered for autocomplete. Any other string is
# forwarded as given, so a task type newer than this list still works.
EmbeddingTaskType: TypeAlias = Literal[
    'RETRIEVAL_QUERY',
    'RETRIEVAL_DOCUMENT',
    'SEMANTIC_SIMILARITY',
    'CLASSIFICATION',
    'CLUSTERING',
    'CODE_RETRIEVAL_QUERY',
    'QUESTION_ANSWERING',
    'FACT_VERIFICATION',
]


class EmbeddingConfig(BaseModel):
    """Settings an embedder accepts, on both backends.

    Pass an instance or a dict of these fields to ``embedding()``; pass a dict
    of them to ``ai.embed(config=...)``. Keys may be written in snake_case or
    camelCase (``task_type`` or ``taskType``); when both spellings of one
    field are given, the camelCase one wins. Unknown keys raise.
    """

    model_config = ConfigDict(extra='forbid', populate_by_name=True, alias_generator=to_camel)

    task_type: EmbeddingTaskType | str | None = Field(
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
    mime_type: str | None = Field(
        default=None,
        description='MIME type of the input. Vertex AI only.',
    )
    auto_truncate: bool | None = Field(
        default=None,
        description='Truncate inputs longer than the model maximum instead of failing. Vertex AI only.',
    )

    @model_validator(mode='before')
    @classmethod
    def _camel_case_key_wins(cls, data: object) -> object:
        """Drop the snake_case key of a field whose camelCase alias is also given.

        A ref config is stored in snake_case and a per-call config is merged
        over it as typed, so a field present in both spellings carries the
        per-call value under the camelCase key.
        """
        if not isinstance(data, dict):
            return data
        raw = cast(dict[str, object], data)
        shadowed = [
            name
            for name, field in cls.model_fields.items()
            if field.alias and field.alias != name and field.alias in raw and name in raw
        ]
        if not shadowed:
            return data
        return {key: value for key, value in raw.items() if key not in shadowed}


# Per-request input limits of the embedding endpoints. The Gemini API's
# batchEmbedContents endpoint accepts up to 100 contents per call; the Vertex AI
# :predict endpoint accepts up to 250 input texts per call for the
# text-embedding models and a single input text for gemini-* models.
GOOGLEAI_EMBED_BATCH_SIZE = 100
VERTEX_EMBED_BATCH_SIZE = 250

# Calls in flight per embed request. Quotas are per-minute request counts, so
# this bounds the burst and the share of the SDK's connection pool one request
# takes, not the quota itself.
EMBED_CONCURRENCY_LIMIT = 10

# Fields only the Vertex AI embedding endpoint accepts: field name and wire alias.
VERTEX_ONLY_OPTIONS: tuple[tuple[str, str], ...] = (('mime_type', 'mimeType'), ('auto_truncate', 'autoTruncate'))


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


def _config_schema(is_vertex: bool) -> dict[str, Any]:
    """JSON schema of ``EmbeddingConfig`` as this backend accepts it."""
    schema = dict(to_json_schema(EmbeddingConfig))
    if not is_vertex:
        properties = dict(schema.get('properties') or {})
        for _name, alias in VERTEX_ONLY_OPTIONS:
            properties.pop(alias, None)
        schema['properties'] = properties
    return schema


def get_embedder_info(name: str, label: str, is_vertex: bool = False) -> EmbedderInfo:
    """Return catalog info for a discovered embedder model.

    Args:
        name: The bare (unprefixed) model name, e.g. 'gemini-embedding-2'.
        label: Human-readable label for the embedder.
        is_vertex: True when resolving for the Vertex backend.

    Returns:
        EmbedderInfo describing the model's label, supported inputs,
        static dimensions and the JSON schema of its config.
    """
    base = _base_name(name)
    supports_map = VERTEX_EMBEDDER_INPUT_SUPPORTS if is_vertex else GOOGLEAI_EMBEDDER_INPUT_SUPPORTS
    supports = supports_map.get(name) or supports_map.get(base) or ['text']
    dimensions = EMBEDDER_DIMENSIONS.get(name) or EMBEDDER_DIMENSIONS.get(base)
    return EmbedderInfo(
        label=label,
        supports=EmbedderSupports(input=supports),
        dimensions=dimensions,
        config_schema=_config_schema(is_vertex),
    )


# Result type of one bounded call.
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
        BaseException: The failure of the earliest call in that order among the
            calls that completed on their own. A call still pending when another
            one fails is cancelled, so its own outcome is never reported. A
            ``CancelledError`` raised by a call itself does not stop the others
            and is re-raised once they finish.
    """
    if not calls:
        return []
    semaphore = asyncio.Semaphore(EMBED_CONCURRENCY_LIMIT)

    async def _bounded(call: Coroutine[Any, Any, _T]) -> _T:
        try:
            async with semaphore:
                return await call
        except asyncio.CancelledError:
            # Cancelled before the call started: close the coroutine so it is
            # never awaited. Once the call has started this is a no-op.
            call.close()
            raise

    tasks = [asyncio.create_task(_bounded(call)) for call in calls]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except BaseException:
        # asyncio.wait leaves the tasks running when whoever awaits it is
        # cancelled, so without this they keep calling the API unobserved.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    # Without this the rest of the batch still bills one call each; at most
    # one queued call can start before the cancel lands. Call order keeps the
    # semaphore's waiter list cheap to unwind.
    for task in tasks:
        task.cancel()
    # Also retrieves every other failed outcome, so none resurfaces later as
    # "Task exception was never retrieved".
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for task, result in zip(tasks, results, strict=False):
        # Call order, not completion order, so the reported failure is
        # deterministic across the calls that were left to run. Only calls that
        # finished on their own count: a call cancelled here may turn the
        # cancellation into another error while cleaning up.
        if task in done and isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            raise result
    for result in results:
        # Nothing failed, so nothing was cancelled here: this one came out of
        # the call itself and must not be returned as a result.
        if isinstance(result, asyncio.CancelledError):
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
        version: str,
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
        split into batches, with at most ``EMBED_CONCURRENCY_LIMIT`` (10) calls
        in flight at once; the response carries one embedding per document, in
        input order. A multimodal document can yield several embeddings, told
        apart by their ``embedType`` metadata.

        Args:
            request: Genkit embed request.

        Returns:
            EmbedResponse

        Raises:
            GenkitError: INVALID_ARGUMENT when the input is empty or the config
                fails validation; INTERNAL when the endpoint returns a different
                number of embeddings than documents were sent.
        """
        request = EmbedRequest.model_validate(request)
        if not request.input:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message='Embed request input is empty: provide at least one document with content '
                '(for example input: [{"content": [{"text": "your text here"}]}]).',
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
        try:
            response = await self._client.aio.models.embed_content(
                model=model,
                contents=cast(genai_types.ContentListUnion, batch),
                config=config,
            )
        except APIError as e:
            raise api_error(e) from e
        except GOOGLE_AUTH_ERRORS as e:
            raise_auth_error(e)
        except TRANSPORT_ERRORS as e:
            raise transport_error(e) from e
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

    def _parse_options(self, options: object) -> EmbeddingConfig:
        """Validate raw request options into an ``EmbeddingConfig``.

        Args:
            options: Request options as a dict, an ``EmbeddingConfig`` or another
                model with these fields, or None.

        Returns:
            The validated config.

        Raises:
            GenkitError: INVALID_ARGUMENT naming the key that failed validation.
        """
        if options is None:
            return EmbeddingConfig()
        if isinstance(options, EmbeddingConfig):
            parsed = options
        else:
            if isinstance(options, BaseModel):
                options = options.model_dump(exclude_none=True)
            try:
                parsed = EmbeddingConfig.model_validate(options)
            except ValidationError as e:
                raise sdk_config_error(action_name=str(self._version), error=e) from e
        self._warn_vertex_only_options(parsed)
        return parsed

    def _warn_vertex_only_options(self, options: EmbeddingConfig) -> None:
        """Log the Vertex-only settings a Gemini Developer API embedder drops.

        Args:
            options: Validated embedding config.
        """
        if self._is_vertex:
            return
        dropped = [alias for name, alias in VERTEX_ONLY_OPTIONS if getattr(options, name) is not None]
        if dropped:
            logger.warning(
                'Dropping %s from the %s embed config: Vertex AI only, not accepted by the Gemini Developer API.',
                ', '.join(dropped),
                self._version,
            )

    def _embed_model(self, options: EmbeddingConfig) -> str:
        """API model id: options.version overlays the action's registered id."""
        if options.version:
            return strip_ref_prefixes(options.version)
        return str(self._version)

    def _embed_batch_size(self, model: str) -> int:
        """Maximum number of documents sent in one ``embed_content`` call.

        Vertex AI serves gemini-* embedding models one input per request;
        other Vertex models accept 250 and the Gemini API 100.
        """
        if not self._is_vertex:
            return GOOGLEAI_EMBED_BATCH_SIZE
        if 'gemini' in model.lower():
            return 1
        return VERTEX_EMBED_BATCH_SIZE

    def _is_multimodal(self, model: str) -> bool:
        """Whether this embedder uses the Vertex multimodal ``:predict`` API.

        The google-genai ``embed_content`` API only accepts text on Vertex (it
        silently drops image/video parts), so multimodal embedders must call the
        ``predict`` endpoint with structured ``{text, image, video}`` instances
        instead.
        """
        return 'multimodalembedding' in model.lower()

    async def _generate_multimodal(self, request: EmbedRequest, model: str, options: EmbeddingConfig) -> EmbedResponse:
        """Embed text/image/video via the Vertex multimodal ``:predict`` endpoint.

        ``multimodalembedding@001`` accepts one instance per ``:predict`` call,
        so every document is sent as its own request, at most
        ``EMBED_CONCURRENCY_LIMIT`` (10) in flight at once, and the resulting
        embeddings are concatenated in document order. All documents are
        validated before the first request is made. Of the config, only
        ``output_dimensionality`` applies to ``:predict``.

        Args:
            request: Genkit embed request.
            model: API model id (action id, or options.version overlay).
            options: Validated embedding config.

        Returns:
            EmbedResponse
        """
        if not self._is_vertex:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'{model} embedding is only available on Vertex AI; '
                'it is not supported by the Gemini Developer API. Use the VertexAI plugin instead.',
            )
        instances = [self._build_multimodal_instance(doc) for doc in request.input]

        # google-genai exposes no typed multimodal-embedding method, so reuse the
        # client's authenticated low-level transport to POST to :predict. For
        # Vertex, the project/location prefix is added by the SDK automatically.
        # These are private SDK internals, so guard against them drifting. An
        # outdated install is a local setup problem that retrying won't fix.
        api_client = getattr(self._client, '_api_client', None)
        if api_client is None or not hasattr(api_client, 'async_request'):
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message='Multimodal embedding relies on google-genai client internals that are '
                'unavailable in the installed google-genai version; install google-genai>=1.63.0.',
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
        options: EmbeddingConfig,
    ) -> list[Embedding]:
        """Send one multimodal instance to ``:predict`` and decode its embeddings.

        Args:
            api_client: The google-genai client's low-level transport.
            model: API model id.
            instance: A single ``{text, image, video}`` instance.
            options: Validated embedding config.

        Returns:
            The embeddings decoded from this instance's predictions.
        """
        payload: dict[str, Any] = {'instances': [instance]}
        if options.output_dimensionality is not None:
            payload['parameters'] = {'dimension': options.output_dimensionality}
        try:
            http_response = await api_client.async_request(
                http_method='post',
                path=f'publishers/google/models/{model}:predict',
                request_dict=payload,
            )
        except APIError as e:
            raise api_error(e) from e
        except GOOGLE_AUTH_ERRORS as e:
            raise_auth_error(e)
        except TRANSPORT_ERRORS as e:
            raise transport_error(e) from e
        try:
            body = json.loads(http_response.body) if http_response.body else {}
        except json.JSONDecodeError as e:
            raise GenkitError(
                status='INTERNAL',
                message='Vertex multimodal embedding returned a body that is not JSON',
                cause=e,
            ) from e
        predictions = body.get('predictions', []) if isinstance(body, dict) else []
        embeddings: list[Embedding] = []
        for prediction in predictions:
            embeddings.extend(self._prediction_to_embeddings(prediction))
        return embeddings

    def _build_multimodal_instance(self, doc: Document) -> dict[str, Any]:
        """Build a Vertex multimodal embedding instance from a Genkit document.

        A Vertex instance accepts at most one text, one image and one video
        field (the three may be combined in a single instance). Multiple text
        parts are concatenated, matching ``Document.text``; multiple images or
        multiple videos raise, since the API would otherwise silently keep only
        the last of each.
        """
        instance: dict[str, Any] = {}
        text_parts: list[str] = []
        for p in doc.content:
            part = p if isinstance(p, Part) else Part.model_validate(p)
            if part.text is not None:
                if part.text:
                    text_parts.append(part.text)
            elif part.media is not None:
                content_type = part.media.content_type or ''
                if content_type.startswith('image/'):
                    if 'image' in instance:
                        raise GenkitError(
                            status='INVALID_ARGUMENT',
                            message='Multimodal embed document cannot contain more than one image.',
                        )
                    instance['image'] = self._media_reference(part.media.url, content_type)
                elif content_type.startswith('video/'):
                    if 'video' in instance:
                        raise GenkitError(
                            status='INVALID_ARGUMENT',
                            message='Multimodal embed document cannot contain more than one video.',
                        )
                    video = self._media_reference(part.media.url, content_type, include_mime_type=False)
                    segment_config = (doc.metadata or {}).get('video_segment_config') or (doc.metadata or {}).get(
                        'videoSegmentConfig'
                    )
                    if segment_config:
                        video['videoSegmentConfig'] = segment_config
                    instance['video'] = video
                else:
                    raise GenkitError(
                        status='INVALID_ARGUMENT',
                        message=f'Unsupported contentType for multimodal embedding: {content_type!r}',
                    )

        if text_parts:
            instance['text'] = ''.join(text_parts)

        if not instance:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message='Multimodal embed document has no text, image, or video content.',
            )
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
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message='Vertex multimodal embedding does not accept http(s) media URLs. '
                'Upload the file to Cloud Storage and pass a gs:// URI, or inline it as a data: URL.',
            )
        elif url.startswith('data:'):
            marker = ';base64,'
            marker_index = url.find(marker)
            if marker_index == -1:
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message='Vertex multimodal embedding requires base64-encoded data: URLs '
                    '(data:<mimeType>;base64,<data>).',
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
            content_parts: list[genai.types.Part] = []
            for p in doc.content:
                part = p if isinstance(p, Part) else Part.model_validate(p)
                converted = await PartConverter.to_gemini(part)
                if isinstance(converted, list):
                    content_parts.extend(converted)
                else:
                    content_parts.append(converted)
            request_contents.append(genai.types.Content(parts=content_parts))

        return request_contents

    def _genkit_to_googleai_cfg(self, options: EmbeddingConfig) -> genai.types.EmbedContentConfig | None:
        """Translate the embedding config into a google-genai EmbedContentConfig.

        ``mime_type`` and ``auto_truncate`` are forwarded on the Vertex AI
        backend only; the Gemini Developer API does not accept them.

        Args:
            options: Validated embedding config.

        Returns:
            Google-genai embed config, or None when no field is set.
        """
        fields: dict[str, Any] = {}
        if options.task_type is not None:
            fields['task_type'] = options.task_type
        if options.title is not None:
            fields['title'] = options.title
        if options.output_dimensionality is not None:
            fields['output_dimensionality'] = options.output_dimensionality
        if self._is_vertex:
            if options.mime_type is not None:
                fields['mime_type'] = options.mime_type
            if options.auto_truncate is not None:
                fields['auto_truncate'] = options.auto_truncate
        if not fields:
            return None
        return genai.types.EmbedContentConfig(**fields)
