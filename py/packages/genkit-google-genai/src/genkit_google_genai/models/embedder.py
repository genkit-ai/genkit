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

import json
import sys
from typing import Any, cast

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from google import genai
from google.genai import types as genai_types

from genkit import DocumentPart, Embedding, EmbedRequest, EmbedResponse
from genkit._core._typing import DocumentData, MediaPart, TextPart
from genkit.embedder import EmbedderOptions, EmbedderSupports
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
    QUESTION_ANSWERING = 'QUESTION_ANSWERING'
    FACT_VERIFICATION = 'FACT_VERIFICATION'


# Static dimensions for known embedders. Keys are version-suffix free
# (e.g. 'multimodalembedding', not 'multimodalembedding@001') because model
# discovery returns the bare name on some accounts/regions; lookups strip the
# '@version' suffix before matching (see get_embedder_options).
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


def get_embedder_options(name: str, label: str, is_vertex: bool = False) -> EmbedderOptions:
    """Return EmbedderOptions metadata for a discovered embedder model.

    Args:
        name: The bare (unprefixed) model name, e.g. 'gemini-embedding-2'.
        label: Human-readable label for the embedder.
        is_vertex: True when resolving for the Vertex backend.

    Returns:
        EmbedderOptions describing the model's label, supported inputs and
        static dimensions.
    """
    base = _base_name(name)
    supports_map = VERTEX_EMBEDDER_INPUT_SUPPORTS if is_vertex else GOOGLEAI_EMBEDDER_INPUT_SUPPORTS
    supports = supports_map.get(name) or supports_map.get(base) or ['text']
    dimensions = EMBEDDER_DIMENSIONS.get(name) or EMBEDDER_DIMENSIONS.get(base)
    return EmbedderOptions(
        label=label,
        supports=EmbedderSupports(input=supports),
        dimensions=dimensions,
    )


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

        Args:
            request: Genkit embed request.

        Returns:
            EmbedResponse
        """
        request = EmbedRequest.model_validate(request)
        if not request.input:
            raise ValueError(
                'Embed request input is empty: provide at least one document with content '
                '(for example input: [{"content": [{"text": "your text here"}]}]).'
            )
        if self._is_multimodal():
            return await self._generate_multimodal(request)
        contents = await self._build_contents(request)
        config = self._genkit_to_googleai_cfg(request)
        response = await self._client.aio.models.embed_content(
            model=self._version,
            contents=cast(genai_types.ContentListUnion, contents),
            config=config,
        )

        embeddings = [Embedding(embedding=em.values or []) for em in (response.embeddings or [])]
        return EmbedResponse(embeddings=embeddings)

    def _is_multimodal(self) -> bool:
        """Whether this embedder uses the Vertex multimodal ``:predict`` API.

        The google-genai ``embed_content`` API only accepts text on Vertex (it
        silently drops image/video parts), so multimodal embedders must call the
        ``predict`` endpoint with structured ``{text, image, video}`` instances
        instead. This mirrors the JS plugin's vertexai embedder.
        """
        return 'multimodalembedding' in str(self._version).lower()

    async def _generate_multimodal(self, request: EmbedRequest) -> EmbedResponse:
        """Embed text/image/video via the Vertex multimodal ``:predict`` endpoint.

        ``multimodalembedding@001`` accepts only one instance per ``:predict``
        call, so multi-document requests (e.g. ``embed_many``) are rejected
        rather than sent as an invalid multi-instance payload. Batching multiple
        documents is not supported yet.

        Args:
            request: Genkit embed request.

        Returns:
            EmbedResponse
        """
        if not self._is_vertex:
            raise ValueError(
                f'{self._version} embedding is only available on Vertex AI; '
                'it is not supported by the Gemini Developer API. Use the VertexAI plugin instead.'
            )
        if len(request.input) > 1:
            raise ValueError(
                'multimodalembedding@001 supports only one document per request; embed documents one at a time.'
            )
        instances = [self._build_multimodal_instance(doc) for doc in request.input]

        payload: dict[str, Any] = {'instances': instances}
        if request.options:
            dimension = request.options.get('output_dimensionality')
            if dimension is not None:
                payload['parameters'] = {'dimension': dimension}

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
        http_response = await api_client.async_request(
            http_method='post',
            path=f'publishers/google/models/{self._version}:predict',
            request_dict=payload,
        )
        body = json.loads(http_response.body) if http_response.body else {}
        predictions = body.get('predictions', []) if isinstance(body, dict) else []

        embeddings: list[Embedding] = []
        for prediction in predictions:
            embeddings.extend(self._prediction_to_embeddings(prediction))
        return EmbedResponse(embeddings=embeddings)

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

        Unlike the JS plugin, http(s) URLs raise instead of being forwarded as a
        ``gcsUri``: Vertex only accepts ``gs://`` URIs there, so passing an
        http(s) URL produces an opaque API error. Failing fast is clearer.
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

    def _genkit_to_googleai_cfg(self, request: EmbedRequest) -> genai.types.EmbedContentConfig | None:
        """Translate EmbedRequest options to Google Ai GenerateContentConfig.

        Args:
            request: Genkit embed request.

        Returns:
            Google Ai embed config or None.
        """
        cfg = None
        if request.options:
            cfg = genai.types.EmbedContentConfig(
                task_type=request.options.get('task_type'),
                title=request.options.get('title'),
                output_dimensionality=request.options.get('output_dimensionality'),
            )

        return cfg
