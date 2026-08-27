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

"""Bedrock reranking implementation (InvokeModel API).

Reranking is a helper, ``Bedrock.rerank()``, not a registered action: Genkit
Python carries only the bare ``ActionKind.RERANKER`` enum member, and the
reranker schema types sit on the codegen denylist (``py/scripts/schema_to_typing.py``),
so there is no primitive to register against. The four types below mirror
``genkit-tools/genkit-schema.json`` field for field, so call sites survive if
core ever re-adds the primitive.

A malformed result entry is an error rather than a zero-filled document, so a
result with no ``index`` never silently scores the first document. The
``amazon.rerank-*`` family is supported by omitting ``api_version``: the Amazon
schema rejects the key (``extraneous key [api_version] is not permitted``) as
firmly as the Cohere schema requires it.

Two behaviours worth stating, because both are visible to callers: results
come back in the service's order, never re-sorted and never truncated
client-side (``top_n`` only caps what the service returns), and a ranked
document carries its score alone, dropping the original document's metadata.
"""

import json
from collections.abc import Mapping
from typing import Any, Literal

import structlog
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from pydantic.alias_generators import to_camel

from genkit import DocumentPart

# DocumentData has no public re-export yet; the reranker types are built on it.
from genkit._core._typing import DocumentData
from genkit.plugin_api import GenkitError
from genkit_amazon_bedrock.embedders import InvokeModelTransport, document_text
from genkit_amazon_bedrock.model_info import strip_inference_profile_prefix
from genkit_amazon_bedrock.models import _from_botocore_error, _from_client_error

logger = structlog.get_logger(__name__)

# The current Bedrock contract for the Cohere Rerank InvokeModel body.
COHERE_RERANK_API_VERSION = 2

RerankFamily = Literal['cohere', 'amazon']

# Substring matches over the base model ID.
_RERANK_PATTERNS: tuple[tuple[str, RerankFamily], ...] = (
    ('cohere.rerank', 'cohere'),
    ('amazon.rerank', 'amazon'),
)


class RankedDocumentMetadata(BaseModel):
    """Metadata carried by a reranked document.

    Mirrors the genkit-schema type of the same name.
    """

    # The only reranker type the schema opens up; the other three forbid extras.
    model_config = ConfigDict(alias_generator=to_camel, extra='allow', populate_by_name=True)

    score: float
    """The model's relevance score for this document against the query."""


class RankedDocumentData(BaseModel):
    """A document with its relevance score attached.

    Mirrors the genkit-schema type of the same name.
    """

    model_config = ConfigDict(alias_generator=to_camel, extra='forbid', populate_by_name=True)

    content: list[DocumentPart]
    """The ranked document's parts, taken verbatim from the input document."""

    metadata: RankedDocumentMetadata
    """The score. The input document's own metadata is deliberately not carried."""


class RerankerRequest(BaseModel):
    """A rerank call: a query, the documents to rank, and driver options.

    Mirrors the genkit-schema type of the same name.
    """

    model_config = ConfigDict(alias_generator=to_camel, extra='forbid', populate_by_name=True)

    query: DocumentData
    """The query to rank the documents against."""

    documents: list[DocumentData]
    """The documents to rank."""

    options: Any | None = None
    """Driver-specific options; this plugin reads ``BedrockRerankOptions``."""


class RerankerResponse(BaseModel):
    """The ranked documents returned by a rerank call.

    Mirrors the genkit-schema type of the same name.
    """

    model_config = ConfigDict(alias_generator=to_camel, extra='forbid', populate_by_name=True)

    documents: list[RankedDocumentData]
    """The ranked documents, in the order the service returned them."""


class BedrockRerankOptions(BaseModel):
    """Per-call options for a Bedrock rerank call."""

    # Unknown keys are ignored rather than forbidden.
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    top_n: int | None = None
    """Caps how many ranked documents the service returns. Unset, ``<= 0``, or
    a value above the document count all mean every input document."""

    @field_validator('top_n', mode='before')
    @classmethod
    def _reject_bool_top_n(cls, value: Any) -> Any:  # noqa: ANN401
        # bool is an int subclass, and True would silently rank down to one document.
        if isinstance(value, bool):
            raise ValueError('top_n must be an integer, not a bool')
        return value


def _rerank_family(model_id: str) -> RerankFamily | None:
    """Routes a model ID to its rerank family, or None if it is not one."""
    base_id = strip_inference_profile_prefix(model_id)
    for pattern, family in _RERANK_PATTERNS:
        if pattern in base_id:
            return family
    return None


def is_rerank_model(model_id: str) -> bool:
    """Reports whether a Bedrock model ID names a reranking model.

    True for both the Cohere and Amazon families: neither has a Converse path,
    so neither may be resolved as a chat model.

    Args:
        model_id: Bedrock model ID, inference-profile ID, or ARN.

    Returns:
        True when the ID routes to a known rerank family.
    """
    return _rerank_family(model_id) is not None


def coerce_rerank_options(options: Any) -> BedrockRerankOptions | None:  # noqa: ANN401
    """Coerces a request's raw options into ``BedrockRerankOptions``.

    Accepts the typed value or a mapping, keyed by either ``topN`` or
    ``top_n``. Unknown keys are ignored.

    Args:
        options: The raw ``RerankerRequest.options`` value.

    Returns:
        The coerced options, or None when none were given.

    Raises:
        GenkitError: INVALID_ARGUMENT for an unsupported type or an
            unparseable mapping.
    """
    if options is None:
        return None
    if isinstance(options, BedrockRerankOptions):
        return options
    if isinstance(options, Mapping):
        try:
            return BedrockRerankOptions.model_validate(dict(options))
        except ValidationError as error:
            raise GenkitError(
                message=f'bedrock rerank: invalid rerank options: {error}',
                status='INVALID_ARGUMENT',
            ) from error
    raise GenkitError(
        message=f'bedrock rerank: unsupported rerank options type {type(options).__name__}',
        status='INVALID_ARGUMENT',
    )


def build_rerank_body(query: str, documents: list[str], top_n: int, *, include_api_version: bool) -> dict[str, Any]:
    """Builds the InvokeModel body for a Bedrock reranking model.

    ``query``, ``documents`` and ``top_n`` are always present and identical for
    both families. ``api_version`` is the only difference, and it is added
    rather than set to None: the Cohere schema rejects a body without the key,
    and the Amazon schema rejects any body carrying it.

    Args:
        query: The query text.
        documents: The document texts, in input order.
        top_n: How many ranked documents to ask for.
        include_api_version: Whether to send ``api_version``; the Cohere family
            requires it, the Amazon family refuses it.

    Returns:
        The request body to marshal.
    """
    body: dict[str, Any] = {
        'query': query,
        'documents': documents,
        'top_n': top_n,
    }
    if include_api_version:
        body['api_version'] = COHERE_RERANK_API_VERSION
    return body


def _result_fields(result: Any, position: int) -> tuple[int, float]:  # noqa: ANN401
    """Reads the index and score off one rerank result."""
    if not isinstance(result, dict):
        raise GenkitError(message=f'bedrock rerank: result {position} is not an object', status='INTERNAL')
    index = result.get('index')
    # bool is an int subclass, and True would silently rank document 1.
    if isinstance(index, bool) or not isinstance(index, int):
        raise GenkitError(message=f'bedrock rerank: result {position} has no integer index', status='INTERNAL')
    score = result.get('relevance_score')
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise GenkitError(
            message=f'bedrock rerank: result {position} has no numeric relevance_score',
            status='INTERNAL',
        )
    return index, float(score)


def build_rerank_response(payload: dict[str, Any], documents: list[DocumentData]) -> RerankerResponse:
    """Maps the scored results back onto the original documents.

    Both families answer with the same ``results`` shape, so one mapping serves
    them. The service's order is preserved verbatim: results are never
    re-sorted and never truncated, so a caller that asked for a smaller
    ``top_n`` than the service honoured still sees everything it sent back.

    Args:
        payload: The parsed InvokeModel response body.
        documents: The documents the request was built from, in input order.

    Returns:
        One ranked document per result, each carrying its relevance score.

    Raises:
        GenkitError: INTERNAL when the results are malformed or reference a
            document that was never sent.
    """
    results = payload.get('results')
    # An absent results field is a valid empty ranking, not a protocol error.
    if results is None:
        return RerankerResponse(documents=[])
    if not isinstance(results, list):
        raise GenkitError(message='bedrock rerank: response results is not a list', status='INTERNAL')

    ranked: list[RankedDocumentData] = []
    for position, result in enumerate(results):
        index, score = _result_fields(result, position)
        if not 0 <= index < len(documents):
            raise GenkitError(
                message=f'bedrock rerank: result index {index} out of range for {len(documents)} documents',
                status='INTERNAL',
            )
        ranked.append(
            RankedDocumentData(
                content=documents[index].content,
                metadata=RankedDocumentMetadata(score=score),
            )
        )
    return RerankerResponse(documents=ranked)


class BedrockReranker:
    """Handles a rerank call for one Bedrock reranking model."""

    def __init__(self, model_id: str, transport: InvokeModelTransport) -> None:
        """Initializes the reranker.

        Args:
            model_id: Bedrock model ID, inference-profile ID, or ARN, sent to
                the InvokeModel API verbatim.
            transport: The shared transport seam owning the boto3 client.
        """
        self._model_id = model_id
        self._transport = transport

    async def rerank(self, request: RerankerRequest) -> RerankerResponse:
        """Ranks the request's documents by relevance to its query.

        The body is built from the model ID, since the two families disagree
        over ``api_version``: Cohere requires it and Amazon rejects it. An ID
        matching neither family is not blocked and gets the Cohere body, that
        being the only shape AWS documents for InvokeModel reranking.

        The model ID itself is sent to InvokeModel verbatim, so inference
        profiles and ARNs work too.

        Args:
            request: The query, documents, and options to rank with.

        Returns:
            The ranked documents, each carrying its relevance score.

        Raises:
            GenkitError: INVALID_ARGUMENT for a query or document with no text
                or for unusable options, INTERNAL for a malformed response, and
                the mapped AWS status for a failed call.
        """
        query = document_text(request.query)
        if not query:
            raise GenkitError(message='bedrock rerank: query has no text content', status='INVALID_ARGUMENT')

        texts: list[str] = []
        for index, document in enumerate(request.documents):
            text = document_text(document)
            # Failing beats skipping: a skipped document desyncs the index mapping.
            if not text:
                raise GenkitError(
                    message=f'bedrock rerank: document {index} has no text content',
                    status='INVALID_ARGUMENT',
                )
            texts.append(text)
        # Before options are coerced: nothing to rank outranks bad options.
        if not texts:
            return RerankerResponse(documents=[])

        top_n = len(texts)
        options = coerce_rerank_options(request.options)
        # Clamped down only; asking for more than was sent is not a request the API takes.
        if options is not None and options.top_n is not None and 0 < options.top_n < top_n:
            top_n = options.top_n

        # Only the Amazon family drops api_version; an unknown ID takes Cohere's body.
        family = _rerank_family(self._model_id)
        include_api_version = family != 'amazon'
        logger.debug(
            'Bedrock rerank request',
            model=self._model_id,
            family=family,
            documents=len(texts),
            top_n=top_n,
        )
        payload = await self._invoke(build_rerank_body(query, texts, top_n, include_api_version=include_api_version))
        response = build_rerank_response(payload, request.documents)
        logger.debug('Bedrock rerank response', model=self._model_id, ranked=len(response.documents))
        return response

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
