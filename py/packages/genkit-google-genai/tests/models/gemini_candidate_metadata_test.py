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

"""Candidate metadata (grounding, URL context, citations, logprobs) on ModelResponse.custom."""

from typing import Any

import pytest
from genkit_google_genai._models._gemini import GeminiModel
from google.genai import types as genai_types
from pytest_mock import MockerFixture

from genkit import Message, Part, Role
from genkit.model import ModelContext, ModelRequest


def _request() -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('Is pesto nut-free?')])])


def _grounding() -> genai_types.GroundingMetadata:
    return genai_types.GroundingMetadata(
        web_search_queries=['pesto pine nuts'],
        grounding_chunks=[genai_types.GroundingChunk(web=genai_types.GroundingChunkWeb(uri='https://a.example', title='A'))],
    )


def _candidate(text: str, **fields: Any) -> genai_types.Candidate:  # noqa: ANN401
    return genai_types.Candidate(content=genai_types.Content(parts=[genai_types.Part(text=text)]), **fields)


@pytest.mark.asyncio
async def test_generate_copies_candidate_metadata_to_custom(mocker: MockerFixture) -> None:
    """google_search / url_context / response_logprobs metadata reaches custom under wire names."""
    candidate = _candidate(
        'Pesto has pine nuts.',
        finish_reason=genai_types.FinishReason.STOP,
        grounding_metadata=_grounding(),
        url_context_metadata=genai_types.UrlContextMetadata(
            url_metadata=[
                genai_types.UrlMetadata(
                    retrieved_url='https://a.example',
                    url_retrieval_status=genai_types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                )
            ]
        ),
        citation_metadata=genai_types.CitationMetadata(citations=[genai_types.Citation(uri='https://a.example')]),
        logprobs_result=genai_types.LogprobsResult(
            chosen_candidates=[genai_types.LogprobsResultCandidate(token='Pesto', log_probability=-0.1)]
        ),
        avg_logprobs=-0.25,
    )
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = genai_types.GenerateContentResponse(candidates=[candidate])
    gemini = GeminiModel('gemini-flash-latest', client)

    response = await gemini.generate(_request(), ModelContext())

    assert response.custom == {
        'groundingMetadata': {
            'webSearchQueries': ['pesto pine nuts'],
            'groundingChunks': [{'web': {'uri': 'https://a.example', 'title': 'A'}}],
        },
        'urlContextMetadata': {
            'urlMetadata': [{'retrievedUrl': 'https://a.example', 'urlRetrievalStatus': 'URL_RETRIEVAL_STATUS_SUCCESS'}]
        },
        'citationMetadata': {'citations': [{'uri': 'https://a.example'}]},
        'logprobsResult': {'chosenCandidates': [{'token': 'Pesto', 'logProbability': -0.1}]},
        'avgLogprobs': -0.25,
    }


@pytest.mark.asyncio
async def test_generate_without_metadata_leaves_custom_unset(mocker: MockerFixture) -> None:
    """A plain answer carries no custom payload."""
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = genai_types.GenerateContentResponse(
        candidates=[_candidate('Yes.', finish_reason=genai_types.FinishReason.STOP)]
    )
    gemini = GeminiModel('gemini-flash-latest', client)

    response = await gemini.generate(_request(), ModelContext())

    assert response.custom in (None, {})


@pytest.mark.asyncio
async def test_streaming_generate_keeps_trailing_grounding_metadata(mocker: MockerFixture) -> None:
    """Grounding metadata on the last chunk survives into the final streamed response."""

    async def chunks() -> Any:  # noqa: ANN401
        yield genai_types.GenerateContentResponse(candidates=[_candidate('Pesto ')])
        yield genai_types.GenerateContentResponse(
            candidates=[
                _candidate('has pine nuts.', finish_reason=genai_types.FinishReason.STOP, grounding_metadata=_grounding())
            ]
        )

    client = mocker.AsyncMock()
    client.aio.models.generate_content_stream.return_value = chunks()
    gemini = GeminiModel('gemini-flash-latest', client)

    response = await gemini.generate(_request(), ModelContext(streaming_callback=mocker.MagicMock()))

    assert response.custom is not None
    assert response.custom['groundingMetadata']['webSearchQueries'] == ['pesto pine nuts']
