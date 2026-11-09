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

"""PartConverter fidelity: thought signatures on every part kind, enum values, stream coalescing."""

import json
from typing import Any

import pytest
from genkit_google_genai._models._gemini import GeminiModel, _coalesce_stream_parts
from genkit_google_genai._models._utils import PartConverter
from google.genai import types as genai_types
from pytest_mock import MockerFixture

from genkit import Message, Part, Role
from genkit.model import ModelContext, ModelRequest

SIG = b'opaque-signature'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'gemini_part',
    [
        genai_types.Part(text='The eigenvalues sum to 12.', thought_signature=SIG),
        genai_types.Part(
            executable_code=genai_types.ExecutableCode(code='print(1)', language=genai_types.Language.PYTHON),
            thought_signature=SIG,
        ),
        genai_types.Part(
            code_execution_result=genai_types.CodeExecutionResult(outcome=genai_types.Outcome.OUTCOME_OK, output='1\n'),
            thought_signature=SIG,
        ),
        genai_types.Part(inline_data=genai_types.Blob(mime_type='image/png', data=b'\x89PNG'), thought_signature=SIG),
    ],
    ids=['text', 'executable_code', 'code_execution_result', 'inline_data'],
)
async def test_thought_signature_survives_round_trip(gemini_part: genai_types.Part) -> None:
    """Gemini 3 signs text and code parts; the signature must come back on the next turn."""
    genkit_part = PartConverter.from_gemini(gemini_part)
    assert genkit_part.metadata is not None and 'thoughtSignature' in genkit_part.metadata

    # JSON persistence (chat history in a DB) must not lose it either.
    rehydrated = Part.model_validate_json(genkit_part.model_dump_json(by_alias=True))
    back = await PartConverter.to_gemini(rehydrated)

    assert not isinstance(back, list)
    assert back.thought_signature == SIG
    assert back == gemini_part


def test_code_execution_enums_are_wire_strings() -> None:
    """Language/Outcome come back as JSON strings, not google-genai enum members."""
    code = PartConverter.from_gemini(
        genai_types.Part(executable_code=genai_types.ExecutableCode(code='x', language=genai_types.Language.PYTHON))
    )
    result = PartConverter.from_gemini(
        genai_types.Part(code_execution_result=genai_types.CodeExecutionResult(outcome=genai_types.Outcome.OUTCOME_OK))
    )

    assert code.custom is not None and result.custom is not None
    assert type(code.custom['executableCode']['language']) is str
    assert type(result.custom['codeExecutionResult']['outcome']) is str
    assert json.loads(code.model_dump_json())['custom']['executableCode']['language'] == 'PYTHON'


def test_signature_only_part_is_kept_as_reasoning() -> None:
    """A streamed part that carries only a signature is not flattened to empty text."""
    part = PartConverter.from_gemini(genai_types.Part(thought_signature=SIG))

    assert part.reasoning == ''
    assert part.metadata is not None and 'thoughtSignature' in part.metadata


def test_file_data_comes_back_as_media() -> None:
    """A fileData part in a response is media, not an empty text part."""
    part = PartConverter.from_gemini(
        genai_types.Part(file_data=genai_types.FileData(file_uri='gs://menus/today.pdf', mime_type='application/pdf'))
    )

    assert part.media is not None
    assert part.media.url == 'gs://menus/today.pdf'
    assert part.media.content_type == 'application/pdf'


def test_coalesce_joins_fragments_and_keeps_closing_signature() -> None:
    """Text and reasoning fragments merge; the signature that closes a block stays; empty text drops."""
    sig = {'thoughtSignature': 'c2ln'}
    parts = [
        Part.from_reasoning('Think '),
        Part.from_reasoning('hard.', metadata=sig),
        Part.from_text('Pesto '),
        Part.from_text('has '),
        Part.from_text('pine nuts.'),
        Part.from_text(''),
    ]

    out = _coalesce_stream_parts(parts)

    assert [p.model_dump(exclude_none=True) for p in out] == [
        {'reasoning': 'Think hard.', 'metadata': sig},
        {'text': 'Pesto has pine nuts.'},
    ]


def test_coalesce_does_not_merge_across_signed_part_or_tool_call() -> None:
    """A signed part ends a block; tool requests split text runs."""
    sig = {'thoughtSignature': 'c2ln'}
    parts = [
        Part.from_text('A', metadata=sig),
        Part.from_text('B'),
        Part.from_tool_request('lookup_menu', {}),
        Part.from_text('C'),
    ]

    out = _coalesce_stream_parts(parts)

    assert [p.text for p in out] == ['A', 'B', None, 'C']


@pytest.mark.asyncio
async def test_streaming_final_message_is_coalesced(mocker: MockerFixture) -> None:
    """Chunks still stream as fragments; the final message has whole parts for guardrails to read."""

    def chunk(text: str, **kw: Any) -> genai_types.GenerateContentResponse:  # noqa: ANN401
        return genai_types.GenerateContentResponse(
            candidates=[genai_types.Candidate(content=genai_types.Content(parts=[genai_types.Part(text=text)]), **kw)]
        )

    async def chunks() -> Any:  # noqa: ANN401
        yield chunk('Pesto ')
        yield chunk('has pine nuts.')
        yield chunk('', finish_reason=genai_types.FinishReason.STOP)

    client = mocker.AsyncMock()
    client.aio.models.generate_content_stream.return_value = chunks()
    on_chunk = mocker.MagicMock()

    response = await GeminiModel('gemini-flash-latest', client).generate(
        ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('Nut-free?')])]),
        ModelContext(streaming_callback=on_chunk),
    )

    assert response.message is not None
    assert [p.text for p in response.message.content] == ['Pesto has pine nuts.']
    assert on_chunk.call_count == 3


@pytest.mark.asyncio
async def test_parts_without_a_gemini_field_are_not_sent(mocker: MockerFixture) -> None:
    """Replaying Part.data (a2ui) used to send an empty Part(), which Gemini rejects."""
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(content=genai_types.Content(parts=[genai_types.Part(text='ok')]))]
    )
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part.from_text('Show the menu.')]),
            Message(role=Role.MODEL, content=[Part.from_text('Here it is.'), Part.from_data({'surface': 'menu'})]),
            Message(role=Role.MODEL, content=[Part.from_data({'surface': 'cart'})]),
            Message(role=Role.USER, content=[Part.from_text('Add pesto.')]),
        ]
    )

    await GeminiModel('gemini-flash-latest', client).generate(request, ModelContext())

    contents = client.aio.models.generate_content.call_args.kwargs['contents']
    assert [[p.text for p in c.parts] for c in contents] == [['Show the menu.'], ['Here it is.'], ['Add pesto.']]
