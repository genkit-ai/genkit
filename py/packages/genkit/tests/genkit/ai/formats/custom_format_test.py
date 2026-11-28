#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A custom format built from public imports only, run end to end through generate."""

import csv
import io

import pytest

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk, Part, Role
from genkit.formats import FormatHandler, Formatter, FormatterConfig
from genkit.testing import define_scripted_model

CSV_REPLY = 'dish,price\nRamen,14.50\nPho,13.00\n'
CSV_INSTRUCTIONS = 'Reply only with CSV. First row is the header.'


class CsvFormat(Formatter):
    """The CSV example from the Formatter docstring."""

    def __init__(self) -> None:
        """Registers as 'csv' with a text/csv content type."""
        super().__init__('csv', FormatterConfig(content_type='text/csv'))

    def handle(self, schema: dict[str, object] | None) -> FormatHandler[list[dict[str, str]], str]:
        """Parses the reply as CSV rows; chunks pass through as text."""
        return FormatHandler(
            message_parser=lambda msg: list(csv.DictReader(io.StringIO(msg.text))),
            chunk_parser=lambda chunk: chunk.text,
            instructions=CSV_INSTRUCTIONS,
        )


def _csv_response() -> ModelResponse:
    return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(CSV_REPLY)]))


@pytest.mark.asyncio
async def test_custom_csv_format_parses_output_and_shapes_request() -> None:
    """output_format='csv' runs message_parser, injects instructions, and sets content_type."""
    ai = Genkit()
    model, _ = define_scripted_model(ai, responses=[_csv_response()])
    ai.define_format(CsvFormat())

    response = await ai.generate(model='scriptedModel', prompt='List 2 dishes with price.', output_format='csv')

    assert response.output == [{'dish': 'Ramen', 'price': '14.50'}, {'dish': 'Pho', 'price': '13.00'}]
    request = model.last_request
    assert request is not None
    assert request.output is not None
    assert request.output.format == 'csv'
    assert request.output.content_type == 'text/csv'
    prompt_texts = [p.text for p in request.messages[-1].content]
    assert CSV_INSTRUCTIONS in prompt_texts


@pytest.mark.asyncio
async def test_custom_csv_format_streams_chunks_through_chunk_parser() -> None:
    """Each streamed chunk's output is what chunk_parser returned."""
    ai = Genkit()
    define_scripted_model(
        ai,
        responses=[_csv_response()],
        chunks=[
            [
                ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('dish,price\n')]),
                ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Ramen,14.50\n')]),
            ]
        ],
    )
    ai.define_format(CsvFormat())

    result = ai.generate_stream(model='scriptedModel', prompt='List dishes.', output_format='csv')
    chunk_outputs = [chunk.output async for chunk in result.stream]
    response = await result.response

    assert chunk_outputs == ['dish,price\n', 'Ramen,14.50\n']
    assert response.output == [{'dish': 'Ramen', 'price': '14.50'}, {'dish': 'Pho', 'price': '13.00'}]
