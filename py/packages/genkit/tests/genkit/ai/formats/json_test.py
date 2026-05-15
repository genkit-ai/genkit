#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JSON format."""

from genkit import Message, ModelResponseChunk
from genkit._ai._formats import JsonFormat
from genkit._core._typing import Part, TextPart


class TestJsonFormatStreaming:
    """Test streaming chunk parsing."""

    def test_parses_complete_json_object(self) -> None:
        """Test parsing a complete JSON object in one chunk."""
        json_fmt = JsonFormat()
        fmt = json_fmt.handle({'type': 'object'})

        chunk = ModelResponseChunk(content=[Part(TextPart(text='{"id": 1, "name": "test"}'))])
        result = fmt.parse_chunk(ModelResponseChunk(chunk, index=0, previous_chunks=[]))
        assert result == {'id': 1, 'name': 'test'}

    def test_handles_partial_json(self) -> None:
        """Test parsing partial JSON across multiple chunks."""
        json_fmt = JsonFormat()
        fmt = json_fmt.handle({'type': 'object'})

        # Chunk 1: partial object
        chunk1 = ModelResponseChunk(content=[Part(TextPart(text='{"id": 1'))])
        result1 = fmt.parse_chunk(ModelResponseChunk(chunk1, index=0, previous_chunks=[]))
        assert result1 == {'id': 1}

        # Chunk 2: complete object
        chunk2 = ModelResponseChunk(content=[Part(TextPart(text=', "name": "test"}'))])
        result2 = fmt.parse_chunk(ModelResponseChunk(chunk2, index=0, previous_chunks=[chunk1]))
        assert result2 == {'id': 1, 'name': 'test'}

    def test_handles_preamble_with_code_fence(self) -> None:
        """Test parsing JSON with preamble text and code fence."""
        json_fmt = JsonFormat()
        fmt = json_fmt.handle({'type': 'object'})

        # Chunk 1: preamble
        chunk1 = ModelResponseChunk(content=[Part(TextPart(text='Here is the JSON:\n\n```json\n'))])
        result1 = fmt.parse_chunk(ModelResponseChunk(chunk1, index=0, previous_chunks=[]))
        assert result1 is None

        # Chunk 2: actual data
        chunk2 = ModelResponseChunk(content=[Part(TextPart(text='{"id": 1}\n```'))])
        result2 = fmt.parse_chunk(ModelResponseChunk(chunk2, index=0, previous_chunks=[chunk1]))
        assert result2 == {'id': 1}


class TestJsonFormatMessage:
    """Test complete message parsing."""

    def test_parses_complete_json_response(self) -> None:
        """Test parsing a complete JSON response."""
        json_fmt = JsonFormat()
        fmt = json_fmt.handle({'type': 'object'})

        result = fmt.parse_message(Message(role='model', content=[Part(TextPart(text='{"id": 1, "name": "test"}'))]))
        assert result == {'id': 1, 'name': 'test'}

    def test_handles_empty_response(self) -> None:
        """Test parsing an empty response."""
        json_fmt = JsonFormat()
        fmt = json_fmt.handle({'type': 'object'})

        result = fmt.parse_message(Message(role='model', content=[Part(TextPart(text=''))]))
        assert result is None

    def test_parses_json_with_preamble_and_code_fence(self) -> None:
        """Test parsing JSON with preamble and code fence."""
        json_fmt = JsonFormat()
        fmt = json_fmt.handle({'type': 'object'})

        result = fmt.parse_message(
            Message(role='model', content=[Part(TextPart(text='Here is the JSON:\n\n```json\n{"id": 1}\n```'))])
        )
        assert result == {'id': 1}

    def test_parses_partial_json_message(self) -> None:
        """Test parsing a message with partial/incomplete JSON."""
        json_fmt = JsonFormat()
        fmt = json_fmt.handle({'type': 'object'})

        result = fmt.parse_message(Message(role='user', content=[Part(TextPart(text='{"foo": "bar"'))]))
        assert result == {'foo': 'bar'}

    def test_parses_complex_nested_json(self) -> None:
        """Test parsing complex nested JSON across multiple parts."""
        json_fmt = JsonFormat()
        fmt = json_fmt.handle({'type': 'object'})

        result = fmt.parse_chunk(
            ModelResponseChunk(
                ModelResponseChunk(content=[Part(TextPart(text='", "baz": [1,2'))]),
                index=0,
                previous_chunks=[
                    ModelResponseChunk(content=[Part(TextPart(text='{"bar":')), Part(TextPart(text='"ba'))]),
                    ModelResponseChunk(content=[Part(TextPart(text='z'))]),
                ],
            )
        )
        assert result == {'bar': 'baz', 'baz': [1, 2]}


class TestJsonFormatInstructions:
    """Test instruction generation."""

    def test_generates_instructions_with_schema(self) -> None:
        """Test that instructions include the schema."""
        json_fmt = JsonFormat()
        fmt = json_fmt.handle({
            'type': 'object',
            'properties': {
                'value': {
                    'description': 'value field',
                    'type': 'string',
                }
            },
        })

        assert fmt.instructions is not None
        assert 'Output should be in JSON format' in fmt.instructions
        assert 'value' in fmt.instructions

    def test_no_instructions_without_schema(self) -> None:
        """Test that no instructions are generated without schema."""
        json_fmt = JsonFormat()
        fmt = json_fmt.handle(None)

        assert fmt.instructions is None

    def test_instructions_format_matches_expected(self) -> None:
        """Test that instructions format matches expected structure."""
        json_fmt = JsonFormat()
        fmt = json_fmt.handle({
            'properties': {
                'value': {
                    'description': 'value field',
                    'type': 'string',
                }
            },
            'type': 'object',
        })

        expected = """Output should be in JSON format and conform to the following schema:

```
{
  "properties": {
    "value": {
      "description": "value field",
      "type": "string"
    }
  },
  "type": "object"
}
```
"""
        assert fmt.instructions == expected
