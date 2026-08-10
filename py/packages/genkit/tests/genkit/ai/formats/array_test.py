#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Array format."""

import pytest
from pydantic import BaseModel, TypeAdapter

from genkit import Message, ModelResponseChunk
from genkit._ai._formats._array import ArrayFormat
from genkit._core._error import GenkitError
from genkit._core._typing import Part, TextPart


class TestArrayFormatStreaming:
    """Test streaming chunk parsing."""

    def test_emits_complete_array_items_as_they_arrive(self) -> None:
        """Test that complete objects are emitted as they arrive in chunks."""
        array_fmt = ArrayFormat()
        fmt = array_fmt.handle({'type': 'array', 'items': {'type': 'object'}})

        # Chunk 1: [{"id": 1,
        chunk1 = ModelResponseChunk(content=[Part(root=TextPart(text='[{"id": 1,'))])
        result1 = fmt.parse_chunk(ModelResponseChunk(chunk1, index=0, previous_chunks=[]))
        assert result1 == []

        # Chunk 2: "name": "first"}
        chunk2 = ModelResponseChunk(content=[Part(root=TextPart(text='"name": "first"}'))])
        result2 = fmt.parse_chunk(ModelResponseChunk(chunk2, index=0, previous_chunks=[chunk1]))
        assert result2 == [{'id': 1, 'name': 'first'}]

        # Chunk 3: , {"id": 2, "name": "second"}]
        chunk3 = ModelResponseChunk(content=[Part(root=TextPart(text=', {"id": 2, "name": "second"}]'))])
        result3 = fmt.parse_chunk(ModelResponseChunk(chunk3, index=0, previous_chunks=[chunk1, chunk2]))
        assert result3 == [{'id': 2, 'name': 'second'}]

    def test_handles_single_item_arrays(self) -> None:
        """Test parsing a single item array in one chunk."""
        array_fmt = ArrayFormat()
        fmt = array_fmt.handle({'type': 'array', 'items': {'type': 'object'}})

        chunk = ModelResponseChunk(content=[Part(root=TextPart(text='[{"id": 1, "name": "single"}]'))])
        result = fmt.parse_chunk(ModelResponseChunk(chunk, index=0, previous_chunks=[]))
        assert result == [{'id': 1, 'name': 'single'}]

    def test_handles_preamble_with_code_fence(self) -> None:
        """Test parsing array with preamble text and code fence."""
        array_fmt = ArrayFormat()
        fmt = array_fmt.handle({'type': 'array', 'items': {'type': 'object'}})

        # Chunk 1: preamble with code fence start
        chunk1 = ModelResponseChunk(
            content=[Part(root=TextPart(text='Here is the array you requested:\n\n```json\n['))]
        )
        result1 = fmt.parse_chunk(ModelResponseChunk(chunk1, index=0, previous_chunks=[]))
        assert result1 == []

        # Chunk 2: the actual data
        chunk2 = ModelResponseChunk(content=[Part(root=TextPart(text='{"id": 1, "name": "item"}]\n```'))])
        result2 = fmt.parse_chunk(ModelResponseChunk(chunk2, index=0, previous_chunks=[chunk1]))
        assert result2 == [{'id': 1, 'name': 'item'}]


class TestArrayFormatMessage:
    """Test complete message parsing."""

    def test_parses_complete_array_response(self) -> None:
        """Test parsing a complete array response."""
        array_fmt = ArrayFormat()
        fmt = array_fmt.handle({'type': 'array', 'items': {'type': 'object'}})

        result = fmt.parse_message(
            Message(Message(role='model', content=[Part(root=TextPart(text='[{"id": 1, "name": "test"}]'))]))
        )
        assert result == [{'id': 1, 'name': 'test'}]

    def test_parses_empty_array(self) -> None:
        """Test parsing an empty array."""
        array_fmt = ArrayFormat()
        fmt = array_fmt.handle({'type': 'array', 'items': {'type': 'object'}})

        result = fmt.parse_message(Message(Message(role='model', content=[Part(root=TextPart(text='[]'))])))
        assert result == []

    def test_parses_array_with_preamble_and_code_fence(self) -> None:
        """Test parsing array with preamble and code fence."""
        array_fmt = ArrayFormat()
        fmt = array_fmt.handle({'type': 'array', 'items': {'type': 'object'}})

        result = fmt.parse_message(
            Message(
                Message(
                    role='model', content=[Part(root=TextPart(text='Here is the array:\n\n```json\n[{"id": 1}]\n```'))]
                )
            )
        )
        assert result == [{'id': 1}]


class TestArrayFormatErrors:
    """Test error handling."""

    def test_throws_error_for_non_array_schema_type(self) -> None:
        """Test that non-array schema type raises error."""
        array_fmt = ArrayFormat()

        with pytest.raises(GenkitError) as exc_info:
            array_fmt.handle({'type': 'string'})
        assert "Must supply an 'array' schema type" in str(exc_info.value)

    def test_throws_error_for_object_schema_type(self) -> None:
        """Test that object schema type raises error."""
        array_fmt = ArrayFormat()

        with pytest.raises(GenkitError) as exc_info:
            array_fmt.handle({'type': 'object'})
        assert "Must supply an 'array' schema type" in str(exc_info.value)


class TestArrayFormatInstructions:
    """Test instruction generation."""

    def test_generates_instructions_with_schema(self) -> None:
        """Test that instructions are generated when schema is provided."""
        array_fmt = ArrayFormat()
        fmt = array_fmt.handle({'type': 'array', 'items': {'type': 'object'}})

        assert fmt.instructions is not None
        assert 'Output should be a JSON array' in fmt.instructions

    def test_no_instructions_without_schema(self) -> None:
        """Test that no instructions are generated without schema."""
        array_fmt = ArrayFormat()
        fmt = array_fmt.handle(None)

        assert fmt.instructions is None

    def test_accepts_ref_based_array_item_schema(self) -> None:
        """Test that array format accepts TypeAdapter schemas with $ref items."""

        class Book(BaseModel):
            title: str

        array_fmt = ArrayFormat()
        fmt = array_fmt.handle(TypeAdapter(list[Book]).json_schema())

        assert fmt.instructions is not None
        assert '"type": "object"' in fmt.instructions
