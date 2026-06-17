#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Enum format."""

import pytest

from genkit import Message, ModelResponseChunk
from genkit._ai._formats._enum import EnumFormat
from genkit._core._error import GenkitError
from genkit._core._typing import Part, TextPart


class TestEnumFormatMessage:
    """Test complete message parsing."""

    def test_parses_simple_enum_value(self) -> None:
        """Test parsing a simple enum value."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['VALUE1', 'VALUE2']})

        result = fmt.parse_message(Message(role='model', content=[Part(TextPart(text='VALUE1'))]))
        assert result == 'VALUE1'

    def test_trims_whitespace(self) -> None:
        """Test that whitespace is trimmed from the result."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['VALUE1', 'VALUE2']})

        result = fmt.parse_message(Message(role='model', content=[Part(TextPart(text='  VALUE2\n'))]))
        assert result == 'VALUE2'

    def test_removes_double_quotes(self) -> None:
        """Test that double quotes are removed."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['foo', 'bar']})

        result = fmt.parse_message(Message(role='model', content=[Part(TextPart(text='"foo"'))]))
        assert result == 'foo'

    def test_removes_single_quotes(self) -> None:
        """Test that single quotes are removed."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['foo', 'bar']})

        result = fmt.parse_message(Message(role='model', content=[Part(TextPart(text="'bar'"))]))
        assert result == 'bar'

    def test_handles_unquoted_value(self) -> None:
        """Test that unquoted values are returned as-is."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['foo', 'bar']})

        result = fmt.parse_message(Message(role='model', content=[Part(TextPart(text='bar'))]))
        assert result == 'bar'


class TestEnumFormatStreaming:
    """Test streaming chunk parsing."""

    def test_parses_accumulated_text_from_chunks(self) -> None:
        """Test that accumulated text is parsed correctly from chunks."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['foo', 'bar']})

        chunk1 = ModelResponseChunk(content=[Part(TextPart(text='"f'))])
        chunk2 = ModelResponseChunk(content=[Part(TextPart(text='oo"'))])

        result = fmt.parse_chunk(
            ModelResponseChunk(
                chunk2,
                index=0,
                previous_chunks=[chunk1],
            )
        )
        assert result == 'foo'


class TestEnumFormatErrors:
    """Test error handling."""

    def test_throws_error_for_number_schema_type(self) -> None:
        """Test that number schema type raises error."""
        enum_fmt = EnumFormat()

        with pytest.raises(GenkitError) as exc_info:
            enum_fmt.handle({'type': 'number'})
        assert "Must supply a schema of type 'string' with an 'enum' property" in str(exc_info.value)

    def test_throws_error_for_array_schema_type(self) -> None:
        """Test that array schema type raises error."""
        enum_fmt = EnumFormat()

        with pytest.raises(GenkitError) as exc_info:
            enum_fmt.handle({'type': 'array'})
        assert "Must supply a schema of type 'string' with an 'enum' property" in str(exc_info.value)

    def test_accepts_enum_schema_type(self) -> None:
        """Test that 'enum' schema type is accepted."""
        enum_fmt = EnumFormat()
        # Should not raise
        fmt = enum_fmt.handle({'type': 'enum', 'enum': ['a', 'b']})
        assert fmt is not None


class TestEnumFormatInstructions:
    """Test instruction generation."""

    def test_generates_instructions_with_enum_values(self) -> None:
        """Test that instructions list enum values."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['foo', 'bar']})

        assert fmt.instructions is not None
        assert 'Output should be ONLY one of the following enum values' in fmt.instructions
        assert 'foo' in fmt.instructions
        assert 'bar' in fmt.instructions

    def test_no_instructions_without_enum(self) -> None:
        """Test that no instructions are generated without enum values."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string'})

        assert fmt.instructions is None

    def test_no_instructions_without_schema(self) -> None:
        """Test that no instructions are generated without schema."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle(None)

        assert fmt.instructions is None
