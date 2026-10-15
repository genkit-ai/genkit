#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Enum format."""

import pytest

from genkit import Message, ModelResponseChunk, Part
from genkit._ai._formats._enum import EnumFormat
from genkit._core._error import GenkitError, RuntimeErrorReason
from genkit._core._model import chunk_for_stream


class TestEnumFormatMessage:
    """Test complete message parsing."""

    def test_parses_simple_enum_value(self) -> None:
        """Test parsing a simple enum value."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['VALUE1', 'VALUE2']})

        result = fmt.parse_message(Message(role='model', content=[Part.from_text('VALUE1')]))
        assert result == 'VALUE1'

    def test_trims_whitespace(self) -> None:
        """Test that whitespace is trimmed from the result."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['VALUE1', 'VALUE2']})

        result = fmt.parse_message(Message(role='model', content=[Part.from_text('  VALUE2\n')]))
        assert result == 'VALUE2'

    def test_removes_double_quotes(self) -> None:
        """Test that double quotes are removed."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['foo', 'bar']})

        result = fmt.parse_message(Message(role='model', content=[Part.from_text('"foo"')]))
        assert result == 'foo'

    def test_removes_single_quotes(self) -> None:
        """Test that single quotes are removed."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['foo', 'bar']})

        result = fmt.parse_message(Message(role='model', content=[Part.from_text("'bar'")]))
        assert result == 'bar'

    def test_handles_unquoted_value(self) -> None:
        """Test that unquoted values are returned as-is."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['foo', 'bar']})

        result = fmt.parse_message(Message(role='model', content=[Part.from_text('bar')]))
        assert result == 'bar'


class TestEnumFormatStreaming:
    """Test streaming chunk parsing."""

    def test_parses_accumulated_text_from_chunks(self) -> None:
        """Test that accumulated text is parsed correctly from chunks."""
        enum_fmt = EnumFormat()
        fmt = enum_fmt.handle({'type': 'string', 'enum': ['foo', 'bar']})

        chunk1 = ModelResponseChunk(content=[Part.from_text('"f')])
        chunk2 = ModelResponseChunk(content=[Part.from_text('oo"')])

        result = fmt.parse_chunk(
            chunk_for_stream(
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
        assert exc_info.value.reason is RuntimeErrorReason.INVALID_SCHEMA
        assert 'INVALID_SCHEMA' not in exc_info.value.original_message

    def test_throws_error_for_array_schema_type(self) -> None:
        """Test that array schema type raises error."""
        enum_fmt = EnumFormat()

        with pytest.raises(GenkitError) as exc_info:
            enum_fmt.handle({'type': 'array'})
        assert "Must supply a schema of type 'string' with an 'enum' property" in str(exc_info.value)
        assert exc_info.value.reason is RuntimeErrorReason.INVALID_SCHEMA
        assert 'INVALID_SCHEMA' not in exc_info.value.original_message

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


@pytest.mark.parametrize('value', ["don't", 'say "yes"'])
@pytest.mark.parametrize('quoted', [False, True])
def test_message_preserves_quotes_inside_enum_value(value: str, quoted: bool) -> None:
    """Removing surrounding quotes must not alter the enum value itself."""
    fmt = EnumFormat().handle({'type': 'string', 'enum': [value]})
    text = f'"{value}"' if quoted else value
    assert fmt.parse_message(Message(role='model', content=[Part.from_text(text)])) == value


def test_streaming_preserves_apostrophe_inside_enum_value() -> None:
    """Accumulated enum text retains the apostrophe while dropping its wrapper."""
    fmt = EnumFormat().handle({'type': 'string', 'enum': ["don't"]})
    first = ModelResponseChunk(content=[Part.from_text('"don')])
    second = ModelResponseChunk(content=[Part.from_text('\'t"')])
    assert fmt.parse_chunk(chunk_for_stream(first, index=0, previous_chunks=[])) == 'don'
    assert fmt.parse_chunk(chunk_for_stream(second, index=0, previous_chunks=[first])) == "don't"


def test_trims_whitespace_inside_output_wrapper() -> None:
    """The existing whitespace cleanup also applies inside output quotes."""
    fmt = EnumFormat().handle({'type': 'string', 'enum': ['foo']})
    assert fmt.parse_message(Message(role='model', content=[Part.from_text('" foo "')])) == 'foo'


@pytest.mark.parametrize('quote', ["'", '"'])
def test_streaming_prefers_complete_enum_over_longer_prefix(quote: str) -> None:
    """A completed value is not mistaken for a prefix of another option."""
    fmt = EnumFormat().handle({'type': 'string', 'enum': ['foo', f'foo{quote}bar']})
    chunk = ModelResponseChunk(content=[Part.from_text(f'{quote}foo{quote}')])
    assert fmt.parse_message(Message(role='model', content=chunk.content)) == 'foo'
    assert fmt.parse_chunk(chunk_for_stream(chunk, index=0, previous_chunks=[])) == 'foo'


@pytest.mark.parametrize('value', ['say "yes"', 'say "yes" please'])
def test_streaming_preserves_inner_quote_before_wrapper_completes(value: str) -> None:
    """A trailing inner quote is still data while the output wrapper is open."""
    fmt = EnumFormat().handle({'type': 'string', 'enum': [value]})
    first = ModelResponseChunk(content=[Part.from_text('"say "yes"')])
    second = ModelResponseChunk(content=[Part.from_text(value[len('say "yes"') :] + '"')])
    assert fmt.parse_chunk(chunk_for_stream(first, index=0, previous_chunks=[])) == 'say "yes"'
    assert fmt.parse_chunk(chunk_for_stream(second, index=0, previous_chunks=[first])) == value
