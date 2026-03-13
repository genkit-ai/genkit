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

"""Tests for the ``conform.validators`` package."""

from __future__ import annotations

from typing import Any

import pytest
from conform.validators import VALIDATORS, ValidationError, get_validator, register
from conform.validators.helpers import get_media_part, get_message_content, get_message_text
from conform.validators.json import valid_json
from conform.validators.media import valid_media
from conform.validators.reasoning import reasoning
from conform.validators.streaming import stream_has_tool_request, stream_text_includes, stream_valid_json
from conform.validators.text import text_includes, text_not_empty, text_starts_with
from conform.validators.tool import has_tool_request

# ---------------------------------------------------------------------------
# Helpers for building response dicts
# ---------------------------------------------------------------------------


def _resp(content: list[dict]) -> dict:
    """Build a minimal response with message.content."""
    return {'message': {'content': content}}


def _resp_candidates(content: list[dict]) -> dict:
    """Build a response using the candidates format."""
    return {'candidates': [{'message': {'content': content}}]}


def _chunk(content: list[dict]) -> dict:
    """Build a streaming chunk."""
    return {'content': content}


# ===========================================================================
# helpers.py
# ===========================================================================


class TestGetMessageContent:
    """Tests for get_message_content."""

    def test_from_message(self) -> None:
        """Extract content from message format."""
        r = _resp([{'text': 'hi'}])
        assert get_message_content(r) == [{'text': 'hi'}]

    def test_from_candidates(self) -> None:
        """Extract content from candidates format."""
        r = _resp_candidates([{'text': 'hi'}])
        assert get_message_content(r) == [{'text': 'hi'}]

    def test_missing_message(self) -> None:
        """Return None when message is missing."""
        assert get_message_content({}) is None

    def test_empty_candidates(self) -> None:
        """Return None when candidates list is empty."""
        assert get_message_content({'candidates': []}) is None


class TestGetMessageText:
    """Tests for get_message_text."""

    def test_returns_first_text(self) -> None:
        """Return the first text part."""
        r = _resp([{'text': 'hello'}, {'text': 'world'}])
        assert get_message_text(r) == 'hello'

    def test_skips_non_text_parts(self) -> None:
        """Skip non-text parts and return the first text."""
        r = _resp([{'toolRequest': {}}, {'text': 'found'}])
        assert get_message_text(r) == 'found'

    def test_no_text_part(self) -> None:
        """Return None when no text part exists."""
        r = _resp([{'toolRequest': {}}])
        assert get_message_text(r) is None

    def test_missing_message(self) -> None:
        """Return None when message is missing."""
        assert get_message_text({}) is None


class TestGetMediaPart:
    """Tests for get_media_part."""

    def test_returns_media_part(self) -> None:
        """Return the first media part."""
        part = {'media': {'url': 'http://example.com/img.png', 'contentType': 'image/png'}}
        r = _resp([part])
        assert get_media_part(r) == part

    def test_no_media(self) -> None:
        """Return None when no media part exists."""
        r = _resp([{'text': 'hi'}])
        assert get_media_part(r) is None

    def test_missing_message(self) -> None:
        """Return None when message is missing."""
        assert get_media_part({}) is None


# ===========================================================================
# Registry
# ===========================================================================


class TestRegistry:
    """Tests for the validator registry."""

    def test_all_validators_registered(self) -> None:
        """All 10 canonical validators are registered."""
        expected = {
            'text-includes',
            'text-starts-with',
            'text-not-empty',
            'valid-json',
            'has-tool-request',
            'valid-media',
            'reasoning',
            'stream-text-includes',
            'stream-has-tool-request',
            'stream-valid-json',
        }
        assert expected.issubset(set(VALIDATORS.keys()))

    def test_get_validator_found(self) -> None:
        """Look up a registered validator by name."""
        v = get_validator('text-includes')
        assert v is text_includes

    def test_get_validator_not_found(self) -> None:
        """Raise KeyError for unknown validator name."""
        with pytest.raises(KeyError, match='Unknown validator'):
            get_validator('nonexistent-validator')

    def test_register_custom(self) -> None:
        """Register and retrieve a custom validator."""

        @register('test-custom-validator')
        def _custom(
            response: dict[str, Any],
            arg: str | None = None,
            chunks: list[dict[str, Any]] | None = None,
        ) -> None:
            pass

        assert VALIDATORS['test-custom-validator'] is _custom
        # Clean up.
        del VALIDATORS['test-custom-validator']


# ===========================================================================
# text.py
# ===========================================================================


class TestTextIncludes:
    """Tests for text-includes validator."""

    def test_passes(self) -> None:
        """Pass when text contains the expected substring."""
        text_includes(_resp([{'text': 'Hello World'}]), arg='hello')

    def test_case_insensitive(self) -> None:
        """Match is case-insensitive."""
        text_includes(_resp([{'text': 'HELLO'}]), arg='hello')

    def test_fails(self) -> None:
        """Fail when text does not contain the expected substring."""
        with pytest.raises(ValidationError, match='does not include'):
            text_includes(_resp([{'text': 'goodbye'}]), arg='hello')

    def test_empty_text_fails(self) -> None:
        """Fail when text is empty."""
        with pytest.raises(ValidationError):
            text_includes(_resp([{'text': ''}]), arg='hello')

    def test_no_message_fails(self) -> None:
        """Fail when message is missing."""
        with pytest.raises(ValidationError):
            text_includes({}, arg='hello')


class TestTextStartsWith:
    """Tests for text-starts-with validator."""

    def test_passes(self) -> None:
        """Pass when text starts with the expected prefix."""
        text_starts_with(_resp([{'text': 'Hello World'}]), arg='Hello')

    def test_strips_whitespace(self) -> None:
        """Leading whitespace is stripped before comparison."""
        text_starts_with(_resp([{'text': '  Hello'}]), arg='Hello')

    def test_fails(self) -> None:
        """Fail when text does not start with the expected prefix."""
        with pytest.raises(ValidationError, match='does not start with'):
            text_starts_with(_resp([{'text': 'Goodbye'}]), arg='Hello')

    def test_empty_text_fails(self) -> None:
        """Fail when text is empty."""
        with pytest.raises(ValidationError):
            text_starts_with(_resp([{'text': ''}]), arg='Hello')


class TestTextNotEmpty:
    """Tests for text-not-empty validator."""

    def test_passes(self) -> None:
        """Pass when text is non-empty."""
        text_not_empty(_resp([{'text': 'hi'}]))

    def test_empty_fails(self) -> None:
        """Fail when text is empty string."""
        with pytest.raises(ValidationError, match='empty'):
            text_not_empty(_resp([{'text': ''}]))

    def test_whitespace_only_fails(self) -> None:
        """Fail when text is whitespace only."""
        with pytest.raises(ValidationError, match='empty'):
            text_not_empty(_resp([{'text': '   '}]))

    def test_no_message_fails(self) -> None:
        """Fail when message is missing."""
        with pytest.raises(ValidationError):
            text_not_empty({})


# ===========================================================================
# json.py
# ===========================================================================


class TestValidJson:
    """Tests for valid-json validator."""

    def test_passes(self) -> None:
        """Pass when text is valid JSON object."""
        valid_json(_resp([{'text': '{"key": "value"}'}]))

    def test_array_passes(self) -> None:
        """Pass when text is valid JSON array."""
        valid_json(_resp([{'text': '[1, 2, 3]'}]))

    def test_invalid_json_fails(self) -> None:
        """Fail when text is not valid JSON."""
        with pytest.raises(ValidationError, match='not valid JSON'):
            valid_json(_resp([{'text': '{bad json'}]))

    def test_no_text_part_fails(self) -> None:
        """Fail when response has no text part."""
        with pytest.raises(ValidationError, match='did not return text'):
            valid_json(_resp([{'toolRequest': {}}]))

    def test_no_content_fails(self) -> None:
        """Fail when message content is missing."""
        with pytest.raises(ValidationError, match='missing message content'):
            valid_json({})


# ===========================================================================
# tool.py
# ===========================================================================


class TestHasToolRequest:
    """Tests for has-tool-request validator."""

    def test_passes(self) -> None:
        """Pass when response contains a tool request."""
        has_tool_request(_resp([{'toolRequest': {'name': 'weather', 'input': {}}}]))

    def test_with_arg_passes(self) -> None:
        """Pass when tool request matches the expected name."""
        has_tool_request(
            _resp([{'toolRequest': {'name': 'weather', 'input': {}}}]),
            arg='weather',
        )

    def test_wrong_tool_name_fails(self) -> None:
        """Fail when tool request name does not match."""
        with pytest.raises(ValidationError, match="Expected tool request 'search'"):
            has_tool_request(
                _resp([{'toolRequest': {'name': 'weather', 'input': {}}}]),
                arg='search',
            )

    def test_no_tool_request_fails(self) -> None:
        """Fail when response has no tool request."""
        with pytest.raises(ValidationError, match='did not return a tool request'):
            has_tool_request(_resp([{'text': 'hi'}]))

    def test_no_content_fails(self) -> None:
        """Fail when message content is missing."""
        with pytest.raises(ValidationError, match='missing message content'):
            has_tool_request({})


# ===========================================================================
# media.py
# ===========================================================================


class TestValidMedia:
    """Tests for valid-media validator."""

    def test_passes_generic(self) -> None:
        """Pass when response contains any media part."""
        valid_media(_resp([{'media': {'url': 'http://example.com/img.png'}}]))

    def test_passes_image_http(self) -> None:
        """Pass for image with HTTP URL."""
        valid_media(
            _resp([{'media': {'url': 'http://example.com/img.png', 'contentType': 'image/png'}}]),
            arg='image',
        )

    def test_passes_image_data_uri(self) -> None:
        """Pass for image with data URI."""
        valid_media(
            _resp([{'media': {'url': 'data:image/png;base64,abc', 'contentType': 'image/png'}}]),
            arg='image',
        )

    def test_no_media_fails(self) -> None:
        """Fail when response has no media part."""
        with pytest.raises(ValidationError, match='did not return'):
            valid_media(_resp([{'text': 'hi'}]), arg='image')

    def test_wrong_content_type_fails(self) -> None:
        """Fail when content type does not match expected."""
        with pytest.raises(ValidationError, match='Expected image'):
            valid_media(
                _resp([{'media': {'url': 'http://x.com/a.mp3', 'contentType': 'audio/mp3'}}]),
                arg='image',
            )

    def test_image_missing_url_fails(self) -> None:
        """Fail when image media part has no URL."""
        with pytest.raises(ValidationError, match='missing URL'):
            valid_media(
                _resp([{'media': {'contentType': 'image/png'}}]),
                arg='image',
            )

    def test_image_bad_data_uri_fails(self) -> None:
        """Fail when data URI content type is not image."""
        with pytest.raises(ValidationError, match='Invalid data URL'):
            valid_media(
                _resp([{'media': {'url': 'data:audio/mp3;base64,abc', 'contentType': 'image/png'}}]),
                arg='image',
            )

    def test_image_unknown_url_format_fails(self) -> None:
        """Fail for non-http, non-data URL schemes."""
        with pytest.raises(ValidationError, match='Unknown URL format'):
            valid_media(
                _resp([{'media': {'url': 'ftp://example.com/img.png', 'contentType': 'image/png'}}]),
                arg='image',
            )


# ===========================================================================
# reasoning.py
# ===========================================================================


class TestReasoning:
    """Tests for reasoning validator."""

    def test_passes(self) -> None:
        """Pass when response contains reasoning content."""
        reasoning(_resp([{'reasoning': 'I think therefore I am'}, {'text': 'result'}]))

    def test_no_reasoning_fails(self) -> None:
        """Fail when response has no reasoning part."""
        with pytest.raises(ValidationError, match='reasoning content not found'):
            reasoning(_resp([{'text': 'hi'}]))

    def test_no_content_fails(self) -> None:
        """Fail when message content is missing."""
        with pytest.raises(ValidationError, match='missing message content'):
            reasoning({})


# ===========================================================================
# streaming.py
# ===========================================================================


class TestStreamTextIncludes:
    """Tests for stream-text-includes validator."""

    def test_passes(self) -> None:
        """Pass when streamed text includes expected substring."""
        chunks = [_chunk([{'text': 'hel'}]), _chunk([{'text': 'lo'}])]
        stream_text_includes(_resp([{'text': 'hello'}]), arg='hello', chunks=chunks)

    def test_fails(self) -> None:
        """Fail when streamed text does not include expected substring."""
        chunks = [_chunk([{'text': 'goodbye'}])]
        with pytest.raises(ValidationError, match='did not include'):
            stream_text_includes(_resp([{'text': 'goodbye'}]), arg='hello', chunks=chunks)

    def test_no_chunks_fails(self) -> None:
        """Fail when no streaming chunks are received."""
        with pytest.raises(ValidationError, match='no chunks'):
            stream_text_includes(_resp([{'text': 'hi'}]), arg='hi', chunks=[])


class TestStreamHasToolRequest:
    """Tests for stream-has-tool-request validator."""

    def test_passes(self) -> None:
        """Pass when streamed chunks contain a tool request."""
        chunks = [_chunk([{'toolRequest': {'name': 'weather'}}])]
        resp = _resp([{'toolRequest': {'name': 'weather', 'input': {}}}])
        stream_has_tool_request(resp, chunks=chunks)

    def test_with_arg_passes(self) -> None:
        """Pass when streamed tool request matches expected name."""
        chunks = [_chunk([{'toolRequest': {'name': 'weather'}}])]
        resp = _resp([{'toolRequest': {'name': 'weather', 'input': {}}}])
        stream_has_tool_request(resp, arg='weather', chunks=chunks)

    def test_no_tool_in_chunks_fails(self) -> None:
        """Fail when no tool request in streamed chunks."""
        chunks = [_chunk([{'text': 'hi'}])]
        with pytest.raises(ValidationError, match='No tool request'):
            stream_has_tool_request(_resp([{'text': 'hi'}]), chunks=chunks)

    def test_no_chunks_fails(self) -> None:
        """Fail when no streaming chunks are received."""
        with pytest.raises(ValidationError, match='no chunks'):
            stream_has_tool_request(_resp([{'text': 'hi'}]), chunks=[])


class TestStreamValidJson:
    """Tests for stream-valid-json validator."""

    def test_passes(self) -> None:
        """Pass when streamed text forms valid JSON."""
        chunks = [_chunk([{'text': '{"ke'}]), _chunk([{'text': 'y": 1}'}])]
        stream_valid_json(_resp([{'text': '{"key": 1}'}]), chunks=chunks)

    def test_invalid_json_fails(self) -> None:
        """Fail when streamed text is not valid JSON."""
        chunks = [_chunk([{'text': '{bad'}])]
        with pytest.raises(ValidationError, match='not valid JSON'):
            stream_valid_json(_resp([{'text': '{bad'}]), chunks=chunks)

    def test_empty_text_fails(self) -> None:
        """Fail when streamed text is empty."""
        chunks = [_chunk([{'toolRequest': {}}])]
        with pytest.raises(ValidationError, match='no text'):
            stream_valid_json(_resp([{'text': ''}]), chunks=chunks)

    def test_no_chunks_fails(self) -> None:
        """Fail when no streaming chunks are received."""
        with pytest.raises(ValidationError, match='no chunks'):
            stream_valid_json(_resp([{'text': '{}'}]), chunks=[])
