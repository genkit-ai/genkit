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

"""Tests for conform.log_redact — data URI truncation."""

from __future__ import annotations

from conform.log_redact import (
    redact_data_uris,
    redact_data_uris_processor,
    truncate_data_uri,
)


class TestTruncateDataUri:
    """Tests for truncate_data_uri."""

    def test_short_uri_unchanged(self) -> None:
        """Short payloads are left intact."""
        short = 'data:image/png;base64,' + 'A' * 50
        assert truncate_data_uri(short) == short

    def test_long_uri_truncated(self) -> None:
        """Long payloads are replaced with byte count."""
        payload = 'A' * 20000
        uri = f'data:image/png;base64,{payload}'
        result = truncate_data_uri(uri)
        assert result == f'data:image/png;base64,...<{len(payload)} bytes>'
        assert len(result) < 100

    def test_jpeg_uri(self) -> None:
        """JPEG media type is handled."""
        payload = 'B' * 500
        uri = f'data:image/jpeg;base64,{payload}'
        result = truncate_data_uri(uri)
        assert '...<500 bytes>' in result

    def test_audio_uri(self) -> None:
        """Audio media type is handled."""
        payload = 'C' * 1000
        uri = f'data:audio/mp3;base64,{payload}'
        result = truncate_data_uri(uri)
        assert '...<1000 bytes>' in result

    def test_no_data_uri_unchanged(self) -> None:
        """Strings without data URIs pass through."""
        text = 'Hello, this is a normal string with no data URIs.'
        assert truncate_data_uri(text) == text

    def test_mixed_text_and_uri(self) -> None:
        """Surrounding text is preserved."""
        payload = 'D' * 200
        text = f'prefix data:image/png;base64,{payload} suffix'
        result = truncate_data_uri(text)
        assert 'prefix' in result
        assert 'suffix' in result
        assert '...<200 bytes>' in result

    def test_multiple_uris_in_one_string(self) -> None:
        """Multiple URIs in one string are all truncated."""
        p1 = 'A' * 300
        p2 = 'B' * 400
        text = f'data:image/png;base64,{p1} and data:image/jpeg;base64,{p2}'
        result = truncate_data_uri(text)
        assert '...<300 bytes>' in result
        assert '...<400 bytes>' in result

    def test_empty_string(self) -> None:
        """Empty string passes through."""
        assert truncate_data_uri('') == ''


class TestRedactDataUris:
    """Tests for redact_data_uris recursive traversal."""

    def test_string(self) -> None:
        """String values are truncated."""
        payload = 'X' * 500
        result = redact_data_uris(f'data:image/png;base64,{payload}')
        assert isinstance(result, str)
        assert '...<500 bytes>' in result

    def test_dict(self) -> None:
        """Dict values are recursively truncated."""
        payload = 'Y' * 300
        obj = {'url': f'data:image/png;base64,{payload}', 'name': 'test'}
        result = redact_data_uris(obj)
        assert isinstance(result, dict)
        assert '...<300 bytes>' in result['url']
        assert result['name'] == 'test'

    def test_nested_dict(self) -> None:
        """Nested dicts are handled."""
        payload = 'Z' * 200
        obj = {'outer': {'inner': f'data:audio/wav;base64,{payload}'}}
        result = redact_data_uris(obj)
        assert isinstance(result, dict)
        inner = result['outer']
        assert isinstance(inner, dict)
        assert '...<200 bytes>' in inner['inner']

    def test_list(self) -> None:
        """List elements are truncated."""
        payload = 'W' * 150
        obj = [f'data:image/gif;base64,{payload}', 'normal']
        result = redact_data_uris(obj)
        assert isinstance(result, list)
        assert '...<150 bytes>' in result[0]
        assert result[1] == 'normal'

    def test_tuple_preserved(self) -> None:
        """Tuple type is preserved in output."""
        payload = 'V' * 200
        obj = (f'data:image/png;base64,{payload}',)
        result = redact_data_uris(obj)
        assert isinstance(result, tuple)

    def test_non_string_passthrough(self) -> None:
        """Non-string primitives pass through unchanged."""
        assert redact_data_uris(42) == 42
        assert redact_data_uris(None) is None
        assert redact_data_uris(3.14) == 3.14

    def test_original_not_mutated(self) -> None:
        """Original object is not modified."""
        payload = 'M' * 500
        original = {'url': f'data:image/png;base64,{payload}'}
        url_before = original['url']
        redact_data_uris(original)
        assert original['url'] == url_before


class TestRedactDataUrisProcessor:
    """Tests for the structlog processor."""

    def test_processor_truncates_values(self) -> None:
        """Processor truncates data URIs in event dict."""
        payload = 'P' * 1000
        event_dict = {
            'event': 'generate request',
            'request': {'media': f'data:image/png;base64,{payload}'},
        }
        result = redact_data_uris_processor(object(), 'debug', event_dict)
        request = result['request']
        assert isinstance(request, dict)
        assert '...<1000 bytes>' in request['media']
        assert result['event'] == 'generate request'

    def test_processor_preserves_non_string_values(self) -> None:
        """Non-string values pass through the processor."""
        event_dict = {'event': 'test', 'count': 42, 'flag': True}
        result = redact_data_uris_processor(object(), 'info', event_dict)
        assert result == event_dict
