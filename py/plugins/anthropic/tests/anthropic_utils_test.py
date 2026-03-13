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

"""Tests for Anthropic plugin utility functions.

Unit tests for the pure-function helpers extracted into utils.py, covering
cache control extraction, document/image block conversion, media routing,
and cache-aware usage building.
"""

import base64

from genkit.plugins.anthropic.utils import (
    DOCUMENT_MIME_TYPES,
    PDF_MIME_TYPE,
    TEXT_MIME_TYPE,
    build_cache_usage,
    get_cache_control,
    to_anthropic_document,
    to_anthropic_image,
    to_anthropic_media,
)
from genkit.types import (
    GenerationUsage,
    Media,
    MediaPart,
    Metadata,
    TextPart,
)

# ---------------------------------------------------------------------------
# get_cache_control tests
# ---------------------------------------------------------------------------


class TestGetCacheControl:
    """Tests for get_cache_control utility."""

    def test_returns_none_for_no_metadata(self) -> None:
        """Returns None when part has no metadata."""
        part = TextPart(text='hello')
        assert get_cache_control(part) is None

    def test_returns_none_for_none_metadata(self) -> None:
        """Returns None when metadata is explicitly None."""
        part = TextPart(text='hello', metadata=None)
        assert get_cache_control(part) is None

    def test_returns_cache_control_with_metadata_rootmodel(self) -> None:
        """Extracts cache_control when metadata is a Metadata RootModel."""
        part = TextPart(text='hello', metadata=Metadata({'cache_control': {'type': 'ephemeral'}}))
        result = get_cache_control(part)
        assert result == {'type': 'ephemeral'}

    def test_returns_none_when_no_cache_control_key(self) -> None:
        """Returns None when metadata has no cache_control key."""
        part = TextPart(text='hello', metadata=Metadata({'other_key': 'value'}))
        assert get_cache_control(part) is None

    def test_returns_none_for_non_dict_cache_control(self) -> None:
        """Returns None when cache_control is not a dict."""
        part = TextPart(text='hello', metadata=Metadata({'cache_control': 'invalid'}))
        assert get_cache_control(part) is None

    def test_works_with_media_part(self) -> None:
        """Works with MediaPart as well as TextPart."""
        part = MediaPart(
            media=Media(url='https://example.com/img.png', content_type='image/png'),
            metadata=Metadata({'cache_control': {'type': 'ephemeral'}}),
        )
        result = get_cache_control(part)
        assert result == {'type': 'ephemeral'}

    def test_works_with_plain_dict(self) -> None:
        """Works when an object has metadata as a plain dict (no .root)."""

        class FakePart:
            metadata = {'cache_control': {'type': 'ephemeral'}}

        result = get_cache_control(FakePart())
        assert result == {'type': 'ephemeral'}


# ---------------------------------------------------------------------------
# to_anthropic_document tests
# ---------------------------------------------------------------------------


class TestToAnthropicDocument:
    """Tests for to_anthropic_document utility."""

    def test_base64_pdf(self) -> None:
        """Converts base64-encoded PDF to document block."""
        pdf_data = base64.b64encode(b'%PDF-fake').decode()
        url = f'data:application/pdf;base64,{pdf_data}'
        result = to_anthropic_document(url, PDF_MIME_TYPE)
        assert result['type'] == 'document'
        assert result['source']['type'] == 'base64'
        assert result['source']['media_type'] == PDF_MIME_TYPE
        assert result['source']['data'] == pdf_data

    def test_base64_text(self) -> None:
        """Converts base64-encoded plain text to document block."""
        text_data = base64.b64encode(b'Hello world').decode()
        url = f'data:text/plain;base64,{text_data}'
        result = to_anthropic_document(url, TEXT_MIME_TYPE)
        assert result['type'] == 'document'
        assert result['source']['type'] == 'base64'
        assert result['source']['media_type'] == TEXT_MIME_TYPE

    def test_url_pdf(self) -> None:
        """Converts PDF URL to URL-based document block."""
        url = 'https://example.com/doc.pdf'
        result = to_anthropic_document(url, PDF_MIME_TYPE)
        assert result['type'] == 'document'
        assert result['source']['type'] == 'url'
        assert result['source']['url'] == url

    def test_url_text_fallback(self) -> None:
        """Falls back to text block for plain text URLs."""
        url = 'https://example.com/readme.txt'
        result = to_anthropic_document(url, TEXT_MIME_TYPE)
        assert result['type'] == 'text'
        assert 'Document:' in result['text']
        assert url in result['text']


# ---------------------------------------------------------------------------
# to_anthropic_image tests
# ---------------------------------------------------------------------------


class TestToAnthropicImage:
    """Tests for to_anthropic_image utility."""

    def test_base64_image(self) -> None:
        """Converts base64-encoded image to image block."""
        img_data = base64.b64encode(b'\x89PNG').decode()
        url = f'data:image/png;base64,{img_data}'
        result = to_anthropic_image(url, 'image/png')
        assert result['type'] == 'image'
        assert result['source']['type'] == 'base64'
        assert result['source']['media_type'] == 'image/png'
        assert result['source']['data'] == img_data

    def test_url_image(self) -> None:
        """Converts image URL to URL-based image block."""
        url = 'https://example.com/image.jpg'
        result = to_anthropic_image(url, 'image/jpeg')
        assert result['type'] == 'image'
        assert result['source']['type'] == 'url'
        assert result['source']['url'] == url

    def test_infers_content_type_from_data_uri(self) -> None:
        """Infers content type from data URI when not provided."""
        img_data = base64.b64encode(b'\x89PNG').decode()
        url = f'data:image/webp;base64,{img_data}'
        result = to_anthropic_image(url, '')
        assert result['source']['media_type'] == 'image/webp'


# ---------------------------------------------------------------------------
# to_anthropic_media tests
# ---------------------------------------------------------------------------


class TestToAnthropicMedia:
    """Tests for to_anthropic_media routing function."""

    def test_routes_pdf_to_document(self) -> None:
        """Routes PDF media to document block."""
        pdf_data = base64.b64encode(b'%PDF-fake').decode()
        part = MediaPart(
            media=Media(url=f'data:application/pdf;base64,{pdf_data}', content_type=PDF_MIME_TYPE),
        )
        result = to_anthropic_media(part)
        assert result['type'] == 'document'

    def test_routes_text_to_document(self) -> None:
        """Routes plain text media to document block."""
        text_data = base64.b64encode(b'Hello').decode()
        part = MediaPart(
            media=Media(url=f'data:text/plain;base64,{text_data}', content_type=TEXT_MIME_TYPE),
        )
        result = to_anthropic_media(part)
        assert result['type'] == 'document'

    def test_routes_image_to_image(self) -> None:
        """Routes image media to image block."""
        part = MediaPart(
            media=Media(url='https://example.com/photo.jpg', content_type='image/jpeg'),
        )
        result = to_anthropic_media(part)
        assert result['type'] == 'image'

    def test_infers_pdf_from_data_uri(self) -> None:
        """Infers PDF type from data URI when content_type is empty."""
        pdf_data = base64.b64encode(b'%PDF-fake').decode()
        part = MediaPart(
            media=Media(url=f'data:application/pdf;base64,{pdf_data}'),
        )
        result = to_anthropic_media(part)
        assert result['type'] == 'document'

    def test_document_mime_types_constant(self) -> None:
        """Verifies DOCUMENT_MIME_TYPES contains expected types."""
        assert PDF_MIME_TYPE in DOCUMENT_MIME_TYPES
        assert TEXT_MIME_TYPE in DOCUMENT_MIME_TYPES
        assert 'image/png' not in DOCUMENT_MIME_TYPES


# ---------------------------------------------------------------------------
# build_cache_usage tests
# ---------------------------------------------------------------------------


class TestBuildCacheUsage:
    """Tests for build_cache_usage utility."""

    def test_basic_usage_without_cache(self) -> None:
        """Builds usage without cache tokens."""
        basic = GenerationUsage(input_characters=10, output_characters=20)
        result = build_cache_usage(
            input_tokens=100,
            output_tokens=50,
            basic_usage=basic,
        )
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
        assert result.input_characters == 10
        assert result.output_characters == 20
        assert result.custom is None

    def test_usage_with_cache_creation(self) -> None:
        """Includes cache_creation_input_tokens in custom."""
        basic = GenerationUsage()
        result = build_cache_usage(
            input_tokens=100,
            output_tokens=50,
            basic_usage=basic,
            cache_creation_input_tokens=200,
        )
        assert result.custom is not None
        assert result.custom['cache_creation_input_tokens'] == 200
        assert 'cache_read_input_tokens' not in result.custom

    def test_usage_with_cache_read(self) -> None:
        """Includes cache_read_input_tokens in custom."""
        basic = GenerationUsage()
        result = build_cache_usage(
            input_tokens=100,
            output_tokens=50,
            basic_usage=basic,
            cache_read_input_tokens=300,
        )
        assert result.custom is not None
        assert result.custom['cache_read_input_tokens'] == 300
        assert 'cache_creation_input_tokens' not in result.custom

    def test_usage_with_both_cache_fields(self) -> None:
        """Includes both cache token fields when both are present."""
        basic = GenerationUsage()
        result = build_cache_usage(
            input_tokens=100,
            output_tokens=50,
            basic_usage=basic,
            cache_creation_input_tokens=200,
            cache_read_input_tokens=300,
        )
        assert result.custom is not None
        assert result.custom['cache_creation_input_tokens'] == 200
        assert result.custom['cache_read_input_tokens'] == 300

    def test_zero_cache_tokens_are_excluded(self) -> None:
        """Zero cache tokens don't appear in custom field."""
        basic = GenerationUsage()
        result = build_cache_usage(
            input_tokens=100,
            output_tokens=50,
            basic_usage=basic,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        assert result.custom is None
