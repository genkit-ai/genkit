# Copyright 2025 Google LLC
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

"""Unit tests for Part and DocumentPart ergonomic factory methods and property getters."""

from genkit._core._typing import DocumentPart, MediaPart, Part, TextPart


def test_part_from_text() -> None:
    """Part.from_text creates a text part with direct property access."""
    p = Part.from_text('hello world', metadata={'source': 'user'})
    assert isinstance(p.root, TextPart)
    assert p.text == 'hello world'
    assert p.media is None
    assert p.tool_request is None
    assert p.tool_response is None
    assert p.metadata == {'source': 'user'}


def test_part_from_media() -> None:
    """Part.from_media creates a media part with direct property access."""
    p = Part.from_media('https://example.com/image.png', content_type='image/png')
    assert isinstance(p.root, MediaPart)
    assert p.media is not None
    assert p.media.url == 'https://example.com/image.png'
    assert p.media.content_type == 'image/png'
    assert p.text is None


def test_part_from_tool_request() -> None:
    """Part.from_tool_request creates a tool request part."""
    p = Part.from_tool_request(name='get_weather', input={'city': 'Paris'}, ref='call-123')
    assert p.tool_request is not None
    assert p.tool_request.name == 'get_weather'
    assert p.tool_request.input == {'city': 'Paris'}
    assert p.tool_request.ref == 'call-123'
    assert p.text is None


def test_part_from_tool_response() -> None:
    """Part.from_tool_response creates a tool response part."""
    p = Part.from_tool_response(name='get_weather', output={'temp': 22}, ref='call-123')
    assert p.tool_response is not None
    assert p.tool_response.name == 'get_weather'
    assert p.tool_response.output == {'temp': 22}
    assert p.tool_response.ref == 'call-123'
    assert p.text is None


def test_part_from_reasoning() -> None:
    """Part.from_reasoning creates a reasoning part."""
    p = Part.from_reasoning('thinking step by step')
    assert p.reasoning == 'thinking step by step'
    assert p.text is None


def test_part_from_data() -> None:
    """Part.from_data creates a data part."""
    p = Part.from_data({'custom': 'payload'})
    assert p.data == {'custom': 'payload'}
    assert p.text is None


def test_part_from_custom() -> None:
    """Part.from_custom creates a custom part."""
    p = Part.from_custom({'vendor_field': True})
    assert p.custom == {'vendor_field': True}
    assert p.text is None


def test_document_part_from_text() -> None:
    """DocumentPart.from_text creates a text document part."""
    dp = DocumentPart.from_text('sample doc text', metadata={'page': 1})
    assert dp.text == 'sample doc text'
    assert dp.media is None
    assert dp.metadata == {'page': 1}


def test_document_part_from_media() -> None:
    """DocumentPart.from_media creates a media document part."""
    dp = DocumentPart.from_media('https://example.com/doc.pdf', content_type='application/pdf')
    assert dp.media is not None
    assert dp.media.url == 'https://example.com/doc.pdf'
    assert dp.media.content_type == 'application/pdf'
    assert dp.text is None
