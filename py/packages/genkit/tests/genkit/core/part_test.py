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

"""Unit tests for Part factory methods and property getters."""

from genkit._core._typing import MediaPart, Part, TextPart

_GETTERS = ('text', 'media', 'tool_request', 'tool_response', 'data', 'reasoning', 'custom')


def assert_inactive(part: Part, *, active: str) -> None:
    for name in _GETTERS:
        if name != active:
            assert getattr(part, name) is None


def test_part_from_text() -> None:
    """Part.from_text creates a text part with direct property access."""
    p = Part.from_text('hello world', metadata={'source': 'user'})
    assert isinstance(p.root, TextPart)
    assert p.text == 'hello world'
    assert p.metadata == {'source': 'user'}
    assert_inactive(p, active='text')


def test_part_from_text_defaults() -> None:
    """from_text with no metadata leaves metadata unset."""
    p = Part.from_text('hi')
    assert p.text == 'hi'
    assert p.metadata is None


def test_part_from_text_empty() -> None:
    """Empty text is still a text part."""
    p = Part.from_text('')
    assert p.text == ''
    assert_inactive(p, active='text')


def test_part_from_media() -> None:
    """Part.from_media creates a media part with direct property access."""
    p = Part.from_media('https://example.com/image.png', content_type='image/png', metadata={'alt': 'dish'})
    assert isinstance(p.root, MediaPart)
    assert p.media is not None
    assert p.media.url == 'https://example.com/image.png'
    assert p.media.content_type == 'image/png'
    assert p.metadata == {'alt': 'dish'}
    assert_inactive(p, active='media')


def test_part_from_media_without_content_type() -> None:
    """content_type is optional."""
    p = Part.from_media('https://example.com/image.png')
    assert p.media is not None
    assert p.media.url == 'https://example.com/image.png'
    assert p.media.content_type is None
    assert p.metadata is None


def test_part_from_tool_request() -> None:
    """Part.from_tool_request creates a tool request part."""
    p = Part.from_tool_request(name='get_weather', input={'city': 'Paris'}, ref='call-123', metadata={'src': 'model'})
    assert p.tool_request is not None
    assert p.tool_request.name == 'get_weather'
    assert p.tool_request.input == {'city': 'Paris'}
    assert p.tool_request.ref == 'call-123'
    assert p.metadata == {'src': 'model'}
    assert_inactive(p, active='tool_request')


def test_part_from_tool_request_defaults() -> None:
    """name-only tool request leaves input and ref unset."""
    p = Part.from_tool_request(name='lookup')
    assert p.tool_request is not None
    assert p.tool_request.name == 'lookup'
    assert p.tool_request.input is None
    assert p.tool_request.ref is None
    assert p.metadata is None


def test_part_from_tool_response() -> None:
    """Part.from_tool_response creates a tool response part."""
    p = Part.from_tool_response(name='get_weather', output={'temp': 22}, ref='call-123')
    assert p.tool_response is not None
    assert p.tool_response.name == 'get_weather'
    assert p.tool_response.output == {'temp': 22}
    assert p.tool_response.ref == 'call-123'
    assert_inactive(p, active='tool_response')


def test_part_from_tool_response_defaults() -> None:
    """name-only tool response leaves output and ref unset."""
    p = Part.from_tool_response(name='lookup')
    assert p.tool_response is not None
    assert p.tool_response.name == 'lookup'
    assert p.tool_response.output is None
    assert p.tool_response.ref is None


def test_part_from_reasoning() -> None:
    """Part.from_reasoning creates a reasoning part."""
    p = Part.from_reasoning('thinking step by step', metadata={'step': 1})
    assert p.reasoning == 'thinking step by step'
    assert p.metadata == {'step': 1}
    assert_inactive(p, active='reasoning')


def test_part_from_data() -> None:
    """Part.from_data creates a data part."""
    p = Part.from_data({'custom': 'payload'})
    assert p.data == {'custom': 'payload'}
    assert_inactive(p, active='data')


def test_part_from_data_string() -> None:
    """data can be a string."""
    p = Part.from_data('just a string')
    assert p.data == 'just a string'


def test_part_from_data_list() -> None:
    """data can be a list."""
    p = Part.from_data([1, 2])
    assert p.data == [1, 2]


def test_part_from_custom() -> None:
    """Part.from_custom creates a custom part."""
    p = Part.from_custom({'vendor_field': True})
    assert p.custom == {'vendor_field': True}
    assert_inactive(p, active='custom')


def test_part_from_custom_empty() -> None:
    """Empty custom payload is still a custom part."""
    p = Part.from_custom({})
    assert p.custom == {}


def test_part_root_constructor_still_works() -> None:
    """Part(root=TextPart(...)) still constructs a text part."""
    p = Part(root=TextPart(text='legacy'))
    assert p.text == 'legacy'
    assert_inactive(p, active='text')


def test_part_from_text_round_trip() -> None:
    """Dumping and re-parsing a factory-built part keeps the payload."""
    p = Part.from_text('hello', metadata={'source': 'user'})
    again = Part.model_validate(p.model_dump())
    assert again.text == 'hello'
    assert again.metadata == {'source': 'user'}
    assert_inactive(again, active='text')
