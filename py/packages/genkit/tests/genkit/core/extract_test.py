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


"""Tests for JSON extraction utilities."""

from typing import Any

import pytest

from genkit._core import _extract_json
from genkit._core._extract_json import (
    extract_json,
    extract_json_array_from_text,
    extract_partial_json,
    is_complete_json,
    parse_partial_json,
)

# TODO(#4356): consider extracting these tests into shared yaml spec. They are already
# duplicated in js/ai/tests/extract_test.ts

test_cases_extract_json_array_from_text = [
    (
        'handles simple array in chunks',
        [
            {'chunk': '[', 'want': []},
            {'chunk': '{"a": 1},', 'want': [{'a': 1}]},
            {'chunk': '{"b": 2}', 'want': [{'b': 2}]},
            {'chunk': ']', 'want': []},
        ],
    ),
    (
        'handles nested objects',
        [
            {'chunk': '[{"outer": {', 'want': []},
            {
                'chunk': '"inner": "value"}},',
                'want': [{'outer': {'inner': 'value'}}],
            },
            {'chunk': '{"next": true}]', 'want': [{'next': True}]},
        ],
    ),
    (
        'handles escaped characters',
        [
            {'chunk': '[{"text": "line1\\n', 'want': []},
            {
                'chunk': 'line2"},',
                'want': [{'text': 'line1\nline2'}],
            },
            {
                'chunk': '{"text": "tab\\there"}]',
                'want': [{'text': 'tab\there'}],
            },
        ],
    ),
    (
        'ignores content before first array',
        [
            {'chunk': 'Here is an array:\n```json\n\n[', 'want': []},
            {'chunk': '{"a": 1},', 'want': [{'a': 1}]},
            {
                'chunk': '{"b": 2}]\n```\nDid you like my array?',
                'want': [{'b': 2}],
            },
        ],
    ),
    (
        'handles whitespace',
        [
            {'chunk': '[\n  ', 'want': []},
            {'chunk': '{"a": 1},\n  ', 'want': [{'a': 1}]},
            {'chunk': '{"b": 2}\n]', 'want': [{'b': 2}]},
        ],
    ),
]


@pytest.mark.parametrize(
    'name, steps',
    test_cases_extract_json_array_from_text,
    ids=[tc[0] for tc in test_cases_extract_json_array_from_text],
)
def test_extract_json_array_from_text(name: str, steps: list[dict[str, Any]]) -> None:
    """Test extraction of incomplete json that can be fixed."""
    text = ''
    cursor = 0
    for step in steps:
        text += step['chunk']
        result = extract_json_array_from_text(text, cursor)
        assert result.items == step['want']
        cursor = result.cursor


test_cases_extract_json = [
    (
        'extracts simple object',
        {'text': 'prefix{"a":1}suffix'},
        {'expected': {'a': 1}},
    ),
    (
        'returns None for empty str',
        {'text': ''},
        {'expected': None},
    ),
    (
        'extracts simple array',
        {'text': 'prefix[1,2,3]suffix'},
        {'expected': [1, 2, 3]},
    ),
    (
        'extracts a JSON number',
        {'text': '42'},
        {'expected': 42},
    ),
    (
        'extracts JSON null',
        {'text': 'null'},
        {'expected': None},
    ),
    (
        'handles nested structures',
        {'text': 'text{"a":{"b":[1,2]}}more'},
        {'expected': {'a': {'b': [1, 2]}}},
    ),
    (
        'handles strings with braces',
        {'text': '{"text": "not {a} json"}'},
        {'expected': {'text': 'not {a} json'}},
    ),
    (
        'returns null for invalid JSON without throw',
        {'text': 'not json at all'},
        {'expected': None},
    ),
    (
        'throws for invalid JSON with throw flag',
        {'text': 'not json at all', 'throwOnBadJson': True},
        {'throws': True},
    ),
]


@pytest.mark.parametrize(
    'name, input_data, expected_data',
    test_cases_extract_json,
    ids=[tc[0] for tc in test_cases_extract_json],
)
def test_extract_json(name: str, input_data: dict[str, Any], expected_data: dict[str, Any]) -> None:
    """Test if input is unfixable raise the correct exception or return the proper error response."""
    if expected_data.get('throws'):
        with pytest.raises(ValueError):
            extract_json(input_data['text'], throw_on_bad_json=True)
    else:
        result = extract_json(
            input_data['text'],
            throw_on_bad_json=input_data.get('throwOnBadJson', False),
        )
        assert result == expected_data['expected']


test_cases_parse_partial_json = [
    (
        'parses complete object',
        '{"a":1,"b":2}',
        {'expected': {'a': 1, 'b': 2}},
    ),
    (
        'parses partial object',
        '{"a":1,"b":',
        {'expected': {'a': 1}},
    ),
    (
        'parses partial array',
        '[1,2,3,',
        {'expected': [1, 2, 3]},
    ),
    # NOTE: this testcase diverges from the one in js/ai/tests/extract_test.ts
    # Specifically, python partial json parser lib doesn't like malformed json.
    # JS one handles input: '{"a":{"b":1,"c":]}}',
    (
        'parses nested partial structures',
        '{"a":{"b":1,"c":[',
        {'expected': {'a': {'b': 1, 'c': []}}},
    ),
]


@pytest.mark.parametrize(
    'name, input_str, expected_data',
    test_cases_parse_partial_json,
    ids=[tc[0] for tc in test_cases_parse_partial_json],
)
def test_parse_partial_json(name: str, input_str: str, expected_data: dict[str, Any]) -> None:
    """Test if it fixes simple malformed json string."""
    result = parse_partial_json(input_str)
    assert result == expected_data['expected']


def test_extract_json_cut_off_object_raises() -> None:
    """extract_json('{"a": 1') raises ValueError because the object is not complete."""
    with pytest.raises(ValueError):
        extract_json('{"a": 1')


def test_extract_json_cut_off_object_no_throw_returns_none() -> None:
    """extract_json('{"a": 1', throw_on_bad_json=False) returns None."""
    assert extract_json('{"a": 1', throw_on_bad_json=False) is None


def test_extract_partial_json_cut_off_object_returns_partial() -> None:
    """extract_partial_json('{"a": 1') returns {'a': 1} for a streaming cut-off."""
    assert extract_partial_json('{"a": 1') == {'a': 1}


def test_extract_json_balanced_non_json_no_throw_returns_none() -> None:
    """extract_json('{1, 2}', throw_on_bad_json=False) returns None."""
    assert extract_json('{1, 2}', throw_on_bad_json=False) is None


def test_extract_json_skips_non_json_braces_and_returns_later_object() -> None:
    """extract_json('{1, 2} then {"a": 1}') returns {'a': 1}."""
    assert extract_json('{1, 2} then {"a": 1}') == {'a': 1}


def test_is_complete_json_edge_cases() -> None:
    """is_complete_json handles open strings with braces, escapes, fences, scalars, and JSON5."""
    assert not is_complete_json('')
    assert not is_complete_json('   ')
    assert not is_complete_json('Here is the JSON:')
    assert not is_complete_json('{"a": 1')
    assert not is_complete_json('[{"a": 1}, {"b": 2')
    assert not is_complete_json('{"a": "unclosed } brace')
    assert not is_complete_json('{"a": "escaped quote \\"} still open')
    assert not is_complete_json('{"a": "trailing backslash \\')
    assert not is_complete_json('```json\n{"a": 1')

    assert is_complete_json('42')
    assert is_complete_json('null')
    assert is_complete_json('true')
    assert is_complete_json('"hello"')
    assert is_complete_json('{"a": 1}')
    assert is_complete_json('{"a": "escaped backslash \\\\"}')
    assert is_complete_json('Here is JSON:\n```json\n{"a": 1}\n```')
    assert is_complete_json("{'a': 'hello } world',}")
    assert extract_json("[{'a': ']'}, {'b': 2}]") == [{'a': ']'}, {'b': 2}]


def test_open_json_stream_does_not_call_json5_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open standard-JSON prefixes must not fall back to slow json5.loads scans."""
    json5_calls = 0
    real_loads = _extract_json.json5.loads

    def counting_loads(s: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        nonlocal json5_calls
        json5_calls += 1
        return real_loads(s, *args, **kwargs)

    monkeypatch.setattr(_extract_json.json5, 'loads', counting_loads)

    partial = '{"title": "Chef\'s Cake", "url": "https://example.com/a/*b*/", "steps": ["mix'
    assert extract_partial_json(partial) == {
        'title': "Chef's Cake",
        'url': 'https://example.com/a/*b*/',
        'steps': ['mix'],
    }
    assert not is_complete_json(partial)
    assert json5_calls == 0
