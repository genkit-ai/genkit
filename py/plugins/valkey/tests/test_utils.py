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

"""Pure unit tests for valkey plugin utilities (no Valkey instance required)."""

import struct

import pytest

from genkit import Document
from genkit.plugins.valkey.plugin import (
    _doc_id,
    _float32_to_bytes,
    _validate_filter,
)


def test_float32_to_bytes():
    """Test float32 encoding matches expected little-endian format."""
    vec = [1.0, 2.0, 3.0]
    result = _float32_to_bytes(vec)
    assert len(result) == 12  # 3 floats * 4 bytes

    # Verify round-trip
    unpacked = struct.unpack('<3f', result)
    assert unpacked == (1.0, 2.0, 3.0)


def test_doc_id_deterministic():
    """Test that doc IDs are deterministic."""
    d1 = Document.from_text('hello')
    d2 = Document.from_text('hello')
    d3 = Document.from_text('world')

    assert _doc_id(d1) == _doc_id(d2)
    assert _doc_id(d1) != _doc_id(d3)
    assert len(_doc_id(d1)) == 32  # MD5 hex


def test_validate_filter_allows_valid_expressions():
    _validate_filter('@price:[100 200]')
    _validate_filter('@tag:{foo}')
    _validate_filter('*')
    _validate_filter('')


@pytest.mark.parametrize('char', [';', '|', '`', '$', '\\'])
def test_validate_filter_blocks_disallowed_chars(char):
    with pytest.raises(ValueError, match='disallowed characters'):
        _validate_filter(f'@field:[0 10]{char}inject')


def test_validate_filter_blocks_knn_injection_sequence():
    """The => token separates filter from KNN clause and must be blocked."""
    with pytest.raises(ValueError, match='disallowed characters'):
        _validate_filter('@tag:{x})=>[KNN 99999 @embedding $query_vec]')
    with pytest.raises(ValueError, match='disallowed characters'):
        _validate_filter('foo=>bar')
