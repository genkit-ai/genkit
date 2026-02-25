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

"""Tests for the codec module."""

from pydantic import BaseModel

from genkit.codec import dump_dict, dump_json


def test_dump_json_basic() -> None:
    """Test basic JSON serialization."""
    # Test dictionary
    assert dump_json({'a': 1, 'b': 'test'}) == '{"a": 1, "b": "test"}'

    # Test list
    assert dump_json([1, 2, 3]) == '[1, 2, 3]'

    # Test nested structures
    assert dump_json({'a': [1, 2], 'b': {'c': 3}}) == '{"a": [1, 2], "b": {"c": 3}}'


def test_dump_json_special_types() -> None:
    """Test JSON serialization of special Python types."""
    # Test None
    assert dump_json(None) == 'null'

    # Test boolean
    assert dump_json(True) == 'true'
    assert dump_json(False) == 'false'


def test_dump_json_numbers() -> None:
    """Test JSON serialization of different number types."""
    # Test integers
    assert dump_json(42) == '42'

    # Test floats
    assert dump_json(3.14) == '3.14'

    # Test scientific notation
    assert dump_json(1e-10) == '1e-10'


def test_dump_json_pydantic() -> None:
    """Test JSON serialization of Pydantic models."""

    class MyModel(BaseModel):
        a: int
        b: str

    assert dump_json(MyModel(a=1, b='test')) == '{"a":1,"b":"test"}'


def test_dump_dict_list_of_models() -> None:
    """Test dump_dict with a list of Pydantic models."""

    class MyModel(BaseModel):
        a: int

    models = [MyModel(a=1), MyModel(a=2)]
    # This was failing before the fix because dump_dict didn't recurse into lists
    assert dump_dict(models) == [{'a': 1}, {'a': 2}]


def test_dump_dict_nested_models() -> None:
    """Test dump_dict with nested structures containing Pydantic models."""

    class MyModel(BaseModel):
        a: int

    data = {'key': [MyModel(a=1)], 'other': MyModel(a=2)}
    assert dump_dict(data) == {'key': [{'a': 1}], 'other': {'a': 2}}
