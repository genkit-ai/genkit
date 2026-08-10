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

"""Tests for the schema module."""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from genkit._core._schema import to_json_schema


def test_to_json_schema_pydantic_model() -> None:
    """Test that a Pydantic model can be converted to a JSON schema."""

    class TestSchema(BaseModel):
        foo: int | None = Field(default=None, description='foo field')
        bar: str | None = Field(default=None, description='bar field')

    assert to_json_schema(TestSchema) == {
        'properties': {
            'bar': {
                'anyOf': [{'type': 'string'}, {'type': 'null'}],
                'default': None,
                'description': 'bar field',
                'title': 'Bar',
            },
            'foo': {
                'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                'default': None,
                'description': 'foo field',
                'title': 'Foo',
            },
        },
        'title': 'TestSchema',
        'type': 'object',
    }


def test_to_json_schema_already_schema() -> None:
    """Test that a JSON schema can be converted to a JSON schema."""
    json_schema = {
        'properties': {
            'bar': {
                'default': None,
                'description': 'bar field',
                'title': 'Bar',
                'type': 'string',
            },
            'foo': {
                'default': None,
                'description': 'foo field',
                'title': 'Foo',
                'type': 'integer',
            },
        },
        'title': 'TestSchema',
        'type': 'object',
    }

    assert to_json_schema(json_schema) == json_schema


# =============================================================================
# JSON Schema Specification-based Tests
# See: https://json-schema.org/understanding-json-schema/reference/type
# =============================================================================


class TestNullType:
    """Tests for null type as per JSON Schema spec.

    See: https://json-schema.org/understanding-json-schema/reference/null
    """

    def test_none_produces_null_type(self) -> None:
        """Python None should produce JSON Schema null type."""
        assert to_json_schema(None) == {'type': 'null'}


class TestStringType:
    """Tests for string type as per JSON Schema spec.

    See: https://json-schema.org/understanding-json-schema/reference/string
    """

    def test_str_type(self) -> None:
        """Python str type should produce JSON Schema string type."""
        assert to_json_schema(str) == {'type': 'string'}


class TestNumericTypes:
    """Tests for numeric types as per JSON Schema spec.

    See: https://json-schema.org/understanding-json-schema/reference/numeric
    Note: JSON Schema has 'integer' and 'number' (floating point).
    """

    @pytest.mark.parametrize(
        'py_type, json_type_name',
        [
            (int, 'integer'),
            (float, 'number'),
        ],
    )
    def test_numeric_types(self, py_type: type, json_type_name: str) -> None:
        """Python numeric types should produce correct JSON Schema numeric types."""
        assert to_json_schema(py_type) == {'type': json_type_name}


class TestBooleanType:
    """Tests for boolean type as per JSON Schema spec.

    See: https://json-schema.org/understanding-json-schema/reference/boolean
    """

    def test_bool_type(self) -> None:
        """Python bool type should produce JSON Schema boolean type."""
        assert to_json_schema(bool) == {'type': 'boolean'}


class TestArrayType:
    """Tests for array type as per JSON Schema spec.

    See: https://json-schema.org/understanding-json-schema/reference/array
    """

    @pytest.mark.parametrize(
        'list_type, item_schema',
        [
            (list[str], {'type': 'string'}),
            (list[int], {'type': 'integer'}),
        ],
    )
    def test_list_types(self, list_type: type, item_schema: dict[str, Any]) -> None:
        """Python list types should produce array schema with correct item types."""
        result = to_json_schema(list_type)
        assert result['type'] == 'array'
        assert result['items'] == item_schema


class TestObjectType:
    """Tests for object type as per JSON Schema spec.

    See: https://json-schema.org/understanding-json-schema/reference/object
    """

    def test_dict_type(self) -> None:
        """Python dict should produce object schema."""
        result = to_json_schema(dict)
        assert result['type'] == 'object'

    def test_pydantic_model(self) -> None:
        """Pydantic BaseModel should produce object schema with properties."""

        class Person(BaseModel):
            name: str
            age: int

        result = to_json_schema(Person)
        assert result['type'] == 'object'
        assert 'properties' in result
        assert result['properties']['name']['type'] == 'string'
        assert result['properties']['age']['type'] == 'integer'
        assert result['required'] == ['name', 'age']


class TestPassthroughBehavior:
    """Tests for passthrough behavior when input is already a JSON Schema dict."""

    @pytest.mark.parametrize(
        'schema',
        [
            {'type': 'string', 'minLength': 1},
            {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'items': {
                        'type': 'array',
                        'items': {'type': 'integer'},
                    },
                },
                'required': ['name'],
            },
        ],
        ids=['simple_schema', 'complex_schema'],
    )
    def test_passthrough_behavior(self, schema: dict[str, Any]) -> None:
        """A dict representing a JSON Schema should be returned as-is."""
        assert to_json_schema(schema) == schema
