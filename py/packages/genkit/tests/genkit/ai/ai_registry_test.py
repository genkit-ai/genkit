#!/usr/bin/env python3
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

"""Tests for the AI registry module."""

import unittest

import pytest

from genkit import Genkit
from genkit._core._dap import DapValue, DynamicActionProvider
from genkit._core._flow import get_func_description


class TestGetFuncDescription(unittest.TestCase):
    def test_get_func_description_with_explicit_description(self) -> None:
        def test_func() -> None:
            """This docstring should be ignored."""
            pass

        description = get_func_description(test_func, 'Explicit description')
        self.assertEqual(description, 'Explicit description')

    def test_get_func_description_with_docstring(self) -> None:
        def test_func() -> None:
            """This is the function's docstring."""
            pass

        description = get_func_description(test_func)
        self.assertEqual(description, "This is the function's docstring.")

    def test_get_func_description_without_docstring(self) -> None:
        def test_func() -> None:
            pass

        description = get_func_description(test_func)
        self.assertEqual(description, '')

    def test_get_func_description_with_none_docstring(self) -> None:
        def test_func() -> None:
            pass

        test_func.__doc__ = None

        description = get_func_description(test_func)
        self.assertEqual(description, '')


class TestDefineJsonSchema:
    def test_define_json_schema_basic(self) -> None:
        ai = Genkit()

        schema = ai.define_json_schema(
            'SimpleObject',
            {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'age': {'type': 'integer'},
                },
                'required': ['name'],
            },
        )

        assert schema is not None
        assert schema['type'] == 'object'
        assert 'properties' in schema

    def test_define_json_schema_complex(self) -> None:
        ai = Genkit()

        schema = ai.define_json_schema(
            'Recipe',
            {
                'type': 'object',
                'properties': {
                    'title': {'type': 'string'},
                    'ingredients': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                    'instructions': {'type': 'string'},
                    'nutrition': {
                        'type': 'object',
                        'properties': {
                            'calories': {'type': 'number'},
                            'protein': {'type': 'number'},
                        },
                    },
                },
                'required': ['title', 'ingredients', 'instructions'],
            },
        )

        assert schema is not None
        assert schema['type'] == 'object'
        properties: dict[str, object] = schema['properties']  # type: ignore[assignment]
        assert isinstance(properties, dict)
        assert 'ingredients' in properties
        ingredients: dict[str, object] = properties['ingredients']  # type: ignore[assignment]
        assert isinstance(ingredients, dict)
        assert ingredients['type'] == 'array'

    def test_define_json_schema_returns_same_schema(self) -> None:
        ai = Genkit()

        input_schema: dict[str, object] = {
            'type': 'string',
            'minLength': 1,
        }

        returned_schema = ai.define_json_schema('StringSchema', input_schema)
        assert returned_schema is input_schema


class TestDefineDynamicActionProvider:
    @pytest.mark.asyncio
    async def test_define_dap_with_string_config(self) -> None:
        ai = Genkit()

        async def dap_fn() -> DapValue:
            return {}

        dap = ai.define_dynamic_action_provider('my-dap', dap_fn)

        assert isinstance(dap, DynamicActionProvider)

    @pytest.mark.asyncio
    async def test_define_dap_with_options(self) -> None:
        ai = Genkit()

        async def dap_fn() -> DapValue:
            return {}

        dap = ai.define_dynamic_action_provider(
            'configured-dap',
            dap_fn,
            description='A configured DAP',
            cache_ttl_millis=5000,
            metadata={'custom': 'value'},
        )

        assert isinstance(dap, DynamicActionProvider)

    @pytest.mark.asyncio
    async def test_define_dap_returns_provider(self) -> None:
        ai = Genkit()

        async def dap_fn() -> DapValue:
            return {}

        result = ai.define_dynamic_action_provider('test-dap', dap_fn)

        assert isinstance(result, DynamicActionProvider)
        assert hasattr(result, 'get_action')
        assert hasattr(result, 'list_action_metadata')
        assert hasattr(result, 'invalidate_cache')
        assert hasattr(result, 'get_action_metadata_record')


if __name__ == '__main__':
    unittest.main()
