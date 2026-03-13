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

"""Tests for MCP conversion utilities."""

import json
import os
import pathlib
import sys
import unittest
from typing import Any

from mcp.types import BlobResourceContents, ImageContent, TextContent, TextResourceContents

from genkit.core.typing import Media, MediaPart, Message, Part, TextPart

# Defer genkit imports to allow mocking. Type annotations help ty understand these are callable.
to_mcp_prompt_arguments: Any = None
to_mcp_prompt_message: Any = None
to_mcp_resource_contents: Any = None
to_mcp_tool_result: Any = None


def setup_mocks() -> None:
    """Set up mocks for testing."""
    global to_mcp_prompt_arguments, to_mcp_prompt_message, to_mcp_resource_contents, to_mcp_tool_result

    # Add test directory to path for fakes
    if pathlib.Path(__file__).parent not in sys.path:
        sys.path.insert(0, str(pathlib.Path(__file__).parent))

    # Add src directory to path if not installed
    src_path = str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, '../src')).resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        # Deferred import: mock_mcp_modules must be called before importing genkit.plugins.mcp
        from fakes import mock_mcp_modules

        mock_mcp_modules()

        # Deferred import: these imports must happen after mock_mcp_modules() is called
        from genkit.plugins.mcp.util import (
            to_mcp_prompt_arguments as _to_mcp_prompt_arguments,
            to_mcp_prompt_message as _to_mcp_prompt_message,
            to_mcp_resource_contents as _to_mcp_resource_contents,
            to_mcp_tool_result as _to_mcp_tool_result,
        )

        to_mcp_prompt_arguments = _to_mcp_prompt_arguments
        to_mcp_prompt_message = _to_mcp_prompt_message
        to_mcp_resource_contents = _to_mcp_resource_contents
        to_mcp_tool_result = _to_mcp_tool_result
    except ImportError:
        pass


class TestMessageConversion(unittest.TestCase):
    """Tests for message conversion utilities."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        setup_mocks()

    def test_convert_user_message(self) -> None:
        """Test converting a user message."""
        message = Message(role='user', content=[Part(root=TextPart(text='Hello, world!'))])

        result = to_mcp_prompt_message(message)

        self.assertEqual(result.role, 'user')
        self.assertEqual(result.content.type, 'text')
        assert isinstance(result.content, TextContent)
        self.assertEqual(result.content.text, 'Hello, world!')
        self.assertEqual(result.content.type, 'text')
        assert isinstance(result.content, TextContent)
        self.assertEqual(result.content.text, 'Hello, world!')

    def test_convert_model_message(self) -> None:
        """Test converting a model message (maps to assistant)."""
        message = Message(role='model', content=[Part(root=TextPart(text='Hi there!'))])

        result = to_mcp_prompt_message(message)

        self.assertEqual(result.role, 'assistant')
        self.assertEqual(result.content.type, 'text')
        assert isinstance(result.content, TextContent)
        self.assertEqual(result.content.text, 'Hi there!')

    def test_convert_message_with_multiple_text_parts(self) -> None:
        """Test converting a message with multiple text parts."""
        message = Message(
            role='user',
            content=[
                Part(root=TextPart(text='Part 1 ')),
                Part(root=TextPart(text='Part 2 ')),
                Part(root=TextPart(text='Part 3')),
            ],
        )

        result = to_mcp_prompt_message(message)

        assert isinstance(result.content, TextContent)
        self.assertEqual(result.content.text, 'Part 1 Part 2 Part 3')

    def test_convert_message_with_invalid_role(self) -> None:
        """Test that converting a message with invalid role raises error."""
        message = Message(role='system', content=[Part(root=TextPart(text='System message'))])

        with self.assertRaises(ValueError) as context:
            to_mcp_prompt_message(message)

        self.assertIn('system', str(context.exception).lower())

    def test_convert_message_with_image(self) -> None:
        """Test converting a message with image content."""
        message = Message(
            role='user',
            content=[
                Part(root=MediaPart(media=Media(url='data:image/png;base64,iVBORw0KG...', content_type='image/png')))
            ],
        )

        result = to_mcp_prompt_message(message)

        self.assertEqual(result.role, 'user')
        self.assertEqual(result.content.type, 'image')
        assert isinstance(result.content, ImageContent)
        self.assertEqual(result.content.mimeType, 'image/png')
        self.assertEqual(result.content.type, 'image')
        assert isinstance(result.content, ImageContent)
        self.assertEqual(result.content.mimeType, 'image/png')

    def test_convert_message_with_non_data_url_fails(self) -> None:
        """Test that non-data URLs raise an error."""
        message = Message(
            role='user',
            content=[Part(root=MediaPart(media=Media(url='http://example.com/image.png')))],
        )

        with self.assertRaises(ValueError) as context:
            to_mcp_prompt_message(message)

        self.assertIn('base64', str(context.exception).lower())


class TestResourceConversion(unittest.TestCase):
    """Tests for resource content conversion."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        setup_mocks()

    def test_convert_text_resource(self) -> None:
        """Test converting text resource content."""
        parts = [Part(root=TextPart(text='Resource content'))]

        result = to_mcp_resource_contents('test://resource', parts)

        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0].uri), 'test://resource')
        assert isinstance(result[0], TextResourceContents)
        self.assertEqual(result[0].text, 'Resource content')

    def test_convert_multiple_text_parts(self) -> None:
        """Test converting multiple text parts."""
        parts = [
            Part(root=TextPart(text='Part 1')),
            Part(root=TextPart(text='Part 2')),
            Part(root=TextPart(text='Part 3')),
        ]

        result = to_mcp_resource_contents('test://resource', parts)

        self.assertEqual(len(result), 3)
        for i, part in enumerate(result, 1):
            assert isinstance(part, TextResourceContents)
            self.assertEqual(part.text, f'Part {i}')

    def test_convert_string_parts(self) -> None:
        """Test converting string parts - strings are not Part objects, function expects list[Part]."""
        # We need to construct Parts even for strings if the function expects list[Part].
        # If the function handles strings (Union[str, Part]), we should check the function signature.
        # Function signature says 'parts: list[Part]'.
        # So we must pass Parts.
        parts = [Part(root=TextPart(text='Text 1')), Part(root=TextPart(text='Text 2'))]

        result = to_mcp_resource_contents('test://resource', parts)

        self.assertEqual(len(result), 2)
        assert isinstance(result[0], TextResourceContents)
        self.assertEqual(result[0].text, 'Text 1')
        assert isinstance(result[1], TextResourceContents)
        self.assertEqual(result[1].text, 'Text 2')

    def test_convert_media_resource(self) -> None:
        """Test converting media resource content."""
        parts = [Part(root=MediaPart(media=Media(url='data:image/png;base64,abc123', content_type='image/png')))]

        result = to_mcp_resource_contents('test://image', parts)

        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0].uri), 'test://image')
        self.assertEqual(result[0].mimeType, 'image/png')
        assert isinstance(result[0], BlobResourceContents)
        self.assertEqual(result[0].blob, 'abc123')

    def test_convert_mixed_content(self) -> None:
        """Test converting mixed text and media content."""
        parts = [
            Part(root=TextPart(text='Description')),
            Part(root=MediaPart(media=Media(url='data:image/png;base64,xyz', content_type='image/png'))),
        ]

        result = to_mcp_resource_contents('test://mixed', parts)

        self.assertEqual(len(result), 2)
        assert isinstance(result[0], TextResourceContents)
        self.assertEqual(result[0].text, 'Description')
        assert isinstance(result[1], BlobResourceContents)
        self.assertEqual(result[1].blob, 'xyz')


class TestToolResultConversion(unittest.TestCase):
    """Tests for tool result conversion."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        setup_mocks()

    def test_convert_string_result(self) -> None:
        """Test converting string result."""
        result = to_mcp_tool_result('Hello, world!')

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].type, 'text')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].type, 'text')
        assert isinstance(result[0], TextContent)
        self.assertEqual(result[0].text, 'Hello, world!')

    def test_convert_dict_result(self) -> None:
        """Test converting dict result."""
        result = to_mcp_tool_result({'key': 'value', 'number': 42})

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].type, 'text')
        # Should be JSON serialized
        assert isinstance(result[0], TextContent)
        parsed = json.loads(result[0].text)
        self.assertEqual(parsed['key'], 'value')
        self.assertEqual(parsed['number'], 42)

    def test_convert_number_result(self) -> None:
        """Test converting number result."""
        result = to_mcp_tool_result(42)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result), 1)
        assert isinstance(result[0], TextContent)
        self.assertEqual(result[0].text, '42')

    def test_convert_boolean_result(self) -> None:
        """Test converting boolean result."""
        result = to_mcp_tool_result(True)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result), 1)
        assert isinstance(result[0], TextContent)
        self.assertEqual(result[0].text, 'True')


class TestSchemaConversion(unittest.TestCase):
    """Tests for schema conversion utilities."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        setup_mocks()

    def test_convert_simple_schema(self) -> None:
        """Test converting simple string schema."""
        schema = {'type': 'object', 'properties': {'name': {'type': 'string', 'description': 'User name'}}}

        result = to_mcp_prompt_arguments(schema)

        assert result is not None
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], 'name')
        self.assertEqual(result[0]['description'], 'User name')

    def test_convert_schema_with_required(self) -> None:
        """Test converting schema with required fields."""
        schema = {
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'age': {'type': 'string'}},
            'required': ['name'],
        }

        result = to_mcp_prompt_arguments(schema)
        assert result is not None

        name_arg = next(arg for arg in result if arg['name'] == 'name')
        age_arg = next(arg for arg in result if arg['name'] == 'age')

        self.assertTrue(name_arg['required'])
        self.assertFalse(age_arg['required'])

    def test_convert_schema_with_non_string_fails(self) -> None:
        """Test that non-string properties raise an error."""
        schema = {'type': 'object', 'properties': {'count': {'type': 'number'}}}

        with self.assertRaises(ValueError) as context:
            to_mcp_prompt_arguments(schema)

        self.assertIn('string', str(context.exception).lower())

    def test_convert_schema_with_union_type(self) -> None:
        """Test converting schema with union type including string."""
        schema = {'type': 'object', 'properties': {'value': {'type': ['string', 'null']}}}

        result = to_mcp_prompt_arguments(schema)

        # Should succeed because string is in the union
        assert result is not None
        self.assertEqual(len(result), 1)

    def test_convert_none_schema(self) -> None:
        """Test converting None schema."""
        result = to_mcp_prompt_arguments(None)

        self.assertIsNone(result)

    def test_convert_schema_without_properties_returns_none(self) -> None:
        """Test that schema without properties returns None (no parameters)."""
        schema = {'type': 'object'}

        # Schema without properties is valid - it just means no input parameters
        result = to_mcp_prompt_arguments(schema)

        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
