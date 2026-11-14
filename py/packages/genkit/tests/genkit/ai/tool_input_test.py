# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for string-tool wire adaptation."""

from genkit.model import ToolDefinition
from genkit.model._tool_input import is_string_tool


def test_is_string_tool_skips_missing_tool_definitions() -> None:
    """A nullable tool entry should not hide a later matching string tool."""
    string_tool = ToolDefinition(name='echo', description='Echo input', input_schema={'type': 'string'})
    object_tool = ToolDefinition(name='other', description='Object input', input_schema={'type': 'object'})

    assert is_string_tool('echo', [None, string_tool])
    assert not is_string_tool('other', [None, object_tool])
