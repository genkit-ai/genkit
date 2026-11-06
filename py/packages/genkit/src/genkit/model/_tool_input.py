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

"""Wire envelope for tools whose Genkit input is a bare string."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from genkit.model import ToolDefinition

TOOL_INPUT_FIELD = 'input'


def wrap_string_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Object-only provider APIs receive ``{'input': <string schema>}``."""
    if not isinstance(schema, dict) or schema.get('type') != 'string':
        return schema
    return {
        'type': 'object',
        'properties': {TOOL_INPUT_FIELD: dict(schema)},
        'required': [TOOL_INPUT_FIELD],
    }


def is_string_tool(name: str, tools: Sequence[ToolDefinition | None] | None) -> bool:
    return any(
        tool is not None
        and tool.name == name
        and isinstance(tool.input_schema, dict)
        and tool.input_schema.get('type') == 'string'
        for tool in tools or ()
    )


def unwrap_string_input(name: str, value: Any, tools: Sequence[ToolDefinition] | None) -> Any:  # noqa: ANN401
    """Restore the original string only for a declared string tool."""
    if is_string_tool(name, tools) and isinstance(value, dict) and TOOL_INPUT_FIELD in value:
        return value[TOOL_INPUT_FIELD]
    return value


def wrap_string_input(name: str, value: Any, tools: Sequence[ToolDefinition] | None) -> Any:  # noqa: ANN401
    """Replay a prior bare-string call using the same object wire schema."""
    if is_string_tool(name, tools) and not (isinstance(value, dict) and TOOL_INPUT_FIELD in value):
        return {TOOL_INPUT_FIELD: value}
    return value
