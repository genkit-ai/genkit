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

"""Tests for ToolApproval middleware."""

import pytest

from genkit._ai._tools import Interrupt
from genkit._core._action import Action
from genkit._core._model import ToolHookParams
from genkit._core._typing import ToolRequest, ToolRequestPart, ToolResponsePart
from genkit.plugins.middleware import ToolApproval


@pytest.mark.asyncio
async def test_tool_approval_allowed_tool():
    """Test that allowed tools pass through without approval."""
    approval = ToolApproval(allowed_tools=['get_weather'])
    
    async def next_fn(params):
        return (ToolResponsePart(tool_response=None), None)
    
    tool = Action(name='get_weather')
    tool_request = ToolRequest(name='get_weather', input={})
    tool_request_part = ToolRequestPart(tool_request=tool_request)
    params = ToolHookParams(tool_request_part=tool_request_part, tool=tool)
    
    result = await approval.wrap_tool(params, next_fn)
    assert result is not None


@pytest.mark.asyncio
async def test_tool_approval_non_allowed_tool():
    """Test that non-allowed tools raise Interrupt."""
    approval = ToolApproval(allowed_tools=['get_weather'])
    
    async def next_fn(params):
        return (ToolResponsePart(tool_response=None), None)
    
    tool = Action(name='delete_database')
    tool_request = ToolRequest(name='delete_database', input={})
    tool_request_part = ToolRequestPart(tool_request=tool_request)
    params = ToolHookParams(tool_request_part=tool_request_part, tool=tool)
    
    with pytest.raises(Interrupt) as exc_info:
        await approval.wrap_tool(params, next_fn)
    assert 'delete_database' in exc_info.value.metadata['message']


@pytest.mark.asyncio
async def test_tool_approval_resumed_with_approval():
    """Test that resumed tools with approval metadata pass through."""
    approval = ToolApproval(allowed_tools=[])
    
    async def next_fn(params):
        return (ToolResponsePart(tool_response=None), None)
    
    tool = Action(name='some_tool')
    tool_request = ToolRequest(name='some_tool', input={})
    tool_request_part = ToolRequestPart(
        tool_request=tool_request,
        metadata={'resumed': {'toolApproved': True}}
    )
    params = ToolHookParams(tool_request_part=tool_request_part, tool=tool)
    
    result = await approval.wrap_tool(params, next_fn)
    assert result is not None


@pytest.mark.asyncio
async def test_tool_approval_empty_allowed_list():
    """Test that empty allowed list requires approval for all tools."""
    approval = ToolApproval(allowed_tools=[])
    
    async def next_fn(params):
        return (ToolResponsePart(tool_response=None), None)
    
    tool = Action(name='any_tool')
    tool_request = ToolRequest(name='any_tool', input={})
    tool_request_part = ToolRequestPart(tool_request=tool_request)
    params = ToolHookParams(tool_request_part=tool_request_part, tool=tool)
    
    with pytest.raises(Interrupt):
        await approval.wrap_tool(params, next_fn)
