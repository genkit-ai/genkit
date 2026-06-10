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

"""Tool approval middleware for Genkit."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from genkit._ai._tools import Interrupt
from genkit._core._tracing import SpanMetadata, run_in_new_span
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, MultipartToolResponse, ToolHookParams


class ToolApprovalConfig(BaseModel):
    """Tools that may run without an approval interrupt."""

    allowed_tools: list[str] = Field(default_factory=list)


class ToolApproval(BaseMiddleware[ToolApprovalConfig]):
    """Tool approval middleware that interrupts execution for non-allowed tools."""

    async def wrap_tool(
        self,
        params: ToolHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ToolHookParams, GenerateMiddlewareContext], Awaitable[MultipartToolResponse]],
    ) -> MultipartToolResponse:
        """Intercept tool execution and require approval if not in allowed list."""
        tool_name = params.tool.name

        if tool_name in self.config.allowed_tools:
            return await next_fn(params, ctx)

        metadata = params.tool_request_part.metadata or {}
        resumed = metadata.get('resumed')
        if isinstance(resumed, dict) and (resumed.get('toolApproved') or resumed.get('tool_approved')):
            return await next_fn(params, ctx)

        tool_input = params.tool_request_part.tool_request.input
        with run_in_new_span(
            SpanMetadata(name=tool_name, type='action', subtype='tool', input=tool_input),
        ) as span:
            if tool_input is not None:
                inp_json = tool_input.model_dump_json() if isinstance(tool_input, BaseModel) else json.dumps(tool_input)
                span.set_attribute('genkit:input', inp_json)
            raise Interrupt({'message': f'Tool not in approved list: {tool_name}'})
