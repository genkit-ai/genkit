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

"""Tool approval middleware for Genkit.

Requires explicit approval for tool calls by interrupting execution and waiting
for the caller to approve and resume. Tools in the allowed list bypass approval.
Useful for sensitive operations or user confirmation flows.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from genkit._ai._tools import Interrupt
from genkit._core._tracing import run_in_new_span
from genkit._core._typing import SpanMetadata, ToolRequestPart
from genkit.middleware import BaseMiddleware, MultipartToolResponse, ToolHookParams, middleware


@middleware(name='tool_approval', description='Requires approval before executing tools')
class ToolApproval(BaseMiddleware):
    """Tool approval middleware that interrupts execution for non-allowed tools.

    Interrupts tool execution unless the tool is in the allowed list or the call
    is being resumed with explicit approval metadata (``resumed.toolApproved: true``).
    An empty ``allowed_tools`` list requires approval for every tool call.

    Allowed-list entries must match the plain tool name as passed to ``define_tool``
    (e.g. ``'search'``), not the full registry key (e.g. ``'/tool/search'``).
    """

    allowed_tools: list[str] = Field(default_factory=list)

    async def wrap_tool(
        self,
        params: ToolHookParams,
        next_fn: Callable[
            [ToolHookParams],
            Awaitable[tuple[MultipartToolResponse | None, ToolRequestPart | None]],
        ],
    ) -> tuple[MultipartToolResponse | None, ToolRequestPart | None]:
        """Intercept tool execution and require approval if not in allowed list."""
        tool_name = params.tool.name

        # Tools in the allowed list run without interruption.
        if tool_name in self.allowed_tools:
            return await next_fn(params)

        # Check if this is an approved resume (caller must opt in explicitly;
        # Bare resume metadata alone is not treated as approval; toolApproved must be explicit.
        metadata = params.tool_request_part.metadata or {}
        resumed = metadata.get('resumed')
        if isinstance(resumed, dict) and resumed.get('toolApproved'):
            return await next_fn(params)

        # Emit a tool-shaped span so the interrupt is attributed to the tool in traces.
        # From the trace tree, the consumer of Dev UI should see that a tool action span
        # was created and immediately interrupted.
        tool_input = params.tool_request_part.tool_request.input
        with run_in_new_span(
            SpanMetadata(name=tool_name, input=tool_input),
            labels={
                'genkit:type': 'action',
                'genkit:metadata:subtype': 'tool',
                'genkit:name': tool_name,
            },
        ) as span:
            if tool_input is not None:
                inp_json = tool_input.model_dump_json() if isinstance(tool_input, BaseModel) else json.dumps(tool_input)
                span.set_attribute('genkit:input', inp_json)
            raise Interrupt({'message': f'Tool not in approved list: {tool_name}'})
