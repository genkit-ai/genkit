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

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from genkit import MultipartToolResponse
from genkit._ai._tools import Interrupt
from genkit._core._action import ActionKind
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, ToolHookParams
from genkit.telemetry import SpanContext, run_in_new_span


class ToolApprovalConfig(BaseModel):
    """Tools that may run without an approval interrupt."""

    allowed_tools: list[str] = Field(default_factory=list)


class ToolApprovalMiddleware(BaseMiddleware):
    """Requires approval before a tool runs, unless the tool is on allowed_tools."""

    def __init__(self, config: ToolApprovalConfig | None = None) -> None:
        self.config = config or ToolApprovalConfig()

    async def on_tool_call(
        self,
        context: GenerateMiddlewareContext,
        params: ToolHookParams,
        next: Callable[[ToolHookParams], Awaitable[MultipartToolResponse]],
    ) -> MultipartToolResponse:
        """Interrupts with tool approval request if tool is not in allowed list."""
        tool_name = params.tool.name if hasattr(params.tool, 'name') else str(params.tool)

        if tool_name in self.config.allowed_tools:
            return await next(params)

        metadata = getattr(params.context, 'resumed_metadata', None)
        if metadata:
            decision = metadata.get('decision')
            if decision == 'approved':
                return await next(params)
            if decision == 'rejected':
                reason = metadata.get('reason', 'Tool execution rejected by user')
                return MultipartToolResponse(output=f'Tool execution rejected: {reason}')

        async def _call(span: SpanContext) -> MultipartToolResponse:
            span.set_metadata({'tool_approval': {'tool': tool_name}})
            raise Interrupt(
                metadata={
                    'type': 'tool_approval',
                    'tool': tool_name,
                    'input': params.input,
                }
            )

        return await run_in_new_span('tool_approval', _call, action_type=ActionKind.CUSTOM)
