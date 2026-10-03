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

"""Mock models for testing Genkit flows, agents, and plugins."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any, cast

from pydantic import BaseModel

from genkit import ActionRunContext, Genkit, Message, ModelResponse, ModelResponseChunk, Part
from genkit._core._action import Action
from genkit._core._typing import Role
from genkit.model import ModelRequest

ResponseCallback = Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]] | ModelResponse[Any]]


class ScriptedModel:
    """Answers turn N with ``responses[N]`` (streaming ``chunks[N]`` first) and records requests."""

    def __init__(
        self,
        responses: list[ModelResponse] | None = None,
        chunks: list[list[ModelResponseChunk]] | None = None,
        response_cb: ResponseCallback | None = None,
    ) -> None:
        self._request_idx: int = 0
        self.request_count: int = 0
        self.responses: list[ModelResponse] = list(responses) if responses is not None else []
        self.chunks: list[list[ModelResponseChunk]] | None = chunks
        self.last_request: ModelRequest | None = None
        self.response_cb: ResponseCallback | None = response_cb

    def reset(self) -> None:
        """Reset request index, count, and recorded requests."""
        self._request_idx = 0
        self.request_count = 0
        self.responses = []
        self.chunks = None
        self.last_request = None
        self.response_cb = None

    async def model_fn(
        self,
        request: ModelRequest,
        ctx: ActionRunContext,
    ) -> ModelResponse:
        """Handle a model execution turn."""
        self.last_request = deepcopy(request)
        self.request_count += 1

        if self.response_cb is not None:
            res = self.response_cb(request)
            if inspect.isawaitable(res):
                response = await res
            else:
                response = res
        else:
            if self._request_idx >= len(self.responses):
                raise IndexError(
                    f'ScriptedModel received request {self._request_idx + 1}, '
                    f'but only {len(self.responses)} responses were configured.'
                )
            response = self.responses[self._request_idx]

        if self.chunks and self._request_idx < len(self.chunks):
            for chunk in self.chunks[self._request_idx]:
                ctx.send_chunk(chunk)

        self._request_idx += 1
        return cast(ModelResponse[object], response)


def define_scripted_model(
    ai: Genkit,
    name: str = 'scriptedModel',
    *,
    responses: list[ModelResponse] | None = None,
    chunks: list[list[ModelResponseChunk]] | None = None,
    response_cb: ResponseCallback | None = None,
) -> tuple[ScriptedModel, Action]:
    """Register a ScriptedModel on the given Genkit instance."""
    model = ScriptedModel(responses=responses, chunks=chunks, response_cb=response_cb)

    async def model_fn(
        request: ModelRequest,
        ctx: ActionRunContext,
    ) -> ModelResponse:
        return await model.model_fn(request, ctx)

    action = ai.define_model(name=name, fn=model_fn)
    return (model, action)


# Backward-compatible aliases for tests written against ProgrammableModel
ProgrammableModel = ScriptedModel


def define_programmable_model(
    ai: Genkit,
    name: str = 'programmableModel',
    *,
    responses: list[ModelResponse] | None = None,
    chunks: list[list[ModelResponseChunk]] | None = None,
    response_cb: ResponseCallback | None = None,
) -> tuple[ProgrammableModel, Action]:
    """Register a ScriptedModel under the legacy ProgrammableModel alias."""
    return define_scripted_model(ai, name=name, responses=responses, chunks=chunks, response_cb=response_cb)


class EchoModel:
    """A model implementation that echoes back the input with metadata."""

    def __init__(self, stream_countdown: bool = False) -> None:
        self.last_request: ModelRequest | None = None
        self.stream_countdown: bool = stream_countdown

    async def model_fn(
        self,
        request: ModelRequest,
        ctx: ActionRunContext,
    ) -> ModelResponse:
        """Echo the request messages, tool choice, and config back in the response text."""
        self.last_request = request

        merged_txt = ''
        for m in request.messages:
            merged_txt += f' {m.role}: ' + ','.join(
                json.dumps(p.text) if p.text is not None else '""' for p in m.content
            )
        echo_resp = f'[ECHO]{merged_txt}'

        if request.config:
            if isinstance(request.config, BaseModel):
                config_json = request.config.model_dump_json()
            else:
                config_json = json.dumps(request.config, separators=(',', ':'))
        else:
            config_json = '{}'
        if request.config and config_json != '{}':
            echo_resp += f' {config_json}'
        if request.tools:
            echo_resp += f' tools={",".join(t.name for t in request.tools)}'
        if request.tool_choice is not None:
            echo_resp += f' tool_choice={request.tool_choice}'
        output_dict: dict[str, object] = {}
        if request.output_format:
            output_dict['format'] = request.output_format
        if request.output_schema:
            output_dict['schema'] = request.output_schema
        if request.output_constrained is not None:
            output_dict['constrained'] = request.output_constrained
        if request.output_content_type:
            output_dict['contentType'] = request.output_content_type
        output_json = json.dumps(output_dict, separators=(',', ':')) if output_dict else '{}'
        if output_dict and output_json != '{}':
            echo_resp += f' output={output_json}'

        if self.stream_countdown:
            for i, countdown in enumerate(['3', '2', '1']):
                ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, index=i, content=[Part.from_text(countdown)]))

        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(echo_resp)]))


def define_echo_model(
    ai: Genkit,
    name: str = 'echoModel',
    *,
    stream_countdown: bool = False,
    config_schema: type[BaseModel] | None = None,
) -> tuple[EchoModel, Action]:
    """Register an EchoModel on the given Genkit instance."""
    echo = EchoModel(stream_countdown=stream_countdown)

    async def model_fn(
        request: ModelRequest,
        ctx: ActionRunContext,
    ) -> ModelResponse:
        return await echo.model_fn(request, ctx)

    action = ai.define_model(name=name, fn=model_fn, config_schema=config_schema)
    return (echo, action)


class StaticResponseModel:
    """A model that always returns the same static response."""

    def __init__(self, message: Message | dict[str, Any] | str) -> None:
        if isinstance(message, str):
            self.response_message = Message(role=Role.MODEL, content=[Part.from_text(message)])
        elif isinstance(message, dict):
            self.response_message = Message.model_validate(message)
        else:
            self.response_message = message
        self.last_request: ModelRequest | None = None
        self.request_count: int = 0

    async def model_fn(
        self,
        request: ModelRequest,
        _ctx: ActionRunContext,
    ) -> ModelResponse:
        """Return the preconfigured static message."""
        self.last_request = request
        self.request_count += 1
        return ModelResponse(message=self.response_message)


def define_static_response_model(
    ai: Genkit,
    message: Message | dict[str, Any] | str,
    name: str = 'staticModel',
) -> tuple[StaticResponseModel, Action]:
    """Register a StaticResponseModel that replies with a fixed message."""
    static = StaticResponseModel(message)

    async def model_fn(
        request: ModelRequest,
        ctx: ActionRunContext,
    ) -> ModelResponse:
        return await static.model_fn(request, ctx)

    action = ai.define_model(name=name, fn=model_fn)
    return (static, action)
