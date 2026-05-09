# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
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

"""Internal testing utilities for Genkit AI (mock models, test_models)."""

import json
from copy import deepcopy
from typing import Any, TypedDict

from pydantic import BaseModel, Field

from genkit._core._action import Action, ActionKind, ActionRunContext
from genkit._core._tracing import run_in_new_span
from genkit._core._typing import (
    Media,
    MediaPart,
    ModelInfo,
    Part,
    Role,
    SpanMetadata,
    TextPart,
)
from genkit.model import Message, ModelRequest, ModelResponse, ModelResponseChunk

from ._aio import Genkit


class ProgrammableModel:
    """A configurable model implementation for testing."""

    def __init__(self) -> None:
        self._request_idx: int = 0
        self.request_count: int = 0
        self.responses: list[ModelResponse] = []
        self.chunks: list[list[ModelResponseChunk]] | None = None
        self.last_request: ModelRequest | None = None

    def reset(self) -> None:
        self._request_idx = 0
        self.request_count = 0
        self.responses = []
        self.chunks = None
        self.last_request = None

    async def model_fn(
        self,
        request: ModelRequest,
        ctx: ActionRunContext,
    ) -> ModelResponse:
        self.last_request = deepcopy(request)
        self.request_count += 1

        response = self.responses[self._request_idx]
        if self.chunks and self._request_idx < len(self.chunks):
            for chunk in self.chunks[self._request_idx]:
                ctx.send_chunk(chunk)
        self._request_idx += 1
        return response


def define_programmable_model(
    ai: Genkit,
    name: str = 'programmableModel',
) -> tuple[ProgrammableModel, Action]:
    pm = ProgrammableModel()

    async def model_fn(
        request: ModelRequest,
        ctx: ActionRunContext,
    ) -> ModelResponse:
        return await pm.model_fn(request, ctx)

    action = ai.define_model(name=name, fn=model_fn)

    return (pm, action)


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
        self.last_request = request

        merged_txt = ''
        messages = request.messages.root if hasattr(request.messages, 'root') else request.messages  # pyright: ignore[reportAttributeAccessIssue]
        for m in messages:  # ty: ignore[not-iterable]
            merged_txt += f' {m.role}: ' + ','.join(
                json.dumps(p.root.text) if p.root.text is not None else '""' for p in m.content
            )
        echo_resp = f'[ECHO]{merged_txt}'

        if request.config:
            if hasattr(request.config, 'model_dump_json'):
                config_json = request.config.model_dump_json()
            else:
                config_json = json.dumps(request.config, separators=(',', ':'))
        else:
            config_json = '{}'
        if request.config and config_json != '{}':
            echo_resp += f' {config_json}'
        tools_list = request.tools.root if hasattr(request.tools, 'root') else request.tools  # pyright: ignore[reportAttributeAccessIssue,reportOptionalMemberAccess]
        if tools_list:
            echo_resp += f' tools={",".join(t.name for t in tools_list)}'  # ty: ignore[not-iterable]
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
                ctx.send_chunk(
                    ModelResponseChunk(role=Role.MODEL, index=i, content=[Part(root=TextPart(text=countdown))])
                )

        return ModelResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text=echo_resp))]))


def define_echo_model(
    ai: Genkit,
    name: str = 'echoModel',
    stream_countdown: bool = False,
) -> tuple[EchoModel, Action]:
    echo = EchoModel(stream_countdown=stream_countdown)

    async def model_fn(
        request: ModelRequest,
        ctx: ActionRunContext,
    ) -> ModelResponse:
        return await echo.model_fn(request, ctx)

    action = ai.define_model(name=name, fn=model_fn)

    return (echo, action)


class StaticResponseModel:
    """A model that always returns the same static response."""

    def __init__(self, message: dict[str, Any]) -> None:
        self.response_message: Message = Message.model_validate(message)
        self.last_request: ModelRequest | None = None
        self.request_count: int = 0

    async def model_fn(
        self,
        request: ModelRequest,
        _ctx: ActionRunContext,
    ) -> ModelResponse:
        self.last_request = request
        self.request_count += 1
        return ModelResponse(message=self.response_message)


def define_static_response_model(
    ai: Genkit,
    message: dict[str, Any],
    name: str = 'staticModel',
) -> tuple[StaticResponseModel, Action]:
    static = StaticResponseModel(message)

    async def model_fn(
        request: ModelRequest,
        ctx: ActionRunContext,
    ) -> ModelResponse:
        return await static.model_fn(request, ctx)

    action = ai.define_model(name=name, fn=model_fn)

    return (static, action)


class SkipTestError(Exception):
    """Exception raised to skip a test case."""


def skip() -> None:
    raise SkipTestError()


class ModelTestError(TypedDict, total=False):
    message: str
    stack: str | None


class ModelTestResult(TypedDict, total=False):
    name: str
    passed: bool
    skipped: bool
    error: ModelTestError


class TestCaseReport(TypedDict):
    description: str
    models: list[ModelTestResult]


TestReport = list[TestCaseReport]


class GablorkenInput(BaseModel):
    value: float = Field(..., description='The value to calculate gablorken for')


async def test_models(ai: Genkit, models: list[str]) -> TestReport:
    """Run a standard test suite against one or more models."""

    @ai.tool(name='gablorkenTool')
    async def gablorken_tool(input: GablorkenInput) -> float:
        """Calculate the gablorken of a value."""
        return (input.value**3) + 1.407

    async def get_model_info(model_name: str) -> ModelInfo | None:
        model_action = await ai.registry.resolve_action(ActionKind.MODEL, model_name)
        if model_action and model_action.metadata:
            info_obj = model_action.metadata.get('model')
            if isinstance(info_obj, ModelInfo):
                return info_obj
        return None

    async def test_basic_hi(model: str) -> None:
        response = await ai.generate(model=model, prompt='just say "Hi", literally')
        got = response.text.strip()
        assert 'hi' in got.lower(), f'Expected "Hi" in response, got: {got}'

    async def test_multimodal(model: str) -> None:
        info = await get_model_info(model)
        if not (info and info.supports and info.supports.media):
            skip()

        test_image = (
            'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2'
            'AAABhGlDQ1BJQ0MgcHJvZmlsZQAAKJF9kT1Iw0AcxV9TpSoVETOIOGSoulgQFXHU'
            'KhShQqgVWnUwufRDaNKQtLg4Cq4FBz8Wqw4uzro6uAqC4AeIs4OToouU+L+k0CLG'
            'g+N+vLv3uHsHCLUi0+22MUA3ylYyHpPSmRUp9IpOhCCiFyMKs81ZWU7Ad3zdI8DX'
            'uyjP8j/35+jWsjYDAhLxDDOtMvE68dRm2eS8TyyygqIRnxOPWnRB4keuqx6/cc67'
            'LPBM0Uol54hFYinfwmoLs4KlE08SRzTdoHwh7bHGeYuzXqywxj35C8NZY3mJ6zQH'
            'EccCFiFDgooKNlBEGVFaDVJsJGk/5uMfcP0yuVRybYCRYx4l6FBcP/gf/O7Wzk2M'
            'e0nhGND+4jgfQ0BoF6hXHef72HHqJ0DwGbgymv5SDZj+JL3a1CJHQM82cHHd1NQ9'
            '4HIH6H8yFUtxpSBNIZcD3s/omzJA3y3Qter11tjH6QOQoq4SN8DBITCcp+w1n3d3'
            'tPb275lGfz9aC3Kd0jYiSQAAAAlwSFlzAAAuIwAALiMBeKU/dgAAAAd0SU1FB+gJ'
            'BxQRO1/5qB8AAAAZdEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIEdJTVBXgQ4XAAAA'
            'sUlEQVQoz61SMQqEMBDcO5SYToUE/IBPyRMCftAH+INUviApUwYjNkKCVcTiQK7I'
            'HSw45czODrMswCOQUkopEQZjzDiOWemdZfu+b5oGYYgx1nWNMPwB2vACAK01Y4wQ'
            '8qGqqirL8jzPlNI9t64r55wQUgBA27be+xDCfaJhGJxzSqnv3UKIn7ne+2VZEB2s'
            'tZRSRLN93+d5RiRs28Y5RySEEI7jyEpFlp2mqeu6Zx75ApQwPdsIcq0ZAAAAAElF'
            'TkSuQmCC'
        )

        response = await ai.generate(
            model=model,
            prompt=[
                Part(root=MediaPart(media=Media(url=test_image))),
                Part(root=TextPart(text='what math operation is this? plus, minus, multiply or divide?')),
            ],
        )
        got = response.text.strip().lower()
        assert 'plus' in got, f'Expected "plus" in response, got: {got}'

    async def test_history(model: str) -> None:
        info = await get_model_info(model)
        if not (info and info.supports and info.supports.multiturn):
            skip()

        response1 = await ai.generate(model=model, prompt='My name is Glorb')
        response2 = await ai.generate(
            model=model,
            prompt="What's my name?",
            messages=response1.messages,
        )
        got = response2.text.strip()
        assert 'Glorb' in got, f'Expected "Glorb" in response, got: {got}'

    async def test_system_prompt(model: str) -> None:
        response = await ai.generate(
            model=model,
            prompt='Hi',
            messages=[
                Message.model_validate({
                    'role': 'system',
                    'content': [{'text': 'If the user says "Hi", just say "Bye"'}],
                }),
            ],
        )
        got = response.text.strip()
        assert 'Bye' in got, f'Expected "Bye" in response, got: {got}'

    async def test_structured_output(model: str) -> None:
        class PersonInfo(BaseModel):
            name: str
            occupation: str

        response = await ai.generate(
            model=model,
            prompt='extract data as json from: Jack was a Lumberjack',
            output_schema=PersonInfo,
        )
        got = response.output
        assert got is not None, 'Expected structured output'
        if isinstance(got, BaseModel):
            got = got.model_dump()

        assert isinstance(got, dict), f'Expected output to be a dict or BaseModel, got {type(got)}'
        assert got.get('name') == 'Jack', f"Expected name='Jack', got: {got.get('name')}"
        assert got.get('occupation') == 'Lumberjack', f"Expected occupation='Lumberjack', got: {got.get('occupation')}"

    async def test_tool_calling(model: str) -> None:
        info = await get_model_info(model)
        if not (info and info.supports and info.supports.tools):
            skip()

        response = await ai.generate(
            model=model,
            prompt='what is a gablorken of 2? use provided tool',
            tools=['gablorkenTool'],
        )
        got = response.text.strip()
        assert '9.407' in got, f'Expected "9.407" in response, got: {got}'

    tests: dict[str, Any] = {
        'basic hi': test_basic_hi,
        'multimodal': test_multimodal,
        'history': test_history,
        'system prompt': test_system_prompt,
        'structured output': test_structured_output,
        'tool calling': test_tool_calling,
    }

    report: TestReport = []

    with run_in_new_span(SpanMetadata(name='testModels'), labels={'genkit:type': 'testSuite'}):
        for test_name, test_fn in tests.items():
            with run_in_new_span(SpanMetadata(name=test_name), labels={'genkit:type': 'testCase'}):
                case_report: TestCaseReport = {
                    'description': test_name,
                    'models': [],
                }

                for model in models:
                    model_result: ModelTestResult = {
                        'name': model,
                        'passed': True,
                    }

                    try:
                        await test_fn(model)
                    except SkipTestError:
                        model_result['passed'] = False
                        model_result['skipped'] = True
                    except AssertionError as e:
                        model_result['passed'] = False
                        model_result['error'] = {
                            'message': str(e),
                            'stack': None,
                        }
                    except Exception as e:
                        model_result['passed'] = False
                        model_result['error'] = {
                            'message': str(e),
                            'stack': None,
                        }

                    case_report['models'].append(model_result)

                report.append(case_report)

    return report
