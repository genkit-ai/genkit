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

from __future__ import annotations

from collections.abc import Mapping

import pytest
from genkit_otel import ContentCapturingMode, GenAiInstrumentation
from genkit_otel._gen_ai_attributes import (
    CAPTURE_CONTENT_ENV_VAR,
    GEN_AI_OPERATION_DETAILS_EVENT,
    GenAiAttr,
    GenkitAttr,
)
from opentelemetry.trace import StatusCode

from genkit import FinishReason
from genkit._core._instrumentation_api import SpanNext
from genkit._core._model import OutputConfig
from genkit._core._typing import (
    Candidate,
    GenerationUsage,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)
from genkit.model import Message, ModelRequest, ModelResponse
from genkit.telemetry import SpanMetadata


def _model_request(
    *,
    config: Mapping[str, object] | None = None,
    messages: list[Message] | None = None,
    output: OutputConfig | None = None,
) -> ModelRequest:
    kwargs: dict[str, object] = {
        'messages': messages or [Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
    }
    if config is not None:
        kwargs['config'] = config
    if output is not None:
        kwargs['output'] = output
    return ModelRequest(**kwargs)


def _model_response(
    *,
    finish_reason: FinishReason | None = None,
    message: Message | None = None,
    usage: GenerationUsage | None = None,
) -> ModelResponse:
    return ModelResponse(
        finish_reason=finish_reason or FinishReason.STOP,
        message=message or Message(role=Role.MODEL, content=[Part(root=TextPart(text='ok'))]),
        usage=usage,
    )


async def _run_model(
    instr: GenAiInstrumentation,
    name: str,
    input: object,
    fn: SpanNext[object],
) -> object:
    return await instr.run_in_new_span(
        SpanMetadata(name=name, action_type='model', input=input),
        fn,
    )


def _event_names(harness) -> list[str | None]:
    names: list[str | None] = []
    for record in harness.logs.get_finished_logs():
        inner = getattr(record, 'log_record', record)
        names.append(getattr(inner, 'event_name', None))
    return names


@pytest.mark.asyncio
async def test_emits_chat_span_with_request_provider_model(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(config={'temperature': 0.5, 'topK': 40, 'maxOutputTokens': 100}),
        lambda span=None: _awaitable(_model_response()),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert harness.attr(span, GenAiAttr.OPERATION_NAME) == 'chat'
    assert harness.attr(span, GenAiAttr.PROVIDER_NAME) == 'gcp.gemini'
    assert harness.attr(span, GenAiAttr.REQUEST_MODEL) == 'gemini-flash-latest'
    assert harness.attr(span, GenAiAttr.REQUEST_TEMPERATURE) == 0.5
    assert harness.attr(span, GenAiAttr.REQUEST_TOP_K) == 40
    assert harness.attr(span, GenAiAttr.REQUEST_MAX_TOKENS) == 100


@pytest.mark.asyncio
async def test_records_usage_and_finish_reason(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'openai/gpt-x',
        _model_request(),
        lambda span=None: _awaitable(
            _model_response(
                finish_reason=FinishReason.LENGTH,
                usage=GenerationUsage(input_tokens=10, output_tokens=20),
            )
        ),
    )
    span = harness.span_named('chat gpt-x')
    assert span is not None
    assert harness.attr(span, GenAiAttr.PROVIDER_NAME) == 'openai'
    assert harness.attr(span, GenAiAttr.USAGE_INPUT_TOKENS) == 10
    assert harness.attr(span, GenAiAttr.USAGE_OUTPUT_TOKENS) == 20
    assert list(harness.attr(span, GenAiAttr.RESPONSE_FINISH_REASONS)) == ['length']


@pytest.mark.asyncio
async def test_reports_tool_calls_when_response_has_tool_request(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(),
        lambda span=None: _awaitable(
            _model_response(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='lookup', input={})))],
                )
            )
        ),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert list(harness.attr(span, GenAiAttr.RESPONSE_FINISH_REASONS)) == ['tool_calls']


@pytest.mark.asyncio
async def test_records_error_status_and_type_on_throw(harness) -> None:
    instr = harness.instrumentation()

    async def boom(span=None):
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await _run_model(instr, 'googleai/gemini-flash-latest', _model_request(), boom)

    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert span.status.status_code == StatusCode.ERROR
    assert harness.attr(span, GenAiAttr.ERROR_TYPE) == 'RuntimeError'


@pytest.mark.asyncio
async def test_does_not_capture_content_by_default(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='secret'))])],
        ),
        lambda span=None: _awaitable(_model_response()),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert harness.attr(span, GenAiAttr.INPUT_MESSAGES) is None


@pytest.mark.asyncio
async def test_span_only_captures_content_on_span_not_event(harness) -> None:
    instr = harness.instrumentation(content_capturing_mode=ContentCapturingMode.SPAN_ONLY)
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(
            messages=[
                Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='Be nice'))]),
                Message(role=Role.USER, content=[Part(root=TextPart(text='hello'))]),
            ]
        ),
        lambda span=None: _awaitable(_model_response()),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    input_messages = harness.attr(span, GenAiAttr.INPUT_MESSAGES)
    assert isinstance(input_messages, str)
    assert 'hello' in input_messages
    assert 'Be nice' in harness.attr(span, GenAiAttr.SYSTEM_INSTRUCTIONS)
    assert 'ok' in harness.attr(span, GenAiAttr.OUTPUT_MESSAGES)
    assert GEN_AI_OPERATION_DETAILS_EVENT not in _event_names(harness)


@pytest.mark.asyncio
async def test_event_only_emits_event_not_span_content(harness) -> None:
    instr = harness.instrumentation(content_capturing_mode=ContentCapturingMode.EVENT_ONLY)
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hello'))])]),
        lambda span=None: _awaitable(_model_response()),
    )
    assert GEN_AI_OPERATION_DETAILS_EVENT in _event_names(harness)
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert harness.attr(span, GenAiAttr.INPUT_MESSAGES) is None


@pytest.mark.asyncio
async def test_span_and_event_writes_both(harness) -> None:
    instr = harness.instrumentation(content_capturing_mode=ContentCapturingMode.SPAN_AND_EVENT)
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hello'))])]),
        lambda span=None: _awaitable(_model_response()),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert 'hello' in harness.attr(span, GenAiAttr.INPUT_MESSAGES)
    assert GEN_AI_OPERATION_DETAILS_EVENT in _event_names(harness)


@pytest.mark.asyncio
async def test_nests_flow_and_model_into_one_trace(harness) -> None:
    instr = harness.instrumentation()

    async def flow_body(span=None):
        return await _run_model(
            instr,
            'googleai/gemini-flash-latest',
            _model_request(),
            lambda inner=None: _awaitable(_model_response()),
        )

    await instr.run_in_new_span(SpanMetadata(name='myFlow', action_type='flow'), flow_body)
    flow = harness.span_named('myFlow')
    model = harness.span_named('chat gemini-flash-latest')
    assert flow is not None and model is not None
    assert harness.attr(flow, GenkitAttr.ACTION_TYPE) == 'flow'
    assert model.context.trace_id == flow.context.trace_id


@pytest.mark.asyncio
async def test_execute_tool_spans_only_when_enabled(harness) -> None:
    off = harness.instrumentation()

    async def sunny(span=None):
        return 'sunny'

    await off.run_in_new_span(SpanMetadata(name='weather', action_type='tool'), sunny)
    assert harness.span_named('execute_tool weather') is None

    on = harness.instrumentation(emit_tool_spans=True)
    await on.run_in_new_span(SpanMetadata(name='weather', action_type='tool'), sunny)
    span = harness.span_named('execute_tool weather')
    assert span is not None
    assert harness.attr(span, GenAiAttr.OPERATION_NAME) == 'execute_tool'
    assert harness.attr(span, GenAiAttr.TOOL_NAME) == 'weather'


@pytest.mark.asyncio
async def test_does_not_capture_action_io_by_default(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(),
        lambda span=None: _awaitable(_model_response()),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert harness.attr(span, GenkitAttr.INPUT) is None
    assert harness.attr(span, GenkitAttr.OUTPUT) is None


@pytest.mark.asyncio
async def test_capture_action_io_records_raw_io_on_all_span_types(harness) -> None:
    instr = harness.instrumentation(capture_action_io=True, emit_tool_spans=True)

    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(),
        lambda span=None: _awaitable(_model_response()),
    )
    model = harness.span_named('chat gemini-flash-latest')
    assert model is not None
    assert isinstance(harness.attr(model, GenkitAttr.INPUT), str)
    assert isinstance(harness.attr(model, GenkitAttr.OUTPUT), str)

    async def sunny(span=None):
        return 'sunny'

    await instr.run_in_new_span(
        SpanMetadata(name='weather', action_type='tool', input='Paris'),
        sunny,
    )
    tool = harness.span_named('execute_tool weather')
    assert tool is not None
    assert 'Paris' in harness.attr(tool, GenkitAttr.INPUT)
    assert 'sunny' in harness.attr(tool, GenkitAttr.OUTPUT)
    assert harness.attr(tool, GenAiAttr.INPUT_MESSAGES) is None
    assert harness.attr(tool, GenAiAttr.OUTPUT_MESSAGES) is None

    async def flow_out(span=None):
        return 'out'

    await instr.run_in_new_span(SpanMetadata(name='myFlow', action_type='flow', input='in'), flow_out)
    flow = harness.span_named('myFlow')
    assert flow is not None
    assert 'in' in harness.attr(flow, GenkitAttr.INPUT)
    assert 'out' in harness.attr(flow, GenkitAttr.OUTPUT)
    assert harness.attr(flow, GenAiAttr.INPUT_MESSAGES) is None


@pytest.mark.asyncio
async def test_captures_legacy_candidates_message(harness) -> None:
    instr = harness.instrumentation(content_capturing_mode=ContentCapturingMode.SPAN_ONLY)
    legacy = ModelResponse(
        finish_reason=FinishReason.STOP,
        candidates=[
            Candidate(
                index=0,
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='legacy answer'))]),
            )
        ],
    )
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(),
        lambda span=None: _awaitable(legacy),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert 'legacy answer' in harness.attr(span, GenAiAttr.OUTPUT_MESSAGES)


@pytest.mark.asyncio
async def test_reports_tool_calls_from_legacy_candidates(harness) -> None:
    instr = harness.instrumentation()
    legacy = ModelResponse(
        finish_reason=FinishReason.STOP,
        candidates=[
            Candidate(
                index=0,
                finish_reason=FinishReason.STOP,
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='lookup', input={})))],
                ),
            )
        ],
    )
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(),
        lambda span=None: _awaitable(legacy),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert list(harness.attr(span, GenAiAttr.RESPONSE_FINISH_REASONS)) == ['tool_calls']


@pytest.mark.asyncio
async def test_classifies_action_subtype_model_as_chat(harness) -> None:
    """Python action leftover: type=action, subtype=model still becomes a chat span."""
    instr = harness.instrumentation()

    async def body(span=None):
        return _model_response()

    await instr.run_in_new_span(
        SpanMetadata(
            name='googleai/gemini-flash-latest',
            action_type='action',
            subtype='model',
            input=_model_request(),
        ),
        body,
    )
    assert harness.span_named('chat gemini-flash-latest') is not None
    assert harness.span_named('googleai/gemini-flash-latest') is None


@pytest.mark.asyncio
async def test_explicit_mode_overrides_env(harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CAPTURE_CONTENT_ENV_VAR, 'SPAN_ONLY')
    instr = harness.instrumentation(content_capturing_mode=ContentCapturingMode.NO_CONTENT)
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='secret'))])]),
        lambda span=None: _awaitable(_model_response()),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert harness.attr(span, GenAiAttr.INPUT_MESSAGES) is None


@pytest.mark.asyncio
async def test_invalid_env_token_defaults_to_no_content(harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CAPTURE_CONTENT_ENV_VAR, 'bogus')
    instr = GenAiInstrumentation(
        tracer=harness.tracer_provider.get_tracer('test'),
        meter=harness.meter_provider.get_meter('test'),
        otel_logger=harness.logger_provider.get_logger('test'),
    )
    assert instr.content_capturing_mode is ContentCapturingMode.NO_CONTENT


@pytest.mark.asyncio
async def test_output_type_json_from_request(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(output=OutputConfig(format='json')),
        lambda span=None: _awaitable(_model_response()),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert harness.attr(span, GenAiAttr.OUTPUT_TYPE) == 'json'


async def _awaitable(value: object) -> object:
    return value
