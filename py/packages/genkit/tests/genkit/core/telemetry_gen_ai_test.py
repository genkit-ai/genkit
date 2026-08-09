#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GenAI semantic-convention mapping from Genkit spans."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genkit._core._model import Message, ModelResponse
from genkit._core._telemetry_handlers import otel_renderer
from genkit._core._trace._attrs import metadata_key
from genkit._core._tracing import (
    SpanMetadata,
    annotate,
    annotate_output,
    clear_genkit_telemetry_handlers,
    register_genkit_telemetry_handler,
    restore_default_telemetry_handlers,
    run_in_new_span,
)
from genkit._core._typing import FinishReason, GenerationUsage, Role, TextPart


@pytest.fixture(autouse=True)
def _reset_handlers() -> Generator[None, None, None]:
    restore_default_telemetry_handlers()
    try:
        yield
    finally:
        restore_default_telemetry_handlers()


@pytest.fixture
def exporter() -> Generator[InMemorySpanExporter, None, None]:
    provider = trace_api.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace_api.set_tracer_provider(provider)
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    exp.clear()
    yield exp
    exp.clear()


@pytest.mark.asyncio
async def test_action_subtype_model_maps_request_and_usage(exporter: InMemorySpanExporter) -> None:
    """Real model actions use type=action + subtype=model (not type=model)."""
    request = {
        'messages': [{'role': 'user', 'content': [{'text': 'hi'}]}],
        'config': {
            'temperature': 0.2,
            'topP': 0.9,
            'topK': 40,
            'maxOutputTokens': 256,
            'stopSequences': ['END'],
        },
        'tools': [
            {
                'name': 'getWeather',
                'description': 'Weather lookup',
                'inputSchema': {'type': 'object'},
            }
        ],
        'output': {'format': 'json'},
    }

    async def work() -> ModelResponse:
        resp = ModelResponse(
            message=Message(role=Role.MODEL, content=[TextPart(text='hello')]),
            finish_reason=FinishReason.STOP,
            usage=GenerationUsage(input_tokens=11, output_tokens=7, thoughts_tokens=3, cached_content_tokens=5),
            custom={'id': 'resp_123'},
        )
        annotate_output(resp)
        return resp

    await run_in_new_span(
        SpanMetadata(
            name='googleai/gemini-2.0-flash',
            type='action',
            subtype='model',
            input=request,
        ),
        work,
    )

    spans = list(exporter.get_finished_spans())
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})

    assert span.name == 'generate_content gemini-2.0-flash'
    assert span.kind == trace_api.SpanKind.CLIENT
    assert attrs['gen_ai.operation.name'] == 'generate_content'
    assert attrs['gen_ai.provider.name'] == 'gcp.gen_ai'
    assert attrs['gen_ai.request.model'] == 'gemini-2.0-flash'
    assert attrs['gen_ai.request.temperature'] == 0.2
    assert attrs['gen_ai.request.top_p'] == 0.9
    assert attrs['gen_ai.request.top_k'] == 40
    assert attrs['gen_ai.request.max_tokens'] == 256
    assert list(attrs['gen_ai.request.stop_sequences']) == ['END']
    assert attrs['gen_ai.output.type'] == 'json'
    assert 'getWeather' in str(attrs['gen_ai.tool.definitions'])
    # Message bodies stay off unless opted in.
    assert 'gen_ai.input.messages' not in attrs

    assert attrs['gen_ai.usage.input_tokens'] == 11
    assert attrs['gen_ai.usage.output_tokens'] == 7
    assert attrs['gen_ai.usage.reasoning.output_tokens'] == 3
    assert attrs['gen_ai.usage.cache_read.input_tokens'] == 5
    assert list(attrs['gen_ai.response.finish_reasons']) == ['stop']
    assert attrs['gen_ai.response.id'] == 'resp_123'
    assert attrs['gen_ai.response.model'] == 'gemini-2.0-flash'


@pytest.mark.asyncio
async def test_openai_model_uses_chat_operation(exporter: InMemorySpanExporter) -> None:
    async def work() -> str:
        return 'ok'

    await run_in_new_span(
        SpanMetadata(name='openai/gpt-4o', type='action', subtype='model', input={'messages': []}),
        work,
    )
    span = list(exporter.get_finished_spans())[0]
    attrs = dict(span.attributes or {})
    assert span.name == 'chat gpt-4o'
    assert attrs['gen_ai.operation.name'] == 'chat'
    assert attrs['gen_ai.provider.name'] == 'openai'
    assert attrs['gen_ai.request.model'] == 'gpt-4o'


@pytest.mark.asyncio
async def test_tool_and_agent_roles(exporter: InMemorySpanExporter) -> None:
    async def noop() -> None:
        return None

    await run_in_new_span(SpanMetadata(name='getWeather', type='action', subtype='tool'), noop)
    await run_in_new_span(SpanMetadata(name='weatherAgent', type='agent-turn'), noop)

    by_name = {s.name: dict(s.attributes or {}) for s in exporter.get_finished_spans()}
    assert by_name['getWeather']['gen_ai.operation.name'] == 'execute_tool'
    assert by_name['getWeather']['gen_ai.tool.name'] == 'getWeather'
    assert by_name['weatherAgent']['gen_ai.operation.name'] == 'invoke_agent'
    assert by_name['weatherAgent']['gen_ai.agent.name'] == 'weatherAgent'


@pytest.mark.asyncio
async def test_mid_annotate_session_projects_conversation_id(exporter: InMemorySpanExporter) -> None:
    """Mid-span Genkit facts project onto gen_ai via the conventions handler."""

    async def work() -> None:
        annotate(metadata_key('agent:sessionId'), 'sess_abc')

    await run_in_new_span(SpanMetadata(name='weatherAgent', type='agent-turn'), work)
    attrs = dict(list(exporter.get_finished_spans())[0].attributes or {})
    assert attrs['gen_ai.operation.name'] == 'invoke_agent'
    assert attrs['gen_ai.conversation.id'] == 'sess_abc'


@pytest.mark.asyncio
async def test_without_gen_ai_handler_annotate_does_not_project(exporter: InMemorySpanExporter) -> None:
    """annotate() stays convention-agnostic — no GenAI handler means no gen_ai.*."""
    clear_genkit_telemetry_handlers()
    register_genkit_telemetry_handler(otel_renderer)

    async def work() -> None:
        annotate(metadata_key('agent:sessionId'), 'sess_abc')

    await run_in_new_span(SpanMetadata(name='weatherAgent', type='agent-turn'), work)
    attrs = dict(list(exporter.get_finished_spans())[0].attributes or {})
    assert attrs[metadata_key('agent:sessionId')] == 'sess_abc'
    assert 'gen_ai.conversation.id' not in attrs
    assert 'gen_ai.operation.name' not in attrs


@pytest.mark.asyncio
async def test_message_content_opt_in(exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT', 'true')
    messages = [{'role': 'user', 'content': [{'text': 'secret'}]}]

    async def work() -> ModelResponse:
        resp = ModelResponse(
            message=Message(role=Role.MODEL, content=[TextPart(text='ok')]),
            finish_reason=FinishReason.STOP,
        )
        annotate_output(resp)
        return resp

    await run_in_new_span(
        SpanMetadata(name='openai/gpt-4o', type='action', subtype='model', input={'messages': messages}),
        work,
    )
    attrs = dict(list(exporter.get_finished_spans())[0].attributes or {})
    assert 'secret' in str(attrs.get('gen_ai.input.messages'))
    assert 'gen_ai.output.messages' in attrs
