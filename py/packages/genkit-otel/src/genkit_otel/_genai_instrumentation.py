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

"""Instrumentation that emits OpenTelemetry GenAI spans and metrics."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from typing import TypeVar

from opentelemetry import metrics as metrics_api, trace as trace_api
from opentelemetry.metrics import Meter
from opentelemetry.trace import Span, SpanKind, StatusCode, Tracer
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel

from genkit._core._instrumentation_api import SpanNext
from genkit.model import ModelRequest, ModelResponse
from genkit.telemetry import SpanMetadata
from genkit_otel._gen_ai_attributes import (
    CAPTURE_CONTENT_ENV_VAR,
    GEN_AI_OPERATION_DETAILS_EVENT,
    ContentCapturingMode,
    GenAiAttr,
    GenAiOperation,
    GenkitAttr,
    as_double,
    as_int,
    as_string_list,
    derive_output_type,
    derive_provider_name,
    map_finish_reason,
    parse_content_capturing_mode,
    split_model_name,
)
from genkit_otel._gen_ai_message_mapping import (
    is_tool_request_part,
    map_output_message,
    normalize_messages,
    resolve_response_message,
)
from genkit_otel._gen_ai_metrics import GenAiMetrics

logger = logging.getLogger('genkit_otel')

T = TypeVar('T')


class GenAiInstrumentation:
    """Emit ``gen_ai.*`` spans and metrics for Genkit model calls.

    The application owns the OpenTelemetry SDK: set a ``TracerProvider``
    (and optional meter/logger providers) before constructing ``Genkit``.
    When nothing is recording, this provider is a no-op.

    It composes with the Developer UI HTTP poster — they export to
    separate pipelines.

    Content capture is off by default. Prompt and reply text may
    contain PII, so a later reader only sees it when the caller opts in.
    """

    def __init__(
        self,
        *,
        content_capturing_mode: ContentCapturingMode | None = None,
        capture_action_io: bool = False,
        emit_tool_spans: bool = False,
        emit_metrics: bool = True,
        scope_name: str = 'genkit-genai',
        tracer: Tracer | None = None,
        meter: Meter | None = None,
        otel_logger: object | None = None,
    ) -> None:
        self.content_capturing_mode = (
            content_capturing_mode if content_capturing_mode is not None else _content_capturing_mode_from_env()
        )
        self.capture_action_io = capture_action_io
        self.emit_tool_spans = emit_tool_spans
        self.emit_metrics = emit_metrics
        self.scope_name = scope_name
        self._injected_tracer = tracer
        self._injected_meter = meter
        self._injected_logger = otel_logger
        self._cached_tracer: Tracer | None = None
        self._cached_metrics: GenAiMetrics | None = None
        self._cached_logger: object | None = None
        self._warned_not_initialized = False

    @property
    def _capture_on_span(self) -> bool:
        return self.content_capturing_mode in {
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        }

    @property
    def _capture_on_event(self) -> bool:
        return self.content_capturing_mode in {
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        }

    @property
    def _tracer(self) -> Tracer:
        if self._cached_tracer is None:
            self._cached_tracer = self._injected_tracer or trace_api.get_tracer(self.scope_name)
        return self._cached_tracer

    @property
    def _metrics(self) -> GenAiMetrics:
        if self._cached_metrics is None:
            meter = self._injected_meter or metrics_api.get_meter(self.scope_name)
            self._cached_metrics = GenAiMetrics(meter)
        return self._cached_metrics

    @property
    def _otel_logger(self) -> object | None:
        if self._injected_logger is not None:
            return self._injected_logger
        if self._cached_logger is None:
            self._cached_logger = _resolve_logger(self.scope_name)
        return self._cached_logger

    async def run_in_new_span(
        self,
        metadata: SpanMetadata,
        next: SpanNext[T],
    ) -> T:
        kind = _action_kind(metadata)
        if kind == 'model':
            return await self._run_model_span(metadata, next)
        if kind == 'tool' and self.emit_tool_spans:
            return await self._run_tool_span(metadata, next)
        return await self._run_generic_span(metadata, next)

    async def _run_model_span(
        self,
        metadata: SpanMetadata,
        next: SpanNext[T],
    ) -> T:
        prefix, model = split_model_name(metadata.name)
        provider = derive_provider_name(prefix)
        request = metadata.input if isinstance(metadata.input, ModelRequest) else None

        attrs: dict[str, AttributeValue] = {
            GenAiAttr.OPERATION_NAME: GenAiOperation.CHAT,
            GenAiAttr.REQUEST_MODEL: model,
        }
        if provider is not None:
            attrs[GenAiAttr.PROVIDER_NAME] = provider
        if request is not None:
            _add_request_config_attributes(attrs, request)

        metric_attrs: dict[str, AttributeValue] = {
            GenAiAttr.OPERATION_NAME: GenAiOperation.CHAT,
            GenAiAttr.REQUEST_MODEL: model,
        }
        if provider is not None:
            metric_attrs[GenAiAttr.PROVIDER_NAME] = provider

        started = time.perf_counter()
        with self._tracer.start_as_current_span(
            f'chat {model}',
            kind=SpanKind.CLIENT,
            attributes=attrs,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            self._maybe_warn_not_recording(span)
            try:
                output = await next(GenAiSpanContext(span))
                response = output if isinstance(output, ModelResponse) else None
                if response is not None:
                    self._add_response_attributes(span, response, failed=False)
                if self.content_capturing_mode != ContentCapturingMode.NO_CONTENT:
                    self._record_content(span, request, response)
                self._maybe_capture_action_io(span, metadata.input, output)
                if self.emit_metrics:
                    self._record_model_metrics(started, metric_attrs, response=response)
                return output
            except Exception as exc:
                self._record_error(span, exc)
                if self.emit_metrics:
                    self._record_model_metrics(
                        started,
                        metric_attrs,
                        error_type=type(exc).__name__,
                    )
                raise

    async def _run_tool_span(
        self,
        metadata: SpanMetadata,
        next: SpanNext[T],
    ) -> T:
        attrs: dict[str, AttributeValue] = {
            GenAiAttr.OPERATION_NAME: GenAiOperation.EXECUTE_TOOL,
            GenAiAttr.TOOL_NAME: metadata.name,
            GenAiAttr.TOOL_TYPE: 'function',
        }
        with self._tracer.start_as_current_span(
            f'execute_tool {metadata.name}',
            kind=SpanKind.INTERNAL,
            attributes=attrs,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            self._maybe_warn_not_recording(span)
            try:
                output = await next(GenAiSpanContext(span))
                self._maybe_capture_action_io(span, metadata.input, output)
                return output
            except Exception as exc:
                self._record_error(span, exc)
                raise

    async def _run_generic_span(
        self,
        metadata: SpanMetadata,
        next: SpanNext[T],
    ) -> T:
        attrs: dict[str, AttributeValue] = {}
        if metadata.action_type:
            attrs[GenkitAttr.ACTION_TYPE] = metadata.action_type
        with self._tracer.start_as_current_span(
            metadata.name,
            kind=SpanKind.INTERNAL,
            attributes=attrs,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            self._maybe_warn_not_recording(span)
            try:
                output = await next(GenAiSpanContext(span))
                self._maybe_capture_action_io(span, metadata.input, output)
                return output
            except Exception as exc:
                self._record_error(span, exc)
                raise

    def _record_model_metrics(
        self,
        started: float,
        base_attrs: dict[str, AttributeValue],
        *,
        response: ModelResponse | None = None,
        error_type: str | None = None,
    ) -> None:
        elapsed = time.perf_counter() - started
        usage = response.usage if response is not None else None
        if usage is not None:
            self._metrics.record_token_usage(
                base_attributes=base_attrs,
                input_tokens=as_int(usage.input_tokens),
                output_tokens=as_int(usage.output_tokens),
            )
        duration_attrs: dict[str, AttributeValue] = dict(base_attrs)
        if error_type is not None:
            duration_attrs[GenAiAttr.ERROR_TYPE] = error_type
        self._metrics.record_duration(elapsed, duration_attrs)

    def _maybe_capture_action_io(self, span: Span, input: object, output: object) -> None:
        if not self.capture_action_io:
            return
        _set_json_attribute(span, GenkitAttr.INPUT, input)
        _set_json_attribute(span, GenkitAttr.OUTPUT, output)

    def _add_response_attributes(
        self,
        span: Span,
        response: ModelResponse,
        *,
        failed: bool,
    ) -> None:
        finish_reasons = _resolve_finish_reasons(response, failed=failed)
        if finish_reasons:
            span.set_attribute(GenAiAttr.RESPONSE_FINISH_REASONS, finish_reasons)
        usage = response.usage
        if usage is None:
            return
        input_tokens = as_int(usage.input_tokens)
        if input_tokens is not None:
            span.set_attribute(GenAiAttr.USAGE_INPUT_TOKENS, input_tokens)
        output_tokens = as_int(usage.output_tokens)
        if output_tokens is not None:
            span.set_attribute(GenAiAttr.USAGE_OUTPUT_TOKENS, output_tokens)
        thoughts = as_int(usage.thoughts_tokens)
        if thoughts is not None:
            span.set_attribute(GenAiAttr.USAGE_REASONING_OUTPUT_TOKENS, thoughts)
        cached = as_int(usage.cached_content_tokens)
        if cached is not None:
            span.set_attribute(GenAiAttr.USAGE_CACHE_READ_INPUT_TOKENS, cached)

    def _record_content(
        self,
        span: Span,
        request: ModelRequest | None,
        response: ModelResponse | None,
    ) -> None:
        input_messages = normalize_messages(request.messages) if request is not None else None
        output_messages: list[dict[str, object]] = []
        output_message = resolve_response_message(response) if response is not None else None
        if output_message is not None and response is not None:
            reason = _resolve_finish_reasons(response, failed=False)[0]
            output_messages.append(map_output_message(output_message, reason))

        if self._capture_on_span:
            if input_messages is not None:
                _set_json_attribute(span, GenAiAttr.INPUT_MESSAGES, input_messages.messages)
                if input_messages.system_instructions:
                    _set_json_attribute(
                        span,
                        GenAiAttr.SYSTEM_INSTRUCTIONS,
                        input_messages.system_instructions,
                    )
            if output_messages:
                _set_json_attribute(span, GenAiAttr.OUTPUT_MESSAGES, output_messages)

        if self._capture_on_event:
            event_attrs: dict[str, AttributeValue] = {}
            if input_messages is not None:
                event_attrs[GenAiAttr.INPUT_MESSAGES] = json.dumps(input_messages.messages)
                if input_messages.system_instructions:
                    event_attrs[GenAiAttr.SYSTEM_INSTRUCTIONS] = json.dumps(input_messages.system_instructions)
            if output_messages:
                event_attrs[GenAiAttr.OUTPUT_MESSAGES] = json.dumps(output_messages)
            _emit_operation_details(self._otel_logger, event_attrs)

    def _record_error(self, span: Span, exc: BaseException) -> None:
        span.set_status(StatusCode.ERROR, str(exc))
        span.set_attribute(GenAiAttr.ERROR_TYPE, type(exc).__name__)
        span.record_exception(exc)

    def _maybe_warn_not_recording(self, span: Span) -> None:
        if span.is_recording() or self._warned_not_initialized:
            return
        self._warned_not_initialized = True
        logger.warning(
            'GenAiInstrumentation is configured but the OpenTelemetry SDK is '
            'not recording, so GenAI telemetry will not be exported. Set a '
            'TracerProvider before constructing Genkit.'
        )


class GenAiSpanContext:
    """SpanContext backed by an OpenTelemetry span."""

    def __init__(self, span: Span) -> None:
        self._span = span

    @property
    def trace_id(self) -> str:
        ctx = self._span.get_span_context()
        if ctx is None or not ctx.trace_id:
            return ''
        return format(ctx.trace_id, '032x')

    @property
    def span_id(self) -> str:
        ctx = self._span.get_span_context()
        if ctx is None or not ctx.span_id:
            return ''
        return format(ctx.span_id, '016x')

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        for key, value in metadata.items():
            try:
                value_string = value if isinstance(value, str) else json.dumps(value)
            except (TypeError, ValueError) as exc:
                value_string = f'Error encoding metadata: {exc}'
            # Keep ad-hoc metadata out of the reserved gen_ai.* namespace.
            self._span.set_attribute(f'genkit:metadata:{key}', value_string)

    def set_output(self, value: object) -> None:
        _set_json_attribute(self._span, GenkitAttr.OUTPUT, value)


def _action_kind(metadata: SpanMetadata) -> str | None:
    """Kind the GenAI provider classifies on.

    A Genkit action span is tagged ``action`` with the real kind on
    ``subtype`` (``model``, ``tool``, …) so the Developer UI still sees
    ``genkit:type=action``. Prefer that subtype when present.
    """
    if metadata.action_type == 'action' and metadata.subtype:
        return metadata.subtype
    return metadata.action_type


def _content_capturing_mode_from_env() -> ContentCapturingMode:
    raw = os.environ.get(CAPTURE_CONTENT_ENV_VAR)
    parsed = parse_content_capturing_mode(raw)
    if parsed is not None:
        return parsed
    logger.warning(
        'Invalid %s=%r; expected one of NO_CONTENT, SPAN_ONLY, EVENT_ONLY, SPAN_AND_EVENT. Defaulting to NO_CONTENT.',
        CAPTURE_CONTENT_ENV_VAR,
        raw,
    )
    return ContentCapturingMode.NO_CONTENT


def _add_request_config_attributes(attrs: dict[str, AttributeValue], request: ModelRequest) -> None:
    config = _config_map(request.config)
    temperature = as_double(_pick(config, 'temperature'))
    if temperature is not None:
        attrs[GenAiAttr.REQUEST_TEMPERATURE] = temperature
    top_p = as_double(_pick(config, 'topP', 'top_p'))
    if top_p is not None:
        attrs[GenAiAttr.REQUEST_TOP_P] = top_p
    top_k = as_int(_pick(config, 'topK', 'top_k'))
    if top_k is not None:
        attrs[GenAiAttr.REQUEST_TOP_K] = top_k
    max_tokens = as_int(_pick(config, 'maxOutputTokens', 'max_output_tokens'))
    if max_tokens is not None:
        attrs[GenAiAttr.REQUEST_MAX_TOKENS] = max_tokens
    stop_sequences = as_string_list(_pick(config, 'stopSequences', 'stop_sequences'))
    if stop_sequences:
        attrs[GenAiAttr.REQUEST_STOP_SEQUENCES] = stop_sequences
    frequency_penalty = as_double(_pick(config, 'frequencyPenalty', 'frequency_penalty'))
    if frequency_penalty is not None:
        attrs[GenAiAttr.REQUEST_FREQUENCY_PENALTY] = frequency_penalty
    presence_penalty = as_double(_pick(config, 'presencePenalty', 'presence_penalty'))
    if presence_penalty is not None:
        attrs[GenAiAttr.REQUEST_PRESENCE_PENALTY] = presence_penalty
    seed = as_int(_pick(config, 'seed'))
    if seed is not None:
        attrs[GenAiAttr.REQUEST_SEED] = seed
    choice_count = as_int(_pick(config, 'candidateCount', 'candidate_count'))
    if choice_count is not None and choice_count != 1:
        attrs[GenAiAttr.REQUEST_CHOICE_COUNT] = choice_count

    output = request.output
    if output is not None:
        output_type = derive_output_type(format=output.format, content_type=output.content_type)
        if output_type is not None:
            attrs[GenAiAttr.OUTPUT_TYPE] = output_type


def _resolve_finish_reasons(response: ModelResponse, *, failed: bool) -> list[str]:
    message = resolve_response_message(response)
    has_tool_request = any(is_tool_request_part(part) for part in (message.content if message else []))
    if has_tool_request:
        # A turn that ends in tool calls is the more informative signal
        # for a later reader than the generic stop reason.
        return ['tool_calls']
    reason = response.finish_reason.value if response.finish_reason is not None else None
    return [map_finish_reason(reason, failed=failed)]


def _config_map(config: object) -> dict[str, object]:
    if config is None:
        return {}
    if isinstance(config, BaseModel):
        dumped = config.model_dump(by_alias=True, exclude_none=True)
        dumped.update(config.model_dump(by_alias=False, exclude_none=True))
        return dumped
    if isinstance(config, Mapping):
        return {str(key): value for key, value in config.items()}
    return {}


def _pick(config: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def _set_json_attribute(span: Span, key: str, value: object) -> None:
    if value is None:
        return
    try:
        encoded = value if isinstance(value, str) else json.dumps(value, default=str)
    except (TypeError, ValueError) as exc:
        encoded = f'Unable to encode: {exc}'
    span.set_attribute(key, encoded)


def _resolve_logger(scope_name: str) -> object | None:
    try:
        from opentelemetry._logs import get_logger

        return get_logger(scope_name)
    except Exception:
        return None


def _emit_operation_details(otel_logger: object | None, attributes: Mapping[str, object]) -> None:
    if otel_logger is None:
        return
    emit = getattr(otel_logger, 'emit', None)
    if emit is None:
        return
    try:
        emit(event_name=GEN_AI_OPERATION_DETAILS_EVENT, attributes=dict(attributes))
    except TypeError:
        return
