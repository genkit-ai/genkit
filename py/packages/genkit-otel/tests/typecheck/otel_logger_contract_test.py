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

# pyright: reportUnnecessaryTypeIgnoreComment=true

"""Typed contract of ``GenAiInstrumentation(otel_logger=...)``.

pyright, pyrefly and ty must accept OpenTelemetry's ``get_logger`` and SDK
``LoggerProvider.get_logger`` handles, plus minimal structural implementations
of ``OtelEventLogger``, with zero suppressions, while rejecting incompatible
objects.
"""

from collections.abc import Mapping

import pytest
from genkit_otel import GenAiInstrumentation
from genkit_otel._gen_ai_attributes import GEN_AI_OPERATION_DETAILS_EVENT
from genkit_otel._genai_instrumentation import OtelEventLogger
from opentelemetry._logs import get_logger
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.util.types import AttributeValue

from genkit import FinishReason, Message, ModelResponse, Part, Role
from genkit.model import ModelRequest
from genkit.telemetry import SpanContext, SpanMetadata


class _RecordedEventLogger:
    """Structural event logger implementing only ``emit(event_name=..., attributes=...)``."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, AttributeValue]]] = []

    def emit(
        self,
        *,
        event_name: str,
        attributes: Mapping[str, AttributeValue],
    ) -> None:
        self.events.append((event_name, dict(attributes)))


class _IncompatibleLogger:
    """Missing keyword ``event_name`` / ``attributes`` signature."""

    def emit(self, message: str) -> None:
        _ = message


def test_real_otel_loggers_satisfy_otel_event_logger_protocol() -> None:
    """Both API ``get_logger`` and SDK ``LoggerProvider.get_logger`` satisfy ``OtelEventLogger``."""
    api_logger = get_logger('dining-api')
    sdk_logger = LoggerProvider().get_logger('dining-sdk')
    structural_logger: OtelEventLogger = _RecordedEventLogger()

    instr_api = GenAiInstrumentation(otel_logger=api_logger)
    instr_sdk = GenAiInstrumentation(otel_logger=sdk_logger)
    instr_custom = GenAiInstrumentation(otel_logger=structural_logger)

    assert instr_api._otel_logger is api_logger
    assert instr_sdk._otel_logger is sdk_logger
    assert instr_custom._otel_logger is structural_logger


def test_incompatible_otel_logger_is_rejected_by_type_checkers() -> None:
    """``otel_logger`` is typed against ``OtelEventLogger``, not ``Any`` or ``object``."""
    bad = _IncompatibleLogger()
    GenAiInstrumentation(otel_logger=bad)  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]


@pytest.mark.asyncio
async def test_structural_event_logger_receives_operation_details() -> None:
    """A custom ``OtelEventLogger`` receives ``gen_ai.client.inference.operation.details``."""
    recorder = _RecordedEventLogger()
    instr = GenAiInstrumentation(
        content_capturing_mode='EVENT_ONLY',
        otel_logger=recorder,
    )
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('Suggest a starter.')])])
    response = ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text('Burrata with heirloom tomatoes.')]),
    )

    async def call_model(span: SpanContext | None = None) -> ModelResponse:
        _ = span
        return response

    await instr.run_in_new_span(
        SpanMetadata(name='googleai/gemini-flash-latest', action_type='model', input=request),
        call_model,
    )

    assert len(recorder.events) == 1
    event_name, attrs = recorder.events[0]
    assert event_name == GEN_AI_OPERATION_DETAILS_EVENT
    assert 'Suggest a starter.' in str(attrs.get('gen_ai.input.messages'))
