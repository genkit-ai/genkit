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

import pytest
from genkit_otel import GenAiInstrumentation
from genkit_otel._gen_ai_attributes import GenAiAttr, GenAiMetric

from genkit import FinishReason, Part
from genkit._core._typing import GenerationUsage, Role
from genkit.model import Message, ModelResponse
from genkit.telemetry import SpanMetadata


def _points_for_model(harness, metric_name: str, model: str) -> list:
    data = harness.metrics.get_metrics_data()
    if data is None:
        return []
    points = []
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != metric_name:
                    continue
                for point in metric.data.data_points:
                    attrs = dict(point.attributes or {})
                    if attrs.get(GenAiAttr.REQUEST_MODEL) == model:
                        points.append(point)
    return points


def _response(*, usage: GenerationUsage | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        usage=usage,
    )


async def _run_model(instr: GenAiInstrumentation, name: str, fn) -> None:
    await instr.run_in_new_span(SpanMetadata(name=name, action_type='model'), fn)


@pytest.mark.asyncio
async def test_records_token_usage_split_by_input_output(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'googleai/metrics-usage',
        lambda span=None: _awaitable(_response(usage=GenerationUsage(input_tokens=12, output_tokens=34))),
    )
    points = _points_for_model(harness, GenAiMetric.TOKEN_USAGE, 'metrics-usage')
    by_type = {dict(p.attributes).get(GenAiAttr.TOKEN_TYPE): p for p in points}
    assert by_type['input'].sum == 12
    assert by_type['output'].sum == 34
    any_point = points[0]
    attrs = dict(any_point.attributes)
    assert attrs[GenAiAttr.OPERATION_NAME] == 'chat'
    assert attrs[GenAiAttr.PROVIDER_NAME] == 'gcp.gemini'


@pytest.mark.asyncio
async def test_records_operation_duration_on_success(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'openai/metrics-duration',
        lambda span=None: _awaitable(_response(usage=GenerationUsage(input_tokens=1))),
    )
    points = _points_for_model(harness, GenAiMetric.OPERATION_DURATION, 'metrics-duration')
    assert len(points) == 1
    attrs = dict(points[0].attributes)
    assert attrs[GenAiAttr.PROVIDER_NAME] == 'openai'
    assert GenAiAttr.ERROR_TYPE not in attrs


@pytest.mark.asyncio
async def test_records_duration_with_error_type_on_failure(harness) -> None:
    instr = harness.instrumentation()

    async def boom(span=None):
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await _run_model(instr, 'googleai/metrics-error', boom)

    duration = _points_for_model(harness, GenAiMetric.OPERATION_DURATION, 'metrics-error')
    assert len(duration) == 1
    assert dict(duration[0].attributes)[GenAiAttr.ERROR_TYPE] == 'RuntimeError'
    assert _points_for_model(harness, GenAiMetric.TOKEN_USAGE, 'metrics-error') == []


@pytest.mark.asyncio
async def test_emit_metrics_false_records_none(harness) -> None:
    instr = harness.instrumentation(emit_metrics=False)
    await _run_model(
        instr,
        'googleai/metrics-disabled',
        lambda span=None: _awaitable(_response(usage=GenerationUsage(input_tokens=5))),
    )
    assert _points_for_model(harness, GenAiMetric.TOKEN_USAGE, 'metrics-disabled') == []
    assert _points_for_model(harness, GenAiMetric.OPERATION_DURATION, 'metrics-disabled') == []


async def _awaitable(value: object) -> object:
    return value
