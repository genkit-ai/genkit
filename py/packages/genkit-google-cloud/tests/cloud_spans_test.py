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

"""enable_google_cloud_telemetry() hangs Cloud; Genkit() under genkit start fills the Traces tab."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from genkit_google_cloud.telemetry.tracing import (
    _reset_google_cloud_telemetry,
    enable_google_cloud_telemetry,
)
from genkit_otel import GenAiInstrumentation
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genkit import ActionKind, Genkit
from genkit._core._action import Action
from genkit._core._environment import GENKIT_ENV
from genkit._core._telemetry._instrumentation import (
    instrumentations,
    is_instrumented_by,
    parent_path_context,
    reset_instrumentation,
)
from genkit._core._telemetry._log_exporter import reset_log_export
from genkit._core._telemetry.http import GenkitBuiltinInstrumentation
from genkit.telemetry import configure_instrumentation


def _hex_id(value: str, length: int) -> bool:
    return len(value) == length and all(c in '0123456789abcdef' for c in value)


def _flush_exporters_in_provider(provider: TracerProvider) -> None:
    active = getattr(provider, '_active_span_processor', None)
    if active is None:
        return
    processors = getattr(active, '_span_processors', [active])
    for proc in processors:
        exp = getattr(proc, 'span_exporter', None) or getattr(proc, 'exporter', None)
        if exp is not None and hasattr(exp, 'force_flush'):
            exp.force_flush()


def _force_flush() -> None:
    provider = trace_api.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.force_flush()
        _flush_exporters_in_provider(provider)


async def _joke() -> str:
    return 'Why did the cat cross the road?'


@contextmanager
def _cloud_enable(**kwargs: Any) -> Generator[InMemorySpanExporter, None, None]:
    """enable_google_cloud_telemetry() with Cloud exporters that stay in memory."""
    _reset_google_cloud_telemetry()
    cloud = InMemorySpanExporter()
    with (
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter'),
        patch(
            'genkit_google_cloud.telemetry.config.GcpAdjustingTraceExporter',
            return_value=cloud,
        ),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
        patch('genkit_google_cloud.telemetry.config.CloudLoggingExporter'),
    ):
        enable_google_cloud_telemetry(**kwargs)
        yield cloud


@pytest.fixture(autouse=True)
def _isolate_telemetry(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Each test starts with no providers, unset collector env, and its own tracer."""
    reset_instrumentation()
    reset_log_export()
    _reset_google_cloud_telemetry()
    monkeypatch.delenv(GENKIT_ENV, raising=False)
    monkeypatch.delenv('GENKIT_TELEMETRY_SERVER', raising=False)
    monkeypatch.setattr(Genkit, '_start_reflection_background', lambda self: None)
    isolated = TracerProvider()
    monkeypatch.setattr(trace_api, 'get_tracer_provider', lambda: isolated)
    monkeypatch.setattr(trace_api, 'set_tracer_provider', lambda _provider: None)
    monkeypatch.setattr(
        'genkit_google_cloud.telemetry.config.trace_api.get_tracer_provider',
        lambda: isolated,
    )
    path_token = parent_path_context.set('')
    try:
        yield
    finally:
        parent_path_context.reset(path_token)
        reset_instrumentation()
        reset_log_export()
        _reset_google_cloud_telemetry()
        isolated.shutdown()


@pytest.mark.asyncio
async def test_enable_google_cloud_telemetry_mints_ids_and_sends_the_action_to_cloud() -> None:
    """enable_google_cloud_telemetry() turns GenAI spans on; Cloud sees the action."""
    with _cloud_enable() as cloud:
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()
        _force_flush()

        assert is_instrumented_by(GenAiInstrumentation)
        assert _hex_id(result.trace_id, 32)
        names = [span.name for span in cloud.get_finished_spans()]
        assert 'joke' in names


@pytest.mark.asyncio
async def test_configure_genai_then_enable_sends_one_span_to_cloud() -> None:
    """configure_instrumentation(GenAiInstrumentation()) then enable(): Cloud sees one joke span."""
    yours = GenAiInstrumentation()
    configure_instrumentation(yours)
    with _cloud_enable() as cloud:
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()
        _force_flush()

        assert [i for i in instrumentations if isinstance(i, GenAiInstrumentation)] == [yours]
        assert _hex_id(result.trace_id, 32)
        joke = [span for span in cloud.get_finished_spans() if span.name == 'joke']
        assert len(joke) == 1


@pytest.mark.asyncio
async def test_configure_genai_on_a_private_tracer_then_enable_does_not_send_that_action_to_cloud() -> None:
    """GenAI on a private tracer then enable(): Cloud hangs on the process tracer, so that action is not there."""
    private = TracerProvider()
    configure_instrumentation(GenAiInstrumentation(tracer=private.get_tracer('test')))
    with _cloud_enable() as cloud:
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()
        _force_flush()
        private.force_flush()

        assert _hex_id(result.trace_id, 32)
        names = [span.name for span in cloud.get_finished_spans()]
        assert 'joke' not in names
    private.shutdown()


def test_enable_when_process_tracer_cannot_attach_does_not_raise() -> None:
    """A dead process tracer stays a log line so enable() does not crash the process."""
    global_provider = trace_api.get_tracer_provider()

    def boom(_processor: object) -> None:
        raise RuntimeError('processor dead')

    global_provider.add_span_processor = boom  # type: ignore[method-assign]
    with _cloud_enable():
        pass


@pytest.mark.asyncio
async def test_enable_under_genkit_start_still_adds_the_ui_poster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """force_dev_export=True under genkit start turns GenAI spans on; Genkit() still adds the Traces tab poster."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')

    with _cloud_enable(force_dev_export=True):
        assert is_instrumented_by(GenAiInstrumentation)

    Genkit()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert is_instrumented_by(GenAiInstrumentation)
    assert is_instrumented_by(GenkitBuiltinInstrumentation)
    assert _hex_id(result.trace_id, 32)
