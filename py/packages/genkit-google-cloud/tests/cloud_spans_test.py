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

import subprocess  # noqa: S404 - runs this interpreter on a fixed script
import sys
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
from opentelemetry import _logs, context as otel_context, trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ParentBased, TraceIdRatioBased
from opentelemetry.trace import (
    NonRecordingSpan,
    NoOpTracer,
    SpanContext,
    TraceFlags,
    TracerProvider as ApiTracerProvider,
    set_span_in_context,
)
from opentelemetry.util._once import Once

from genkit import (
    ActionRunContext,
    FinishReason,
    Genkit,
    GenkitError,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Part,
    Role,
)
from genkit._core._action import Action
from genkit.model import ModelRequest, ModelUsage
from genkit.plugin_api import ActionKind
from genkit.telemetry import (
    DirectHttpInstrumentation,
    configure_instrumentation,
    is_instrumented_by,
    reset_instrumentation,
)


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


@contextmanager
def _cloud_trace(**kwargs: Any) -> Generator[InMemorySpanExporter, None, None]:
    """enable_google_cloud_telemetry() with the real Cloud exporter over an in-memory client."""
    _reset_google_cloud_telemetry()
    cloud = InMemorySpanExporter()
    with (
        patch('genkit_google_cloud.telemetry.config.GenkitGCPExporter', return_value=cloud),
        patch('genkit_google_cloud.telemetry.config.GoogleCloudResourceDetector'),
        patch('genkit_google_cloud.telemetry.config.CloudMonitoringMetricsExporter'),
        patch('genkit_google_cloud.telemetry.config.GenkitMetricExporter'),
        patch('genkit_google_cloud.telemetry.config.PeriodicExportingMetricReader'),
        patch('genkit_google_cloud.telemetry.config.metrics'),
        patch('genkit_google_cloud.telemetry.config.CloudLoggingExporter'),
    ):
        enable_google_cloud_telemetry(project_id='p', **kwargs)
        yield cloud


_CAPTURE_ENV = 'OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'
_PROMPT = 'ssn-999 do not leak'
_REPLY = 'reply-secret-42'
_SYSTEM = 'do not store this system line'
_TOOL_RESULT = 'classified-result-42'
_MODEL = 'echo'
_INPUT_TOKENS = 12
_OUTPUT_TOKENS = 4
_CONTENT_KEYS = (
    'gen_ai.input.messages',
    'gen_ai.output.messages',
    'gen_ai.system_instructions',
    'genkit.input',
    'genkit.output',
    'genkit/input',
    'genkit/output',
    'gen_ai.tool.call.arguments',
    'gen_ai.tool.call.result',
)


def _attrs(span: object) -> dict[str, Any]:
    attributes = getattr(span, 'attributes', None)
    return dict(attributes) if attributes else {}


def _spans_named(exporter: InMemorySpanExporter, name: str) -> list[Any]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


def _one_span(exporter: InMemorySpanExporter, name: str) -> Any:
    matches = _spans_named(exporter, name)
    assert len(matches) == 1, [span.name for span in exporter.get_finished_spans()]
    return matches[0]


def _assert_model_and_tokens(attrs: dict[str, Any]) -> None:
    assert attrs['gen_ai.request.model'] == _MODEL
    assert attrs['gen_ai.usage.input_tokens'] == _INPUT_TOKENS


def _echo_response() -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text(_REPLY)]),
        usage=ModelUsage(input_tokens=_INPUT_TOKENS, output_tokens=_OUTPUT_TOKENS),
    )


def _echo_app() -> Genkit:
    ai = Genkit()

    async def echo_model(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(_REPLY)]))
        return _echo_response()

    ai.define_model(_MODEL, echo_model)
    return ai


def _reset_otel_globals() -> None:
    """Clear the process tracer and logger so the next test looks like a new process."""
    from opentelemetry._logs import _internal as logs_internal

    current = getattr(trace_api, '_TRACER_PROVIDER', None)
    if isinstance(current, TracerProvider):
        current.shutdown()
    trace_api._TRACER_PROVIDER = None
    trace_api._TRACER_PROVIDER_SET_ONCE = Once()

    current_logs = getattr(logs_internal, '_LOGGER_PROVIDER', None)
    if current_logs is not None and hasattr(current_logs, 'shutdown'):
        current_logs.shutdown()
    logs_internal._LOGGER_PROVIDER = None
    logs_internal._LOGGER_PROVIDER_SET_ONCE = Once()


def _span_names(exporter: InMemorySpanExporter) -> list[str]:
    return [span.name for span in exporter.get_finished_spans()]


def _processors(provider: TracerProvider) -> tuple[object, ...]:
    active = getattr(provider, '_active_span_processor', None)
    if active is None:
        return ()
    return tuple(getattr(active, '_span_processors', (active,)))


class NonSdkTracerProvider(ApiTracerProvider):
    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = None,
        schema_url: str | None = None,
        attributes: object | None = None,
    ) -> NoOpTracer:
        return NoOpTracer()


class NonSdkLoggerProvider:
    def get_logger(self, *args: object, **kwargs: object) -> object:
        return object()


@pytest.fixture
def real_otel_globals() -> None:
    """Opt a test into the process OpenTelemetry globals instead of a stub provider."""


@pytest.fixture(autouse=True)
def _no_adc_project() -> Generator[None, None, None]:
    """Keep the developer's own ADC project out of these tests."""
    with patch('genkit_google_cloud.telemetry.config._adc_project_id', return_value=None):
        yield


@pytest.fixture(autouse=True)
def _isolate_telemetry(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Each test starts with no providers, unset collector env, and its own tracer."""
    reset_instrumentation()
    _reset_google_cloud_telemetry()
    monkeypatch.delenv('GENKIT_ENV', raising=False)
    monkeypatch.delenv('GENKIT_TELEMETRY_SERVER', raising=False)
    monkeypatch.delenv('OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT', raising=False)
    monkeypatch.setattr(Genkit, '_start_reflection_background', lambda self: None)
    if 'real_otel_globals' in request.fixturenames:
        _reset_otel_globals()
        try:
            yield
        finally:
            reset_instrumentation()
            _reset_google_cloud_telemetry()
            _reset_otel_globals()
        return

    isolated = TracerProvider()
    monkeypatch.setattr(trace_api, 'get_tracer_provider', lambda: isolated)
    monkeypatch.setattr(trace_api, 'set_tracer_provider', lambda _provider: None)
    monkeypatch.setattr(
        'genkit_google_cloud.telemetry.config.trace_api.get_tracer_provider',
        lambda: isolated,
    )
    try:
        yield
    finally:
        reset_instrumentation()
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
async def test_enable_google_cloud_telemetry_does_not_install_genai_instrumentation_twice() -> None:
    """configure_instrumentation(GenAiInstrumentation()) then enable(): Cloud sees one joke span."""
    yours = GenAiInstrumentation()
    configure_instrumentation(yours)
    with _cloud_enable() as cloud:
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()
        _force_flush()

        assert is_instrumented_by(GenAiInstrumentation)
        assert yours.emit_metrics is True
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
    monkeypatch.setenv('GENKIT_ENV', 'dev')
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')

    with _cloud_enable(force_dev_export=True):
        assert is_instrumented_by(GenAiInstrumentation)

    Genkit()
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert is_instrumented_by(GenAiInstrumentation)
    assert is_instrumented_by(DirectHttpInstrumentation)
    assert _hex_id(result.trace_id, 32)


@pytest.mark.asyncio
async def test_enable_with_no_tracer_creates_one_and_cloud_gets_the_flow_span(
    real_otel_globals: None,
) -> None:
    """On a fresh process the helper installs a tracer and Cloud gets the flow span."""
    with _cloud_enable() as cloud:
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        await action.run()
        _force_flush()

        provider = trace_api.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        assert 'joke' in _span_names(cloud)


@pytest.mark.asyncio
async def test_enable_with_no_tracer_and_always_off_sampler_sends_nothing_to_cloud(
    real_otel_globals: None,
) -> None:
    """sampler=ALWAYS_OFF on a fresh process means Cloud receives no spans."""
    with _cloud_enable(sampler=ALWAYS_OFF) as cloud:
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        await action.run()
        _force_flush()

        provider = trace_api.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        assert provider.sampler is ALWAYS_OFF
        assert _span_names(cloud) == []


@pytest.mark.asyncio
async def test_enable_with_parent_based_ratio_sampler_drops_new_roots_and_keeps_sampled_parents(
    real_otel_globals: None,
) -> None:
    """ParentBased(TraceIdRatioBased(0.0)): a new trace is dropped; a request whose caller sampled it is kept."""
    sampler = ParentBased(TraceIdRatioBased(0.0))
    with _cloud_enable(sampler=sampler) as cloud:
        # 1. No upstream trace: the ratio decides, and 0.0 drops it.
        await Action(name='joke', kind=ActionKind.FLOW, fn=_joke).run()
        _force_flush()
        assert _span_names(cloud) == []

        # 2. Upstream caller sent a sampled traceparent: the parent decides, so Cloud gets it.
        upstream = SpanContext(
            trace_id=0x4BF92F3577B34DA6A3CE929D0E0E4736,
            span_id=0x00F067AA0BA902B7,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        token = otel_context.attach(set_span_in_context(NonRecordingSpan(upstream)))
        try:
            await Action(name='joke', kind=ActionKind.FLOW, fn=_joke).run()
        finally:
            otel_context.detach(token)
        _force_flush()

        provider = trace_api.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        assert provider.sampler is sampler
        joke = [span for span in cloud.get_finished_spans() if span.name == 'joke']
        assert [span.context.trace_id if span.context else None for span in joke] == [upstream.trace_id]


@pytest.mark.asyncio
async def test_enable_after_app_sets_tracer_sends_spans_to_their_exporter_and_cloud(
    real_otel_globals: None,
) -> None:
    """The app's exporter and Cloud both get the flow span; their tracer stays installed."""
    theirs = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(theirs))
    trace_api.set_tracer_provider(provider)

    with _cloud_enable() as cloud:
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        await action.run()
        _force_flush()

        assert trace_api.get_tracer_provider() is provider
        assert 'joke' in _span_names(theirs)
        assert 'joke' in _span_names(cloud)


@pytest.mark.asyncio
async def test_enable_after_app_sets_tracer_with_always_off_keeps_their_sampler(
    real_otel_globals: None,
) -> None:
    """The app's ALWAYS_OFF sampler stays in charge, so neither exporter gets the span."""
    theirs = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_OFF)
    provider.add_span_processor(SimpleSpanProcessor(theirs))
    trace_api.set_tracer_provider(provider)

    with _cloud_enable() as cloud:
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        await action.run()
        _force_flush()

        assert trace_api.get_tracer_provider() is provider
        assert provider.sampler is ALWAYS_OFF
        assert _span_names(theirs) == []
        assert _span_names(cloud) == []


@pytest.mark.asyncio
async def test_enable_with_sampler_after_app_sets_tracer_raises_invalid_argument(
    real_otel_globals: None,
) -> None:
    """sampler= after the app set a tracer raises INVALID_ARGUMENT and installs nothing."""
    theirs = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(theirs))
    trace_api.set_tracer_provider(provider)
    before = _processors(provider)

    with pytest.raises(GenkitError, match='TracerProvider\\(sampler=') as raised:
        with _cloud_enable(sampler=ALWAYS_OFF):
            pass

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert trace_api.get_tracer_provider() is provider
    assert _processors(provider) == before
    assert _span_names(theirs) == []


@pytest.mark.asyncio
async def test_enable_with_sampler_after_app_sets_tracer_then_without_sampler_adds_cloud(
    real_otel_globals: None,
) -> None:
    """After the sampler= raise, a second call without sampler= adds Cloud and the flow span arrives."""
    theirs = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(theirs))
    trace_api.set_tracer_provider(provider)

    with pytest.raises(GenkitError) as raised:
        with _cloud_enable(sampler=ALWAYS_OFF):
            pass
    assert raised.value.status == 'INVALID_ARGUMENT'

    with _cloud_enable() as cloud:
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        await action.run()
        _force_flush()

        assert trace_api.get_tracer_provider() is provider
        assert 'joke' in _span_names(theirs)
        assert 'joke' in _span_names(cloud)


def test_enable_with_sampler_after_app_sets_tracer_under_genkit_start_still_raises(
    real_otel_globals: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GENKIT_ENV=dev still raises INVALID_ARGUMENT when sampler= meets the app's tracer."""
    monkeypatch.setenv('GENKIT_ENV', 'dev')
    provider = TracerProvider()
    trace_api.set_tracer_provider(provider)

    with pytest.raises(GenkitError, match='TracerProvider\\(sampler=') as raised:
        with _cloud_enable(sampler=ALWAYS_OFF):
            pass

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert trace_api.get_tracer_provider() is provider


def test_enable_with_non_sdk_tracer_raises_failed_precondition_and_does_not_log_initialized(
    real_otel_globals: None,
) -> None:
    """A non-SDK process tracer raises FAILED_PRECONDITION and does not log a full init."""
    trace_api.set_tracer_provider(NonSdkTracerProvider())

    with patch('genkit_google_cloud.telemetry.config.logger.info') as mock_info:
        with pytest.raises(GenkitError) as raised:
            with _cloud_enable():
                pass

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert not any('Telemetry fully initialized' in str(call) for call in mock_info.call_args_list)
    assert not isinstance(trace_api.get_tracer_provider(), TracerProvider)


def test_enable_with_non_sdk_logger_raises_failed_precondition_and_does_not_log_initialized(
    real_otel_globals: None,
) -> None:
    """A non-SDK process logger raises FAILED_PRECONDITION and does not log a full init."""
    _logs.set_logger_provider(NonSdkLoggerProvider())  # type: ignore[arg-type]

    with patch('genkit_google_cloud.telemetry.config.logger.info') as mock_info:
        with pytest.raises(GenkitError) as raised:
            with _cloud_enable():
                pass

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert not any('Telemetry fully initialized' in str(call) for call in mock_info.call_args_list)


def test_enable_with_non_sdk_tracer_and_disable_traces_runs_and_leaves_their_tracer(
    real_otel_globals: None,
) -> None:
    """disable_traces=True never adds Cloud Trace, so a non-SDK process tracer doesn't raise and stays installed."""
    theirs = NonSdkTracerProvider()
    trace_api.set_tracer_provider(theirs)

    with patch('genkit_google_cloud.telemetry.config.logger.info') as mock_info:
        with _cloud_enable(disable_traces=True):
            pass

    assert trace_api.get_tracer_provider() is theirs
    assert any('Telemetry fully initialized' in str(call) for call in mock_info.call_args_list)


@pytest.mark.asyncio
async def test_enable_then_app_sets_tracer_cloud_keeps_spans_and_their_exporter_gets_none(
    real_otel_globals: None,
) -> None:
    """Helper first, then the app's tracer: Cloud still gets the flow span; their exporter does not."""
    with _cloud_enable() as cloud:
        theirs = InMemorySpanExporter()
        later = TracerProvider()
        later.add_span_processor(SimpleSpanProcessor(theirs))
        trace_api.set_tracer_provider(later)

        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        await action.run()
        _force_flush()
        later.force_flush()

        assert 'joke' in _span_names(cloud)
        assert _span_names(theirs) == []


def test_enable_with_sampler_and_disable_traces_raises_invalid_argument(
    real_otel_globals: None,
) -> None:
    """sampler= with disable_traces=True raises INVALID_ARGUMENT."""
    with pytest.raises(GenkitError) as raised:
        with _cloud_enable(sampler=ALWAYS_OFF, disable_traces=True):
            pass

    assert raised.value.status == 'INVALID_ARGUMENT'


def test_enable_twice_with_sampler_on_a_fresh_process_raises_already_called(
    real_otel_globals: None,
) -> None:
    """A repeat call with sampler= hits the once-per-process guard, not the app-tracer sampler error."""
    with _cloud_enable(sampler=ALWAYS_OFF):
        installed = trace_api.get_tracer_provider()
        with pytest.raises(GenkitError, match='already called') as raised:
            enable_google_cloud_telemetry(sampler=ALWAYS_OFF)

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert isinstance(installed, TracerProvider)
    assert installed.sampler is ALWAYS_OFF


def test_nothing_registered_matches_only_the_default_otel_proxies(
    real_otel_globals: None,
) -> None:
    """The default tracer and logger proxies count as unset; a same-named class elsewhere does not."""
    from genkit_google_cloud.telemetry.config import _nothing_registered

    class ProxyLoggerProvider:  # same name as OTel's, but the app's module
        pass

    assert _nothing_registered(trace_api.get_tracer_provider())
    assert _nothing_registered(_logs.get_logger_provider())
    assert not _nothing_registered(ProxyLoggerProvider())
    assert not _nothing_registered(NonSdkLoggerProvider())


def test_import_survives_otel_moving_the_logger_proxy() -> None:
    """genkit_google_cloud imports even if OTel drops ProxyLoggerProvider from _logs._internal."""
    script = (
        'import opentelemetry._logs._internal as m\n'
        'del m.ProxyLoggerProvider\n'
        'import genkit_google_cloud.telemetry.config\n'
        "print('imported')\n"
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)  # noqa: S603

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'imported'


def _chat_attrs(exporter: InMemorySpanExporter) -> dict[str, Any]:
    return _attrs(_one_span(exporter, f'chat {_MODEL}'))


@pytest.mark.asyncio
async def test_generate_with_span_only_env_shows_redacted_prompt_and_reply_in_cloud_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the env var at SPAN_ONLY, Cloud Trace shows <redacted> for both message keys."""
    monkeypatch.setenv(_CAPTURE_ENV, 'SPAN_ONLY')
    with _cloud_trace() as cloud:
        await _echo_app().generate(prompt=_PROMPT, model=_MODEL)
        _force_flush()

    attrs = _chat_attrs(cloud)
    _assert_model_and_tokens(attrs)
    assert attrs['gen_ai.input.messages'] == '<redacted>'
    assert attrs['gen_ai.output.messages'] == '<redacted>'
    assert _PROMPT not in attrs['gen_ai.input.messages']
    assert _REPLY not in attrs['gen_ai.output.messages']


@pytest.mark.asyncio
async def test_generate_with_span_and_event_env_shows_redacted_prompt_in_cloud_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPAN_AND_EVENT redacts the same two message keys on the Cloud span."""
    monkeypatch.setenv(_CAPTURE_ENV, 'SPAN_AND_EVENT')
    with _cloud_trace() as cloud:
        await _echo_app().generate(prompt=_PROMPT, model=_MODEL)
        _force_flush()

    attrs = _chat_attrs(cloud)
    _assert_model_and_tokens(attrs)
    assert attrs['gen_ai.input.messages'] == '<redacted>'
    assert attrs['gen_ai.output.messages'] == '<redacted>'


@pytest.mark.asyncio
async def test_generate_with_system_prompt_and_span_only_env_shows_redacted_system_instructions_in_cloud_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With system= set, Cloud Trace shows gen_ai.system_instructions as <redacted>."""
    monkeypatch.setenv(_CAPTURE_ENV, 'SPAN_ONLY')
    with _cloud_trace() as cloud:
        await _echo_app().generate(prompt=_PROMPT, system=_SYSTEM, model=_MODEL)
        _force_flush()

    attrs = _chat_attrs(cloud)
    _assert_model_and_tokens(attrs)
    assert attrs['gen_ai.system_instructions'] == '<redacted>'
    assert _SYSTEM not in ' '.join(str(value) for value in attrs.values())


@pytest.mark.asyncio
async def test_generate_with_span_only_set_in_code_shows_redacted_prompt_in_cloud_trace() -> None:
    """Registering content_capturing_mode='SPAN_ONLY' in code still redacts Cloud Trace."""
    configure_instrumentation(GenAiInstrumentation(content_capturing_mode='SPAN_ONLY'))
    with _cloud_trace() as cloud:
        await _echo_app().generate(prompt=_PROMPT, model=_MODEL)
        _force_flush()

    attrs = _chat_attrs(cloud)
    _assert_model_and_tokens(attrs)
    assert attrs['gen_ai.input.messages'] == '<redacted>'
    assert attrs['gen_ai.output.messages'] == '<redacted>'


@pytest.mark.asyncio
async def test_generate_with_capture_action_io_shows_redacted_genkit_input_and_output_in_cloud_trace() -> None:
    """capture_action_io=True writes genkit.input and genkit.output; Cloud Trace shows both as <redacted>."""
    configure_instrumentation(GenAiInstrumentation(capture_action_io=True))
    with _cloud_trace() as cloud:
        await _echo_app().generate(prompt=_PROMPT, model=_MODEL)
        _force_flush()

    attrs = _chat_attrs(cloud)
    _assert_model_and_tokens(attrs)
    assert attrs['genkit.input'] == '<redacted>'
    assert attrs['genkit.output'] == '<redacted>'
    assert 'gen_ai.input.messages' not in attrs
    assert _PROMPT not in ' '.join(str(value) for value in attrs.values())


@pytest.mark.asyncio
async def test_flow_with_capture_action_io_shows_redacted_flow_input_in_cloud_trace() -> None:
    """A flow's own span has genkit.input / genkit.output redacted."""
    configure_instrumentation(GenAiInstrumentation(capture_action_io=True))
    with _cloud_trace() as cloud:
        ai = Genkit()

        @ai.flow()
        async def payroll(secret: str) -> str:
            return 'paid ' + secret

        await payroll(_PROMPT)
        _force_flush()

    attrs = _attrs(_one_span(cloud, 'payroll'))
    assert attrs['genkit.input'] == '<redacted>'
    assert attrs['genkit.output'] == '<redacted>'
    assert _PROMPT not in ' '.join(str(value) for value in attrs.values())


@pytest.mark.asyncio
async def test_generate_stream_with_span_only_env_shows_redacted_reply_in_cloud_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streamed reply is redacted the same way as a non-streamed one."""
    monkeypatch.setenv(_CAPTURE_ENV, 'SPAN_ONLY')
    with _cloud_trace() as cloud:
        stream = _echo_app().generate_stream(prompt=_PROMPT, model=_MODEL)
        chunks = [chunk.text async for chunk in stream]
        await stream.response
        _force_flush()

    assert _REPLY in ''.join(chunks)
    attrs = _chat_attrs(cloud)
    _assert_model_and_tokens(attrs)
    assert attrs['gen_ai.output.messages'] == '<redacted>'
    assert _REPLY not in attrs['gen_ai.output.messages']


@pytest.mark.asyncio
async def test_generate_with_tool_spans_and_span_only_env_shows_redacted_tool_arguments_and_result_in_cloud_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool arguments and the tool result are <redacted> on the Cloud tool span."""
    monkeypatch.setenv(_CAPTURE_ENV, 'SPAN_ONLY')
    configure_instrumentation(GenAiInstrumentation(emit_tool_spans=True))
    ai = Genkit()

    @ai.tool(name='lookup')
    async def lookup(query: str) -> str:
        """Look up a record."""
        return _TOOL_RESULT

    async def tool_model(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        if any(message.role == Role.TOOL for message in request.messages):
            return _echo_response()
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[Part.from_tool_request(name='lookup', input=_PROMPT, ref='r1')],
            ),
            usage=ModelUsage(input_tokens=_INPUT_TOKENS, output_tokens=_OUTPUT_TOKENS),
        )

    ai.define_model(_MODEL, tool_model)
    with _cloud_trace() as cloud:
        await ai.generate(prompt=_PROMPT, model=_MODEL, tools=['lookup'])
        _force_flush()

    tool_attrs = _attrs(_one_span(cloud, 'execute_tool lookup'))
    assert tool_attrs['gen_ai.tool.call.arguments'] == '<redacted>'
    assert tool_attrs['gen_ai.tool.call.result'] == '<redacted>'
    assert _PROMPT not in tool_attrs['gen_ai.tool.call.arguments']
    assert _TOOL_RESULT not in tool_attrs['gen_ai.tool.call.result']
    for span in _spans_named(cloud, f'chat {_MODEL}'):
        attrs = _attrs(span)
        _assert_model_and_tokens(attrs)
        assert attrs['gen_ai.input.messages'] == '<redacted>'


@pytest.mark.asyncio
async def test_generate_with_span_only_env_still_sends_prompt_text_to_their_own_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exporter the app added to the same provider gets the raw prompt while Cloud gets <redacted>."""
    monkeypatch.setenv(_CAPTURE_ENV, 'SPAN_ONLY')
    with _cloud_trace() as cloud:
        own = InMemorySpanExporter()
        provider = trace_api.get_tracer_provider()
        assert isinstance(provider, TracerProvider)
        provider.add_span_processor(SimpleSpanProcessor(own))
        await _echo_app().generate(prompt=_PROMPT, model=_MODEL)
        _force_flush()

    cloud_attrs = _chat_attrs(cloud)
    _assert_model_and_tokens(cloud_attrs)
    assert cloud_attrs['gen_ai.input.messages'] == '<redacted>'
    own_attrs = _attrs(_one_span(own, f'chat {_MODEL}'))
    assert _PROMPT in str(own_attrs['gen_ai.input.messages'])
    assert _REPLY in str(own_attrs['gen_ai.output.messages'])
    _assert_model_and_tokens(own_attrs)


@pytest.mark.asyncio
async def test_generate_with_no_capture_has_no_message_keys_in_cloud_trace() -> None:
    """By default the content keys are absent from the Cloud span, not set to <redacted>."""
    with _cloud_trace() as cloud:
        await _echo_app().generate(prompt=_PROMPT, model=_MODEL)
        _force_flush()

    attrs = _chat_attrs(cloud)
    _assert_model_and_tokens(attrs)
    for key in _CONTENT_KEYS:
        assert key not in attrs


@pytest.mark.asyncio
async def test_generate_with_span_only_env_keeps_token_counts_and_model_in_cloud_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redaction leaves gen_ai.request.model and the usage counts as they were."""
    monkeypatch.setenv(_CAPTURE_ENV, 'SPAN_ONLY')
    with _cloud_trace() as cloud:
        await _echo_app().generate(prompt=_PROMPT, model=_MODEL)
        _force_flush()

    attrs = _chat_attrs(cloud)
    _assert_model_and_tokens(attrs)
    assert attrs['gen_ai.usage.output_tokens'] == _OUTPUT_TOKENS
    assert attrs['gen_ai.input.messages'] == '<redacted>'
    assert attrs['gen_ai.output.messages'] == '<redacted>'


@pytest.mark.asyncio
async def test_generate_that_raises_with_span_only_env_has_no_prompt_in_cloud_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the model raises, the Cloud span has no prompt text in any content key."""
    monkeypatch.setenv(_CAPTURE_ENV, 'SPAN_ONLY')
    ai = Genkit()

    async def boom(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        raise RuntimeError('model exploded')

    ai.define_model(_MODEL, boom)
    with _cloud_trace() as cloud:
        response = await ai.generate(prompt=_PROMPT, model=_MODEL)
        _force_flush()

    assert response.finish_reason == FinishReason.FAILED
    attrs = _chat_attrs(cloud)
    assert attrs['gen_ai.request.model'] == _MODEL
    for key in _CONTENT_KEYS:
        assert key not in attrs
    assert _PROMPT not in ' '.join(str(value) for value in attrs.values())
