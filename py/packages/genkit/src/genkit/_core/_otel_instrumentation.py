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

"""Built-in OpenTelemetry Instrumentation provider."""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from typing import Any, TypeVar

from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import Link, NoOpTracer, NoOpTracerProvider, ProxyTracerProvider, SpanKind, StatusCode
from opentelemetry.util import types
from pydantic import BaseModel

from ._environment import is_dev_environment
from ._error import GenkitError, GenkitInterrupt
from ._instrumentation import configure_instrumentation, instrumentations, is_instrumented_by
from ._instrumentation_api import Instrumentation, SpanContext, SpanMetadata
from ._logger import get_logger
from ._trace._attrs import Attr, State, metadata_key
from ._trace._default_exporter import TraceServerExporter, create_span_processor, init_telemetry_server_exporter
from ._trace._path import build_path

logger = get_logger(__name__)

T = TypeVar('T')

parent_path_context: ContextVar[str] = ContextVar('genkit_parent_path', default='')


def to_json_attr(value: object) -> str:
    """Serialize an arbitrary object for an input/output span attribute."""
    if isinstance(value, BaseModel):
        return value.model_dump_json(by_alias=True, exclude_none=True)
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def start_attributes(
    metadata: SpanMetadata,
    *,
    qualified_path: str,
) -> dict[str, Any]:
    """Attrs known when the span begins (identity/shape + input).

    Live-trace export snapshots the span the instant it starts, so these have to
    be on the span *before* start returns; otherwise Dev UI shows a blank
    in-progress entry until the span ends. State/output stay out — they aren't
    known until the body finishes.
    """
    attrs: dict[str, Any] = {}
    if metadata.attributes:
        attrs.update(metadata.attributes)
    attrs.update({
        Attr.NAME: metadata.name,
        Attr.PATH: qualified_path,
        Attr.QUALIFIED_PATH: qualified_path,
    })
    if metadata.action_type:
        attrs[Attr.TYPE] = metadata.action_type
    if metadata.subtype:
        attrs[Attr.SUBTYPE] = metadata.subtype
    if metadata.is_root:
        attrs[Attr.IS_ROOT] = True
    if metadata.metadata:
        for meta_key, meta_value in metadata.metadata.items():
            attrs[metadata_key(meta_key)] = str(meta_value)
    if metadata.input is not None:
        attrs[Attr.INPUT] = to_json_attr(metadata.input)
    if metadata.init is not None:
        attrs[Attr.INIT] = to_json_attr(metadata.init)
    return attrs


class OtelSpanContext:
    """SpanContext backed by an OpenTelemetry span."""

    def __init__(self, span: trace_api.Span) -> None:
        self._span = span
        self._output: object | None = None
        self._output_set = False

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
        if not self._span.is_recording():
            return
        for key, value in metadata.items():
            try:
                value_string = value if isinstance(value, str) else to_json_attr(value)
            except Exception as e:
                value_string = f'Error encoding metadata: {e}'
            self._span.set_attribute(metadata_key(key), value_string)

    @property
    def output_was_set(self) -> bool:
        return self._output_set

    def set_output(self, value: object) -> None:
        self._output = value
        self._output_set = True
        if self._span.is_recording():
            self._span.set_attribute(Attr.OUTPUT, to_json_attr(value))


class OtelInstrumentation:
    """Built-in OpenTelemetry provider.

    Records each action as a span with ``genkit:*`` attributes. Pass
    ``tracer_provider`` to mint on your provider; Cloud Trace exporters
    hang there too. Omit it to use the process-global provider.

    ``genkit start`` installs this for you when a collector URL is set.
    Construct it yourself only to add a backend or to own the provider.
    """

    def __init__(self, *, tracer_provider: TracerProvider | None = None) -> None:
        if tracer_provider is not None and not isinstance(tracer_provider, TracerProvider):
            cls = type(tracer_provider)
            raise TypeError(f'OtelInstrumentation expected a TracerProvider, got {cls.__module__}.{cls.__qualname__}')
        self._tracer_provider = tracer_provider
        self._cached_tracer: trace_api.Tracer | None = None

    @property
    def tracer(self) -> trace_api.Tracer:
        if self._cached_tracer is None:
            provider = self._tracer_provider or trace_api.get_tracer_provider()
            self._cached_tracer = provider.get_tracer('genkit-tracer', 'v1')
        return self._cached_tracer

    async def run_in_new_span(
        self,
        metadata: SpanMetadata,
        next: Callable[[SpanContext], Awaitable[T]],
    ) -> T:
        qualified_path = build_path(
            metadata.name,
            parent_path_context.get(),
            metadata.action_type or '',
            metadata.subtype,
        )
        start_attrs = start_attributes(metadata, qualified_path=qualified_path)
        path_token = parent_path_context.set(qualified_path)
        try:
            with self.tracer.start_as_current_span(
                name=metadata.name,
                attributes=start_attrs,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                ctx = OtelSpanContext(span)
                try:
                    result = await next(ctx)
                    if not ctx.output_was_set and result is not None:
                        span.set_attribute(Attr.OUTPUT, to_json_attr(result))
                    span.set_attribute(Attr.STATE, State.SUCCESS)
                    return result
                except GenkitInterrupt:
                    span.set_attribute(Attr.STATE, State.SUCCESS)
                    raise
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as e:
                    logger.debug(f'Error in run_in_new_span: {e!s}')
                    logger.debug(traceback.format_exc())
                    span.set_attribute(Attr.STATE, State.ERROR)
                    err_text = e.original_message if isinstance(e, GenkitError) else str(e)
                    span.set_attribute(Attr.ERROR, err_text)
                    span.set_status(StatusCode.ERROR, str(e))
                    span.record_exception(e)
                    raise
        finally:
            parent_path_context.reset(path_token)


def init_provider() -> TracerProvider:
    """Init and return the global tracer provider."""
    tracer_provider = trace_api.get_tracer_provider()

    if tracer_provider is None or not isinstance(tracer_provider, TracerProvider):  # pyright: ignore[reportUnnecessaryComparison]
        tracer_provider = TracerProvider()
        trace_api.set_tracer_provider(tracer_provider)
        # Booting a tracer shouldn't rewrite the process log format.
        # pyrefly: ignore[missing-attribute]
        LoggingInstrumentor().instrument(set_logging_format=False)
        logger.debug('Creating a new global tracer provider for telemetry.')

    if not isinstance(tracer_provider, TracerProvider):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f'The current trace provider is not an instance of TracerProvider.  It is of type: {type(tracer_provider)}'
        )

    return tracer_provider


def is_placeholder_provider(provider: object) -> bool:
    """True when the global provider is still OTel's unset proxy / no-op."""
    return isinstance(provider, (ProxyTracerProvider, NoOpTracerProvider))


def provider_for_exporters() -> tuple[TracerProvider, bool]:
    """Provider minting Genkit OTel spans, and whether they handed it to us.

    If they already configured ``OtelInstrumentation(tracer_provider=theirs)``,
    exporters have to land on ``theirs`` or Cloud Trace stays empty. Otherwise
    the global provider.
    """
    for inst in instrumentations:
        if not isinstance(inst, OtelInstrumentation):
            continue
        provider = inst._tracer_provider
        if provider is None:
            continue
        if not isinstance(provider, TracerProvider):
            raise TypeError(
                'Cannot attach an exporter: OtelInstrumentation is using '
                f'{type(provider).__name__}, not a TracerProvider.'
            )
        return provider, True
    return init_provider(), False


def add_custom_exporter(exporter: SpanExporter | None, name: str = 'last') -> None:
    """Attach a span exporter to the provider minting Genkit spans.

    If you passed ``tracer_provider=`` to ``OtelInstrumentation``, the
    exporter hangs there. Otherwise the process-global provider. This
    does not turn tracing on. Call ``configure_instrumentation`` so
    spans exist for the exporter to see. Under ``genkit start``,
    ``Genkit()`` does that when a collector is configured.
    """
    if exporter is None:
        logger.warn(f'{name} exporter is None')
        return

    provider, theirs = provider_for_exporters()
    try:
        provider.add_span_processor(create_span_processor(exporter))
        logger.debug(f'{name} exporter added successfully.')
    except Exception:
        logger.error(f'tracing.add_custom_exporter: failed to add exporter {name}')
        logger.exception('Failed to add custom exporter')
        if theirs:
            raise


def maybe_configure_otel_for_exporters() -> None:
    """Turn on OTel when nothing else will mint Genkit spans.

    Skip if you already configured ``OtelInstrumentation``, or if
    ``Genkit()`` under ``genkit start`` will still attach the Developer
    UI collector (``GENKIT_ENV=dev`` and a collector URL). Call after
    exporters are attached.
    """
    if is_instrumented_by(OtelInstrumentation):
        return
    if is_dev_environment() and os.environ.get('GENKIT_TELEMETRY_SERVER'):
        return
    configure_instrumentation(OtelInstrumentation())


developer_ui_collector_connected = False


def reset_developer_ui_collector() -> None:
    """Forget the handshake / notify collector so tests can reconnect."""
    global developer_ui_collector_connected
    developer_ui_collector_connected = False


def otel_for_collector(*, url: str) -> OtelInstrumentation:
    """Mint Genkit spans and POST them to this Developer UI collector."""
    provider = TracerProvider()
    provider.add_span_processor(create_span_processor(TraceServerExporter(telemetry_server_url=url)))
    return OtelInstrumentation(tracer_provider=provider)


def connect_developer_ui_collector(*, url: str) -> None:
    """Point the Developer UI collector at this URL."""
    global developer_ui_collector_connected
    if not url or developer_ui_collector_connected:
        return
    developer_ui_collector_connected = True
    configure_instrumentation(otel_for_collector(url=url))


def genkit_dev_instrumentation() -> Instrumentation | None:
    """OTel provider for the Developer UI, or None when no collector URL is set.

    ``genkit start -- python app.py`` sets ``GENKIT_TELEMETRY_SERVER``
    before spawn; ``Genkit()`` calls this.
    """
    exporter = init_telemetry_server_exporter()
    if exporter is None:
        return None
    provider = TracerProvider()
    provider.add_span_processor(create_span_processor(exporter))
    return OtelInstrumentation(tracer_provider=provider)


class PluginTracer:
    """Follows the provider minting Genkit spans. No-op when uninstrumented."""

    def inner(self) -> trace_api.Tracer:
        if not is_instrumented_by(OtelInstrumentation):
            return NoOpTracer()
        for inst in instrumentations:
            if isinstance(inst, OtelInstrumentation):
                return inst.tracer
        return NoOpTracer()

    def start_as_current_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: types.Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Any:  # noqa: ANN401
        # Imagen and Veo open their spans on this name. Keep it a real method
        # so those call sites stay valid even when no provider is minting yet.
        return self.inner().start_as_current_span(
            name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
            end_on_exit=end_on_exit,
        )

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self.inner(), name)


# Plugins import this as ``from genkit.plugin_api import tracer``.
tracer = PluginTracer()
