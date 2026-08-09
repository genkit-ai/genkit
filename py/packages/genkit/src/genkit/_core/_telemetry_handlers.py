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

"""OpenTelemetry-backed Genkit telemetry handlers.

Owns span creation, ``gen_ai.*`` enrichment, and Dev UI provider wiring.
Registered on the dispatcher in ``_tracing``.
"""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import Token
from typing import Any

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import NoOpTracerProvider, ProxyTracerProvider, StatusCode

from ._error import GenkitError, GenkitInterrupt
from ._logger import get_logger
from ._telemetry_contract import (
    SpanHookParams,
    bind_annotate_flush,
    bind_annotate_projector,
    current_frame,
    unbind_annotate_flush,
    unbind_annotate_projector,
)
from ._telemetry_gen_ai import (
    apply_gen_ai_start_attrs,
    gen_ai_span_name,
    is_model_span,
    project_gen_ai_from_frame,
)
from ._trace._attrs import Attr, State
from ._trace._default_exporter import create_span_processor, init_telemetry_server_exporter

logger = get_logger(__name__)

tracer = trace_api.get_tracer('genkit-tracer', 'v1')

# Second ensure_dev_telemetry must not attach another realtime processor.
dev_telemetry_installed = False


def current_span_ids() -> tuple[str, str]:
    """Return ``(trace_id_hex, span_id_hex)`` for the active span."""
    ctx = trace_api.get_current_span().get_span_context()
    return format(ctx.trace_id, '032x'), format(ctx.span_id, '016x')


def otel_kind(kind: str) -> trace_api.SpanKind:
    if kind == 'client':
        return trace_api.SpanKind.CLIENT
    return trace_api.SpanKind.INTERNAL


async def otel_ai_semantic_conventions(
    params: SpanHookParams,
    next_fn: Callable[[], Awaitable[Any]],
) -> Any:  # noqa: ANN401
    """Map Genkit span facts onto ``gen_ai.*``; create nothing; call next_fn.

    Installs a span-scoped annotate projector so mid/end Genkit writes are
    projected too — without baking GenAI into :func:`annotate` itself. Drop or
    replace this handler to swap conventions.
    """
    apply_gen_ai_start_attrs(name=params.name, attrs=params.attributes)
    # Model spans are client calls to an inference provider.
    if is_model_span(params.attributes):
        params.kind = 'client'

    def project() -> None:
        frame = current_frame()
        if frame is not None:
            project_gen_ai_from_frame(frame)

    token = bind_annotate_projector(project)
    try:
        return await next_fn()
    finally:
        unbind_annotate_projector(token)


def span_name(name: str, attrs: dict[str, Any]) -> str:
    return gen_ai_span_name(name, attrs)


@contextmanager
def start_genkit_span(
    name: str,
    attrs: dict[str, Any],
    *,
    links: list[Any] | None = None,
    kind: str = 'internal',
) -> Generator[trace_api.Span, None, None]:
    """Open a Genkit OTel span as a context manager (tests / second renderers)."""
    with tracer.start_as_current_span(
        span_name(name, attrs),
        kind=otel_kind(kind),
        links=links,
        attributes=attrs,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        yield span


def bind_span_annotate_flush(span: trace_api.Span) -> Token[Callable[[str, Any], None] | None]:
    """Mirror frame annotate() writes onto this span while it is open."""

    def flush(key: str, value: Any) -> None:  # noqa: ANN401
        if span.is_recording():
            span.set_attribute(key, value)

    return bind_annotate_flush(flush)


def unbind_span_annotate_flush(flush_token: Token[Callable[[str, Any], None] | None]) -> None:
    unbind_annotate_flush(flush_token)


async def otel_renderer(
    params: SpanHookParams,
    next_fn: Callable[[], Awaitable[Any]],
) -> Any:  # noqa: ANN401
    """The one span-creating handler: run next_fn inside an OTel span.

    Also stamps Dev UI outcome attrs (``genkit:state`` / ``genkit:error``) after
    the body finishes. Call sites write ``genkit:output`` via ``annotate_output``.
    """
    with tracer.start_as_current_span(
        span_name(params.name, params.attributes),
        kind=otel_kind(params.kind),
        links=params.links,
        attributes=params.attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        flush_token = bind_span_annotate_flush(span)
        try:
            # TODO(design): span-end ownership for streams — if next_fn returns a
            # generator/async-generator it escapes and the span ends before the
            # first chunk is consumed.
            result = await next_fn()
            span.set_attribute(Attr.STATE, State.SUCCESS)
            return result
        except GenkitInterrupt:
            # HITL / tool pause — control flow, not a failed span.
            span.set_attribute(Attr.STATE, State.SUCCESS)
            raise
        except (asyncio.CancelledError, KeyboardInterrupt):
            # Abort/timeout — leave genkit:state unset so metrics don't count
            # unfinished work as success or failure.
            raise
        except Exception as e:
            logger.debug(f'Error in run_in_new_span: {e!s}')
            logger.debug(traceback.format_exc())
            span.set_attribute(Attr.STATE, State.ERROR)
            err_text = e.original_message if isinstance(e, GenkitError) else str(e)
            span.set_attribute(Attr.ERROR, err_text)
            span.set_status(StatusCode.ERROR)
            span.record_exception(e)
            span.set_attribute('error.type', type(e).__name__)
            raise
        finally:
            unbind_span_annotate_flush(flush_token)


def init_provider() -> TracerProvider:
    """Inits and returns the tracer global provider."""
    tracer_provider = trace_api.get_tracer_provider()

    if tracer_provider is None or not isinstance(tracer_provider, TracerProvider):  # pyright: ignore[reportUnnecessaryComparison]
        tracer_provider = TracerProvider()
        trace_api.set_tracer_provider(tracer_provider)
        # pyrefly: ignore[missing-attribute] - LoggingInstrumentor has instrument() method
        LoggingInstrumentor().instrument(set_logging_format=True)
        logger.debug('Creating a new global tracer provider for telemetry.')

    if not isinstance(tracer_provider, TracerProvider):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f'The current trace provider is not an instance of TracerProvider.  It is of type: {type(tracer_provider)}'
        )

    return tracer_provider


def add_custom_exporter(exporter: SpanExporter | None, name: str = 'last') -> None:
    """Adds custom span exporter to current tracer provider."""
    current_provider = init_provider()

    try:
        if exporter is None:
            logger.warn(f'{name} exporter is None')
            return

        processor = create_span_processor(exporter)
        current_provider.add_span_processor(processor)
        logger.debug(f'{name} exporter added successfully.')
    except Exception:
        logger.error(f'tracing.add_custom_exporter: failed to add exporter {name}')
        logger.exception('Failed to add custom exporter')


def is_placeholder_provider(provider: object) -> bool:
    """True when the global provider is still OTel's unset proxy / no-op."""
    return isinstance(provider, (ProxyTracerProvider, NoOpTracerProvider))


def ensure_dev_telemetry() -> None:
    """Wire Dev UI export. Only runs when GENKIT_ENV=dev.

    Claims the global provider only while it is still OTel's placeholder. A
    real provider (SDK or custom) is left alone and piggybacked via
    ``add_span_processor`` when available so we never wipe someone else's
    telemetry setup. Idempotent: a second call does not attach again.
    """
    global dev_telemetry_installed
    if dev_telemetry_installed:
        return

    exporter = init_telemetry_server_exporter()
    if exporter is None:
        return

    processor = create_span_processor(exporter)
    current = trace_api.get_tracer_provider()

    if is_placeholder_provider(current):
        provider = TracerProvider(sampler=ALWAYS_ON)
        trace_api.set_tracer_provider(provider)
        claimed = trace_api.get_tracer_provider()
        if claimed is provider:
            provider.add_span_processor(processor)
            # pyrefly: ignore[missing-attribute]
            LoggingInstrumentor().instrument(set_logging_format=True)
            dev_telemetry_installed = True
            logger.debug('Installed minimal dev TracerProvider for telemetry.')
            return
        # Once-guard ignored our set — fall through and piggyback on whatever won.
        logger.warning(
            'set_tracer_provider was ignored (provider already set); '
            'falling through to attach dev processor on the active provider.'
        )
        current = claimed

    add_span_processor = getattr(current, 'add_span_processor', None)
    if callable(add_span_processor):
        add_span_processor(processor)
        dev_telemetry_installed = True
        logger.debug('Attached dev telemetry processor to existing tracer provider.')
        return

    logger.warning(
        'Cannot attach Genkit dev telemetry: tracer provider %s has no add_span_processor',
        type(current).__name__,
    )
