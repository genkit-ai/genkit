# Copyright 2025 Google LLC
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


"""Telemetry and tracing functionality for the Genkit framework."""

import json
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, Literal

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from ._base import GenkitModel
from ._environment import is_dev_environment
from ._error import GenkitError
from ._logger import get_logger
from ._trace._attrs import Attr, State, metadata_key
from ._trace._default_exporter import create_span_processor, init_telemetry_server_exporter
from ._trace._path import build_path

logger = get_logger(__name__)


class SpanMetadata(GenkitModel):
    """Input parameters for opening a Genkit span via :func:`run_in_new_span`.

    Mapping from SpanMetadata to span attributes (see ``Attr`` for wire names):
      - name                 -> Attr.NAME (and span name)
      - input / output       -> Attr.INPUT / Attr.OUTPUT (JSON-serialized)
      - type                 -> Attr.TYPE
      - subtype              -> Attr.SUBTYPE
      - metadata[k]          -> metadata_key(k)
      - telemetry_labels[k]  -> <k> verbatim (caller-controlled keys)

    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel, extra='forbid', populate_by_name=True, arbitrary_types_allowed=True
    )

    name: str = Field(...)
    state: Literal['success', 'error'] | None = None
    input: Any | None = Field(default=None)
    output: Any | None = Field(default=None)
    is_root: bool | None = None
    metadata: dict[str, Any] | None = None
    path: str | None = None
    type: str | None = None
    subtype: str | None = None
    telemetry_labels: dict[str, str] | None = None


tracer = trace_api.get_tracer('genkit-tracer', 'v1')


# Qualified ``genkit:path`` of the active span; pushed by ``run_in_new_span`` so
# nested spans can build child paths.
_parent_path_context: ContextVar[str] = ContextVar('genkit_parent_path', default='')


@contextmanager
def push_parent_path(path: str) -> Generator[None, None, None]:
    """Push ``path`` as the active parent path for nested spans; restore on exit."""
    token = _parent_path_context.set(path)
    try:
        yield
    finally:
        _parent_path_context.reset(token)


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
    """Adds custom span exporter to current tracer provider.

    Args:
        exporter: Custom or dedicated span exporter.
        name: Name of the span exporter. Only for logging purposes.
    """
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


if is_dev_environment():
    add_custom_exporter(init_telemetry_server_exporter(), 'local_telemetry_server')


def _to_json_attr(value: object) -> str:
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
) -> dict[str, str | bool]:
    """Attrs known when the span begins (identity/shape + input).

    Live-trace export snapshots the span the instant it starts, so these have to
    be on the span *before* start returns; otherwise Dev UI shows a blank
    in-progress entry until the span ends. State/output are excluded — they
    aren't known until the body finishes.
    """
    attrs: dict[str, str | bool] = {}
    if metadata.telemetry_labels:
        attrs.update(metadata.telemetry_labels)
    attrs.update({
        Attr.NAME: metadata.name,
        Attr.PATH: qualified_path,
        Attr.QUALIFIED_PATH: qualified_path,
    })
    if metadata.type:
        attrs[Attr.TYPE] = metadata.type
    if metadata.subtype:
        attrs[Attr.SUBTYPE] = metadata.subtype
    if metadata.is_root:
        attrs[Attr.IS_ROOT] = True
    if metadata.metadata:
        for meta_key, meta_value in metadata.metadata.items():
            attrs[metadata_key(meta_key)] = str(meta_value)
    if metadata.input is not None:
        attrs[Attr.INPUT] = _to_json_attr(metadata.input)
    return attrs


@contextmanager
def record_span_outcome(
    span: trace_api.Span,
    metadata: SpanMetadata,
) -> Generator[None, None, None]:
    """Write success or error attrs after the span body finishes."""
    try:
        yield
    except Exception as e:
        logger.debug(f'Error in run_in_new_span: {e!s}')
        logger.debug(traceback.format_exc())
        span.set_attribute(Attr.STATE, State.ERROR)
        err_text = e.original_message if isinstance(e, GenkitError) else str(e)
        span.set_attribute(Attr.ERROR, err_text)
        span.set_status(status=trace_api.StatusCode.ERROR, description=str(e))
        span.record_exception(e)
        raise

    if metadata.output is not None:
        span.set_attribute(Attr.OUTPUT, _to_json_attr(metadata.output))
    span.set_attribute(Attr.STATE, State.SUCCESS)


@contextmanager
def run_in_new_span(
    metadata: SpanMetadata,
    links: list[trace_api.Link] | None = None,
) -> Generator[trace_api.Span, None, None]:
    """Starts a new span context under the current trace.

    All Genkit-specific attributes are derived from ``metadata``; caller-controlled
    passthrough attributes go via ``metadata.telemetry_labels``.

    Args:
        metadata: Span metadata. See :class:`SpanMetadata` for field routing.
        links: Optional span links.

    Yields:
        The OpenTelemetry Span object.
    """
    qualified_path = build_path(metadata.name, _parent_path_context.get(), metadata.type or '', metadata.subtype)
    # Seed start-known attrs as span-start options so RealtimeSpanProcessor's
    # on_start export already carries them for the Dev UI.
    start_attrs = start_attributes(metadata, qualified_path=qualified_path)

    with push_parent_path(qualified_path):
        with tracer.start_as_current_span(name=metadata.name, links=links, attributes=start_attrs) as span:
            with record_span_outcome(span, metadata):
                yield span
