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

"""Developer UI telemetry that POSTs OTLP/JSON. No OpenTelemetry runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import threading
import time
import urllib.request
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TypeVar
from urllib.parse import urljoin, urlparse

from .._environment import is_dev_environment
from .._error import GenkitError, GenkitInterrupt
from .._logger import get_logger
from ._attrs import Attr, State, metadata_key
from ._instrumentation import (
    Instrumentation,
    SpanMetadata,
    SpanNext,
    configure_instrumentation,
    instrumentations,
    is_instrumented_by,
    parent_path_context,
    start_attributes,
    to_json_attr,
)
from ._path import build_path

logger = get_logger(__name__)

T = TypeVar('T')

TRACE_HEADERS = {'Content-Type': 'application/json', 'Accept': 'application/json'}
EXPORT_TIMEOUT_SECONDS = 300
SINK_LOGGER_NAME = 'CollectorHttpSink'


class GenkitBuiltinInstrumentation:
    """Marker on the Developer UI poster so we never inject it twice."""


@dataclass
class ActiveSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time_unix_nano: int
    end_time_unix_nano: int = 0
    attributes: dict[str, object] = field(default_factory=dict)
    status_code: int = 0
    status_message: str | None = None
    output_was_set: bool = False


PARENT_SPAN: ContextVar[ActiveSpan | None] = ContextVar('genkit_direct_http_parent', default=None)


class DirectSpanContext:
    """Span handle that writes ``genkit:*`` attributes onto the live span."""

    def __init__(self, span: ActiveSpan) -> None:
        self._span = span

    @property
    def trace_id(self) -> str:
        return self._span.trace_id

    @property
    def span_id(self) -> str:
        return self._span.span_id

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        for key, value in metadata.items():
            try:
                encoded = value if isinstance(value, str) else to_json_attr(value)
            except Exception as e:
                encoded = f'Error encoding metadata: {e}'
            self._span.attributes[metadata_key(str(key))] = encoded

    def set_output(self, value: object) -> None:
        self._span.output_was_set = True
        self._span.attributes[Attr.OUTPUT] = to_json_attr(value)

    def set_state(self, state: str) -> None:
        self._span.attributes[Attr.STATE] = state
        if state == State.ERROR:
            self._span.status_code = 2


def now_unix_nano() -> int:
    return time.time_ns()


def new_trace_id() -> str:
    return secrets.token_hex(16)


def new_span_id() -> str:
    return secrets.token_hex(8)


def encode_attribute_value(value: object) -> dict[str, object]:
    if isinstance(value, str):
        return {'stringValue': value}
    if isinstance(value, bool):
        return {'boolValue': value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {'intValue': value}
    if isinstance(value, float):
        return {'doubleValue': value}
    return {'stringValue': str(value)}


def encode_attributes(attributes: dict[str, object]) -> list[dict[str, object]]:
    return [{'key': key, 'value': encode_attribute_value(value)} for key, value in attributes.items()]


def encode_span(span: ActiveSpan, *, resource_attributes: dict[str, object]) -> dict[str, object]:
    body: dict[str, object] = {
        'traceId': span.trace_id,
        'spanId': span.span_id,
        'name': span.name,
        'kind': 1,
        'startTimeUnixNano': str(span.start_time_unix_nano),
        'endTimeUnixNano': str(span.end_time_unix_nano),
        'attributes': encode_attributes(span.attributes),
        'droppedAttributesCount': 0,
        'events': [],
        'droppedEventsCount': 0,
        'status': {'code': span.status_code, 'message': span.status_message},
        'links': [],
        'droppedLinksCount': 0,
    }
    if span.parent_span_id:
        body['parentSpanId'] = span.parent_span_id
    return {
        'resource': {
            'attributes': encode_attributes(resource_attributes),
            'droppedAttributesCount': 0,
        },
        'scopeSpans': [
            {
                'scope': {'name': 'genkit-python', 'version': ''},
                'spans': [body],
            }
        ],
    }


def encode_log(
    *,
    time_unix_nano: int,
    severity_number: int,
    severity_text: str,
    body: object,
    attributes: dict[str, object],
    trace_id: str,
    span_id: str,
    resource_attributes: dict[str, object],
) -> dict[str, object]:
    record: dict[str, object] = {
        'timeUnixNano': str(time_unix_nano),
        'severityNumber': severity_number,
        'severityText': severity_text,
        'body': encode_attribute_value(body if isinstance(body, (str, bool, int, float)) else str(body)),
        'attributes': encode_attributes(attributes),
    }
    if trace_id:
        record['traceId'] = trace_id
    if span_id:
        record['spanId'] = span_id
    return {
        'resource': {
            'attributes': encode_attributes(resource_attributes),
            'droppedAttributesCount': 0,
        },
        'scopeLogs': [
            {
                'scope': {'name': 'genkit-python', 'version': ''},
                'logRecords': [record],
            }
        ],
    }


def collector_otlp_url(server: str) -> str:
    return urljoin(server.rstrip('/') + '/', 'api/otlp')


def post_json(*, url: str, body: str) -> None:
    if urlparse(url).scheme not in ('http', 'https'):
        raise ValueError(f'invalid telemetry server URL {url!r}')
    request = urllib.request.Request(  # noqa: S310 — scheme checked above
        url,
        data=body.encode(),
        headers=TRACE_HEADERS,
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=EXPORT_TIMEOUT_SECONDS) as response:  # noqa: S310
        response.read()


class CollectorHttpSink:
    """Fire-and-forget POST of OTLP/JSON spans and logs to the collector."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.closed = False
        self._inflight: list[threading.Thread] = []
        self._lock = threading.Lock()

    def export_spans(self, spans: list[ActiveSpan], *, resource_attributes: dict[str, object]) -> None:
        if self.closed or not spans:
            return
        payload = {'resourceSpans': [encode_span(span, resource_attributes=resource_attributes) for span in spans]}
        self._post(json.dumps(payload), what='spans')

    def export_logs(self, payload: dict[str, object]) -> None:
        if self.closed:
            return
        self._post(json.dumps(payload), what='logs')

    def flush(self) -> None:
        with self._lock:
            threads = list(self._inflight)
        for thread in threads:
            thread.join(timeout=2)

    def shutdown(self) -> None:
        self.closed = True
        self.flush()

    def _post(self, body: str, *, what: str) -> None:
        thread = threading.Thread(target=self._send, args=(body, what), daemon=True)
        with self._lock:
            self._inflight.append(thread)
        thread.start()

    def _send(self, body: str, what: str) -> None:
        try:
            post_json(url=self.url, body=body)
        except Exception as e:
            logger.debug('Failed to export %s: %s', what, e)


class _LogHandler(logging.Handler):
    def __init__(self, instrumentation: DirectHttpInstrumentation) -> None:
        super().__init__()
        self.instrumentation = instrumentation

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == SINK_LOGGER_NAME:
            return
        self.instrumentation.export_log_record(record)


class DirectHttpInstrumentation:
    """Mints its own ids and POSTs finished spans to the Developer UI collector."""

    def __init__(
        self,
        sink: CollectorHttpSink,
        *,
        resource_attributes: dict[str, object] | None = None,
        capture_logs: bool = True,
    ) -> None:
        self.sink = sink
        self.resource_attributes = resource_attributes or {'service.name': 'genkit-python'}
        self._handler: logging.Handler | None = None
        if capture_logs:
            self._handler = _LogHandler(self)
            logging.getLogger().addHandler(self._handler)

    async def run_in_new_span(
        self,
        metadata: SpanMetadata,
        next: SpanNext[T],
    ) -> T:
        parent = PARENT_SPAN.get()
        qualified_path = build_qualified_path(metadata)
        span = ActiveSpan(
            trace_id=parent.trace_id if parent is not None else new_trace_id(),
            span_id=new_span_id(),
            parent_span_id=parent.span_id if parent is not None else None,
            name=metadata.name,
            start_time_unix_nano=now_unix_nano(),
            attributes=start_attributes(metadata, qualified_path=qualified_path),
        )
        self.sink.export_spans([span], resource_attributes=self.resource_attributes)
        path_token = parent_path_context.set(qualified_path)
        parent_token = PARENT_SPAN.set(span)
        ctx = DirectSpanContext(span)
        try:
            try:
                result = await next(ctx)
                if not span.output_was_set and result is not None:
                    span.attributes[Attr.OUTPUT] = to_json_attr(result)
                if Attr.STATE not in span.attributes:
                    span.attributes[Attr.STATE] = State.SUCCESS
                    span.status_code = 1
                return result
            except GenkitInterrupt:
                span.attributes[Attr.STATE] = State.SUCCESS
                span.status_code = 1
                raise
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as e:
                span.attributes[Attr.STATE] = State.ERROR
                err_text = e.original_message if isinstance(e, GenkitError) else str(e)
                span.attributes[Attr.ERROR] = err_text
                span.status_code = 2
                span.status_message = str(e)
                raise
        finally:
            span.end_time_unix_nano = now_unix_nano()
            self.sink.export_spans([span], resource_attributes=self.resource_attributes)
            PARENT_SPAN.reset(parent_token)
            parent_path_context.reset(path_token)

    def export_log_record(self, record: logging.LogRecord) -> None:
        span = PARENT_SPAN.get()
        attributes: dict[str, object] = {'loggerName': record.name}
        if record.exc_info:
            attributes['error'] = logging.Formatter().formatException(record.exc_info)
        self.sink.export_logs({
            'resourceLogs': [
                encode_log(
                    time_unix_nano=int(record.created * 1_000_000_000),
                    severity_number=severity_number(record.levelno),
                    severity_text=severity_text(record.levelno),
                    body=record.getMessage(),
                    attributes=attributes,
                    trace_id=span.trace_id if span is not None else '',
                    span_id=span.span_id if span is not None else '',
                    resource_attributes=self.resource_attributes,
                )
            ]
        })

    def dispose(self) -> None:
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler = None
        self.sink.shutdown()

    def flush(self) -> None:
        self.sink.flush()


class DirectBuiltin(DirectHttpInstrumentation, GenkitBuiltinInstrumentation):
    """The auto-injected Developer UI poster."""


def build_qualified_path(metadata: SpanMetadata) -> str:
    return build_path(
        metadata.name,
        parent_path_context.get(),
        metadata.action_type or '',
        metadata.subtype,
    )


def severity_number(levelno: int) -> int:
    if levelno >= logging.CRITICAL:
        return 21
    if levelno >= logging.ERROR:
        return 17
    if levelno >= logging.WARNING:
        return 13
    if levelno >= logging.INFO:
        return 9
    return 5


def severity_text(levelno: int) -> str:
    if levelno >= logging.CRITICAL:
        return 'FATAL'
    if levelno >= logging.ERROR:
        return 'ERROR'
    if levelno >= logging.WARNING:
        return 'WARN'
    if levelno >= logging.INFO:
        return 'INFO'
    return 'DEBUG'


def telemetry_server_url() -> str | None:
    url = os.environ.get('GENKIT_TELEMETRY_SERVER')
    return url or None


def direct_http_for_collector(*, url: str) -> DirectBuiltin:
    return DirectBuiltin(CollectorHttpSink(collector_otlp_url(url)))


def genkit_dev_instrumentation() -> Instrumentation | None:
    """Developer UI poster, or None when no collector URL is set.

    ``genkit start -- python app.py`` sets ``GENKIT_TELEMETRY_SERVER``
    before spawn; ``Genkit()`` calls this. Does not boot OpenTelemetry.
    """
    url = telemetry_server_url()
    if url is None:
        return None
    return direct_http_for_collector(url=url)


def enable_dev_instrumentation_for_server(*, url: str) -> None:
    """Turn on the Developer UI poster from a handshake / notify URL.

    No-op when the URL is empty or the poster is already registered, so
    ``Genkit()`` under ``genkit start`` and a later notify cannot double-post.
    A ``GENKIT_TELEMETRY_SERVER`` already in the production shell (no poster
    yet) does not block this — today's handshake URL still fills the Traces tab.
    """
    if not url:
        return
    if is_instrumented_by(GenkitBuiltinInstrumentation):
        return
    configure_instrumentation(direct_http_for_collector(url=url))


def connect_developer_ui_collector(*, url: str) -> None:
    """Handshake / notify entry. Same wiring as ``enable_dev_instrumentation_for_server``."""
    enable_dev_instrumentation_for_server(url=url)


def flush_direct_http_instrumentations() -> None:
    """Wait for in-flight collector POSTs. Tests."""
    for inst in instrumentations:
        flush = getattr(inst, 'flush', None)
        if callable(flush):
            flush()


def maybe_inject_dev_instrumentation() -> None:
    """``Genkit()`` in dev installs the poster once when a collector URL is set."""
    if not is_dev_environment():
        return
    if is_instrumented_by(GenkitBuiltinInstrumentation):
        return
    inst = genkit_dev_instrumentation()
    if inst is not None:
        configure_instrumentation(inst)
