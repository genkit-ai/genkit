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

"""Developer UI trace poster that POSTs OTLP/JSON. No OpenTelemetry runtime.

Logs reach the Developer UI through Genkit's own log exporter, not here.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
import time
import urllib.request
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from queue import Full, Queue
from typing import Literal, TypeVar
from urllib.parse import urljoin, urlparse

from .._environment import is_dev_environment
from .._error import GenkitError, Interrupt
from .._logger import get_logger
from .._reflection_config import reflection_enabled
from ._attrs import Attr, State, metadata_key
from ._instrumentation import (
    Instrumentation,
    SpanMetadata,
    SpanNext,
    configure_instrumentation,
    is_instrumented_by,
    parent_path_context,
    span_is_action,
    start_attributes,
    to_json_attr,
)
from ._log_exporter import QUEUE_SIZE, put_poison_pill
from ._path import build_path

logger = get_logger(__name__)

T = TypeVar('T')

TRACE_HEADERS = {'Content-Type': 'application/json', 'Accept': 'application/json'}
EXPORT_TIMEOUT_SECONDS = 300
# The Developer UI flushes right before it reads a trace. A hung collector
# should cost it a couple of seconds, not the full export timeout.
FLUSH_TIMEOUT_SECONDS = 2.0


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


_parent_span: ContextVar[ActiveSpan | None] = ContextVar('genkit_direct_http_parent', default=None)


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

    def set_state(self, state: Literal['success', 'error']) -> None:
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
    """POSTs OTLP/JSON spans to the collector from one background worker."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.closed = False
        self.queue: Queue[str | None] = Queue(maxsize=QUEUE_SIZE)
        self.worker = threading.Thread(target=self.run_worker, name='genkit-trace-export', daemon=True)
        self.worker.start()

    def export_spans(self, spans: list[ActiveSpan], *, resource_attributes: dict[str, object]) -> None:
        if self.closed or not spans:
            return
        payload = {'resourceSpans': [encode_span(span, resource_attributes=resource_attributes) for span in spans]}
        try:
            self.queue.put_nowait(json.dumps(payload))
        except Full:
            logger.debug('Developer UI trace export queue is full; dropping spans')

    def run_worker(self) -> None:
        while True:
            body = self.queue.get()
            try:
                if body is None:
                    return
                post_json(url=self.url, body=body)
            except Exception as e:
                logger.debug('Failed to export spans: %s', e)
            finally:
                self.queue.task_done()

    def flush(self) -> None:
        with self.queue.all_tasks_done:
            self.queue.all_tasks_done.wait_for(lambda: self.queue.unfinished_tasks == 0, timeout=FLUSH_TIMEOUT_SECONDS)

    def shutdown(self) -> None:
        self.closed = True
        self.flush()
        put_poison_pill(queue=self.queue)


class DirectHttpInstrumentation:
    """HTTP poster that mints Genkit spans and writes them to a sink.

    Testers pass a recording sink. The Developer UI Traces tab uses
    :class:`DevUIInstrumentation`, a distinct type, so a test sink
    does not count as the tab already being on.
    """

    def __init__(
        self,
        sink: CollectorHttpSink,
        *,
        resource_attributes: dict[str, object] | None = None,
    ) -> None:
        self.sink = sink
        self.resource_attributes = resource_attributes or {'service.name': 'genkit-python'}

    async def run_in_new_span(
        self,
        metadata: SpanMetadata,
        next: SpanNext[T],
    ) -> T:
        parent = _parent_span.get()
        is_action = span_is_action.get()
        qualified_path = build_qualified_path(metadata, is_action=is_action)
        span = ActiveSpan(
            trace_id=parent.trace_id if parent is not None else new_trace_id(),
            span_id=new_span_id(),
            parent_span_id=parent.span_id if parent is not None else None,
            name=metadata.name,
            start_time_unix_nano=now_unix_nano(),
            attributes=start_attributes(metadata, qualified_path=qualified_path, is_action=is_action),
        )
        self.sink.export_spans([span], resource_attributes=self.resource_attributes)
        path_token = parent_path_context.set(qualified_path)
        parent_token = _parent_span.set(span)
        ctx = DirectSpanContext(span)
        try:
            try:
                result = await next(ctx)
                if result is not None:
                    span.attributes[Attr.OUTPUT] = to_json_attr(result)
                if Attr.STATE not in span.attributes:
                    span.attributes[Attr.STATE] = State.SUCCESS
                    span.status_code = 1
                return result
            except Interrupt:
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
            _parent_span.reset(parent_token)
            parent_path_context.reset(path_token)

    def dispose(self) -> None:
        self.sink.shutdown()

    def flush(self) -> None:
        self.sink.flush()


class DevUIInstrumentation(DirectHttpInstrumentation):
    """The poster that fills the Developer UI Traces tab.

    ``genkit start`` and handshake/notify install this. A recording sink
    used in tests is a plain DirectHttpInstrumentation and does not
    count as the Traces tab already being on.
    """


def reset_parent_span() -> None:
    _parent_span.set(None)


def build_qualified_path(metadata: SpanMetadata, *, is_action: bool = False) -> str:
    if is_action:
        return build_path(metadata.name, parent_path_context.get(), 'action', metadata.action_type)
    return build_path(metadata.name, parent_path_context.get(), metadata.action_type or '')


def telemetry_server_url() -> str | None:
    url = os.environ.get('GENKIT_TELEMETRY_SERVER')
    return url or None


def direct_http_for_collector(*, url: str) -> DevUIInstrumentation:
    return DevUIInstrumentation(CollectorHttpSink(collector_otlp_url(url)))


def genkit_dev_instrumentation() -> Instrumentation | None:
    """Developer UI poster, or None when no collector URL is set.

    ``genkit start -- python app.py`` sets ``GENKIT_TELEMETRY_SERVER``
    before spawn; ``Genkit()`` calls this. Does not boot OpenTelemetry.
    """
    url = telemetry_server_url()
    if url is None:
        return None
    return direct_http_for_collector(url=url)


def connect_developer_ui_collector(*, url: str) -> None:
    """Turn on the Developer UI poster from a handshake / notify URL.

    No-op when the URL is empty or the poster is already registered, so
    ``Genkit()`` under ``genkit start`` and a later notify cannot double-post.
    A ``GENKIT_TELEMETRY_SERVER`` already in the production shell (no poster
    yet) does not block this — today's handshake URL still fills the Traces tab.
    """
    if not url:
        return
    if is_instrumented_by(DevUIInstrumentation):
        return
    configure_instrumentation(direct_http_for_collector(url=url))


def maybe_inject_dev_instrumentation() -> None:
    """``Genkit()`` installs the poster once when a collector URL is set.

    Keyed on reflection being on rather than GENKIT_ENV alone, so a non-dev
    runtime with reflection enabled still posts traces to the server it was
    given.
    """
    if not is_dev_environment() and not reflection_enabled():
        return
    if is_instrumented_by(DevUIInstrumentation):
        return
    inst = genkit_dev_instrumentation()
    if inst is not None:
        configure_instrumentation(inst)
