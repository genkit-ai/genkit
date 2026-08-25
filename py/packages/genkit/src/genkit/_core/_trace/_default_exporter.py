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

"""Telemetry and tracing default exporter for the Genkit framework."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from queue import Queue
from typing import Any, cast
from urllib.parse import urljoin, urlparse

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from pydantic import BaseModel

from genkit._core._compat import override
from genkit._core._environment import is_dev_environment
from genkit._core._logger import get_logger
from genkit._core._typing import (
    Annotation,
    InstrumentationLibrary,
    SpanData,
    SpanStatus,
    TimeEvent,
    TimeEvents,
    TraceData,
)

from ._attrs import Attr, Subtype
from ._realtime_processor import RealtimeSpanProcessor

logger = get_logger(__name__)

INSTRUMENTATION = InstrumentationLibrary(name='genkit-tracer', version='v1')
TRACE_HEADERS = {'Content-Type': 'application/json', 'Accept': 'application/json'}
# Five minutes is long enough for a slow local Dev UI and short enough that a
# wedged collector eventually gets a Failed to save trace line.
EXPORT_TIMEOUT_SECONDS = 300
# The export worker is a daemon: once the interpreter starts tearing down it
# is killed, so we give the last span a couple of seconds to land. Ctrl-C of
# a wedged collector still shouldn't sit for minutes.
SHUTDOWN_FLUSH_TIMEOUT_MILLIS = 2_000


def post_trace(*, url: str, body: str) -> None:
    """POST one collector document without going through the process httpx logger."""
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


def resolve_telemetry_server_url(*, telemetry_server_url: str, telemetry_server_endpoint: str) -> str:
    """A typo'd collector URL should fail when tracing starts, not as missing Dev UI traces later."""
    url = telemetry_server_url.strip()
    try:
        joined = urljoin(url, telemetry_server_endpoint)
    except ValueError as error:
        raise ValueError(f'invalid telemetry server URL {telemetry_server_url!r}') from error
    parsed = urlparse(joined)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError(f'invalid telemetry server URL {telemetry_server_url!r}')
    return url


def _ns_to_ms(ns: int | None) -> float:
    return ns / 1_000_000 if ns is not None else 0


def _otel_event_attributes_to_json(attrs: object | None) -> dict[str, Any]:
    """Flatten OTel event attributes for JSON / Dev UI (expects string keys and JSON-safe values)."""
    if attrs is None:
        return {}
    out: dict[str, Any] = {}
    try:
        items_getter = getattr(attrs, 'items', None)
        if callable(items_getter):
            items = cast(Callable[[], Iterable[tuple[Any, Any]]], items_getter)()
        else:
            items = ()
        for k, v in items:
            key = str(k)
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[key] = v
            else:
                out[key] = str(v)
    except (TypeError, ValueError):
        pass
    return out


def json_safe_attributes(*, attrs: Mapping[Any, Any] | None) -> dict[str, Any]:
    """Drop values the collector cannot store so the rest of the trace still lands."""
    if not attrs:
        return {}
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        name = str(key)
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            logger.warning(f'skipped span attribute {name!r} ({type(value).__name__})')
            continue
        out[name] = value
    return out


def ensure_exception_message(*, span: SpanData) -> None:
    """Dev UI reads the first exception event's message and shows "Error" if it's missing.

    Status uses ``message``, not OTel's ``description``. When the span failed
    but nobody recorded an exception event, copy the status or error attr onto
    one so the trace still names what broke.
    """
    status = span.status
    if status is None or status.code != 2:
        return
    message = status.message or span.attributes.get(Attr.ERROR)
    if not message:
        return
    if not status.message:
        status.message = message
    time_events = span.time_events or TimeEvents(time_event=[])
    span.time_events = time_events
    event_list = time_events.time_event or []
    time_events.time_event = event_list
    for event in event_list:
        if event.annotation.description != 'exception':
            continue
        if event.annotation.attributes.get('exception.message'):
            return
        event.annotation.attributes['exception.message'] = message
        return
    event_list.append(
        TimeEvent(
            time=span.end_time,
            annotation=Annotation(
                description='exception',
                attributes={
                    'exception.type': 'Error',
                    'exception.message': message,
                },
            ),
        )
    )


def events_to_time_events(*, span: ReadableSpan) -> TimeEvents:
    """Copy OTel events onto the collector document so Dev UI can show them."""
    events = getattr(span, 'events', None) or ()
    time_event: list[TimeEvent] = []
    for ev in events:
        name = getattr(ev, 'name', None) or 'event'
        ts = getattr(ev, 'timestamp', None)
        raw_attrs = getattr(ev, 'attributes', None) or {}
        time_event.append(
            TimeEvent(
                time=_ns_to_ms(ts),
                annotation=Annotation(
                    description=name,
                    attributes=_otel_event_attributes_to_json(raw_attrs),
                ),
            )
        )
    return TimeEvents(time_event=time_event)


def extract_span_data(span: ReadableSpan) -> TraceData:
    """Convert a finished span into the collector document.

    Requires a span context. Values the collector cannot store are dropped
    so generate() still returns and the rest of the document still POSTs.
    """
    ctx = span.context
    if ctx is None:
        raise TypeError('span context is required')
    trace_id = format(ctx.trace_id, '032x')
    span_id = format(ctx.span_id, '016x')
    parent_id = format(span.parent.span_id, '016x') if span.parent else None
    start = _ns_to_ms(span.start_time)
    end = _ns_to_ms(span.end_time)

    status: SpanStatus | None = None
    if span.status:
        status = SpanStatus(
            code=trace_api.StatusCode(span.status.status_code).value,
            message=span.status.description,
        )

    entry = SpanData(
        span_id=span_id,
        trace_id=trace_id,
        start_time=start,
        end_time=end,
        attributes=json_safe_attributes(attrs=span.attributes),
        display_name=span.name,
        span_kind=trace_api.SpanKind(span.kind).name,
        instrumentation_library=INSTRUMENTATION,
        time_events=events_to_time_events(span=span),
        parent_span_id=parent_id,
        status=status,
    )
    ensure_exception_message(span=entry)

    result = TraceData(trace_id=trace_id, spans={span_id: entry})
    if not span.parent:
        result.display_name = span.name
        result.start_time = start
        result.end_time = end
    return result


def build_trace_payload(*, spans: Sequence[ReadableSpan]) -> TraceData:
    """One collector document for a batch of spans that share a trace id.

    Production flushes a whole flow at once. The store keys documents by
    trace id, so one POST per id is enough.
    """
    payload = TraceData(trace_id='', spans={})
    for span in spans:
        part = extract_span_data(span)
        payload.trace_id = part.trace_id
        payload.spans.update(part.spans)
        if part.display_name is not None:
            payload.display_name = part.display_name
            payload.start_time = part.start_time
            payload.end_time = part.end_time
    return payload


def encode_trace(*, trace: TraceData) -> str:
    """JSON for the collector. Attributes that cannot encode were already dropped."""
    return json.dumps(BaseModel.model_dump(trace, by_alias=True, exclude_none=True))


DEFAULT_SPAN_FILTERS: dict[str, str] = {
    # Suppress prompt runner preview traces (triggered on every keystroke in Dev UI)
    Attr.SUBTYPE: Subtype.PROMPT,
}


class TraceServerExporter(SpanExporter):
    """Exports spans to Genkit telemetry server (DevUI)."""

    def __init__(
        self,
        telemetry_server_url: str,
        telemetry_server_endpoint: str = '/api/traces',
        filters: dict[str, str] | None = None,
    ) -> None:
        self.telemetry_server_url = resolve_telemetry_server_url(
            telemetry_server_url=telemetry_server_url,
            telemetry_server_endpoint=telemetry_server_endpoint,
        )
        self.telemetry_server_endpoint = telemetry_server_endpoint
        self.filters = filters if filters is not None else DEFAULT_SPAN_FILTERS
        self.last_result = SpanExportResult.SUCCESS
        self.stopped = False
        self.failed_traces: set[str] = set()
        # A hung collector holds the HTTP timeout on the worker. generate()
        # has already returned — traces are best-effort for the Dev UI.
        self.queue: Queue[list[tuple[str, str]] | None] = Queue()
        self.worker = threading.Thread(target=self.run_worker, name='genkit-trace-export', daemon=True)
        self.worker.start()

    def run_worker(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                try:
                    self.post_jobs(jobs=item)
                except Exception as error:
                    # The worker has to survive a surprise exception or every
                    # later trace is silently lost and generate() looks fine.
                    trace_id = item[0][0] if item else 'unknown'
                    self.note_transport_failure(trace_id=trace_id, error=error)
            finally:
                self.queue.task_done()

    def note_transport_failure(self, *, trace_id: str, error: BaseException) -> None:
        self.last_result = SpanExportResult.FAILURE
        # start+end of a 2-span flow is four posts of the same id; one line
        # is enough to find it without a wall. The reason is so a down
        # collector doesn't look like their flow crashed.
        if trace_id in self.failed_traces:
            return
        self.failed_traces.add(trace_id)
        logger.error(f'Failed to save trace {trace_id}: {error}')

    def post_jobs(self, *, jobs: list[tuple[str, str]]) -> None:
        """POST each serialized trace.

        Saving is not atomic: a transport failure stops the rest of the
        batch. Traces already sent stay sent; a down collector is one
        error line, not a retry storm.
        """
        url = urljoin(self.telemetry_server_url, self.telemetry_server_endpoint)
        try:
            # generate() already returned; this wait lives on the worker.
            for trace_id, body in jobs:
                try:
                    post_trace(url=url, body=body)
                except (urllib.error.URLError, TimeoutError, OSError) as error:
                    self.note_transport_failure(trace_id=trace_id, error=error)
                    return
                self.failed_traces.discard(trace_id)
            self.last_result = SpanExportResult.SUCCESS
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            trace_id = jobs[0][0] if jobs else 'unknown'
            self.note_transport_failure(trace_id=trace_id, error=error)

    @override
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if self.stopped:
            return SpanExportResult.FAILURE

        # Collect trace IDs that should be filtered out entirely
        filtered_trace_ids: set[str] = set()
        for span in spans:
            attrs = span.attributes or {}
            if any(attrs.get(k) == v for k, v in self.filters.items()):
                if span.context:
                    filtered_trace_ids.add(format(span.context.trace_id, '032x'))

        # Encode here so the worker only does I/O. Group by trace so a
        # batch flush is one POST per flow, not per span.
        by_trace: dict[str, list[ReadableSpan]] = {}
        for span in spans:
            ctx = span.context
            if ctx is None:
                raise TypeError('span context is required')
            trace_id = format(ctx.trace_id, '032x')
            if trace_id in filtered_trace_ids:
                continue
            by_trace.setdefault(trace_id, []).append(span)

        jobs: list[tuple[str, str]] = []
        for trace_id, group in by_trace.items():
            jobs.append((trace_id, encode_trace(trace=build_trace_payload(spans=group))))

        if jobs:
            self.queue.put(jobs)
        return SpanExportResult.SUCCESS

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        finished = threading.Event()

        def wait() -> None:
            self.queue.join()
            finished.set()

        waiter = threading.Thread(target=wait, name='genkit-trace-flush', daemon=True)
        waiter.start()
        return finished.wait(timeout_millis / 1000.0)

    @override
    def shutdown(self) -> None:
        self.stopped = True
        self.force_flush(timeout_millis=SHUTDOWN_FLUSH_TIMEOUT_MILLIS)
        self.queue.put(None)


def init_telemetry_server_exporter() -> SpanExporter | None:
    """Return TraceServerExporter if GENKIT_TELEMETRY_SERVER is set, else None."""
    url = os.environ.get('GENKIT_TELEMETRY_SERVER')
    if not url:
        logger.warn(
            'GENKIT_TELEMETRY_SERVER is not set. If running with `genkit start`, make sure `genkit-cli` is up to date.'
        )
        return None
    try:
        return TraceServerExporter(telemetry_server_url=url)
    except ValueError as error:
        logger.error(f'GENKIT_TELEMETRY_SERVER is not a valid URL: {error}')
        return None


def create_span_processor(exporter: SpanExporter) -> SpanProcessor:
    """RealtimeSpanProcessor in dev, BatchSpanProcessor in production."""
    if is_dev_environment():
        return RealtimeSpanProcessor(exporter)
    return BatchSpanProcessor(exporter)
