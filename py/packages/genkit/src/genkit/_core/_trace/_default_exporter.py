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
from collections.abc import Callable, Iterable, Sequence
from queue import Queue
from typing import Any, cast
from urllib.parse import urljoin

import httpx
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import SpanContext

from genkit._core._compat import override
from genkit._core._environment import is_dev_environment
from genkit._core._logger import get_logger

from ._attrs import Attr, Subtype
from ._realtime_processor import RealtimeSpanProcessor

logger = get_logger(__name__)

INSTRUMENTATION = {'name': 'genkit-tracer', 'version': 'v1'}
TRACE_HEADERS = {'Content-Type': 'application/json', 'Accept': 'application/json'}


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


def _ensure_exception_message_for_dev_ui(span_entry: dict[str, Any]) -> None:
    r"""Ensure exception timeEvents carry exception.message for Dev UI / evaluate.ts.

    TraceData SpanStatusSchema uses `message` (not OTel's `description`). Dev UI and
    evaluate.ts read the first `exception` timeEvent's `exception.message` and fall
    back to the literal "Error" if missing. Synthesize from status.message or
    the error attr when events are empty or incomplete.
    """
    st = span_entry.get('status')
    if not st or st.get('code') != 2:
        return
    attrs = span_entry.get('attributes') or {}
    msg = st.get('message') or attrs.get(Attr.ERROR)
    if not msg:
        return
    if not st.get('message'):
        span_entry.setdefault('status', {})['message'] = msg
    te = span_entry.get('timeEvents')
    events = (te or {}).get('timeEvent') or []
    for ev in events:
        ann = ev.get('annotation') or {}
        if ann.get('description') != 'exception':
            continue
        ann_attrs = ann.get('attributes') or {}
        if ann_attrs.get('exception.message'):
            return
        ann_attrs['exception.message'] = msg
        ann['attributes'] = ann_attrs
        ev['annotation'] = ann
        return
    if not te:
        span_entry['timeEvents'] = {'timeEvent': []}
        te = span_entry['timeEvents']
    te.setdefault('timeEvent', []).append({
        'time': span_entry.get('endTime', 0),
        'annotation': {
            'description': 'exception',
            'attributes': {
                'exception.type': 'Error',
                'exception.message': msg,
            },
        },
    })


def _events_to_time_events(span: ReadableSpan) -> dict[str, Any]:
    """Build Genkit trace `timeEvents` from OTel span events (matches JS TraceServerExporter).

    Always includes `timeEvent` (possibly empty) so the payload matches JS and
    `_ensure_exception_message_for_dev_ui` can append a synthetic exception event.
    """
    events = getattr(span, 'events', None) or ()
    time_event: list[dict[str, Any]] = []
    for ev in events:
        name = getattr(ev, 'name', None) or 'event'
        ts = getattr(ev, 'timestamp', None)
        raw_attrs = getattr(ev, 'attributes', None) or {}
        time_event.append({
            'time': _ns_to_ms(ts),
            'annotation': {
                'attributes': _otel_event_attributes_to_json(raw_attrs),
                'description': name,
            },
        })
    return {'timeEvent': time_event}


def extract_span_data(span: ReadableSpan) -> dict[str, Any]:
    """Convert ReadableSpan to Genkit telemetry server JSON format."""
    ctx = cast(SpanContext, span.context)
    trace_id = format(ctx.trace_id, '032x')
    span_id = format(ctx.span_id, '016x')
    parent_id = format(span.parent.span_id, '016x') if span.parent else None
    start = _ns_to_ms(span.start_time)
    end = _ns_to_ms(span.end_time)

    span_entry: dict[str, Any] = {
        'spanId': span_id,
        'traceId': trace_id,
        'startTime': start,
        'endTime': end,
        'attributes': dict(span.attributes or {}),
        'displayName': span.name,
        'spanKind': trace_api.SpanKind(span.kind).name,
        'instrumentationLibrary': INSTRUMENTATION,
        'timeEvents': _events_to_time_events(span),
    }
    if parent_id:
        span_entry['parentSpanId'] = parent_id
    if span.status:
        code = trace_api.StatusCode(span.status.status_code).value
        desc = span.status.description
        # SpanStatusSchema only has code + message; omit nulls (Zod rejects null for optional strings).
        status_obj: dict[str, Any] = {'code': code}
        if desc is not None:
            status_obj['message'] = desc
        span_entry['status'] = status_obj
    _ensure_exception_message_for_dev_ui(span_entry)

    result: dict[str, Any] = {'traceId': trace_id, 'spans': {span_id: span_entry}}
    if not span.parent:
        result['displayName'] = span.name
        result['startTime'] = start
        result['endTime'] = end

    return result


def build_trace_payload(spans: Sequence[ReadableSpan]) -> dict[str, Any]:
    """One collector document for a batch of spans that share a trace id.

    Production uses BatchSpanProcessor, which can flush a whole flow at once.
    The store keys documents by trace id, so one POST per id is enough.
    """
    payload: dict[str, Any] = {'spans': {}}
    for span in spans:
        part = extract_span_data(span)
        payload['traceId'] = part['traceId']
        payload['spans'].update(part['spans'])
        if 'displayName' in part:
            payload['displayName'] = part['displayName']
            payload['startTime'] = part['startTime']
            payload['endTime'] = part['endTime']
    return payload


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
        self.telemetry_server_url = telemetry_server_url
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
                self.post_jobs(jobs=item)
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
        url = urljoin(self.telemetry_server_url, self.telemetry_server_endpoint)
        try:
            # No timeout: a slow local collector must still get the span.
            # generate() already returned; this wait lives on the worker.
            with httpx.Client(timeout=None) as client:  # noqa: S113
                for trace_id, body in jobs:
                    try:
                        client.post(url, content=body, headers=TRACE_HEADERS)
                    except (httpx.RequestError, OSError) as error:
                        self.note_transport_failure(trace_id=trace_id, error=error)
                        return
                    self.failed_traces.discard(trace_id)
            self.last_result = SpanExportResult.SUCCESS
        except (httpx.RequestError, OSError) as error:
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

        # Serialize on this thread so a span that can't encode fails in
        # export(), not as a silent miss after generate() has moved on.
        # Group by trace so a batch flush is one POST per flow, not per span.
        by_trace: dict[str, list[ReadableSpan]] = {}
        for span in spans:
            ctx = span.context
            if ctx is None:
                extract_span_data(span)
                raise TypeError('span context is required')
            trace_id = format(ctx.trace_id, '032x')
            if trace_id in filtered_trace_ids:
                continue
            by_trace.setdefault(trace_id, []).append(span)

        jobs: list[tuple[str, str]] = []
        for trace_id, group in by_trace.items():
            jobs.append((trace_id, json.dumps(build_trace_payload(group))))

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
        self.force_flush()
        self.stopped = True
        self.queue.put(None)


def init_telemetry_server_exporter() -> SpanExporter | None:
    """Return TraceServerExporter if GENKIT_TELEMETRY_SERVER is set, else None."""
    url = os.environ.get('GENKIT_TELEMETRY_SERVER')
    if not url:
        logger.warn(
            'GENKIT_TELEMETRY_SERVER is not set. If running with `genkit start`, make sure `genkit-cli` is up to date.'
        )
        return None
    return TraceServerExporter(telemetry_server_url=url)


def create_span_processor(exporter: SpanExporter) -> SpanProcessor:
    """RealtimeSpanProcessor in dev, BatchSpanProcessor in production."""
    if is_dev_environment():
        return RealtimeSpanProcessor(exporter)
    return BatchSpanProcessor(exporter)
