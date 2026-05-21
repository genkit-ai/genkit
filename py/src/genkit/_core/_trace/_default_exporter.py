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

import os
from collections.abc import Callable, Iterable, Sequence
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
    genkit:error when events are empty or incomplete.
    """
    st = span_entry.get('status')
    if not st or st.get('code') != 2:
        return
    attrs = span_entry.get('attributes') or {}
    msg = st.get('message') or attrs.get('genkit:error')
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


DEFAULT_SPAN_FILTERS: dict[str, str] = {
    # Suppress prompt runner preview traces (triggered on every keystroke in Dev UI)
    'genkit:metadata:subtype': 'prompt',
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

    @override
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        # Collect trace IDs that should be filtered out entirely
        filtered_trace_ids: set[str] = set()
        for span in spans:
            attrs = span.attributes or {}
            if any(attrs.get(k) == v for k, v in self.filters.items()):
                if span.context:
                    filtered_trace_ids.add(format(span.context.trace_id, '032x'))

        url = urljoin(self.telemetry_server_url, self.telemetry_server_endpoint)
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        with httpx.Client() as client:
            for span in spans:
                if span.context and format(span.context.trace_id, '032x') in filtered_trace_ids:
                    continue
                client.post(url, json=extract_span_data(span), headers=headers)
        return SpanExportResult.SUCCESS

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


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
