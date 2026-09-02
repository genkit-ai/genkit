# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Dev UI log export — OTLP-JSON POSTs to the telemetry server.

Console level stays on ``GENKIT_LOG``. This sink is debug and above, and it
never waits on the collector: ``generate()`` has already returned.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from queue import Empty, Full, Queue
from urllib.parse import urljoin

import httpx
from opentelemetry import trace as trace_api

from genkit._core._constants import GENKIT_VERSION
from genkit._core._environment import is_dev_environment

GENKIT_OTEL_ENABLE_LOGS = 'GENKIT_OTEL_ENABLE_LOGS'
GENKIT_TELEMETRY_SERVER = 'GENKIT_TELEMETRY_SERVER'

LOG_ENDPOINT = '/api/otlp'
LOG_HEADERS = {'Content-Type': 'application/json', 'Accept': 'application/json'}
SCOPE_NAME = 'genkit-py'

QUEUE_SIZE = 1024
BATCH_SIZE = 64
BATCH_DELAY_S = 0.1
# Long enough for a slow local Dev UI to accept the batch; short enough that
# a hung collector does not pin this worker for the rest of the process.
EXPORT_TIMEOUT_S = 300.0
# The export worker is a daemon: once the interpreter starts tearing down it
# is killed, so we give the last batch a couple of seconds to land. Ctrl-C of
# a wedged collector still shouldn't sit for minutes.
SHUTDOWN_FLUSH_TIMEOUT_S = 2.0

# Keep the Dev UI log panel readable — a generate() that binds 200 keys or a
# megabyte prompt shouldn't serialize that onto the generate thread.
MAX_ATTRIBUTES = 32
MAX_ATTR_CHARS = 2048
REDACT_KEYS = frozenset({
    'apikey',
    'authtoken',
    'accesstoken',
    'authorization',
    'password',
    'secret',
    'token',
})
REDACT_SUFFIXES = ('key', 'token', 'secret', 'password')

SEVERITY: dict[int, tuple[int, str]] = {
    logging.DEBUG: (5, 'DEBUG'),
    logging.INFO: (9, 'INFO'),
    logging.WARNING: (13, 'WARN'),
    logging.ERROR: (17, 'ERROR'),
    logging.CRITICAL: (21, 'FATAL'),
}

_exporter: LogServerExporter | None = None
_exporter_lock = threading.Lock()
_skip_export = threading.local()
_atexit_registered = False


def logs_opted_out() -> bool:
    """True only when ``GENKIT_OTEL_ENABLE_LOGS`` is a ParseBool false."""
    raw = os.environ.get(GENKIT_OTEL_ENABLE_LOGS, '').strip().lower()
    return raw in {'false', '0', 'f'}


def log_export_is_enabled() -> bool:
    """True when a destination is configured and accepting records."""
    current = _exporter
    return current is not None and not current.stopped


def enable_log_export(*, url: str) -> None:
    """Start (or retarget) log export. No-op when opted out, empty, or not dev."""
    global _exporter
    if not url or logs_opted_out() or not is_dev_environment():
        return
    # Late import: _default_exporter pulls get_logger, and get_logger tees here.
    from genkit._core._trace._default_exporter import resolve_telemetry_server_url

    try:
        resolved = resolve_telemetry_server_url(
            telemetry_server_url=url,
            telemetry_server_endpoint=LOG_ENDPOINT,
        )
    except ValueError as error:
        _diag(level=logging.ERROR, event=f'GENKIT_TELEMETRY_SERVER is not a valid URL: {error}')
        return
    with _exporter_lock:
        if _exporter is not None:
            _exporter.set_url(url=resolved)
            return
        _exporter = LogServerExporter(telemetry_server_url=resolved)
        _register_atexit()


def reset_log_export() -> None:
    """Tear down the process-wide exporter. Tests only."""
    global _exporter
    with _exporter_lock:
        current = _exporter
        _exporter = None
    if current is not None:
        current.shutdown()


def emit_log(*, level: int, event: str, attrs: dict[str, object] | None = None) -> None:
    """Queue one record. Never blocks. Encode bugs raise on this thread."""
    if getattr(_skip_export, 'on', False):
        return
    current = _exporter
    if current is None or current.stopped or level < logging.DEBUG:
        return
    current.enqueue(record=build_log_record(level=level, event=event, attrs=attrs or {}))


def build_log_record(*, level: int, event: str, attrs: dict[str, object]) -> dict[str, object]:
    """One OTLP-JSON ``logRecords`` item."""
    number, text = _severity(level=level)
    encoded, dropped = _otlp_attributes(attrs=attrs)
    record: dict[str, object] = {
        'timeUnixNano': str(time.time_ns()),
        'severityNumber': number,
        'severityText': text,
        'body': {'stringValue': event},
        'attributes': encoded,
    }
    if dropped:
        record['droppedAttributesCount'] = dropped
    span = trace_api.get_current_span()
    ctx = span.get_span_context()
    if ctx is not None and ctx.is_valid:
        record['traceId'] = format(ctx.trace_id, '032x')
        record['spanId'] = format(ctx.span_id, '016x')
    return record


def build_payload(*, records: list[dict[str, object]]) -> dict[str, object]:
    """One ``resourceLogs`` document for a batch."""
    return {
        'resourceLogs': [
            {
                'resource': {'attributes': [], 'droppedAttributesCount': 0},
                'scopeLogs': [
                    {
                        'scope': {'name': SCOPE_NAME, 'version': GENKIT_VERSION},
                        'logRecords': records,
                    }
                ],
            }
        ]
    }


def _severity(*, level: int) -> tuple[int, str]:
    for threshold in (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG):
        if level >= threshold:
            return SEVERITY[threshold]
    return SEVERITY[logging.DEBUG]


def _should_redact(*, key: str) -> bool:
    # apiKey / authToken / apikey all collapse to the same compact form so
    # a plugin that logs the key under any of those names does not POST it
    # to the telemetry server.
    compact = key.lower().replace('_', '')
    if compact in REDACT_KEYS:
        return True
    return any(compact.endswith(suffix) for suffix in REDACT_SUFFIXES)


def _otlp_attributes(*, attrs: dict[str, object]) -> tuple[list[dict[str, object]], int]:
    encoded: list[dict[str, object]] = []
    dropped = 0
    for key, value in attrs.items():
        name = str(key)
        if name.startswith('_') or value is None:
            continue
        if len(encoded) >= MAX_ATTRIBUTES:
            dropped += 1
            continue
        if _should_redact(key=name):
            encoded.append({'key': name, 'value': {'stringValue': '[redacted]'}})
            continue
        encoded.append({'key': name, 'value': _otlp_value(value=value)})
    return encoded, dropped


def _otlp_value(*, value: object) -> dict[str, object]:
    if isinstance(value, bool):
        return {'boolValue': value}
    if isinstance(value, int):
        return {'intValue': value}
    if isinstance(value, float):
        return {'doubleValue': value}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {'stringValue': f'<{len(value)} bytes>'}
    if isinstance(value, str):
        return {'stringValue': _clip_attr(value)}
    try:
        return {'stringValue': _clip_attr(json.dumps(value))}
    except (TypeError, ValueError):
        return {'stringValue': _clip_attr(str(value))}


def _clip_attr(value: str) -> str:
    if len(value) <= MAX_ATTR_CHARS:
        return value
    return f'{value[:MAX_ATTR_CHARS]}...<{len(value) - MAX_ATTR_CHARS} chars>'


def _diag(*, level: int, event: str) -> None:
    """Console-only line — must not re-enter export."""
    _skip_export.on = True
    try:
        # Late import: _logger.get_logger tees back into emit_log.
        from genkit._core._logger import get_logger

        logger = get_logger(__name__)
        if level >= logging.ERROR:
            logger.error(event)
        else:
            logger.warning(event)
    finally:
        _skip_export.on = False


def _register_atexit() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    atexit.register(reset_log_export)
    _atexit_registered = True


def put_poison_pill(*, queue: Queue[dict[str, object] | None]) -> None:
    """Make sure the worker sees the stop token even when the queue is full."""
    try:
        queue.put_nowait(None)
        return
    except Full:
        pass
    try:
        queue.get_nowait()
        queue.task_done()
    except Empty:
        pass
    try:
        queue.put_nowait(None)
    except Full:
        pass


class LogServerExporter:
    """Background OTLP log sink. Same worker shape as the quiet-TTY span exporter."""

    def __init__(self, *, telemetry_server_url: str) -> None:
        self.telemetry_server_url = telemetry_server_url
        self.stopped = False
        self.dropped = 0
        self.warned_overflow = False
        self.warned_unreachable = False
        self.last_result_ok = True
        self.queue: Queue[dict[str, object] | None] = Queue(maxsize=QUEUE_SIZE)
        self.worker = threading.Thread(target=self.run_worker, name='genkit-log-export', daemon=True)
        self.worker.start()

    def set_url(self, *, url: str) -> None:
        self.telemetry_server_url = url

    def enqueue(self, *, record: dict[str, object]) -> None:
        if self.stopped:
            return
        try:
            self.queue.put_nowait(record)
        except Full:
            self.dropped += 1
            if not self.warned_overflow:
                self.warned_overflow = True
                _diag(
                    level=logging.WARNING,
                    event='log export queue is full; dropping records from the Dev UI log view',
                )

    def run_worker(self) -> None:
        try:
            with httpx.Client(timeout=EXPORT_TIMEOUT_S) as client:
                while True:
                    item = self.queue.get()
                    try:
                        if item is None:
                            return
                        batch = [item]
                        deadline = time.monotonic() + BATCH_DELAY_S
                        while len(batch) < BATCH_SIZE:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            try:
                                nxt = self.queue.get(timeout=remaining)
                            except Empty:
                                break
                            if nxt is None:
                                self.queue.task_done()
                                self.post_batch(client=client, records=batch)
                                return
                            batch.append(nxt)
                            self.queue.task_done()
                        self.post_batch(client=client, records=batch)
                    finally:
                        self.queue.task_done()
        except Exception as error:
            # Client() or context-exit can raise before post_batch's guard.
            # Stop accepting records so debug branches stop building payloads
            # that will never export.
            self.stopped = True
            _diag(level=logging.ERROR, event=f'log export worker stopped: {error}')

    def post_batch(self, *, client: httpx.Client, records: list[dict[str, object]]) -> None:
        url = urljoin(self.telemetry_server_url, LOG_ENDPOINT)
        try:
            body = json.dumps(build_payload(records=records))
            # The caller already returned; this wait lives on the worker.
            response = client.post(url, content=body, headers=LOG_HEADERS)
            if response.status_code != 200:
                self.note_transport_failure(error=RuntimeError(f'HTTP {response.status_code}'))
                return
            self.last_result_ok = True
            self.warned_unreachable = False
        except Exception as error:
            self.note_transport_failure(error=error)

    def note_transport_failure(self, *, error: BaseException) -> None:
        self.last_result_ok = False
        if self.warned_unreachable:
            return
        self.warned_unreachable = True
        _diag(level=logging.ERROR, event=f'Failed to export logs: {error}')

    def force_flush(self, *, timeout_s: float = 30.0) -> bool:
        finished = threading.Event()

        def wait() -> None:
            self.queue.join()
            finished.set()

        waiter = threading.Thread(target=wait, name='genkit-log-flush', daemon=True)
        waiter.start()
        return finished.wait(timeout_s)

    def shutdown(self) -> None:
        self.stopped = True
        self.force_flush(timeout_s=SHUTDOWN_FLUSH_TIMEOUT_S)
        put_poison_pill(queue=self.queue)
