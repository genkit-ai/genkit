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

"""Realtime span processor for live trace visualization.

This module provides a SpanProcessor that exports spans both when they start
and when they end, enabling real-time trace visualization in the DevUI.

Overview:
    Standard OpenTelemetry processors (SimpleSpanProcessor, BatchSpanProcessor)
    only export spans when they complete. This is efficient but means the DevUI
    cannot show in-progress operations.

    RealtimeSpanProcessor exports spans immediately on start (without endTime),
    then exports again when the span completes with full data. This enables:

    - Live progress visualization in DevUI
    - Real-time debugging during development
    - Immediate feedback on long-running operations

Key Concepts:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Processor Type         │ Export on Start │ Export on End │ Use Case    │
    ├────────────────────────┼─────────────────┼───────────────┼─────────────┤
    │ SimpleSpanProcessor    │ No              │ Yes           │ Dev testing │
    │ BatchSpanProcessor     │ No              │ Yes (batched) │ Production  │
    │ RealtimeSpanProcessor  │ Yes             │ Yes           │ DevUI live  │
    └────────────────────────┴─────────────────┴───────────────┴─────────────┘

Usage:
    Enable realtime telemetry by setting the environment variable:

    ```bash
    export GENKIT_ENABLE_REALTIME_TELEMETRY=true
    genkit start -- python main.py
    ```

    Or programmatically configure the processor:

    ```python
    from opentelemetry.sdk.trace import TracerProvider
    from genkit.core.trace import RealtimeSpanProcessor, TelemetryServerSpanExporter

    exporter = TelemetryServerSpanExporter(telemetry_server_url='http://localhost:4000')
    processor = RealtimeSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    ```

Caveats:
    - Doubles network traffic (each span exported twice)
    - Not recommended for production use
    - Should only be used with GENKIT_ENABLE_REALTIME_TELEMETRY=true

See Also:
    - JavaScript RealtimeSpanProcessor: js/core/src/tracing/realtime-span-processor.ts
"""

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter

from genkit.core._compat import override
from genkit.core.logging import get_logger

logger = get_logger(__name__)


class RealtimeSpanProcessor(SpanProcessor):
    """Exports spans both when they start and when they end.

    This processor enables real-time trace visualization by exporting spans
    immediately when they start (without endTime), then again when they
    complete with full timing and status data.

    Attributes:
        exporter: The SpanExporter to use for exporting span data.

    Example:
        ```python
        from opentelemetry.sdk.trace import TracerProvider
        from genkit.core.trace import RealtimeSpanProcessor

        exporter = TelemetryServerSpanExporter(url='http://localhost:4000')
        processor = RealtimeSpanProcessor(exporter)

        provider = TracerProvider()
        provider.add_span_processor(processor)
        ```
    """

    def __init__(self, exporter: SpanExporter) -> None:
        """Initialize the RealtimeSpanProcessor.

        Args:
            exporter: The SpanExporter to use for exporting spans.
        """
        self._exporter: SpanExporter = exporter

    @override
    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Called when a span is started.

        Exports the span immediately for real-time updates. The span will
        not have endTime set yet, allowing the DevUI to show it as in-progress.

        Args:
            span: The span that was just started.
            parent_context: The parent context (unused).
        """
        # Export the span immediately (it won't have endTime yet).
        # Catch all exceptions so a failing exporter (e.g. Jaeger not
        # running) never propagates into the application call stack.
        # Without this guard, a ConnectionError here bubbles through
        # the OTel gRPC server interceptor and kills the actual RPC.
        try:
            self._exporter.export([span])
        except ConnectionError:
            logger.debug(
                'RealtimeSpanProcessor: export failed on_start (collector unreachable)',
                exc_info=True,
            )
        except Exception:  # noqa: BLE001 — must never crash the caller
            logger.warning(
                'RealtimeSpanProcessor: unexpected error during export on_start',
                exc_info=True,
            )

    @override
    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span ends.

        Exports the completed span with full timing and status data.

        Args:
            span: The span that just ended.
        """
        try:
            self._exporter.export([span])
        except ConnectionError:
            logger.debug(
                'RealtimeSpanProcessor: export failed on_end (collector unreachable)',
                exc_info=True,
            )
        except Exception:  # noqa: BLE001 — must never crash the caller
            logger.warning(
                'RealtimeSpanProcessor: unexpected error during export on_end',
                exc_info=True,
            )

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force the exporter to flush any buffered spans.

        Args:
            timeout_millis: Maximum time to wait for flush in milliseconds.

        Returns:
            True if flush succeeded, False otherwise.
        """
        if hasattr(self._exporter, 'force_flush'):
            return self._exporter.force_flush(timeout_millis)
        return True

    @override
    def shutdown(self) -> None:
        """Shut down the processor and exporter."""
        self._exporter.shutdown()
