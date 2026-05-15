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

"""Realtime span processor for live trace visualization."""

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from genkit._core._compat import override
from genkit._core._logger import get_logger
from genkit._core._trace._suppress import suppress_telemetry

logger = get_logger(__name__)


class RealtimeSpanProcessor(SimpleSpanProcessor):
    """Exports spans on start (real-time) and on end, unlike SimpleSpanProcessor (end only)."""

    @override
    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Export span immediately so DevUI can show in-progress traces."""
        if suppress_telemetry.get():
            return
        try:
            self.span_exporter.export([span])
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
        """Skip export entirely for suppressed traces (e.g. prompt keystroke previews)."""
        if suppress_telemetry.get():
            return
        super().on_end(span)
