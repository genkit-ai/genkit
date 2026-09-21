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

"""A minimal Genkit app wired to OpenTelemetry GenAI instrumentation.

Run the local telemetry stack first (`python tool/telemetry.py`), then
this app. Traces land in Jaeger (http://localhost:16686) and metrics in
the collector debug log. See README.md.
"""

from __future__ import annotations

import os

from genkit_google_genai import GoogleAI
from genkit_otel import ContentCapturingMode, GenAiInstrumentation
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from genkit import Genkit
from genkit.telemetry import configure_instrumentation


def _init_otel() -> None:
    """Own the SDK. Defaults to OTLP http/protobuf on localhost:4318."""
    resource = Resource.create({SERVICE_NAME: os.environ.get('OTEL_SERVICE_NAME', 'genkit-otel-sample')})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
    )
    metrics.set_meter_provider(meter_provider)


def _flush() -> None:
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.force_flush()
        provider.shutdown()
    meter = metrics.get_meter_provider()
    shutdown = getattr(meter, 'shutdown', None)
    if shutdown is not None:
        shutdown()


if __name__ == '__main__':
    _init_otel()

    # Content capture is opt-in (it may contain PII). SPAN_ONLY is the
    # easiest to eyeball in Jaeger; EVENT_ONLY emits a logs-signal event
    # Jaeger can't show. Unset consults
    # OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT.
    configure_instrumentation(
        GenAiInstrumentation(content_capturing_mode=ContentCapturingMode.SPAN_ONLY),
    )

    ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))

    async def main() -> None:
        response = await ai.generate(prompt='Explain OpenTelemetry in one sentence.')
        print(response.text)
        _flush()

    ai.run_main(main())
