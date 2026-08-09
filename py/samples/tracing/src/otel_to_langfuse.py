#!/usr/bin/env python3
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

"""Hero: Genkit spans show up in Langfuse with zero Genkit-specific wiring.

Genkit instruments with the OpenTelemetry API. Your app owns the OTel SDK /
exporters. Point that exporter at Langfuse the normal way — Genkit spans arrive.

No custom Genkit telemetry handler required for the baseline path. A native
Langfuse handler is an optional upgrade later (richer observations), not the
on-ramp.

Setup::

    # AUTH=$(echo -n "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" | base64)
    export OTEL_EXPORTER_OTLP_ENDPOINT="https://cloud.langfuse.com/api/public/otel"
    export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic ${AUTH},x-langfuse-ingestion-version=4"

    uv run src/otel_to_langfuse.py
"""

from __future__ import annotations

import asyncio

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from genkit import Genkit
from genkit._core._tracing import SpanMetadata, annotate_output, run_in_new_span

# 1) App turns on OpenTelemetry the normal way (Langfuse = just an OTLP sink).
provider = TracerProvider(resource=Resource.create({'service.name': 'genkit-demo'}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

# 2) Genkit instruments via OTel. No Langfuse import. No Genkit knob.
ai = Genkit()


async def main() -> None:
    async def work() -> dict[str, str]:
        annotate_output({'hello': 'langfuse'})
        return {'status': 'ok'}

    result = await run_in_new_span(
        SpanMetadata(name='helloLangfuse', type='util', input={'msg': 'ping'}),
        work,
    )
    print(result)  # noqa: T201
    provider.force_flush()


if __name__ == '__main__':
    asyncio.run(main())
