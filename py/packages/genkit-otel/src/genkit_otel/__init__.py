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

"""OpenTelemetry GenAI semantic-conventions instrumentation for Genkit.

Configure ``GenAiInstrumentation`` before creating ``Genkit``, and let
the application own the OpenTelemetry SDK:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from genkit import Genkit
from genkit.telemetry import configure_instrumentation
from genkit_otel import GenAiInstrumentation

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

configure_instrumentation(GenAiInstrumentation())
ai = Genkit()
```

It emits ``gen_ai.*`` spans and metrics following the
[OTel GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai).
It composes freely with the built-in Developer UI poster (they export
to separate pipelines).
"""

from genkit_otel._gen_ai_attributes import ContentCapturingMode
from genkit_otel._genai_instrumentation import GenAiInstrumentation

__all__ = [
    'ContentCapturingMode',
    'GenAiInstrumentation',
]
