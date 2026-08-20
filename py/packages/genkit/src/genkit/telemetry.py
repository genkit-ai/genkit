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

"""Turn tracing on, or add your own backend.

A plain ``Genkit()`` script does not record traces. Action results have
empty ``trace_id`` and ``span_id``.

Tracing turns on when you call ``configure_instrumentation`` with a
provider (usually ``OtelInstrumentation``), or when you run under
``genkit start``. ``genkit start -- python app.py`` sets
``GENKIT_ENV=dev`` and a collector URL before spawn; ``Genkit()``
installs ``OtelInstrumentation`` so the Traces tab works. If you start
the app yourself (``genkit start``, then ``GENKIT_ENV=dev python
app.py``), the collector URL arrives on ``/api/notify`` or the
reflection v2 handshake and turns tracing on the same way.

``enable_google_cloud_telemetry()`` is enough for Cloud Trace in
production. Under ``genkit start``, ``Genkit()`` still owns the
Developer UI collector.
"""

from genkit._core._instrumentation import (
    configure_instrumentation,
    is_instrumented_by,
    reset_instrumentation,
    run_in_new_span,
    set_custom_metadata_attributes,
)
from genkit._core._instrumentation_api import Instrumentation, SpanContext, SpanMetadata
from genkit._core._otel_instrumentation import OtelInstrumentation, genkit_dev_instrumentation

__all__ = [
    'Instrumentation',
    'OtelInstrumentation',
    'SpanContext',
    'SpanMetadata',
    'configure_instrumentation',
    'genkit_dev_instrumentation',
    'is_instrumented_by',
    'reset_instrumentation',
    'run_in_new_span',
    'set_custom_metadata_attributes',
]
