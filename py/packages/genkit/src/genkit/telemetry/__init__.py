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

"""On-switch for Genkit traces, and the protocol for a recording backend.

Application code should call :class:`genkit.Genkit`. ``genkit start``
records to the Developer UI. ``configure_instrumentation`` is the hook
for a recording backend; ``reset_instrumentation`` starts over between tests.
"""

from genkit._core._telemetry._http import GenkitBuiltinInstrumentation
from genkit._core._telemetry._instrumentation import (
    DisposableInstrumentation,
    FlushableInstrumentation,
    Instrumentation,
    SpanAttributeValue,
    SpanContext,
    SpanMetadata,
    SpanNext,
    SpanState,
    configure_instrumentation,
    flush_instrumentations,
    instrumentations,
    is_instrumented_by,
    reset_instrumentation,
    run_in_new_span,
    set_custom_metadata_attributes,
    set_span_state,
    to_json_attr,
)

__all__ = [
    # Runtime & provider SPI
    'DisposableInstrumentation',
    'FlushableInstrumentation',
    'Instrumentation',
    'SpanAttributeValue',
    'SpanContext',
    'SpanMetadata',
    'SpanNext',
    'SpanState',
    'configure_instrumentation',
    'flush_instrumentations',
    'run_in_new_span',
    'set_custom_metadata_attributes',
    'set_span_state',
    'to_json_attr',
    # Testing & test-inspection helpers
    'GenkitBuiltinInstrumentation',
    'instrumentations',
    'is_instrumented_by',
    'reset_instrumentation',
]
