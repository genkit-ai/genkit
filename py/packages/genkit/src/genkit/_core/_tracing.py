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

"""Re-export of the telemetry dispatcher and OTel helpers.

New code should import from ``genkit.telemetry`` or ``_instrumentation``.
"""

from ._instrumentation import (
    SpanContext,
    SpanMetadata,
    configure_instrumentation,
    reset_instrumentation,
    run_in_new_span,
    set_custom_metadata_attributes,
)
from ._otel_instrumentation import (
    add_custom_exporter,
    init_provider,
    parent_path_context,
    start_attributes,
    tracer,
)

# Older tests imported this private name.
_parent_path_context = parent_path_context

__all__ = [
    'SpanContext',
    'SpanMetadata',
    'add_custom_exporter',
    'configure_instrumentation',
    'init_provider',
    'parent_path_context',
    'reset_instrumentation',
    'run_in_new_span',
    'set_custom_metadata_attributes',
    'start_attributes',
    'tracer',
]
