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
for a recording backend.
"""

from genkit._core._telemetry._instrumentation import (
    Instrumentation,
    SpanContext,
    SpanMetadata,
    configure_instrumentation,
)

__all__ = [
    'Instrumentation',
    'SpanContext',
    'SpanMetadata',
    'configure_instrumentation',
]
