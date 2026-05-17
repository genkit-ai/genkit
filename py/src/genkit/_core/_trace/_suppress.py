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

"""Telemetry suppression context variable.

Kept in its own module to avoid circular imports between _action, _tracing,
and the span processor chain.
"""

from contextvars import ContextVar

# Set to True when telemetryLabels contain 'genkitx:ignore-trace': 'true' so
# RealtimeSpanProcessor skips on_start/on_end exports for those traces.
suppress_telemetry: ContextVar[bool] = ContextVar('suppress_telemetry', default=False)

# TODO(https://github.com/genkit-ai/genkit/issues/5019): Investigate whether
# JS also needs this ContextVar approach or if it avoids the problem through
# a different mechanism (e.g. server-side batching, context baggage, or span
# attribute timing differences in the JS RealtimeSpanProcessor). In JS, when
# the root prompt span arrives filtered (possibleRoot=true in file-trace-store),
# child spans are orphaned and re-indexed without 'genkitx:ignore-trace', yet
# the Dev UI still hides them. The exact reason is not yet understood.
