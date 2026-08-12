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

"""Constants for GCP telemetry.

This module centralizes all constants used across the GCP telemetry
implementation, matching the pattern from JS/Go implementations.
"""

# Metric export intervals (matching JS/Go implementations)
MIN_METRIC_EXPORT_INTERVAL_MS = 5000
DEFAULT_METRIC_EXPORT_INTERVAL_MS = 300000
DEV_METRIC_EXPORT_INTERVAL_MS = 5000
PROD_METRIC_EXPORT_INTERVAL_MS = 300000

# Project ID environment variables (resolution order)
# Priority: FIREBASE_PROJECT_ID > GOOGLE_CLOUD_PROJECT > GCLOUD_PROJECT
PROJECT_ID_ENV_VARS = (
    'FIREBASE_PROJECT_ID',
    'GOOGLE_CLOUD_PROJECT',
    'GCLOUD_PROJECT',
)

# Retry configuration for trace export to Cloud Trace
TRACE_RETRY_INITIAL = 0.1
TRACE_RETRY_MAXIMUM = 30.0
TRACE_RETRY_MULTIPLIER = 2
TRACE_RETRY_DEADLINE = 120.0

# Time adjustment for GCP span requirements
# GCP requires end_time > start_time, so we add 1 microsecond minimum duration
MIN_SPAN_DURATION_NS = 1000  # 1 microsecond in nanoseconds

# Start time adjustment for metrics to prevent DELTA->CUMULATIVE overlap
# Cloud Monitoring converts DELTA to CUMULATIVE, causing overlap issues
# We add 1 millisecond to ensure discrete export timeframes
METRIC_START_TIME_ADJUSTMENT_NS = 1_000_000  # 1 millisecond in nanoseconds
