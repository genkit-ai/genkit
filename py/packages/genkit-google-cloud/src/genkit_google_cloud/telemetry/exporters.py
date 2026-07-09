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

"""Base exporter utilities for GCP telemetry.

This module provides reusable error handling and base utilities for
trace and metrics exporters, eliminating code duplication.
"""

import structlog

logger = structlog.get_logger(__name__)


class ErrorHandler:
    """Manages error handling for telemetry exports.

    Ensures error messages are logged only once to avoid spam, while still
    logging subsequent errors without the detailed help text.

    This replaces the previous pattern of module-level boolean flags and
    separate error handler functions for tracing and metrics.
    """

    def __init__(self) -> None:
        """Initialize the error handler."""
        self._logged = False

    def handle(
        self,
        error: Exception,
        error_message: str,
        help_text: str,
    ) -> None:
        """Handle export error with one-time detailed logging.

        Args:
            error: The exception that occurred.
            error_message: Brief description of what failed.
            help_text: Detailed help text shown only on first error.
        """
        if not self._logged:
            self._logged = True
            logger.error(f'{error_message}\n{help_text}\nError: {error}')
        else:
            logger.error(f'{error_message}: {error}')


# Singleton error handlers for tracing and metrics
_tracing_error_handler = ErrorHandler()
_metrics_error_handler = ErrorHandler()

# Help text for tracing errors (GCP IAM requirements)
TRACING_HELP_TEXT = 'Ensure the service account has the "Cloud Trace Agent" (roles/cloudtrace.agent) role.'

# Help text for metrics errors (GCP IAM requirements)
METRICS_HELP_TEXT = (
    'Ensure the service account has the "Monitoring Metric Writer" '
    '(roles/monitoring.metricWriter) or "Cloud Telemetry Metrics Writer" '
    '(roles/telemetry.metricsWriter) role.'
)


def handle_tracing_error(error: Exception) -> None:
    """Handle trace export errors with helpful messages.

    Only logs detailed instructions once to avoid spam.

    Args:
        error: The export error.
    """
    error_str = str(error).lower()
    if 'permission' in error_str or 'denied' in error_str or '403' in error_str:
        _tracing_error_handler.handle(
            error,
            'Unable to send traces to Google Cloud.',
            TRACING_HELP_TEXT,
        )
    else:
        logger.error('Error exporting traces to GCP', error=str(error))


def handle_metric_error(error: Exception) -> None:
    """Handle metrics export errors with helpful messages.

    Only logs detailed instructions once to avoid spam.

    Args:
        error: The export error.
    """
    error_str = str(error).lower()
    if 'permission' in error_str or 'denied' in error_str or '403' in error_str:
        _metrics_error_handler.handle(
            error,
            'Unable to send metrics to Google Cloud.',
            METRICS_HELP_TEXT,
        )
    else:
        logger.error('Error exporting metrics to GCP', error=str(error))
