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

"""GCP Cloud Logging integration for Genkit telemetry.

This module provides a logger that writes structured logs directly to
Google Cloud Logging with trace correlation, enabling visibility in
the Firebase AIM dashboard.

This is analogous to the JavaScript implementation in:
- js/plugins/google-cloud/src/gcpLogger.ts

Usage:
    from genkit.plugins.google_cloud.telemetry.gcp_logger import gcp_logger

    # Initialize during telemetry setup
    gcp_logger.initialize(project_id="my-project", credentials=creds, export=True)

    # Write structured logs
    gcp_logger.log_structured("Input[path, feature]", {"content": "...", "traceId": "..."})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from google.auth.credentials import Credentials
from opentelemetry import trace

if TYPE_CHECKING:
    from google.cloud.logging_v2 import Logger as CloudLogger

logger = structlog.get_logger(__name__)


class GcpLogger:
    """Logger for writing structured logs to Cloud Logging.

    This class provides a simple interface for writing logs that appear
    in the Firebase AIM dashboard. It writes directly to Cloud Logging
    using the google-cloud-logging client library.
    """

    def __init__(self) -> None:
        """Initialize logger state."""
        self._initialized = False
        self._export = False
        self._project_id: str | None = None
        self._cloud_logger: CloudLogger | None = None

    def initialize(
        self,
        *,
        project_id: str | None = None,
        credentials: Credentials | dict[str, Any] | None = None,
        export: bool = False,
    ) -> None:
        """Initialize the GCP logger.

        This method MUST be called before any log_structured() calls.

        Args:
            project_id: GCP project ID (required if export=True).
            credentials: GCP credentials (Credentials object or dict).
            export: Whether to export logs to Cloud Logging (GCP).

        Behavior:
            - export=False: Local logging only (console output)
            - export=True: Logs sent to Cloud Logging for AIM visibility
        """
        if self._initialized:
            logger.debug('GcpLogger already initialized, skipping re-initialization')
            return

        self._export = export
        self._project_id = project_id

        if not export:
            logger.info(
                'GcpLogger initialized in LOCAL mode',
                export=False,
                project_id=project_id or '<not-set>',
                note='Logs written to console only, not exported to GCP',
            )
            self._initialized = True
            return

        # Export mode: validate required configuration
        if not project_id:
            logger.error(
                'GcpLogger initialization FAILED: project_id required for export=True',
                export=True,
                project_id=None,
                consequence='Telemetry logs will NOT appear in Cloud Logging or AIM dashboard',
                fix='Provide project_id when calling add_firebase_telemetry()',
            )
            self._initialized = True  # Mark initialized to prevent repeated errors
            return

        try:
            from google.cloud import logging as cloud_logging

            # Cloud Logging Client accepts Credentials object or None
            # If credentials is a dict, let it use Application Default Credentials
            creds = credentials if isinstance(credentials, Credentials) else None

            client = cloud_logging.Client(
                project=project_id,
                credentials=creds,
            )
            self._cloud_logger = client.logger('genkit_log')
            logger.info(
                'GcpLogger initialized for CLOUD LOGGING export',
                export=True,
                project_id=project_id,
                log_name='genkit_log',
                consequence='Logs will appear in Cloud Logging and AIM dashboard',
            )
        except ImportError:
            logger.error(
                'GcpLogger initialization FAILED: google-cloud-logging not installed',
                export=True,
                project_id=project_id,
                consequence='Telemetry logs will NOT be exported to Cloud Logging',
                fix='Install with: pip install google-cloud-logging>=3.10.0',
            )
        except Exception as e:
            logger.error(
                'GcpLogger initialization FAILED: Cloud Logging client error',
                export=True,
                project_id=project_id,
                error=str(e),
                error_type=type(e).__name__,
                consequence='Telemetry logs will NOT be exported to Cloud Logging',
                fix='Check credentials and project_id, ensure GCP access is configured',
            )

        self._initialized = True

    def _get_trace_context(self) -> dict[str, str]:
        """Extract trace context from current span if available.

        Returns:
            Dictionary with trace fields for Cloud Logging, empty if no trace.
        """
        span = trace.get_current_span()
        if not (span and span.is_recording()):
            return {}

        ctx = span.get_span_context()
        if not (ctx and ctx.trace_id):
            return {}

        trace_id = format(ctx.trace_id, '032x')
        span_id = format(ctx.span_id, '016x')

        return {
            'logging.googleapis.com/trace': (
                f'projects/{self._project_id}/traces/{trace_id}' if self._project_id else trace_id
            ),
            'logging.googleapis.com/spanId': span_id,
            'logging.googleapis.com/trace_sampled': str(ctx.trace_flags.sampled),
        }

    def _write(self, message: str, payload: dict[str, Any], severity: str) -> None:
        """Write log to Cloud Logging or fallback to console.

        Args:
            message: Log message for fallback logging.
            payload: Structured payload with all metadata.
            severity: Cloud Logging severity (INFO, ERROR).
        """
        if self._export and self._cloud_logger:
            try:
                self._cloud_logger.log_struct(payload, severity=severity, labels={'module': 'genkit'})
            except Exception as e:
                logger.error('Failed to write to Cloud Logging', error=str(e), message=message)
                # Fallback to console
                if severity == 'ERROR':
                    logger.error(message, **payload)
                else:
                    logger.info(message, **payload)
        else:
            if severity == 'ERROR':
                logger.error(message, **payload)
            else:
                logger.info(message, **payload)

    def log_structured(self, message: str, metadata: dict[str, Any] | None = None) -> None:
        """Write a structured log entry.

        This method is called by telemetry handlers to write structured logs.
        If not initialized, logs an error once and falls back to console logging.

        Args:
            message: Log message.
            metadata: Additional structured metadata.
        """
        if not self._initialized:
            # Log error ONCE to avoid spam
            if not hasattr(self, '_logged_init_error'):
                logger.error(
                    'gcp_logger.log_structured() called before initialization',
                    message=message,
                    hint='Ensure GcpTelemetry.initialize() was called',
                )
                self._logged_init_error = True

            # Fall back to console logging for debugging
            logger.warning(f'[FALLBACK] {message}', **(metadata or {}))
            return

        payload = metadata.copy() if metadata else {}
        payload['message'] = message
        payload.update(self._get_trace_context())
        self._write(message, payload, 'INFO')

    def log_structured_error(self, message: str, metadata: dict[str, Any] | None = None) -> None:
        """Write a structured error log entry.

        Args:
            message: Log message.
            metadata: Additional structured metadata.
        """
        payload = metadata.copy() if metadata else {}
        payload['message'] = message
        payload.update(self._get_trace_context())
        self._write(message, payload, 'ERROR')


# Singleton instance
gcp_logger = GcpLogger()
