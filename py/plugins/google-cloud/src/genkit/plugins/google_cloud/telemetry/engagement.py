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

"""Engagement telemetry for GCP.

This module tracks user feedback and acceptance metrics,
matching the JavaScript implementation.

Metrics Recorded:
    - genkit/engagement/feedback: Counter for user feedback events
    - genkit/engagement/acceptance: Counter for user acceptance events

Cross-Language Parity:
    - JavaScript: js/plugins/google-cloud/src/telemetry/engagement.ts
    - Go: go/plugins/googlecloud/engagement.go

See Also:
    - Cloud Monitoring Custom Metrics: https://cloud.google.com/monitoring/custom-metrics
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from opentelemetry import metrics
from opentelemetry.sdk.trace import ReadableSpan

from genkit.core import GENKIT_VERSION

from .gcp_logger import gcp_logger
from .utils import create_common_log_attributes, truncate

logger = structlog.get_logger(__name__)

# Lazy-initialized metrics
_feedback_counter: metrics.Counter | None = None
_acceptance_counter: metrics.Counter | None = None


def _get_feedback_counter() -> metrics.Counter:
    """Get or create the user feedback counter."""
    global _feedback_counter
    if _feedback_counter is None:
        meter = metrics.get_meter('genkit')
        _feedback_counter = meter.create_counter(
            'genkit/engagement/feedback',
            description='Counts user feedback events.',
            unit='1',
        )
    return _feedback_counter


def _get_acceptance_counter() -> metrics.Counter:
    """Get or create the user acceptance counter."""
    global _acceptance_counter
    if _acceptance_counter is None:
        meter = metrics.get_meter('genkit')
        _acceptance_counter = meter.create_counter(
            'genkit/engagement/acceptance',
            description='Tracks user acceptance events.',
            unit='1',
        )
    return _acceptance_counter


class EngagementTelemetry:
    """Telemetry handler for user engagement (feedback, acceptance)."""

    def tick(
        self,
        span: ReadableSpan,
        log_input_and_output: bool,
        project_id: str | None = None,
    ) -> None:
        """Record telemetry for a user engagement span.

        Args:
            span: The span to record telemetry for.
            log_input_and_output: Whether to log input/output (unused here).
            project_id: Optional GCP project ID.
        """
        attrs: dict[str, Any] = dict(span.attributes) if span.attributes else {}
        subtype = str(attrs.get('genkit:metadata:subtype', ''))

        if subtype == 'userFeedback':
            self._write_user_feedback(span, attrs, project_id)
        elif subtype == 'userAcceptance':
            self._write_user_acceptance(span, attrs, project_id)
        else:
            logger.warning('Unknown user engagement subtype', subtype=subtype)

    def _write_user_feedback(
        self,
        span: ReadableSpan,
        attrs: dict[str, Any],
        project_id: str | None,
    ) -> None:
        """Record user feedback metrics and logs."""
        name = self._extract_trace_name(attrs)
        feedback_value = attrs.get('genkit:metadata:feedbackValue')
        text_feedback = attrs.get('genkit:metadata:textFeedback')

        dimensions = {
            'name': str(name)[:256],
            'value': str(feedback_value)[:256] if feedback_value else '',
            'hasText': str(bool(text_feedback)),
            'source': 'py',
            'sourceVersion': GENKIT_VERSION,
        }
        _get_feedback_counter().add(1, dimensions)

        metadata: dict[str, Any] = {
            **create_common_log_attributes(span, project_id),
            'feedbackValue': feedback_value,
        }
        if text_feedback:
            metadata['textFeedback'] = truncate(str(text_feedback))

        gcp_logger.log_structured(f'UserFeedback[{name}]', metadata)

    def _write_user_acceptance(
        self,
        span: ReadableSpan,
        attrs: dict[str, Any],
        project_id: str | None,
    ) -> None:
        """Record user acceptance metrics and logs."""
        name = self._extract_trace_name(attrs)
        acceptance_value = attrs.get('genkit:metadata:acceptanceValue')

        dimensions = {
            'name': str(name)[:256],
            'value': str(acceptance_value)[:256] if acceptance_value else '',
            'source': 'py',
            'sourceVersion': GENKIT_VERSION,
        }
        _get_acceptance_counter().add(1, dimensions)

        metadata = {
            **create_common_log_attributes(span, project_id),
            'acceptanceValue': acceptance_value,
        }
        gcp_logger.log_structured(f'UserAcceptance[{name}]', metadata)

    def _extract_trace_name(self, attrs: dict[str, Any]) -> str:
        """Extract the trace name from span attributes."""
        path = str(attrs.get('genkit:path', ''))
        if not path or path == '<unknown>':
            return '<unknown>'

        match = re.search(r'/{(.+)}', path)
        return match.group(1) if match else '<unknown>'


# Singleton instance
engagement_telemetry = EngagementTelemetry()
