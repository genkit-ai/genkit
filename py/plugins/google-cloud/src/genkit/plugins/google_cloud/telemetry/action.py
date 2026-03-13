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

"""Action telemetry for GCP.

This module logs input/output for tool and generate actions,
matching the JavaScript implementation.

Logging:
    When log_input_and_output=True, logs action inputs and outputs to
    Cloud Logging with structured attributes for correlation.

Cross-Language Parity:
    - JavaScript: js/plugins/google-cloud/src/telemetry/action.ts
    - Go: go/plugins/googlecloud/action.go

See Also:
    - Cloud Logging: https://cloud.google.com/logging/docs
    - Structured Logging: https://cloud.google.com/logging/docs/structured-logging
"""

from __future__ import annotations

import structlog
from opentelemetry.sdk.trace import ReadableSpan

from .gcp_logger import gcp_logger
from .utils import (
    create_common_log_attributes,
    extract_outer_feature_name_from_path,
    to_display_path,
    truncate,
    truncate_path,
)

logger = structlog.get_logger(__name__)


class ActionTelemetry:
    """Telemetry handler for Genkit actions (tools, generate)."""

    def tick(
        self,
        span: ReadableSpan,
        log_input_and_output: bool,
        project_id: str | None = None,
    ) -> None:
        """Record telemetry for an action span.

        Only logs input/output if log_input_and_output is True.

        Args:
            span: The span to record telemetry for.
            log_input_and_output: Whether to log input/output.
            project_id: Optional GCP project ID.
        """
        if not log_input_and_output:
            return

        attrs = span.attributes or {}
        action_name = str(attrs.get('genkit:name', '')) or '<unknown>'
        subtype = str(attrs.get('genkit:metadata:subtype', ''))

        # Only log for tools and generate actions
        if subtype != 'tool' and action_name != 'generate':
            return

        path = str(attrs.get('genkit:path', '')) or '<unknown>'
        input_val = truncate(str(attrs.get('genkit:input', '')))
        output_val = truncate(str(attrs.get('genkit:output', '')))
        session_id = str(attrs.get('genkit:sessionId', '')) or None
        thread_name = str(attrs.get('genkit:threadName', '')) or None

        feature_name = extract_outer_feature_name_from_path(path)
        if not feature_name or feature_name == '<unknown>':
            feature_name = action_name

        if input_val:
            self._write_log(span, 'Input', feature_name, path, input_val, project_id, session_id, thread_name)
        if output_val:
            self._write_log(span, 'Output', feature_name, path, output_val, project_id, session_id, thread_name)

    def _write_log(
        self,
        span: ReadableSpan,
        tag: str,
        feature_name: str,
        qualified_path: str,
        content: str,
        project_id: str | None,
        session_id: str | None,
        thread_name: str | None,
    ) -> None:
        """Write a structured log entry to Cloud Logging."""
        path = truncate_path(to_display_path(qualified_path))
        metadata = {
            **create_common_log_attributes(span, project_id),
            'path': path,
            'qualifiedPath': qualified_path,
            'featureName': feature_name,
            'content': content,
        }
        if session_id:
            metadata['sessionId'] = session_id
        if thread_name:
            metadata['threadName'] = thread_name

        gcp_logger.log_structured(f'{tag}[{path}, {feature_name}]', metadata)


# Singleton instance
action_telemetry = ActionTelemetry()
