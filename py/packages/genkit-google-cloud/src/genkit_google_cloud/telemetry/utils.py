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

"""Utility functions for GCP telemetry.

This module provides utility functions used by the telemetry handlers,
matching the JavaScript implementation in js/plugins/google-cloud/src/utils.ts.

Functions:
    - truncate(): Limit string length for log content
    - truncate_path(): Limit Genkit path string length
    - extract_outer_feature_name_from_path(): Get root feature from path
    - create_common_log_attributes(): Build log attributes dict
    - extract_error_*(): Error info extraction helpers

See Also:
    - Cloud Logging Limits: https://cloud.google.com/logging/quotas
"""

from __future__ import annotations

import re
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import TraceFlags

# Constants matching JS implementation
MAX_LOG_CONTENT_CHARS = 128_000
MAX_PATH_CHARS = 4096


def truncate(text: str | None, limit: int = MAX_LOG_CONTENT_CHARS) -> str:
    """Truncate text to a maximum length.

    Args:
        text: The text to truncate.
        limit: Maximum length (default: 128,000 chars).

    Returns:
        The truncated text or empty string if None.
    """
    if not text:
        return ''
    return text[:limit]


def truncate_path(path: str) -> str:
    """Truncate a path to the maximum path length.

    Args:
        path: The path to truncate.

    Returns:
        The truncated path.
    """
    return truncate(path, MAX_PATH_CHARS)


def extract_outer_flow_name_from_path(path: str) -> str:
    """Extract the outer flow name from a Genkit path.

    Args:
        path: The Genkit path (e.g., '/{myFlow,t:flow}').

    Returns:
        The flow name or '<unknown>'.
    """
    if not path or path == '<unknown>':
        return '<unknown>'

    match = re.search(r'/{(.+),t:flow}', path)
    return match.group(1) if match else '<unknown>'


def extract_outer_feature_name_from_path(path: str) -> str:
    """Extract the outer feature name from a Genkit path.

    Extracts the first feature name from paths like:
    '/{myFlow,t:flow}/{myStep,t:flowStep}/{googleai/gemini-pro,t:action,s:model}'
    Returns 'myFlow'.

    Args:
        path: The Genkit path.

    Returns:
        The feature name or '<unknown>'.
    """
    if not path or path == '<unknown>':
        return '<unknown>'

    parts = path.split('/')
    if len(parts) < 2:
        return '<unknown>'

    first = parts[1]
    match = re.match(r'\{(.+),t:(flow|action|prompt|dotprompt|helper)', first)
    return match.group(1) if match else '<unknown>'


def extract_error_name(events: list[Any]) -> str | None:
    """Extract the error name from span events.

    Args:
        events: List of span events.

    Returns:
        The error type name or None.
    """
    for event in events:
        if event.name == 'exception':
            attrs = event.attributes or {}
            error_type = attrs.get('exception.type')
            if error_type:
                return truncate(str(error_type), 1024)
    return None


def extract_error_message(events: list[Any]) -> str | None:
    """Extract the error message from span events.

    Args:
        events: List of span events.

    Returns:
        The error message or None.
    """
    for event in events:
        if event.name == 'exception':
            attrs = event.attributes or {}
            error_msg = attrs.get('exception.message')
            if error_msg:
                return truncate(str(error_msg), 4096)
    return None


def extract_error_stack(events: list[Any]) -> str | None:
    """Extract the error stack trace from span events.

    Args:
        events: List of span events.

    Returns:
        The stack trace or None.
    """
    for event in events:
        if event.name == 'exception':
            attrs = event.attributes or {}
            stack = attrs.get('exception.stacktrace')
            if stack:
                return truncate(str(stack), 32_768)
    return None


def create_common_log_attributes(span: ReadableSpan, project_id: str | None = None) -> dict[str, Any]:
    """Create common log attributes for GCP structured logging.

    These attributes link logs to traces in Google Cloud.

    Args:
        span: The span to extract context from.
        project_id: Optional GCP project ID.

    Returns:
        Dictionary with logging.googleapis.com attributes.
    """
    span_context = span.context
    if span_context is None:
        return {}
    is_sampled = bool(span_context.trace_flags & TraceFlags.SAMPLED)

    return {
        'logging.googleapis.com/spanId': format(span_context.span_id, '016x'),
        'logging.googleapis.com/trace': f'projects/{project_id}/traces/{format(span_context.trace_id, "032x")}',
        'logging.googleapis.com/trace_sampled': '1' if is_sampled else '0',
    }
