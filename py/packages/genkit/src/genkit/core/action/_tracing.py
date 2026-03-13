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

"""Action tracing module for defining and managing action tracing."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import quote

from opentelemetry.trace import Span
from opentelemetry.util import types as otel_types

from genkit.codec import dump_json

# Type alias for span attribute values
SpanAttributeValue = otel_types.AttributeValue

# Context variable to track parent path across nested spans
_parent_path_context: ContextVar[str] = ContextVar('genkit_parent_path', default='')


def build_path(
    name: str,
    parent_path: str,
    type_str: str,
    subtype: str | None = None,
) -> str:
    """Build a hierarchical path with type annotations.

    Args:
        name: The name of the action/flow/step.
        parent_path: The path of the parent span (empty string for root).
        type_str: The type (e.g., 'flow', 'action', 'flowStep').
        subtype: Optional subtype (e.g., 'tool', 'model', 'flow').

    Returns:
        Annotated path string.

    Examples:
        >>> build_path('myFlow', '', 'flow')
        '/{myFlow,t:flow}'

        >>> build_path('myTool', '/{myFlow,t:flow}', 'action', 'tool')
        '/{myFlow,t:flow}/{myTool,t:action,s:tool}'
    """
    # URL-encode name to handle special characters
    name = quote(name, safe='')

    # Build the path segment with type annotation
    if type_str:
        path_segment = f'{name},t:{type_str}'
    else:
        path_segment = name

    # Add subtype if provided
    if subtype:
        path_segment = f'{path_segment},s:{subtype}'

    # Wrap in braces and append to parent path
    path_segment = '{' + path_segment + '}'
    return parent_path + '/' + path_segment


def decorate_path_with_subtype(path: str, subtype: str) -> str:
    """Add subtype annotation to the leaf node of a path.

    Args:
        path: The path to decorate.
        subtype: The subtype to add (e.g., 'tool', 'model', 'flow').

    Returns:
        Decorated path string.

    Examples:
        >>> decorate_path_with_subtype('/{myFlow,t:flow}/{myTool,t:action}', 'tool')
        '/{myFlow,t:flow}/{myTool,t:action,s:tool}'
    """
    if not path or not subtype:
        return path

    # Find the last opening brace
    last_brace_idx = path.rfind('{')
    if last_brace_idx == -1:
        return path  # No braces found

    # Find the closing brace after the last opening brace
    closing_brace_idx = path.find('}', last_brace_idx)
    if closing_brace_idx == -1:
        return path  # No closing brace found

    # Extract the content of the last segment (without braces)
    segment_content = path[last_brace_idx + 1 : closing_brace_idx]

    # Check if subtype already exists
    if any(p.strip().startswith('s:') for p in segment_content.split(',')[1:]):
        return path

    # Add subtype annotation
    decorated_content = segment_content + ',s:' + subtype

    # Rebuild the path with the decorated last segment
    return path[: last_brace_idx + 1] + decorated_content + path[closing_brace_idx:]


@contextmanager
def save_parent_path() -> Generator[None, None, None]:
    """Context manager to save and restore parent path.

    Yields:
        None
    """
    saved = _parent_path_context.get()
    try:
        yield
    finally:
        _parent_path_context.set(saved)


def record_input_metadata(
    span: Span,
    kind: str,
    name: str,
    span_metadata: dict[str, SpanAttributeValue] | None,
    input: object | None,
) -> None:
    """Records input metadata onto an OpenTelemetry span for a Genkit action.

    Sets standard Genkit attributes like action type, subtype (kind), name,
    path, qualifiedPath, and the JSON representation of the input. Also adds
    any custom span metadata provided.

    Args:
        span: The OpenTelemetry Span object to add attributes to.
        kind: The kind (e.g., 'model', 'flow', 'tool') of the action.
        name: The specific name of the action.
        span_metadata: An optional dictionary of custom key-value pairs to add
                       as span attributes.
        input: The input data provided to the action.
    """
    span.set_attribute('genkit:type', 'action')
    span.set_attribute('genkit:metadata:subtype', kind)
    span.set_attribute('genkit:name', name)
    if input is not None:
        span.set_attribute('genkit:input', dump_json(input))

    # Build and set path attributes (qualified path with full annotations)
    parent_path = _parent_path_context.get()
    qualified_path = build_path(name, parent_path, 'action', kind)

    # IMPORTANT: Span attributes store the QUALIFIED path (full annotated)
    # Telemetry handlers will derive display path using to_display_path()
    span.set_attribute('genkit:path', qualified_path)
    span.set_attribute('genkit:qualifiedPath', qualified_path)

    # Update context for nested spans
    _parent_path_context.set(qualified_path)

    if span_metadata is not None:
        for meta_key in span_metadata:
            span.set_attribute(meta_key, span_metadata[meta_key])


def record_output_metadata(span: Span, output: object) -> None:
    """Records output metadata onto an OpenTelemetry span for a Genkit action.

    Marks the span state as 'success' and records the JSON representation of
    the action's output.

    Args:
        span: The OpenTelemetry Span object to add attributes to.
        output: The output data returned by the action.
    """
    span.set_attribute('genkit:state', 'success')
    try:
        span.set_attribute('genkit:output', dump_json(output))
    except Exception:
        # Fallback for non-serializable output
        span.set_attribute('genkit:output', str(output))
