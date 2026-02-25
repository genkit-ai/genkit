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

"""Tests for google-cloud telemetry utility functions.

Tests cover truncation, path parsing, error extraction, and log attribute
creation — matching the JS implementation in js/plugins/google-cloud/src/utils.ts.
"""

from unittest.mock import MagicMock

from opentelemetry.trace import TraceFlags

from genkit.plugins.google_cloud.telemetry.utils import (
    MAX_LOG_CONTENT_CHARS,
    MAX_PATH_CHARS,
    create_common_log_attributes,
    extract_error_message,
    extract_error_name,
    extract_error_stack,
    extract_outer_feature_name_from_path,
    extract_outer_flow_name_from_path,
    to_display_path,
    truncate,
    truncate_path,
)


# ---------------------------------------------------------------------------
# truncate()
# ---------------------------------------------------------------------------
class TestTruncate:
    """Tests for Truncate."""

    def test_none_returns_empty(self) -> None:
        """None returns empty."""
        assert truncate(None) == ''

    def test_empty_string_returns_empty(self) -> None:
        """Empty string returns empty."""
        assert truncate('') == ''

    def test_short_text_unchanged(self) -> None:
        """Short text unchanged."""
        assert truncate('hello') == 'hello'

    def test_text_at_limit_unchanged(self) -> None:
        """Text at limit unchanged."""
        text = 'x' * MAX_LOG_CONTENT_CHARS
        assert truncate(text) == text

    def test_text_exceeding_limit_truncated(self) -> None:
        """Text exceeding limit truncated."""
        text = 'x' * (MAX_LOG_CONTENT_CHARS + 100)
        result = truncate(text)
        assert len(result) == MAX_LOG_CONTENT_CHARS

    def test_custom_limit(self) -> None:
        """Custom limit."""
        assert truncate('abcdef', limit=3) == 'abc'


# ---------------------------------------------------------------------------
# truncate_path()
# ---------------------------------------------------------------------------
class TestTruncatePath:
    """Tests for TruncatePath."""

    def test_short_path_unchanged(self) -> None:
        """Short path unchanged."""
        assert truncate_path('/foo/bar') == '/foo/bar'

    def test_long_path_truncated(self) -> None:
        """Long path truncated."""
        path = '/' * (MAX_PATH_CHARS + 100)
        result = truncate_path(path)
        assert len(result) == MAX_PATH_CHARS


# ---------------------------------------------------------------------------
# extract_outer_flow_name_from_path()
# ---------------------------------------------------------------------------
class TestExtractOuterFlowName:
    """Tests for ExtractOuterFlowName."""

    def test_standard_flow_path(self) -> None:
        """Standard flow path."""
        path = '/{myFlow,t:flow}'
        assert extract_outer_flow_name_from_path(path) == 'myFlow'

    def test_nested_path(self) -> None:
        """Nested path."""
        path = '/{orderFlow,t:flow}/{step,t:flowStep}'
        assert extract_outer_flow_name_from_path(path) == 'orderFlow'

    def test_empty_string_returns_unknown(self) -> None:
        """Empty string returns unknown."""
        assert extract_outer_flow_name_from_path('') == '<unknown>'

    def test_unknown_string_returns_unknown(self) -> None:
        """Unknown string returns unknown."""
        assert extract_outer_flow_name_from_path('<unknown>') == '<unknown>'

    def test_no_flow_type_returns_unknown(self) -> None:
        """No flow type returns unknown."""
        path = '/{myAction,t:action}'
        assert extract_outer_flow_name_from_path(path) == '<unknown>'


# ---------------------------------------------------------------------------
# extract_outer_feature_name_from_path()
# ---------------------------------------------------------------------------
class TestExtractOuterFeatureName:
    """Tests for ExtractOuterFeatureName."""

    def test_flow_path(self) -> None:
        """Flow path."""
        path = '/{myFlow,t:flow}/{step,t:flowStep}'
        assert extract_outer_feature_name_from_path(path) == 'myFlow'

    def test_action_path(self) -> None:
        """Action path."""
        path = '/{myAction,t:action}'
        assert extract_outer_feature_name_from_path(path) == 'myAction'

    def test_prompt_path(self) -> None:
        """Prompt path."""
        path = '/{myPrompt,t:prompt}'
        assert extract_outer_feature_name_from_path(path) == 'myPrompt'

    def test_dotprompt_path(self) -> None:
        """Dotprompt path."""
        path = '/{myDotPrompt,t:dotprompt}'
        assert extract_outer_feature_name_from_path(path) == 'myDotPrompt'

    def test_helper_path(self) -> None:
        """Helper path."""
        path = '/{myHelper,t:helper}'
        assert extract_outer_feature_name_from_path(path) == 'myHelper'

    def test_empty_string_returns_unknown(self) -> None:
        """Empty string returns unknown."""
        assert extract_outer_feature_name_from_path('') == '<unknown>'

    def test_unknown_string_returns_unknown(self) -> None:
        """Unknown string returns unknown."""
        assert extract_outer_feature_name_from_path('<unknown>') == '<unknown>'

    def test_single_segment_returns_unknown(self) -> None:
        """Single segment returns unknown."""
        assert extract_outer_feature_name_from_path('no-braces') == '<unknown>'

    def test_unrecognized_type_returns_unknown(self) -> None:
        """Unrecognized type returns unknown."""
        path = '/{thing,t:somethingElse}'
        assert extract_outer_feature_name_from_path(path) == '<unknown>'

    def test_complex_nested_path(self) -> None:
        """Complex nested path."""
        path = '/{myFlow,t:flow}/{myStep,t:flowStep}/{googleai/gemini-pro,t:action,s:model}'
        assert extract_outer_feature_name_from_path(path) == 'myFlow'


# ---------------------------------------------------------------------------
# extract_error_name / extract_error_message / extract_error_stack
# ---------------------------------------------------------------------------
def _make_event(name: str, attrs: dict) -> MagicMock:
    event = MagicMock()
    event.name = name
    event.attributes = attrs
    return event


class TestExtractErrorName:
    """Tests for ExtractErrorName."""

    def test_extracts_error_type(self) -> None:
        """Extracts error type."""
        events = [_make_event('exception', {'exception.type': 'ValueError'})]
        assert extract_error_name(events) == 'ValueError'

    def test_no_exception_returns_none(self) -> None:
        """No exception returns none."""
        events = [_make_event('other', {})]
        assert extract_error_name(events) is None

    def test_empty_events_returns_none(self) -> None:
        """Empty events returns none."""
        assert extract_error_name([]) is None

    def test_truncates_long_error_type(self) -> None:
        """Truncates long error type."""
        long_type = 'E' * 2000
        events = [_make_event('exception', {'exception.type': long_type})]
        result = extract_error_name(events)
        assert result is not None
        assert len(result) == 1024

    def test_missing_type_attribute_returns_none(self) -> None:
        """Missing type attribute returns none."""
        events = [_make_event('exception', {'exception.message': 'oops'})]
        assert extract_error_name(events) is None


class TestExtractErrorMessage:
    """Tests for ExtractErrorMessage."""

    def test_extracts_message(self) -> None:
        """Extracts message."""
        events = [_make_event('exception', {'exception.message': 'something went wrong'})]
        assert extract_error_message(events) == 'something went wrong'

    def test_no_exception_returns_none(self) -> None:
        """No exception returns none."""
        assert extract_error_message([]) is None

    def test_truncates_long_message(self) -> None:
        """Truncates long message."""
        long_msg = 'M' * 5000
        events = [_make_event('exception', {'exception.message': long_msg})]
        result = extract_error_message(events)
        assert result is not None
        assert len(result) == 4096


class TestExtractErrorStack:
    """Tests for ExtractErrorStack."""

    def test_extracts_stacktrace(self) -> None:
        """Extracts stacktrace."""
        events = [_make_event('exception', {'exception.stacktrace': 'Traceback...'})]
        assert extract_error_stack(events) == 'Traceback...'

    def test_no_exception_returns_none(self) -> None:
        """No exception returns none."""
        assert extract_error_stack([]) is None

    def test_truncates_long_stack(self) -> None:
        """Truncates long stack."""
        long_stack = 'S' * 40_000
        events = [_make_event('exception', {'exception.stacktrace': long_stack})]
        result = extract_error_stack(events)
        assert result is not None
        assert len(result) == 32_768


# ---------------------------------------------------------------------------
# create_common_log_attributes()
# ---------------------------------------------------------------------------
class TestCreateCommonLogAttributes:
    """Tests for CreateCommonLogAttributes."""

    def test_creates_attributes_with_project(self) -> None:
        """Creates attributes with project."""
        span = MagicMock()
        span.context.trace_id = 0x12345678901234567890123456789012
        span.context.span_id = 0x1234567890123456
        span.context.trace_flags = TraceFlags.SAMPLED

        attrs = create_common_log_attributes(span, project_id='my-project')
        assert attrs['logging.googleapis.com/spanId'] == '1234567890123456'
        assert 'my-project' in attrs['logging.googleapis.com/trace']
        assert '12345678901234567890123456789012' in attrs['logging.googleapis.com/trace']
        assert attrs['logging.googleapis.com/trace_sampled'] == '1'

    def test_unsampled_trace(self) -> None:
        """Unsampled trace."""
        span = MagicMock()
        span.context.trace_id = 0xAABBCCDDEEFF0011AABBCCDDEEFF0011
        span.context.span_id = 0xAABBCCDDEEFF0011
        span.context.trace_flags = TraceFlags.DEFAULT  # Not sampled

        attrs = create_common_log_attributes(span, project_id='p')
        assert attrs['logging.googleapis.com/trace_sampled'] == '0'

    def test_none_context_returns_empty(self) -> None:
        """None context returns empty."""
        span = MagicMock()
        span.context = None
        assert create_common_log_attributes(span) == {}


# ---------------------------------------------------------------------------
# to_display_path()
# ---------------------------------------------------------------------------
class TestToDisplayPath:
    """Tests for ToDisplayPath."""

    def test_simple_flow_path(self) -> None:
        """Simple flow path."""
        assert to_display_path('/{myFlow,t:flow}') == 'myFlow'

    def test_nested_path(self) -> None:
        """Nested path."""
        result = to_display_path('/{myFlow,t:flow}/{step,t:flowStep}')
        assert result == 'myFlow/step'

    def test_three_level_path(self) -> None:
        """Three level path."""
        result = to_display_path('/{myFlow,t:flow}/{step,t:flowStep}/{googleai/gemini-pro,t:action,s:model}')
        # The function extracts the name part before the first comma in braces,
        # but nested slashes create extra segments.
        assert 'myFlow' in result
        assert 'step' in result

    def test_empty_string_returns_empty(self) -> None:
        """Empty string returns empty."""
        assert to_display_path('') == ''

    def test_plain_segments_preserved(self) -> None:
        """Plain segments preserved."""
        assert to_display_path('foo/bar') == 'foo/bar'
