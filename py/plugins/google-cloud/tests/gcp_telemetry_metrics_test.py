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

"""Tests for Google Cloud telemetry metrics helpers."""

from genkit.plugins.google_cloud.telemetry.metrics import (
    _extract_feature_name,
    _metric,
)


class TestMetricHelper:
    """Tests for _metric name prefix function."""

    def test_basic_name(self) -> None:
        """Test Basic name."""
        name, desc, unit = _metric('generate/requests', 'Generate requests')
        assert name == 'genkit/ai/generate/requests'
        assert desc == 'Generate requests'
        assert unit == '1'

    def test_custom_unit(self) -> None:
        """Test Custom unit."""
        name, desc, unit = _metric('generate/latency', 'Latency', 'ms')
        assert unit == 'ms'

    def test_default_unit_is_one(self) -> None:
        """Test Default unit is one."""
        _, _, unit = _metric('test', 'test')
        assert unit == '1'

    def test_nested_name(self) -> None:
        """Test Nested name."""
        name, _, _ = _metric('generate/input/tokens', 'Input tokens')
        assert name == 'genkit/ai/generate/input/tokens'


class TestExtractFeatureName:
    """Tests for _extract_feature_name path parsing."""

    def test_simple_flow_path(self) -> None:
        """Test Simple flow path."""
        result = _extract_feature_name('/{myFlow,t:flow}')
        assert result == 'myFlow'

    def test_nested_path_extracts_outer(self) -> None:
        """Test Nested path extracts outer."""
        result = _extract_feature_name('/{outer,t:flow}/{inner,t:flow}')
        assert result == 'outer'

    def test_empty_path(self) -> None:
        """Test Empty path."""
        result = _extract_feature_name('')
        assert result == '<unknown>'

    def test_no_slash(self) -> None:
        """Test No slash."""
        result = _extract_feature_name('something')
        assert result == '<unknown>'

    def test_single_slash(self) -> None:
        """Test Single slash."""
        result = _extract_feature_name('/')
        assert result == '<unknown>'

    def test_malformed_path(self) -> None:
        """Test Malformed path."""
        result = _extract_feature_name('/no-braces')
        assert result == '<unknown>'

    def test_model_action_path(self) -> None:
        """Test Model action path."""
        result = _extract_feature_name('/{chatFlow,t:flow}/{google-genai/gemini-2.0-flash,t:model}')
        assert result == 'chatFlow'

    def test_path_with_special_chars(self) -> None:
        """Test Path with special chars."""
        result = _extract_feature_name('/{my-flow-name,t:flow}')
        assert result == 'my-flow-name'

    def test_path_with_dots(self) -> None:
        """Test Path with dots."""
        result = _extract_feature_name('/{my.dotted.flow,t:flow}')
        assert result == 'my.dotted.flow'
