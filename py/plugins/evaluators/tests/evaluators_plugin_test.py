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

"""Tests for Evaluators plugin."""

from genkit.plugins.evaluators import (
    GenkitMetricType,
    MetricConfig,
    PluginOptions,
    define_genkit_evaluators,
    evaluators_name,
    package_name,
)


def test_package_name() -> None:
    """Test package_name returns correct value."""
    assert package_name() == 'genkit.plugins.evaluators'


def test_evaluators_name() -> None:
    """Test evaluators_name helper function."""
    result = evaluators_name('answer_relevancy')
    assert 'answer_relevancy' in result


def test_genkit_metric_type_enum() -> None:
    """Test GenkitMetricType enum has expected values."""
    # Check that the enum has at least the core metrics
    assert hasattr(GenkitMetricType, 'ANSWER_RELEVANCY')
    assert hasattr(GenkitMetricType, 'FAITHFULNESS')


def test_metric_config_instantiation() -> None:
    """Test MetricConfig can be instantiated."""
    config = MetricConfig(metric_type=GenkitMetricType.ANSWER_RELEVANCY)
    assert config.metric_type == GenkitMetricType.ANSWER_RELEVANCY


def test_plugin_options_instantiation() -> None:
    """Test PluginOptions can be instantiated."""
    # PluginOptions is a RootModel that wraps a list of MetricConfig
    options = PluginOptions([
        MetricConfig(metric_type=GenkitMetricType.ANSWER_RELEVANCY),
    ])
    assert len(options.root) == 1
    assert options.root[0].metric_type == GenkitMetricType.ANSWER_RELEVANCY


def test_define_genkit_evaluators_callable() -> None:
    """Test define_genkit_evaluators is callable."""
    assert callable(define_genkit_evaluators)
