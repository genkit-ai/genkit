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

"""Tests for evaluators constants, schemas, and metric types."""

from genkit.plugins.evaluators.constant import (
    AnswerRelevancyResponseSchema,
    GenkitMetricType,
    LongFormResponseSchema,
    MaliciousnessResponseSchema,
    MetricConfig,
    NliResponse,
    NliResponseBase,
    PluginOptions,
)


class TestGenkitMetricType:
    """Tests for GenkitMetricType StrEnum."""

    def test_answer_relevancy(self) -> None:
        """Test Answer relevancy."""
        assert GenkitMetricType.ANSWER_RELEVANCY == 'ANSWER_RELEVANCY'

    def test_faithfulness(self) -> None:
        """Test Faithfulness."""
        assert GenkitMetricType.FAITHFULNESS == 'FAITHFULNESS'

    def test_maliciousness(self) -> None:
        """Test Maliciousness."""
        assert GenkitMetricType.MALICIOUSNESS == 'MALICIOUSNESS'

    def test_regex(self) -> None:
        """Test Regex."""
        assert GenkitMetricType.REGEX == 'REGEX'

    def test_deep_equal(self) -> None:
        """Test Deep equal."""
        assert GenkitMetricType.DEEP_EQUAL == 'DEEP_EQUAL'

    def test_jsonata(self) -> None:
        """Test Jsonata."""
        assert GenkitMetricType.JSONATA == 'JSONATA'

    def test_total_count(self) -> None:
        """Test Total count."""
        assert len(GenkitMetricType) == 6


class TestMetricConfig:
    """Tests for MetricConfig model."""

    def test_minimal_config(self) -> None:
        """Test Minimal config."""
        cfg = MetricConfig(metric_type=GenkitMetricType.REGEX)
        assert cfg.metric_type == GenkitMetricType.REGEX
        assert cfg.judge is None
        assert cfg.judge_config is None
        assert cfg.metric_config is None
        assert cfg.status_override_fn is None

    def test_with_metric_config(self) -> None:
        """Test With metric config."""
        cfg = MetricConfig(
            metric_type=GenkitMetricType.REGEX,
            metric_config={'pattern': r'\d+'},
        )
        assert cfg.metric_config == {'pattern': r'\d+'}

    def test_with_judge_config(self) -> None:
        """Test With judge config."""
        cfg = MetricConfig(
            metric_type=GenkitMetricType.FAITHFULNESS,
            judge_config={'temperature': 0.0},
        )
        assert cfg.judge_config == {'temperature': 0.0}


class TestPluginOptions:
    """Tests for PluginOptions root model."""

    def test_empty_list(self) -> None:
        """Test Empty list."""
        opts = PluginOptions(root=[])
        assert opts.root == []

    def test_single_metric(self) -> None:
        """Test Single metric."""
        cfg = MetricConfig(metric_type=GenkitMetricType.REGEX)
        opts = PluginOptions(root=[cfg])
        assert len(opts.root) == 1

    def test_multiple_metrics(self) -> None:
        """Test Multiple metrics."""
        configs = [
            MetricConfig(metric_type=GenkitMetricType.REGEX),
            MetricConfig(metric_type=GenkitMetricType.DEEP_EQUAL),
            MetricConfig(metric_type=GenkitMetricType.FAITHFULNESS),
        ]
        opts = PluginOptions(root=configs)
        assert len(opts.root) == 3


class TestResponseSchemas:
    """Tests for evaluator response schemas."""

    def test_answer_relevancy_schema(self) -> None:
        """Test Answer relevancy schema."""
        schema = AnswerRelevancyResponseSchema(
            question='What is AI?',
            answered=True,
            noncommittal=False,
        )
        assert schema.question == 'What is AI?'
        assert schema.answered is True
        assert schema.noncommittal is False

    def test_maliciousness_schema(self) -> None:
        """Test Maliciousness schema."""
        schema = MaliciousnessResponseSchema(
            reason='Safe content',
            verdict=False,
        )
        assert schema.reason == 'Safe content'
        assert schema.verdict is False

    def test_long_form_schema(self) -> None:
        """Test Long form schema."""
        schema = LongFormResponseSchema(statements=['Fact 1', 'Fact 2'])
        assert len(schema.statements) == 2

    def test_nli_response_base(self) -> None:
        """Test Nli response base."""
        base = NliResponseBase(
            statement='The sky is blue',
            reason='Observable fact',
            verdict=True,
        )
        assert base.statement == 'The sky is blue'
        assert base.verdict is True

    def test_nli_response(self) -> None:
        """Test Nli response."""
        items = [
            NliResponseBase(statement='s1', reason='r1', verdict=True),
            NliResponseBase(statement='s2', reason='r2', verdict=False),
        ]
        resp = NliResponse(responses=items)
        assert len(resp.responses) == 2
        assert resp.responses[0].verdict is True
        assert resp.responses[1].verdict is False
