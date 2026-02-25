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

"""Tests for evaluator helper functions and schema types."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genkit.ai import Genkit
from genkit.blocks.model import ModelReference
from genkit.core.typing import BaseDataPoint, EvalStatusEnum, Score
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
from genkit.plugins.evaluators.helpers import evaluators_name, fill_scores


class TestEvaluatorsName:
    """Tests for the evaluators_name helper."""

    def test_produces_genkiteval_prefix(self) -> None:
        """Name is prefixed with 'genkitEval/'."""
        result = evaluators_name('answer_relevancy')
        assert result == 'genkitEval/answer_relevancy'

    def test_preserves_casing(self) -> None:
        """Casing of the input name is preserved."""
        assert evaluators_name('MyMetric') == 'genkitEval/MyMetric'

    def test_empty_name(self) -> None:
        """Empty name still produces the prefix."""
        assert evaluators_name('') == 'genkitEval/'


class TestFillScores:
    """Tests for the fill_scores helper."""

    def test_without_status_override(self) -> None:
        """Without override, the score's original status is preserved."""
        dp = BaseDataPoint(input='q', output='a', test_case_id='tc1')
        score = Score(score=True, status=EvalStatusEnum.PASS_)

        result = fill_scores(dp, score)

        assert result.test_case_id == 'tc1'
        assert isinstance(result.evaluation, Score)
        assert result.evaluation.status == EvalStatusEnum.PASS_
        assert result.evaluation.score is True

    def test_with_status_override(self) -> None:
        """Override function replaces the status."""
        dp = BaseDataPoint(input='q', output='a', test_case_id='tc2')
        score = Score(score=0.3, status=EvalStatusEnum.PASS_)

        # Override: always FAIL regardless of original status
        def override_fn(s: Score) -> EvalStatusEnum:
            return EvalStatusEnum.FAIL

        result = fill_scores(dp, score, override_fn)

        assert isinstance(result.evaluation, Score)
        assert result.evaluation.status == EvalStatusEnum.FAIL
        assert result.evaluation.score == 0.3

    def test_missing_test_case_id_defaults_to_empty(self) -> None:
        """Missing test_case_id defaults to empty string."""
        dp = BaseDataPoint(input='q', output='a')
        score = Score(score=True, status=EvalStatusEnum.PASS_)

        result = fill_scores(dp, score)
        assert result.test_case_id == ''


class TestGenkitMetricType:
    """Tests for GenkitMetricType enum values and properties."""

    def test_all_expected_metrics_exist(self) -> None:
        """All six metric types are defined."""
        expected = {
            'ANSWER_RELEVANCY',
            'FAITHFULNESS',
            'MALICIOUSNESS',
            'REGEX',
            'DEEP_EQUAL',
            'JSONATA',
        }
        actual = {m.name for m in GenkitMetricType}
        assert actual == expected

    def test_metric_type_is_string_enum(self) -> None:
        """Each metric type value is a string (StrEnum)."""
        for metric in GenkitMetricType:
            assert isinstance(metric, str)

    def test_metric_count(self) -> None:
        """Exactly 6 metric types are defined."""
        assert len(GenkitMetricType) == 6


class TestMetricConfig:
    """Tests for MetricConfig Pydantic model."""

    def test_minimal_config(self) -> None:
        """MetricConfig requires only metric_type."""
        config = MetricConfig(metric_type=GenkitMetricType.REGEX)
        assert config.metric_type == GenkitMetricType.REGEX
        assert config.judge is None
        assert config.judge_config is None
        assert config.status_override_fn is None
        assert config.metric_config is None

    def test_config_with_judge(self) -> None:
        """MetricConfig accepts a judge ModelReference."""
        from genkit.blocks.model import ModelReference

        judge = ModelReference(name='googleai/gemini-2.0-flash')
        config = MetricConfig(
            metric_type=GenkitMetricType.FAITHFULNESS,
            judge=judge,
            judge_config={'temperature': 0.0},
        )
        assert config.judge is not None
        assert config.judge.name == 'googleai/gemini-2.0-flash'
        assert config.judge_config == {'temperature': 0.0}


class TestPluginOptions:
    """Tests for PluginOptions RootModel."""

    def test_wraps_list_of_metric_config(self) -> None:
        """PluginOptions wraps a list of MetricConfig."""
        configs = [
            MetricConfig(metric_type=GenkitMetricType.REGEX),
            MetricConfig(metric_type=GenkitMetricType.DEEP_EQUAL),
        ]
        opts = PluginOptions(root=configs)
        assert len(opts.root) == 2
        assert opts.root[0].metric_type == GenkitMetricType.REGEX
        assert opts.root[1].metric_type == GenkitMetricType.DEEP_EQUAL

    def test_empty_list(self) -> None:
        """PluginOptions accepts an empty list."""
        opts = PluginOptions([])
        assert len(opts.root) == 0


class TestResponseSchemas:
    """Tests for evaluator response Pydantic schemas."""

    def test_answer_relevancy_schema_fields(self) -> None:
        """AnswerRelevancyResponseSchema has question, answered, noncommittal."""
        resp = AnswerRelevancyResponseSchema(
            question='What is AI?',
            answered=True,
            noncommittal=False,
        )
        assert resp.question == 'What is AI?'
        assert resp.answered is True
        assert resp.noncommittal is False

    def test_answer_relevancy_schema_json_roundtrip(self) -> None:
        """Schema serializes and deserializes correctly."""
        resp = AnswerRelevancyResponseSchema(question='test', answered=False, noncommittal=True)
        data = resp.model_dump()
        restored = AnswerRelevancyResponseSchema.model_validate(data)
        assert restored == resp

    def test_maliciousness_schema(self) -> None:
        """MaliciousnessResponseSchema with reason and verdict."""
        resp = MaliciousnessResponseSchema(
            reason='Contains harmful instructions',
            verdict=True,
        )
        assert resp.reason == 'Contains harmful instructions'
        assert resp.verdict is True

    def test_long_form_schema(self) -> None:
        """LongFormResponseSchema holds list of statements."""
        resp = LongFormResponseSchema(statements=['The sky is blue', 'Water is wet'])
        assert len(resp.statements) == 2
        assert 'sky' in resp.statements[0]

    def test_long_form_empty_statements(self) -> None:
        """LongFormResponseSchema accepts empty list."""
        resp = LongFormResponseSchema(statements=[])
        assert resp.statements == []

    def test_nli_response(self) -> None:
        """NliResponse holds list of NliResponseBase entries."""
        entries = [
            NliResponseBase(
                statement='The earth orbits the sun',
                reason='Supported by context',
                verdict=True,
            ),
            NliResponseBase(
                statement='The moon is made of cheese',
                reason='Not supported',
                verdict=False,
            ),
        ]
        nli = NliResponse(responses=entries)
        assert len(nli.responses) == 2
        assert nli.responses[0].verdict is True
        assert nli.responses[1].verdict is False
        assert 'Not supported' in nli.responses[1].reason

    def test_nli_response_faithfulness_score_calculation(self) -> None:
        """Verify faithfulness score calculation logic from NLI responses.

        This mirrors the actual scoring logic in _faithfulness_eval:
        faithful_count / total_responses.
        """
        entries = [
            NliResponseBase(statement='s1', reason='ok', verdict=True),
            NliResponseBase(statement='s2', reason='ok', verdict=True),
            NliResponseBase(statement='s3', reason='no', verdict=False),
        ]
        # Reproduce the actual scoring logic from helpers.py
        faithful_count = sum(1 for r in entries if r.verdict)
        score_val = faithful_count / len(entries)

        assert faithful_count == 2
        assert abs(score_val - 2 / 3) < 1e-9
        # Score > 0.5 means PASS
        status = EvalStatusEnum.PASS_ if score_val > 0.5 else EvalStatusEnum.FAIL
        assert status == EvalStatusEnum.PASS_

    def test_nli_all_unfaithful_means_fail(self) -> None:
        """When all NLI verdicts are False, the score is 0 and status is FAIL."""
        entries = [
            NliResponseBase(statement='s1', reason='no', verdict=False),
            NliResponseBase(statement='s2', reason='no', verdict=False),
        ]
        faithful_count = sum(1 for r in entries if r.verdict)
        score_val = faithful_count / len(entries)

        assert score_val == 0.0
        status = EvalStatusEnum.PASS_ if score_val > 0.5 else EvalStatusEnum.FAIL
        assert status == EvalStatusEnum.FAIL


class TestEvaluatorConfiguration:
    """Tests for evaluator configuration logic."""

    @pytest.mark.asyncio
    async def test_answer_relevancy_loads_correct_prompt(self) -> None:
        """ANSWER_RELEVANCY metric should load 'answer_relevancy.prompt'."""
        # Mock AI and Model
        mock_ai = MagicMock(spec=Genkit)
        mock_ai.define_evaluator = MagicMock()
        mock_ai.generate = AsyncMock()

        # Configuration for ANSWER_RELEVANCY
        config = MetricConfig(
            metric_type=GenkitMetricType.ANSWER_RELEVANCY, judge=ModelReference(name='test-judge'), judge_config={}
        )

        # Patch load_prompt_file AND render_text to verify arguments
        with (
            patch('genkit.plugins.evaluators.helpers.load_prompt_file', new_callable=AsyncMock) as mock_load_prompt,
            patch('genkit.plugins.evaluators.helpers.render_text', new_callable=AsyncMock) as mock_render_text,
        ):
            # We need to configure the evaluator, which defines the function
            from genkit.plugins.evaluators.helpers import _configure_evaluator

            _configure_evaluator(mock_ai, config)

            # Get the defined evaluator function
            evaluator_fn = mock_ai.define_evaluator.call_args.kwargs['fn']

            # Call the evaluator function
            datapoint = BaseDataPoint(input='test_question', output='test_answer', context=['test_context'])

            try:
                await evaluator_fn(datapoint, None)
            except Exception:  # noqa: S110 - intentionally silent, we only check mock calls
                pass

            # Check prompt file
            assert mock_load_prompt.called
            file_path = mock_load_prompt.call_args[0][0]
            assert 'answer_relevancy.prompt' in file_path

            # Check render_text arguments are correct (question/answer, NOT input/output)
            assert mock_render_text.called
            call_args = mock_render_text.call_args
            # render_text(prompt, variables)
            render_variables = call_args[0][1]

            assert 'question' in render_variables
            assert render_variables['question'] == 'test_question'
            assert 'answer' in render_variables
            assert render_variables['answer'] == 'test_answer'
