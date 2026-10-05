#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluator datapoint typing helpers."""

from genkit._ai._evaluator import _as_eval_datapoint
from genkit._core._typing import BaseDataPoint, BaseEvalDataPoint


def test_as_eval_datapoint_preserves_existing_id() -> None:
    dp = BaseDataPoint(input='in', output='out', test_case_id='case-1')
    eval_dp = _as_eval_datapoint(dp)
    assert isinstance(eval_dp, BaseEvalDataPoint)
    assert eval_dp.test_case_id == 'case-1'
    assert eval_dp.input == 'in'
    assert eval_dp.output == 'out'


def test_as_eval_datapoint_generates_id_when_missing() -> None:
    dp = BaseDataPoint(input='in', output='out')
    assert dp.test_case_id is None
    eval_dp = _as_eval_datapoint(dp)
    assert isinstance(eval_dp, BaseEvalDataPoint)
    assert isinstance(eval_dp.test_case_id, str)
    assert eval_dp.test_case_id
    # Does not mutate the dataset datapoint.
    assert dp.test_case_id is None
