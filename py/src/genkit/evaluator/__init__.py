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

"""Evaluator namespace module for Genkit.

This module provides evaluator-related types and utilities for plugin authors
and advanced users who need access to the evaluator protocol types.

Example:
    from genkit.evaluator import (
        EvalRequest,
        EvalResponse,
        evaluator_action_metadata,
    )
"""

from genkit._ai._evaluator import (
    EvaluatorRef,
    evaluator_action_metadata,
    evaluator_ref,
)
from genkit._core._typing import (
    BaseDataPoint,
    BaseEvalDataPoint,
    Details,
    EvalFnResponse,
    EvalRequest,
    EvalResponse,
    EvalStatusEnum,
    Score,
)

__all__ = [
    # Request/Response types
    'EvalRequest',
    'EvalResponse',
    'EvalFnResponse',
    # Score types
    'Score',
    'Details',
    # Data point types
    'BaseEvalDataPoint',
    'BaseDataPoint',
    # Status
    'EvalStatusEnum',
    # Factory functions and metadata
    'evaluator_action_metadata',
    'evaluator_ref',
    # Reference types
    'EvaluatorRef',
]
