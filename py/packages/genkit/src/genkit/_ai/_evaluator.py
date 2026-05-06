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

"""Evaluator type definitions for the Genkit framework."""

import json
import traceback
import uuid
from collections.abc import Callable, Coroutine
from typing import Any, ClassVar, TypeVar, cast

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from genkit._core._action import Action, ActionKind
from genkit._core._logger import get_logger
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._tracing import run_in_new_span
from genkit._core._typing import (
    ActionMetadata,
    BaseDataPoint,
    EvalFnResponse,
    EvalRequest,
    EvalResponse,
    EvalStatusEnum,
    Score,
    SpanMetadata,
)

logger = get_logger(__name__)

EVALUATOR_METADATA_KEY_DISPLAY_NAME = 'evaluatorDisplayName'
EVALUATOR_METADATA_KEY_DEFINITION = 'evaluatorDefinition'
EVALUATOR_METADATA_KEY_IS_BILLED = 'evaluatorIsBilled'

T = TypeVar('T')

# User-provided evaluator function that evaluates a single datapoint.
# Must be async (coroutine function).
EvaluatorFn = Callable[[BaseDataPoint, T], Coroutine[Any, Any, EvalFnResponse]]

# User-provided batch evaluator function that evaluates an EvaluationRequest
BatchEvaluatorFn = Callable[[EvalRequest, T], Coroutine[Any, Any, list[EvalFnResponse]]]


class EvaluatorRef(BaseModel):
    """Reference to an evaluator."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True, alias_generator=to_camel)

    name: str
    config_schema: dict[str, object] | None = None


def evaluator_ref(name: str, config_schema: dict[str, object] | None = None) -> EvaluatorRef:
    """Create an EvaluatorRef."""
    return EvaluatorRef(name=name, config_schema=config_schema)


def evaluator_action_metadata(
    name: str,
    config_schema: type | dict[str, Any] | None = None,
) -> ActionMetadata:
    """Create ActionMetadata for an evaluator action."""
    return ActionMetadata(
        action_type=ActionKind.EVALUATOR,
        name=name,
        input_json_schema=to_json_schema(EvalRequest),
        output_json_schema=to_json_schema(list[EvalFnResponse]),
        metadata={'evaluator': {'customOptions': to_json_schema(config_schema) if config_schema else None}},
    )


def _get_func_description(func: Callable[..., Any], description: str | None = None) -> str:
    """Return description if provided, otherwise use the function's docstring."""
    if description is not None:
        return description
    if func.__doc__ is not None:
        return func.__doc__
    return ''


def define_evaluator(
    registry: Registry,
    name: str,
    display_name: str,
    definition: str,
    fn: EvaluatorFn[Any],
    is_billed: bool = False,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    description: str | None = None,
) -> Action:
    """Register an evaluator that runs the callback on each dataset sample."""
    evaluator_meta: dict[str, object] = dict(metadata) if metadata else {}
    evaluator_info: dict[str, object]
    existing_evaluator = evaluator_meta.get('evaluator')
    if isinstance(existing_evaluator, dict):
        evaluator_info = {str(key): value for key, value in existing_evaluator.items()}
    else:
        evaluator_info = {}
    evaluator_meta['evaluator'] = evaluator_info
    evaluator_info[EVALUATOR_METADATA_KEY_DEFINITION] = definition
    evaluator_info[EVALUATOR_METADATA_KEY_DISPLAY_NAME] = display_name
    evaluator_info[EVALUATOR_METADATA_KEY_IS_BILLED] = is_billed
    label_value = evaluator_info.get('label')
    if not isinstance(label_value, str) or not label_value:
        evaluator_info['label'] = name
    if config_schema:
        evaluator_info['customOptions'] = to_json_schema(config_schema)

    evaluator_description = _get_func_description(fn, description)

    async def eval_stepper_fn(req: EvalRequest) -> EvalResponse:
        eval_responses: list[EvalFnResponse] = []
        for index in range(len(req.dataset)):
            datapoint = req.dataset[index]
            if datapoint.test_case_id is None:
                datapoint.test_case_id = str(uuid.uuid4())
            span_metadata = SpanMetadata(
                name=f'Test Case {datapoint.test_case_id}',
                metadata={'evaluator:evalRunId': req.eval_run_id},
            )
            try:
                # Try to run with tracing, but fallback if tracing infrastructure fails
                # (e.g., in environments with NonRecordingSpans like pre-commit)
                try:
                    with run_in_new_span(span_metadata, labels={'genkit:type': 'evaluator'}) as span:
                        span_id = format(span.get_span_context().span_id, '016x')
                        trace_id = format(span.get_span_context().trace_id, '032x')
                        try:
                            input_json = (
                                datapoint.model_dump_json(by_alias=True, exclude_none=True)
                                if isinstance(datapoint, BaseModel)
                                else json.dumps(datapoint)
                            )
                            span.set_attribute('genkit:input', input_json)
                            test_case_output = await fn(datapoint, req.options)
                            test_case_output.span_id = span_id
                            test_case_output.trace_id = trace_id
                            output_json = (
                                test_case_output.model_dump_json(by_alias=True, exclude_none=True)
                                if isinstance(test_case_output, BaseModel)
                                else json.dumps(test_case_output)
                            )
                            span.set_attribute('genkit:output', output_json)
                            eval_responses.append(test_case_output)
                        except Exception as e:
                            logger.debug(f'eval_stepper_fn error: {e!s}')
                            logger.debug(traceback.format_exc())
                            evaluation = Score(
                                error=f'Evaluation of test case {datapoint.test_case_id} failed: \n{e!s}',
                                status=EvalStatusEnum.FAIL,
                            )
                            eval_responses.append(
                                # The ty type checker only recognizes aliases, so we use them
                                # to pass both ty check and runtime validation.
                                EvalFnResponse(
                                    span_id=span_id,
                                    trace_id=trace_id,
                                    test_case_id=datapoint.test_case_id,
                                    evaluation=evaluation,
                                )
                            )
                            # Raise to mark span as failed
                            raise e
                except (AttributeError, UnboundLocalError):
                    # Fallback: run without span
                    try:
                        test_case_output = await fn(datapoint, req.options)
                        eval_responses.append(test_case_output)
                    except Exception as e:
                        logger.debug(f'eval_stepper_fn error: {e!s}')
                        logger.debug(traceback.format_exc())
                        evaluation = Score(
                            error=f'Evaluation of test case {datapoint.test_case_id} failed: \n{e!s}',
                            status=EvalStatusEnum.FAIL,
                        )
                        eval_responses.append(
                            EvalFnResponse(
                                test_case_id=datapoint.test_case_id,
                                evaluation=evaluation,
                            )
                        )
            except Exception:  # noqa: S112 - intentionally continue processing other datapoints
                # Continue to process other points
                continue
        return EvalResponse(eval_responses)

    return registry.register_action(
        name=name,
        kind=ActionKind.EVALUATOR,
        fn=eval_stepper_fn,
        metadata=evaluator_meta,
        description=evaluator_description,
    )


def define_batch_evaluator(
    registry: Registry,
    name: str,
    display_name: str,
    definition: str,
    fn: BatchEvaluatorFn[Any],
    is_billed: bool = False,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    description: str | None = None,
) -> Action:
    """Register a batch evaluator that runs the callback on the entire dataset."""
    evaluator_meta: dict[str, object] = metadata.copy() if metadata else {}
    if 'evaluator' not in evaluator_meta:
        evaluator_meta['evaluator'] = {}
    # Cast to dict for nested operations - pyrefly doesn't narrow nested dict types
    evaluator_dict = cast(dict[str, object], evaluator_meta['evaluator'])
    evaluator_dict[EVALUATOR_METADATA_KEY_DEFINITION] = definition
    evaluator_dict[EVALUATOR_METADATA_KEY_DISPLAY_NAME] = display_name
    evaluator_dict[EVALUATOR_METADATA_KEY_IS_BILLED] = is_billed
    if 'label' not in evaluator_dict or not evaluator_dict['label']:
        evaluator_dict['label'] = name
    if config_schema:
        evaluator_dict['customOptions'] = to_json_schema(config_schema)

    evaluator_description = _get_func_description(fn, description)
    return registry.register_action(
        name=name,
        kind=ActionKind.EVALUATOR,
        fn=fn,
        metadata=evaluator_meta,
        description=evaluator_description,
    )
