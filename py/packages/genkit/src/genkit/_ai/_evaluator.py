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

import inspect
import traceback
import uuid
from collections.abc import Callable, Coroutine
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing_extensions import Never

from genkit._core._action import Action, ActionKind, ActionRunContext
from genkit._core._logger import get_logger
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._telemetry._attrs import metadata_key
from genkit._core._telemetry._instrumentation import SpanContext, run_in_new_span
from genkit._core._typing import (
    BaseDataPoint,
    EvalResponse,
    EvaluatorRequest,
    EvaluatorResponse,
    Score,
    ScoreStatus,
)

logger = get_logger(__name__)

EVALUATOR_METADATA_KEY_DISPLAY_NAME = 'evaluatorDisplayName'
EVALUATOR_METADATA_KEY_DEFINITION = 'evaluatorDefinition'
EVALUATOR_METADATA_KEY_IS_BILLED = 'evaluatorIsBilled'

T = TypeVar('T')

# User-provided evaluator function that evaluates a single datapoint.
# Must be async (coroutine function).
EvaluatorFn = Callable[[BaseDataPoint, T], Coroutine[Any, Any, EvaluatorResponse]]

# User-provided batch evaluator: one EvaluatorRequest. Returns list[EvaluatorResponse].
BatchEvaluatorFn = Callable[[EvaluatorRequest], Coroutine[Any, Any, list[EvaluatorResponse]]]


class EvaluatorRef(BaseModel):
    """Reference to an evaluator."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True, alias_generator=to_camel)

    name: str
    config: dict[str, object] | None = None


def _evaluator_metadata(
    name: str,
    display_name: str,
    definition: str,
    is_billed: bool,
    config_schema: type[BaseModel] | dict[str, object] | None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the action metadata that the Dev UI and `genkit eval:run` read."""
    evaluator_meta: dict[str, object] = dict(metadata) if metadata else {}
    existing = evaluator_meta.get('evaluator')
    info: dict[str, object] = {str(k): v for k, v in existing.items()} if isinstance(existing, dict) else {}
    evaluator_meta['evaluator'] = info
    info[EVALUATOR_METADATA_KEY_DEFINITION] = definition
    info[EVALUATOR_METADATA_KEY_DISPLAY_NAME] = display_name
    info[EVALUATOR_METADATA_KEY_IS_BILLED] = is_billed
    label = info.get('label')
    if not isinstance(label, str) or not label:
        info['label'] = name
    if config_schema:
        info['customOptions'] = to_json_schema(config_schema)
    return evaluator_meta


def _get_func_description(func: Callable[..., Any], description: str | None = None) -> str:
    """Return description if provided, otherwise use the function's docstring."""
    if description is not None:
        return description
    if func.__doc__ is not None:
        return func.__doc__
    return ''


class Evaluator(Action[EvaluatorRequest, EvalResponse, Never]):
    """Evaluator action that runs ``fn`` on each dataset sample, without registering it.

    Plugin ``init`` / ``resolve`` / ``list_actions`` return this.
    ``define_evaluator`` registers it.
    """

    def __init__(
        self,
        name: str,
        fn: EvaluatorFn[Any],
        *,
        display_name: str,
        definition: str,
        is_billed: bool = False,
        config_schema: type[BaseModel] | dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        description: str | None = None,
    ) -> None:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"Action handlers must be async functions. Got sync function for '{name}'.")

        evaluator_meta = _evaluator_metadata(name, display_name, definition, is_billed, config_schema, metadata)
        evaluator_description = _get_func_description(fn, description)

        async def eval_stepper_fn(req: EvaluatorRequest) -> EvalResponse:
            eval_responses: list[EvaluatorResponse] = []
            for index in range(len(req.dataset)):
                datapoint = req.dataset[index]
                case_id = datapoint.test_case_id or str(uuid.uuid4())
                datapoint.test_case_id = case_id
                try:

                    async def body(
                        span: SpanContext, point: BaseDataPoint = datapoint, test_case_id: str = case_id
                    ) -> EvaluatorResponse:
                        try:
                            test_case_output = await fn(point, req.options)
                            test_case_output.span_id = span.span_id
                            test_case_output.trace_id = span.trace_id
                            return test_case_output
                        except Exception as e:
                            logger.debug(f'eval_stepper_fn error: {e!s}')
                            logger.debug(traceback.format_exc())
                            evaluation = Score(
                                error=f'Evaluation of test case {test_case_id} failed: \n{e!s}',
                                status=ScoreStatus.FAIL,
                            )
                            eval_responses.append(
                                EvaluatorResponse(
                                    span_id=span.span_id,
                                    trace_id=span.trace_id,
                                    test_case_id=test_case_id,
                                    evaluation=[evaluation],
                                )
                            )
                            raise e

                    eval_responses.append(
                        await run_in_new_span(
                            f'Test Case {datapoint.test_case_id}',
                            body,
                            action_type='evaluator',
                            input=datapoint,
                            attributes={metadata_key('evaluator:evalRunId'): str(req.eval_run_id)},
                        )
                    )
                except Exception:  # noqa: S112 - intentionally continue processing other datapoints
                    continue
            return EvalResponse(eval_responses)

        super().__init__(
            kind=ActionKind.EVALUATOR,
            name=name,
            fn=eval_stepper_fn,
            metadata=evaluator_meta,
            description=evaluator_description,
        )


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
) -> Evaluator:
    """Register an evaluator that runs the callback on each dataset sample."""
    action = Evaluator(
        name,
        fn,
        display_name=display_name,
        definition=definition,
        is_billed=is_billed,
        config_schema=config_schema,
        metadata=metadata,
        description=description,
    )
    registry.register_action_from_instance(action)
    return action


class BatchEvaluator(Action[EvaluatorRequest, EvalResponse, Never]):
    """Batch evaluator action that runs ``fn`` once on the whole ``EvaluatorRequest``, without registering it.

    ``fn`` returns ``list[EvaluatorResponse]``. The action wraps the list so
    ``action.run(...).response`` is an ``EvalResponse``.
    Plugin ``init`` / ``resolve`` / ``list_actions`` return this.
    ``define_batch_evaluator`` registers it.
    """

    def __init__(
        self,
        name: str,
        fn: BatchEvaluatorFn,
        *,
        display_name: str,
        definition: str,
        is_billed: bool = False,
        config_schema: type[BaseModel] | dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        description: str | None = None,
    ) -> None:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"Action handlers must be async functions. Got sync function for '{name}'.")

        evaluator_meta = _evaluator_metadata(name, display_name, definition, is_billed, config_schema, metadata)
        evaluator_description = _get_func_description(fn, description)

        # the action hands back the rows as one model so the Dev UI and
        # `genkit eval:run` get a JSON array, the same as a per-row evaluator.
        # fn stays the metadata_fn, so its signature is still checked when built.
        async def batch_fn(req: EvaluatorRequest, ctx: ActionRunContext) -> EvalResponse:
            return EvalResponse(await self.params.call(fn, req, ctx))

        super().__init__(
            kind=ActionKind.EVALUATOR,
            name=name,
            fn=batch_fn,
            metadata_fn=fn,
            metadata=evaluator_meta,
            description=evaluator_description,
        )


def define_batch_evaluator(
    registry: Registry,
    name: str,
    display_name: str,
    definition: str,
    fn: BatchEvaluatorFn,
    is_billed: bool = False,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    description: str | None = None,
) -> BatchEvaluator:
    """Register a batch evaluator that runs ``fn`` once on the whole ``EvaluatorRequest``."""
    action = BatchEvaluator(
        name,
        fn,
        display_name=display_name,
        definition=definition,
        is_billed=is_billed,
        config_schema=config_schema,
        metadata=metadata,
        description=description,
    )
    registry.register_action_from_instance(action)
    return action
