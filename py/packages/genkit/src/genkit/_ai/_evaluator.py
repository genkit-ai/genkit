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
from typing import Any, ClassVar, TypeVar, cast, get_origin

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from genkit._core._action import (
    Action,
    ActionKind,
    ActionRunContext,
    known_annotation_names,
    resolve_type_hints,
    signature_of,
    with_request_annotation,
)
from genkit._core._logger import get_logger
from genkit._core._model import EvalRequest
from genkit._core._registry import Registry
from genkit._core._schema import to_json_schema
from genkit._core._telemetry._attrs import metadata_key
from genkit._core._telemetry._instrumentation import SpanContext, run_in_new_span
from genkit._core._typing import (
    ActionMetadata,
    BaseDataPoint,
    EvalFnResponse,
    EvalResponse,
    EvalStatusEnum,
    Score,
)

logger = get_logger(__name__)

EVALUATOR_METADATA_KEY_DISPLAY_NAME = 'evaluatorDisplayName'
EVALUATOR_METADATA_KEY_DEFINITION = 'evaluatorDefinition'
EVALUATOR_METADATA_KEY_IS_BILLED = 'evaluatorIsBilled'

T = TypeVar('T')

# User-provided evaluator function that evaluates a single datapoint.
# Must be async (coroutine function).
EvaluatorFn = Callable[[BaseDataPoint, T], Coroutine[Any, Any, EvalFnResponse]]

# User-provided batch evaluator: one EvalRequest. Returns the rows as a list
# or as an EvalResponse.
BatchEvaluatorFn = Callable[[EvalRequest[Any]], Coroutine[Any, Any, list[EvalFnResponse] | EvalResponse]]


class EvaluatorRef(BaseModel):
    """Reference to an evaluator."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra='forbid', populate_by_name=True, alias_generator=to_camel)

    name: str
    config: dict[str, object] | None = None


def evaluator_ref(name: str, *, config: dict[str, object] | None = None) -> EvaluatorRef:
    """Create an EvaluatorRef whose config is merged under ai.evaluate's config=.

    Settings are named. A value in the second position is a TypeError so it
    cannot be stored as config.
    """
    return EvaluatorRef(name=name, config=config)


def _options_class(fn: Callable[..., Any]) -> type[BaseModel] | None:
    """The class on a per-row evaluator's second parameter, e.g. ``options: JudgeConfig``.

    None for ``dict``, ``Any``, a union, or no annotation: the fn takes a dict.
    """
    params = [
        p
        for p in signature_of(fn).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(params) < 2:
        return None
    annotation = resolve_type_hints(fn, known_annotation_names(ActionKind.EVALUATOR)).get(params[1].name)
    if isinstance(annotation, type) and get_origin(annotation) is None and issubclass(annotation, BaseModel):
        return annotation
    return None


def _set_custom_options(evaluator_meta: dict[str, object], action: Action) -> None:
    """Annotation only: the Dev UI form comes from the options class."""
    info = evaluator_meta.get('evaluator')
    if isinstance(info, dict) and 'customOptions' not in info and action.config_schema is not None:
        cast(dict[str, object], info)['customOptions'] = to_json_schema(action.config_schema)


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


def evaluator_action_metadata(
    name: str,
    *,
    display_name: str,
    definition: str,
    is_billed: bool = False,
    config_schema: type[BaseModel] | dict[str, object] | None = None,
) -> ActionMetadata:
    """Describe an evaluator in a plugin's list_actions.

    The metadata matches what define_evaluator registers, so the Dev UI shows
    the same display name and definition before and after the action resolves.
    """
    return ActionMetadata(
        action_type=ActionKind.EVALUATOR,
        name=name,
        input_json_schema=to_json_schema(EvalRequest),
        output_json_schema=to_json_schema(list[EvalFnResponse]),
        metadata=_evaluator_metadata(name, display_name, definition, is_billed, config_schema),
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
    """Register an evaluator that runs the callback on each dataset sample.

    ``fn(datapoint, options)``. Annotate ``options`` with a BaseModel class
    to get it validated, with defaults; otherwise it is a dict. Given both
    that annotation and ``config_schema``, they must be the same class.
    """
    evaluator_meta = _evaluator_metadata(name, display_name, definition, is_billed, config_schema, metadata)

    evaluator_description = _get_func_description(fn, description)

    async def eval_stepper_fn(req: EvalRequest) -> EvalResponse:
        eval_responses: list[EvalFnResponse] = []
        for index in range(len(req.dataset)):
            datapoint = req.dataset[index]
            case_id = datapoint.test_case_id or str(uuid.uuid4())
            datapoint.test_case_id = case_id
            try:

                async def body(
                    span: SpanContext, point: BaseDataPoint = datapoint, test_case_id: str = case_id
                ) -> EvalFnResponse:
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
                            status=EvalStatusEnum.FAIL,
                        )
                        eval_responses.append(
                            EvalFnResponse(
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

    # The stepper is the action, so it carries the per-row fn's options class.
    options_cls = _options_class(fn)
    if options_cls is not None:
        eval_stepper_fn.__annotations__['req'] = cast(Any, EvalRequest)[options_cls]

    action = registry.register_action(
        name=name,
        kind=ActionKind.EVALUATOR,
        fn=eval_stepper_fn,
        metadata=evaluator_meta,
        description=evaluator_description,
        config_schema=config_schema,
    )
    _set_custom_options(evaluator_meta, action)
    return action


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
) -> Action:
    """Register a batch evaluator that runs ``fn`` once on the whole ``EvalRequest``.

    ``fn`` returns the rows as a list or an ``EvalResponse``. The action wraps
    them so ``action.run(...).response`` is always an ``EvalResponse``.
    """
    evaluator_meta = _evaluator_metadata(name, display_name, definition, is_billed, config_schema, metadata)

    evaluator_description = _get_func_description(fn, description)

    if not inspect.iscoroutinefunction(fn):
        raise TypeError(f"Action handlers must be async functions. Got sync function for '{name}'.")

    # the action hands back the rows as one model so the Dev UI and
    # `genkit eval:run` get a JSON array, the same as a per-row evaluator.
    # fn stays the metadata_fn, so its signature is still checked when defined.
    # model_validate takes a list or an EvalResponse; the constructor rejects the latter.
    async def batch_fn(req: EvalRequest, ctx: ActionRunContext) -> EvalResponse:
        return EvalResponse.model_validate(await action.params.call(fn, req, ctx))

    action = registry.register_action(
        name=name,
        kind=ActionKind.EVALUATOR,
        fn=batch_fn,
        # An unannotated fn still gets an EvalRequest, not the raw JSON dict.
        metadata_fn=with_request_annotation(fn, EvalRequest, kind=ActionKind.EVALUATOR),
        metadata=evaluator_meta,
        description=evaluator_description,
        config_schema=config_schema,
    )
    _set_custom_options(evaluator_meta, action)
    return action
