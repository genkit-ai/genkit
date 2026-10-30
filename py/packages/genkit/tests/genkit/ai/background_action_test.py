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

"""BackgroundAction is the generic core; BackgroundModel subclasses it."""

from typing import Any, cast

import pytest
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse, Operation, Part
from genkit._ai._background_model import lookup_background_model
from genkit._core._action import Action, ActionRunContext
from genkit._core._background import operation_actions
from genkit._core._error import GenkitError
from genkit._core._schema import to_json_schema
from genkit._core._typing import Role
from genkit.model import BackgroundModel, ModelContext, ModelInfo, ModelRequest
from genkit.plugin_api import ActionKind, BackgroundAction


class RenderRequest(BaseModel):
    """Which quarterly report to render."""

    quarter: str


class Report(BaseModel):
    """A rendered report."""

    url: str
    pages: int


class ReportJob(BackgroundAction[RenderRequest, Report]):
    """A non-model background job: render a PDF report."""


class TableConfig(BaseModel):
    """Booking options for a table."""

    party_size: int = 2


def _reply(text: str) -> ModelResponse:
    return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(text)]))


def _register_report_job(ai: Genkit) -> ReportJob:
    async def start(request: RenderRequest, ctx: ActionRunContext) -> Operation:
        return Operation(id=f'render-{request.quarter}', done=False)

    async def check(op: Operation[Report], ctx: ActionRunContext) -> Operation:
        return Operation(id=op.id, done=True, output={'url': f'https://reports.example/{op.id}.pdf', 'pages': 12})

    job = ReportJob(
        *operation_actions(ReportJob, 'render-report', kind=ActionKind.CUSTOM, start=start, check=check),
    )
    for each in job.actions:
        ai._registry.register_action_from_instance(each)
    return job


def test_type_arguments_set_input_and_output_types() -> None:
    """class X(BackgroundAction[In, Out]) is all it takes to type the handle."""
    assert ReportJob._input_type is RenderRequest
    assert ReportJob._output_type is Report
    assert BackgroundModel._input_type is ModelRequest
    assert BackgroundModel._output_type is ModelResponse
    assert BackgroundAction._output_type is None


def test_background_action_is_a_composite_not_an_action() -> None:
    """Plugins list it next to Action; it is not one."""
    assert not issubclass(BackgroundAction, Action)
    assert issubclass(BackgroundModel, BackgroundAction)


@pytest.mark.asyncio
async def test_non_model_job_round_trips_through_check_operation() -> None:
    """A custom-kind job's typed handle comes back with its own output type."""
    ai = Genkit()
    job = _register_report_job(ai)

    op = await job.start(RenderRequest(quarter='q3'))
    assert op.action == '/custom/render-report'

    done = await ai.check_operation(op)

    assert done.done is True
    assert isinstance(done.output, Report)
    assert done.output.pages == 12


@pytest.mark.asyncio
async def test_non_model_job_from_stored_dict_returns_validated_output() -> None:
    """A stored dict is checked against the job's output type, not ModelResponse."""
    ai = Genkit()
    _register_report_job(ai)
    saved = {'id': 'render-q4', 'action': '/custom/render-report', 'done': False}

    done = await ai.check_operation(saved)

    assert isinstance(done.output, Report)
    assert done.output.url == 'https://reports.example/render-q4.pdf'


@pytest.mark.asyncio
async def test_non_model_job_rejects_bad_stored_output() -> None:
    """The check action validates the caller's output against Report."""
    ai = Genkit()
    _register_report_job(ai)
    saved = {'id': 'render-q4', 'action': '/custom/render-report', 'output': {'url': 'x', 'pages': 'many'}}

    with pytest.raises(GenkitError) as exc:
        await ai.check_operation(saved)

    assert exc.value.status == 'INVALID_ARGUMENT'
    assert 'output.pages' in str(exc.value)


@pytest.mark.asyncio
async def test_non_model_job_cancel_unsupported_names_a_background_action() -> None:
    """No model wording on a job that isn't a model."""
    ai = Genkit()
    job = _register_report_job(ai)

    with pytest.raises(GenkitError) as exc:
        await job.cancel(Operation(id='render-q1', action='/custom/render-report'))

    assert exc.value.status == 'UNIMPLEMENTED'
    assert exc.value.original_message == 'Background action /custom/render-report does not support cancellation.'
    assert exc.value.reason is None


def test_check_and_cancel_carry_start_metadata() -> None:
    """Like JS/Go: check and cancel spread the start metadata."""
    job = ReportJob(
        *operation_actions(
            ReportJob,
            'render-report',
            kind=ActionKind.CUSTOM,
            start=_noop_start,
            check=_noop_check,
            cancel=_noop_check,
            metadata={'team': 'finance'},
        )
    )
    for each in job.actions:
        assert each.metadata['team'] == 'finance'


async def _noop_start(request: RenderRequest, ctx: ActionRunContext) -> Operation:
    return Operation(id='noop')


async def _noop_check(op: Operation, ctx: ActionRunContext) -> Operation:
    return op


def _define_bistro_video(ai: Genkit) -> BackgroundModel:
    async def start(request: ModelRequest, ctx: ModelContext) -> Operation:
        return Operation(id='video-1', done=False)

    async def check(op: Operation, ctx: ModelContext) -> Operation:
        return Operation(id=op.id, done=True, output=_reply('plated'))

    async def cancel(op: Operation, ctx: ModelContext) -> Operation:
        return Operation(id=op.id, done=True)

    return ai.define_background_model(
        name='bistro-video',
        start=start,
        check=check,
        cancel=cancel,
        info=ModelInfo(label='Bistro Video'),
        config_schema=TableConfig,
        metadata={'team': 'kitchen'},
    )


@pytest.mark.asyncio
async def test_looked_up_background_model_matches_the_defined_one() -> None:
    """ai.check_operation rebuilds the model from the registry; nothing is lost."""
    ai = Genkit()
    defined = _define_bistro_video(ai)

    looked_up = await lookup_background_model(ai._registry, '/background-model/bistro-video')

    assert looked_up is not None
    assert type(looked_up) is BackgroundModel
    assert type(looked_up)._output_type is ModelResponse
    for prop in ('name', 'kind', 'metadata', 'input_schema', 'output_schema', 'supports_cancel', 'label'):
        assert getattr(looked_up, prop) == getattr(defined, prop), prop
    assert looked_up.config_schema is defined.config_schema is TableConfig
    assert looked_up.label == 'Bistro Video'


@pytest.mark.asyncio
async def test_looked_up_background_model_rejects_bad_output() -> None:
    """A rebuilt model still validates the handle's output as a ModelResponse."""
    ai = Genkit()
    _define_bistro_video(ai)
    looked_up = await lookup_background_model(ai._registry, '/background-model/bistro-video')
    assert looked_up is not None

    with pytest.raises(GenkitError) as exc:
        await looked_up.check({'id': 'video-1', 'action': '/background-model/bistro-video', 'output': {'bogus': 1}})

    assert exc.value.status == 'INVALID_ARGUMENT'


def test_annotation_alone_sets_custom_options_on_all_three_actions() -> None:
    """ModelRequest[Cfg] without config_schema still fills the Dev UI form."""
    ai = Genkit()

    async def start(request: ModelRequest[TableConfig], ctx: ModelContext) -> Operation:
        return Operation(id='video-1')

    async def check(op: Operation, ctx: ModelContext) -> Operation:
        return op

    model = ai.define_background_model(name='bistro-video', start=start, check=check)

    expected = to_json_schema(TableConfig)
    assert model.metadata['model']['customOptions'] == expected
    assert cast(dict[str, Any], model.check_action.metadata['model'])['customOptions'] == expected
    assert model.config_schema is TableConfig


@pytest.mark.asyncio
async def test_check_operation_typed_model_handle_round_trips() -> None:
    """Operation[ModelResponse] in, Operation[ModelResponse] out."""
    ai = Genkit()
    _define_bistro_video(ai)
    op = await ai.generate_operation(model='bistro-video', prompt='A chef plating risotto.')

    done = await ai.check_operation(op)

    assert isinstance(done.output, ModelResponse)
    assert done.output.text == 'plated'


@pytest.mark.asyncio
async def test_check_operation_model_dict_returns_model_response() -> None:
    """A stored dict for a model handle comes back with a ModelResponse output."""
    ai = Genkit()
    _define_bistro_video(ai)

    done = await ai.check_operation({'id': 'video-1', 'action': '/background-model/bistro-video'})

    assert isinstance(done.output, ModelResponse)


@pytest.mark.asyncio
async def test_bad_model_output_fails_before_any_action_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """INVALID_ARGUMENT, and no action (so no span) ever starts."""
    ai = Genkit()
    _define_bistro_video(ai)
    runs: list[str] = []
    original_run = Action.run

    async def counting_run(self: Action, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        runs.append(self.name)
        return await original_run(self, *args, **kwargs)

    monkeypatch.setattr(Action, 'run', counting_run)

    with pytest.raises(GenkitError) as exc:
        await ai.check_operation({
            'id': 'video-1',
            'action': '/background-model/bistro-video',
            'output': {'bogus': 1},
        })

    assert exc.value.status == 'INVALID_ARGUMENT'
    assert exc.value.original_message.startswith('invalid Operation: output')
    assert runs == []


@pytest.mark.asyncio
async def test_model_cancel_unsupported_keeps_model_wording() -> None:
    """Model handles keep the UNSUPPORTED_BY_MODEL reason and message."""
    ai = Genkit()

    async def start(request: ModelRequest, ctx: ModelContext) -> Operation:
        return Operation(id='video-1')

    async def check(op: Operation, ctx: ModelContext) -> Operation:
        return op

    ai.define_background_model(name='bistro-video', start=start, check=check)

    with pytest.raises(GenkitError) as exc:
        await ai.cancel_operation({'id': 'video-1', 'action': '/background-model/bistro-video'})

    assert exc.value.status == 'UNIMPLEMENTED'
    expected = 'Background model /background-model/bistro-video does not support cancellation.'
    assert exc.value.original_message == expected
    assert exc.value.reason == 'UNSUPPORTED_BY_MODEL'
