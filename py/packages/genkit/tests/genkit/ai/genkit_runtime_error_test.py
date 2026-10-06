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

"""What a caller reads off ``response.error``, and how a blocked reply looks."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from genkit import FinishReason, Genkit, GenkitRuntimeError, Message, ModelResponse, Part, Role, RuntimeErrorReason
from genkit._core._typing import AgentFinishReason, ToolRequest
from genkit.exp import Genkit as ExpGenkit
from genkit.testing import define_scripted_model


class City(BaseModel):
    name: str


def _reply(text: str, finish_reason: FinishReason = FinishReason.STOP) -> ModelResponse:
    return ModelResponse(
        finish_reason=finish_reason,
        finish_message='safety' if finish_reason == FinishReason.BLOCKED else None,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
    )


@pytest.mark.asyncio
async def test_generate_schema_miss_error_is_genkit_runtime_error_with_invalid_output_reason() -> None:
    """A prose reply to ``output_schema=City`` returns a GenkitRuntimeError with reason INVALID_OUTPUT."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('Paris is lovely')]

    res = await ai.generate(prompt='extract', output_schema=City)

    assert isinstance(res.error, GenkitRuntimeError)
    assert res.error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert res.output is None


@pytest.mark.asyncio
async def test_generate_failed_tool_error_has_status_message_details_and_reason() -> None:
    """A failing tool returns an error with status INTERNAL, the finish message, details, and reason TOOL_FAILED."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        raise RuntimeError('db down')

    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='lookup', input={}, ref='r1'))],
            ),
        )
    ]

    res = await ai.generate(prompt='hi', tools=['lookup'])

    assert isinstance(res.error, GenkitRuntimeError)
    assert res.error.status == 'INTERNAL'
    assert res.error.message == res.finish_message
    assert isinstance(res.error.details, dict)
    assert res.error.details['reason'] == 'TOOL_FAILED'
    assert res.error.reason is RuntimeErrorReason.TOOL_FAILED


@pytest.mark.asyncio
async def test_genkit_runtime_error_is_not_an_exception() -> None:
    """``res.error`` is not a BaseException instance."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('Paris is lovely')]

    res = await ai.generate(prompt='extract', output_schema=City)

    assert res.error is not None
    assert not isinstance(res.error, BaseException)


def test_genkit_runtime_error_is_read_only() -> None:
    """Assigning ``res.error.message = 'x'`` raises a validation error."""
    res = ModelResponse(error=GenkitRuntimeError(status='INTERNAL', message='bad'))
    assert res.error is not None

    with pytest.raises(ValidationError):
        res.error.message = 'x'

    assert res.error.message == 'bad'


def test_genkit_runtime_error_with_unknown_reason_reads_none() -> None:
    """``details={'reason': 'NOT_A_REASON'}`` gives ``.reason is None``."""
    error = GenkitRuntimeError(status='INTERNAL', message='bad', details={'reason': 'NOT_A_REASON'})

    assert error.reason is None
    assert error.details == {'reason': 'NOT_A_REASON'}


def test_model_response_from_wire_dict_builds_genkit_runtime_error() -> None:
    """``ModelResponse.model_validate({'error': {...}})`` gives a GenkitRuntimeError."""
    res = ModelResponse.model_validate({
        'error': {'status': 'INTERNAL', 'message': 'x', 'details': {'reason': 'TOOL_FAILED'}},
    })

    assert isinstance(res.error, GenkitRuntimeError)
    assert res.error.message == 'x'
    assert res.error.reason is RuntimeErrorReason.TOOL_FAILED


def test_model_response_json_schema_names_genkit_runtime_error() -> None:
    """``ModelResponse.model_json_schema()`` defines the error as ``GenkitRuntimeError``."""
    schema = ModelResponse.model_json_schema()

    assert 'GenkitRuntimeError' in schema['$defs']
    assert schema['$defs']['GenkitRuntimeError']['title'] == 'GenkitRuntimeError'
    assert schema['properties']['error']['anyOf'][0] == {'$ref': '#/$defs/GenkitRuntimeError'}


def test_snapshot_and_agent_output_decode_the_same_error_type() -> None:
    """A persisted turn and a live response expose the same ``.reason``."""
    from genkit._core._model import AgentOutput, SessionSnapshot

    wire = {'status': 'ABORTED', 'message': 'stopped', 'details': {'reason': 'MAX_TURNS_EXCEEDED'}}
    snapshot = SessionSnapshot.model_validate({'snapshotId': 's1', 'createdAt': '2026-10-06T00:00:00Z', 'error': wire})
    output = AgentOutput.model_validate({'error': wire})

    assert isinstance(snapshot.error, GenkitRuntimeError)
    assert isinstance(output.error, GenkitRuntimeError)
    assert snapshot.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED
    assert output.error.reason is RuntimeErrorReason.MAX_TURNS_EXCEEDED


def test_snapshot_accepts_generated_wire_error() -> None:
    """A store holding the generated ``_typing`` class still builds a snapshot with ``.reason``."""
    from genkit._core._model import SessionSnapshot
    from genkit._core._typing import GenkitRuntimeError as WireError

    wire = WireError(status='NOT_FOUND', message='gone', details={'reason': 'TOOL_NOT_FOUND'})
    snapshot = SessionSnapshot.model_validate({'snapshotId': 's1', 'createdAt': '2026-10-06T00:00:00Z', 'error': wire})

    assert isinstance(snapshot.error, GenkitRuntimeError)
    assert snapshot.error.reason is RuntimeErrorReason.TOOL_NOT_FOUND


def test_model_response_has_no_assert_valid() -> None:
    """``ModelResponse`` has no ``assert_valid`` attribute."""
    with pytest.raises(AttributeError):
        ModelResponse().assert_valid()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_agent_response_has_no_assert_valid() -> None:
    """``AgentResponse`` has no ``assert_valid`` attribute."""
    ai = ExpGenkit()
    pm, _ = define_scripted_model(ai)
    ai.define_prompt(name='helper', model='scriptedModel')
    agent = ai.define_prompt_agent(name='helper')
    pm.responses = [_reply('hello')]

    res = await agent.chat().send('hi')

    with pytest.raises(AttributeError):
        res.assert_valid()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_blocked_generate_returns_response_with_blocked_finish_reason() -> None:
    """A blocked reply returns with ``finish_reason == 'blocked'`` and no exception."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_reply('nope', FinishReason.BLOCKED)]

    res = await ai.generate(prompt='hi')

    assert res.finish_reason == 'blocked'
    assert res.finish_message == 'safety'


@pytest.mark.asyncio
async def test_blocked_agent_turn_returns_response_with_blocked_finish_reason() -> None:
    """A blocked agent turn returns with ``finish_reason == 'blocked'`` and no exception."""
    ai = ExpGenkit()
    pm, _ = define_scripted_model(ai)
    ai.define_prompt(name='helper', model='scriptedModel')
    agent = ai.define_prompt_agent(name='helper')
    pm.responses = [_reply('nope', FinishReason.BLOCKED)]

    res = await agent.chat().send('hi')

    assert res.finish_reason == AgentFinishReason.BLOCKED
    assert res.finish_reason == 'blocked'
