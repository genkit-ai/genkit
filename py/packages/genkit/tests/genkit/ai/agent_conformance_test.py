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

"""Agent conformance test runner.

Reads the shared spec from tests/specs/agent.yaml and executes each test case
against harness-provided agent implementations. See
docs/agents-conformance-testing.md for the full spec format reference and
harness requirements.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeVar

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.alias_generators import to_camel

from genkit._ai._agents._runtime import SessionRunner
from genkit._ai._agents._session_stores._inmemory_store import InMemorySessionStore
from genkit._ai._agents._types import TurnContext, TurnResult
from genkit._ai._aio import Genkit
from genkit._ai._testing import ProgrammableModel, define_programmable_model
from genkit._ai._tools import Interrupt, ToolRunContext
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import ModelResponse, ModelResponseChunk
from genkit._core._typing import (
    AgentFinishReason,
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentResult,
    Artifact,
    GenkitRuntimeError,
    MessageData,
    Part,
    Role,
    SessionSnapshot as SessionSnapshotSchema,
    TextPart,
)
from genkit.agent import Agent

TERMINAL_STATUSES = {'completed', 'failed', 'aborted'}
DEFAULT_STEP_TIMEOUT_S = 5.0


def spec_path() -> pathlib.Path:
    """Walk up from this file to the repo's tests/specs/agent.yaml."""
    for parent in pathlib.Path(__file__).resolve().parents:
        candidate = parent / 'tests' / 'specs' / 'agent.yaml'
        if candidate.is_file():
            return candidate
    raise AssertionError('tests/specs/agent.yaml not found from agent_conformance_test.py')


# ---------------------------------------------------------------------------
# Spec models (discriminated on step.type)
# ---------------------------------------------------------------------------


class SpecModel(BaseModel):
    # Forbid unknown keys so a typo in agent.yaml fails at load, not as a skip.
    model_config = ConfigDict(extra='forbid', populate_by_name=True, alias_generator=to_camel)


class SpecInit(SpecModel):
    """Send init. ``state`` stays untyped — it can be an object or ``{{state1}}``."""

    session_id: str | None = None
    snapshot_id: str | None = None
    state: Any | None = None


class SpecInput(SpecModel):
    detach: bool | None = None
    message: dict[str, Any] | None = None
    resume: dict[str, Any] | None = None


class TurnEndExpect(SpecModel):
    finish_reason: str | None = None
    snapshot_id: str | None = None


class ExpectChunk(SpecModel):
    turn_end: TurnEndExpect | None = None
    model_chunk: dict[str, Any] | None = None
    artifact: dict[str, Any] | None = None
    custom_patch: Any | None = None

    @model_validator(mode='after')
    def exactly_one_payload(self) -> ExpectChunk:
        kinds = [n for n in ('turn_end', 'model_chunk', 'artifact', 'custom_patch') if n in self.model_fields_set]
        if len(kinds) != 1:
            raise ValueError('expectChunks item must have exactly one of turnEnd, modelChunk, artifact, customPatch')
        return self


class PartialState(SpecModel):
    """Subset match on session state. Extra keys are allowed so the spec can grow."""

    model_config = ConfigDict(extra='allow', populate_by_name=True, alias_generator=to_camel)
    session_id: str | None = None
    messages: list[Any] | None = None
    custom: Any | None = None
    artifacts: list[Any] | None = None


class OutputAssertions(SpecModel):
    message: dict[str, Any] | None = None
    has_snapshot_id: bool | None = None
    has_session_id: bool | None = None
    state_contains: PartialState | None = None
    artifacts_contain: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    error_contains: dict[str, Any] | None = None


class SnapshotAssertions(SpecModel):
    parent_id: str | None = None
    status: str | None = None
    finish_reason: str | None = None
    has_session_id: bool | None = None
    state_contains: PartialState | None = None
    error_contains: dict[str, Any] | None = None


class SendExpectError(SpecModel):
    status: str | None = None
    message: str | None = None


class SendStep(SpecModel):
    type: Literal['send']
    init: SpecInit | None = None
    inputs: list[SpecInput] | None = None
    model_responses: list[dict[str, Any]] | None = None
    stream_chunks: list[list[dict[str, Any]]] | None = None
    expect_chunks: list[ExpectChunk] | None = None
    expect_output: OutputAssertions | None = None
    expect_error: SendExpectError | None = None
    capture_snapshot_id: str | None = None
    capture_state: str | None = None
    capture_session_id: str | None = None


class GetSnapshotDataStep(SpecModel):
    type: Literal['getSnapshotData']
    snapshot_id: str | None = None
    session_id: str | None = None
    expect_snapshot: SnapshotAssertions | None = None
    expect_error: str | None = None


class AbortStep(SpecModel):
    type: Literal['abort']
    snapshot_id: str
    expect_previous_status: str | None = None


class WaitUntilCompletedStep(SpecModel):
    type: Literal['waitUntilCompleted']
    snapshot_id: str
    timeout_ms: float | None = None
    expect_snapshot: SnapshotAssertions | None = None


SpecStep = Annotated[
    SendStep | GetSnapshotDataStep | AbortStep | WaitUntilCompletedStep,
    Field(discriminator='type'),
]


class SpecTest(SpecModel):
    name: str
    description: str | None = None
    agent: str
    # Capabilities the test depends on. A test naming a capability outside
    # SUPPORTED_REQUIRES is skipped, so the shared spec can carry cases for
    # features this runtime has not adopted yet; one outside the suite's own
    # capabilities list is a typo and fails.
    requires: list[str] | None = None
    steps: list[SpecStep]


class SpecSuite(SpecModel):
    # Every name a test may put in `requires`. It is the spec's own registry,
    # so a misspelled capability fails here rather than skipping the test in
    # every SDK at once.
    capabilities: list[str] = []
    tests: list[SpecTest]


def load_spec() -> SpecSuite:
    path = spec_path()
    with path.open() as f:
        raw = yaml.safe_load(f)
    try:
        suite = SpecSuite.model_validate({} if raw is None else raw)
    except ValidationError as e:
        raise AssertionError(f'agent.yaml: {e}') from e
    if not suite.tests:
        raise AssertionError('agent.yaml contains no tests')
    return suite


SPEC_SUITE = load_spec()
SPEC_TESTS = SPEC_SUITE.tests
KNOWN_REQUIRES: frozenset[str] = frozenset(SPEC_SUITE.capabilities)


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------

_FULL_TEMPLATE = re.compile(r'^\{\{(\w+)\}\}$')
_INLINE_TEMPLATE = re.compile(r'\{\{(\w+)\}\}')


def resolve_templates(*, value: Any, captures: dict[str, Any]) -> Any:
    """Recursively resolve ``{{name}}`` references using the captures map."""
    if isinstance(value, str):
        m = _FULL_TEMPLATE.match(value)
        if m:
            name = m.group(1)
            if name not in captures:
                raise AssertionError(f"Template reference '{{{{{name}}}}}' not found in captures")
            return captures[name]

        def sub(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in captures:
                raise AssertionError(f"Template reference '{{{{{name}}}}}' not found in captures")
            v = captures[name]
            return v if isinstance(v, str) else json.dumps(v, separators=(',', ':'))

        return _INLINE_TEMPLATE.sub(sub, value)
    if isinstance(value, list):
        return [resolve_templates(value=item, captures=captures) for item in value]
    if isinstance(value, dict):
        return {k: resolve_templates(value=v, captures=captures) for k, v in value.items()}
    return value


def resolve_step(*, step: SpecStep, captures: dict[str, Any]) -> SpecStep:
    """Apply ``{{name}}`` captures, keeping explicit YAML nulls (including ``~``)."""
    raw = step.model_dump(by_alias=True, exclude_unset=True)
    resolved = resolve_templates(value=raw, captures=captures)
    try:
        return type(step).model_validate(resolved)
    except ValidationError as e:
        raise AssertionError(f'spec step after template resolve: {e}') from e


# ---------------------------------------------------------------------------
# "Contains" assertion helpers
# ---------------------------------------------------------------------------


def assert_contains(*, actual: Any, expected: Any, path: str = '') -> None:
    """Assert that ``actual`` contains all fields specified in ``expected``.

    Dicts are matched key-by-key (extra keys in actual are allowed). Lists are
    matched as an in-order (not necessarily contiguous) subsequence. Scalars
    must match exactly.
    """
    if expected is None:
        # A missing/null expected value is "not specified", not "must be null".
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), f'Expected list at {path}, got {type(actual).__name__}: {actual!r}'
        assert_contains_subsequence(actual=actual, expected=expected, path=path)
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f'Expected dict at {path}, got {type(actual).__name__}: {actual!r}'
        for key, val in expected.items():
            assert_contains(actual=actual.get(key), expected=val, path=f'{path}.{key}')
        return

    assert actual == expected, f'Mismatch at {path}: expected {expected!r}, got {actual!r}'


def assert_contains_subsequence(*, actual: list[Any], expected: list[Any], path: str) -> None:
    """Assert all ``expected`` items appear in ``actual`` in the same relative order."""
    actual_idx = 0
    for i, exp_item in enumerate(expected):
        found = False
        while actual_idx < len(actual):
            try:
                assert_contains(actual=actual[actual_idx], expected=exp_item, path=f'{path}[{actual_idx}]')
                found = True
                actual_idx += 1
                break
            except AssertionError:
                actual_idx += 1
        if not found:
            raise AssertionError(
                f'Expected item at {path}[{i}] not found in actual array.\n'
                f'  Expected: {exp_item!r}\n'
                f'  Actual array: {actual!r}'
            )


def dump(*, model: BaseModel) -> dict[str, Any]:
    """Serialize a wire model to its camelCase JSON form for spec comparison.

    Unset fields stay off the object. A field that was set to null stays null,
    so a spec can tell "not present" from an explicit ``~``. Wire models
    default to dropping nulls, so this has to opt out.
    """
    return model.model_dump(by_alias=True, exclude_unset=True, exclude_none=False, mode='json')


WireT = TypeVar('WireT', bound=BaseModel)


def validate_wire(*, model_type: type[WireT], data: Any, path: str) -> WireT:  # noqa: ANN401
    """Build a runtime wire object; name the YAML path if the shape is wrong."""
    try:
        return model_type.model_validate(data)
    except ValidationError as e:
        raise AssertionError(f'{path}: {e}') from e


def field_was_set(*, model: BaseModel, name: str) -> bool:
    return name in model.model_fields_set


def wire_scalar(*, live: Any) -> Any:  # noqa: ANN401
    if live is None:
        return None
    value = getattr(live, 'value', None)
    return value if isinstance(value, str) else live


def assert_pinned_scalar(*, live: Any, expect_model: BaseModel, name: str, path: str) -> None:  # noqa: ANN401
    """If the spec pinned this field (including YAML ~), compare the live value."""
    if not field_was_set(model=expect_model, name=name):
        return
    want = getattr(expect_model, name)
    got = wire_scalar(live=live)
    assert got == want, f'{path}: expected {want!r}, got {got!r}'


def assert_has_id(*, actual: Any, want: bool, path: str) -> None:  # noqa: ANN401
    present = isinstance(actual, str) and bool(actual)
    if want:
        assert present, f'Expected {path} to be a non-empty string, got: {actual!r}'
    else:
        assert not present, f'Expected {path} to be absent, got: {actual!r}'


# ---------------------------------------------------------------------------
# Harness setup
# ---------------------------------------------------------------------------


def _model_text(*, text: str) -> MessageData:
    return MessageData(role=Role.MODEL, content=[Part(root=TextPart(text=text))])


class InterruptQuery(BaseModel):
    query: str


class RestartInput(BaseModel):
    action: str


class RestartOutput(BaseModel):
    result: str


@dataclass(frozen=True)
class PromptAgentDef:
    name: str
    tools: tuple[str, ...] = ()
    store: bool = False


PROMPT_AGENTS = (
    PromptAgentDef(name='promptAgent'),
    PromptAgentDef(name='promptAgentWithStore', store=True),
    PromptAgentDef(name='promptAgentWithTools', tools=('testTool',)),
    PromptAgentDef(name='promptAgentWithInterrupt', tools=('interruptTool',), store=True),
    PromptAgentDef(name='promptAgentWithRestartTool', tools=('restartTool',), store=True),
)

TurnBody = Callable[[SessionRunner, ActionRunContext, AgentInput, TurnContext], Awaitable[None]]


@dataclass
class Harness:
    ai: Genkit
    pm: ProgrammableModel
    agents: dict[str, Agent] = field(default_factory=dict)


def setup_harness() -> Harness:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    h = Harness(ai=ai, pm=pm)
    _register_tools(ai=ai)
    _register_prompt_agents(ai=ai, agents=h.agents)
    _register_custom_agents(ai=ai, agents=h.agents)
    return h


def _register_tools(*, ai: Genkit) -> None:
    @ai.tool(name='testTool', description='A simple test tool')
    async def test_tool(_: dict) -> str:  # noqa: ARG001
        return 'tool called'

    # interruptTool always interrupts (human-in-the-loop checkpoint).
    ai.define_interrupt(
        name='interruptTool',
        description='An interrupt tool',
        input_schema=InterruptQuery,
    )

    # restartTool interrupts on first call, succeeds when restarted with
    # resumed metadata.
    @ai.tool(name='restartTool', description='A tool that requires confirmation before executing')
    async def restart_tool(input: RestartInput, ctx: ToolRunContext) -> RestartOutput:  # noqa: A002
        if not ctx.is_resumed():
            raise Interrupt({'requiresConfirmation': True})
        return RestartOutput(result=f'confirmed: {input.action}')


def _register_prompt_agents(*, ai: Genkit, agents: dict[str, Agent]) -> None:
    for spec in PROMPT_AGENTS:
        agents[spec.name] = ai.define_agent(
            name=spec.name,
            model='programmableModel',
            config={'temperature': 1},
            tools=list(spec.tools) if spec.tools else None,
            store=InMemorySessionStore() if spec.store else None,
        )


def _run_turns(*, turn_body: TurnBody) -> Callable[[SessionRunner, ActionRunContext], Awaitable[AgentResult]]:
    """Wrap a per-turn body into the canonical custom AgentFn shape."""

    async def agent_fn(session_runner: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput, turn_ctx: TurnContext) -> TurnResult | None:
            await turn_body(session_runner, ctx, inp, turn_ctx)
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    return agent_fn


async def _blocking_turn(sr: SessionRunner, ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext) -> None:
    await ctx.abort_signal.wait()
    await sr.add_messages([_model_text(text='unblocked')])


async def _artifacts_turn(sr: SessionRunner, _ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext) -> None:
    await sr.add_artifacts([Artifact(name='doc1', parts=[Part(root=TextPart(text='v1'))])])
    await sr.add_artifacts([Artifact(name='doc1', parts=[Part(root=TextPart(text='v2'))])])
    await sr.add_artifacts([Artifact(name='doc2', parts=[Part(root=TextPart(text='other'))])])
    await sr.add_messages([_model_text(text='done')])


async def _counter_turn(sr: SessionRunner, _ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext) -> None:
    prev = await sr.get_custom() or {}
    counter = (prev.get('counter') or 0) + 1
    await sr.update_custom(lambda _prev: {'counter': counter})
    await sr.add_messages([_model_text(text='done')])


async def _multi_custom_turn(sr: SessionRunner, _ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext) -> None:
    # First patch is a whole-doc replace; later ones are incremental diffs.
    await sr.update_custom(lambda _prev: {'counter': 1, 'status': 'working'})
    await sr.update_custom(lambda prev: {**(prev or {}), 'counter': 2})
    await sr.update_custom(lambda prev: {**(prev or {}), 'status': 'done'})
    await sr.add_messages([_model_text(text='done')])


async def _artifacts_store_turn(sr: SessionRunner, _ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext) -> None:
    existing = await sr.get_artifacts()
    count = len(existing) + 1
    await sr.add_artifacts([Artifact(name=f'doc{count}', parts=[Part(root=TextPart(text=f'content{count}'))])])
    await sr.add_messages([_model_text(text='done')])


CUSTOM_TURN_AGENTS: tuple[tuple[str, TurnBody, bool], ...] = (
    ('customAgentBlocking', _blocking_turn, True),
    ('customAgentWithArtifacts', _artifacts_turn, False),
    ('customAgentWithCustomState', _counter_turn, False),
    ('customAgentWithMultiCustomState', _multi_custom_turn, False),
    ('customAgentWithArtifactsStore', _artifacts_store_turn, True),
    ('customAgentWithCustomStateStore', _counter_turn, True),
)


def _register_custom_agents(*, ai: Genkit, agents: dict[str, Agent]) -> None:
    for name, turn, store in CUSTOM_TURN_AGENTS:
        agents[name] = ai.define_custom_agent(
            name=name,
            fn=_run_turns(turn_body=turn),
            store=InMemorySessionStore() if store else None,
        )

    # Raise inside the turn so run() records last_turn_error, then return
    # result() so an attached caller gets a graceful failed output.
    async def failing_agent_fn(session_runner: SessionRunner, _ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(_inp: AgentInput, _tc: TurnContext) -> TurnResult | None:
            raise RuntimeError('intentional failure')

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agents['customAgentFailing'] = ai.define_custom_agent(
        name='customAgentFailing',
        fn=failing_agent_fn,
        store=InMemorySessionStore(),
    )


# ---------------------------------------------------------------------------
# Step executors
# ---------------------------------------------------------------------------


def program_model(*, pm: ProgrammableModel, step: SendStep) -> None:
    pm.reset()
    responses = step.model_responses or []
    pm.responses = [
        validate_wire(model_type=ModelResponse, data=r, path=f'modelResponses[{i}]') for i, r in enumerate(responses)
    ]
    if step.stream_chunks:
        pm.chunks = [
            [
                validate_wire(model_type=ModelResponseChunk, data=c, path=f'streamChunks[{i}][{j}]')
                for j, c in enumerate(group)
            ]
            for i, group in enumerate(step.stream_chunks)
        ]


def assert_chunks(*, actual_chunks: list[Any], expected_chunks: list[ExpectChunk]) -> None:
    """Strict ordered chunk comparison per the spec's expectChunks contract."""
    actual = [dump(model=c) for c in actual_chunks]
    expected_dump = [dump(model=c) for c in expected_chunks]
    assert len(actual) == len(expected_chunks), (
        f'Expected {len(expected_chunks)} chunks, got {len(actual)}.\n'
        f'  Actual: {actual!r}\n'
        f'  Expected: {expected_dump!r}'
    )
    for i, expected in enumerate(expected_chunks):
        got = actual[i]
        if field_was_set(model=expected, name='turn_end'):
            # turnEnd carries a dynamic snapshotId; only assert presence, plus
            # finishReason exactly when the spec pins it (key present, including YAML ~).
            assert 'turnEnd' in got, f'Chunk {i}: expected turnEnd, got {got!r}'
            turn_end = expected.turn_end
            if turn_end is not None and field_was_set(model=turn_end, name='finish_reason'):
                te_model = actual_chunks[i].turn_end
                want_fr = turn_end.finish_reason
                got_fr = wire_scalar(live=te_model.finish_reason if te_model is not None else None)
                assert got_fr == want_fr, f'Chunk {i}: expected turnEnd.finishReason {want_fr!r}, got {got_fr!r}'
        elif field_was_set(model=expected, name='model_chunk'):
            assert_contains(actual=got.get('modelChunk'), expected=expected.model_chunk, path=f'chunk[{i}].modelChunk')
        elif field_was_set(model=expected, name='artifact'):
            assert_contains(actual=got.get('artifact'), expected=expected.artifact, path=f'chunk[{i}].artifact')
        else:
            assert_contains(
                actual=got.get('customPatch'), expected=expected.custom_patch, path=f'chunk[{i}].customPatch'
            )


def assert_output(*, output: AgentOutput, expect: OutputAssertions) -> None:
    out = dump(model=output)
    if expect.message is not None:
        # Contains / subsequence — extra keys and extra parts are allowed, so a
        # spec that pins only content still matches a live message that has role.
        assert_contains(actual=out.get('message'), expected=expect.message, path='output.message')

    if field_was_set(model=expect, name='has_snapshot_id'):
        assert_has_id(actual=out.get('snapshotId'), want=bool(expect.has_snapshot_id), path='output.snapshotId')

    if field_was_set(model=expect, name='has_session_id'):
        state = out.get('state') or {}
        assert_has_id(actual=state.get('sessionId'), want=bool(expect.has_session_id), path='output.state.sessionId')

    if expect.state_contains is not None:
        assert out.get('state') is not None, 'Expected output to have state'
        assert_contains(actual=out['state'], expected=dump(model=expect.state_contains), path='output.state')

    if expect.artifacts_contain is not None:
        artifacts = out.get('artifacts')
        assert artifacts is not None, 'Expected output to have artifacts'
        for expected_art in expect.artifacts_contain:
            found = next((a for a in artifacts if a.get('name') == expected_art.get('name')), None)
            assert found is not None, f'Expected artifact {expected_art.get("name")!r} not found in output'
            assert_contains(actual=found, expected=expected_art, path=f'artifact({expected_art.get("name")})')

    assert_pinned_scalar(
        live=output.finish_reason,
        expect_model=expect,
        name='finish_reason',
        path='output.finishReason',
    )

    if expect.error_contains is not None:
        err = out.get('error')
        assert err, f'Expected output to have an error, got: {err!r}'
        want = expect.error_contains
        if 'status' in want:
            assert err.get('status') == want['status'], (
                f'Expected output.error.status {want["status"]!r}, got {err.get("status")!r}'
            )
        if 'message' in want:
            assert want['message'] in (err.get('message') or ''), (
                f'Expected output.error.message to contain {want["message"]!r}, got: {err.get("message")!r}'
            )


def assert_snapshot(*, snap: SessionSnapshotSchema, expect: SnapshotAssertions) -> None:
    dumped = dump(model=snap)
    assert_pinned_scalar(live=snap.parent_id, expect_model=expect, name='parent_id', path='snapshot.parentId')
    assert_pinned_scalar(
        live=snap.status.value if snap.status is not None else None,
        expect_model=expect,
        name='status',
        path='snapshot.status',
    )
    assert_pinned_scalar(
        live=snap.finish_reason,
        expect_model=expect,
        name='finish_reason',
        path='snapshot.finishReason',
    )
    if field_was_set(model=expect, name='has_session_id'):
        state = dumped.get('state') or {}
        assert_has_id(
            actual=state.get('sessionId'),
            want=bool(expect.has_session_id),
            path='snapshot.state.sessionId',
        )
    if expect.state_contains is not None:
        assert_contains(actual=dumped.get('state'), expected=dump(model=expect.state_contains), path='snapshot.state')
    if expect.error_contains is not None:
        err = dumped.get('error')
        assert err, 'Expected snapshot to have error'
        # Snapshot errorContains is subset matching (a string field is exact).
        # Output errorContains is different: status exact, message substring.
        assert_contains(actual=err, expected=expect.error_contains, path='snapshot.error')


async def _close_quietly(*, conn: Any) -> None:  # noqa: ANN401
    if conn is None:
        return
    with contextlib.suppress(Exception):
        await conn.close()


def _thrown_message(*, thrown: BaseException) -> str:
    """``STATUS: text`` for a GenkitError — the status prefix, not a cause suffix."""
    if isinstance(thrown, GenkitError):
        return f'{thrown.status}: {thrown.original_message}'
    return str(thrown)


def _assert_expect_error(*, thrown: BaseException | None, expect_err: SendExpectError) -> None:
    assert thrown is not None, 'Expected the turn to throw an error, but it resolved successfully.'
    if expect_err.status is not None:
        status = getattr(thrown, 'status', None)
        assert status == expect_err.status, (
            f'Expected thrown error.status {expect_err.status!r}, got {status!r} (message: {thrown})'
        )
    if expect_err.message is not None:
        thrown_msg = _thrown_message(thrown=thrown)
        assert expect_err.message in thrown_msg, (
            f'Expected thrown error.message to contain {expect_err.message!r}, got: {thrown_msg!r}'
        )


async def execute_send(*, agent: Agent, pm: ProgrammableModel, step: SendStep, captures: dict[str, Any]) -> None:
    resolved = resolve_step(step=step, captures=captures)
    assert isinstance(resolved, SendStep)
    program_model(pm=pm, step=resolved)

    async def run_turn() -> tuple[list[Any], AgentOutput]:
        conn = await agent.stream_bidi(
            validate_wire(
                model_type=AgentInit,
                data=dump(model=resolved.init) if resolved.init is not None else {},
                path='init',
            ),
        )
        try:
            for i, inp in enumerate(resolved.inputs or []):
                await conn.send(validate_wire(model_type=AgentInput, data=dump(model=inp), path=f'inputs[{i}]'))
            await conn.close()
            chunks = [c async for c in conn.receive()]
            output = await conn.output()
            return chunks, output
        finally:
            await _close_quietly(conn=conn)

    # expectError: the turn throws (API misuse) rather than resolving with a
    # graceful finishReason='failed' output. Cover stream_bidi / send as well
    # as receive / output — an init rejection can surface before the stream.
    if field_was_set(model=resolved, name='expect_error'):
        thrown: BaseException | None = None
        try:
            await asyncio.wait_for(run_turn(), timeout=DEFAULT_STEP_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise AssertionError(f'send step timed out after {DEFAULT_STEP_TIMEOUT_S}s') from None
        except Exception as e:  # noqa: BLE001 - spec asserts on the raised error
            thrown = e
        _assert_expect_error(thrown=thrown, expect_err=resolved.expect_error or SendExpectError())
        return

    try:
        chunks, output = await asyncio.wait_for(run_turn(), timeout=DEFAULT_STEP_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise AssertionError(f'send step timed out after {DEFAULT_STEP_TIMEOUT_S}s') from None
    out = dump(model=output)

    if resolved.expect_chunks is not None:
        assert_chunks(actual_chunks=chunks, expected_chunks=resolved.expect_chunks)

    if resolved.expect_output is not None:
        assert_output(output=output, expect=resolved.expect_output)

    # Capture names come from the unresolved step so they are never themselves
    # template-substituted.
    if step.capture_snapshot_id:
        assert out.get('snapshotId'), (
            f'captureSnapshotId {step.capture_snapshot_id!r} requested but output has no snapshotId'
        )
        captures[step.capture_snapshot_id] = out['snapshotId']
    if step.capture_state:
        assert out.get('state'), f'captureState {step.capture_state!r} requested but output has no state'
        captures[step.capture_state] = out['state']
    if step.capture_session_id:
        state = out.get('state') or {}
        assert state.get('sessionId'), (
            f'captureSessionId {step.capture_session_id!r} requested but output has no state.sessionId'
        )
        captures[step.capture_session_id] = state['sessionId']


async def execute_get_snapshot_data(*, agent: Agent, step: GetSnapshotDataStep, captures: dict[str, Any]) -> None:
    resolved = resolve_step(step=step, captures=captures)
    assert isinstance(resolved, GetSnapshotDataStep)
    snapshot_id = resolved.snapshot_id
    session_id = resolved.session_id
    assert bool(snapshot_id) != bool(session_id), 'getSnapshotData step requires exactly one of snapshotId or sessionId'

    if field_was_set(model=resolved, name='expect_error'):
        expect_err = resolved.expect_error or ''
        try:
            snap = await agent.get_snapshot_data(snapshot_id=snapshot_id, session_id=session_id)
        except Exception as e:  # noqa: BLE001 - spec asserts on the raised error
            thrown_msg = _thrown_message(thrown=e)
            assert expect_err in thrown_msg, (
                f'Expected getSnapshotData error.message to contain {expect_err!r}, got: {thrown_msg!r}'
            )
            return
        if snap is None:
            raise AssertionError(
                f'Expected error containing {expect_err!r} but getSnapshotData returned None '
                '(a miss is not a throw; branching reject must raise)'
            )
        raise AssertionError(f'Expected error containing {expect_err!r} but getSnapshotData succeeded')

    snap = await agent.get_snapshot_data(snapshot_id=snapshot_id, session_id=session_id)
    assert snap is not None, f'Snapshot not found for snapshotId={snapshot_id!r} sessionId={session_id!r}'

    if resolved.expect_snapshot is not None:
        assert_snapshot(snap=snap, expect=resolved.expect_snapshot)


async def execute_abort(*, agent: Agent, step: AbortStep, captures: dict[str, Any]) -> None:
    resolved = resolve_step(step=step, captures=captures)
    assert isinstance(resolved, AbortStep)
    assert resolved.snapshot_id, 'abort step requires snapshotId'

    previous = await agent.abort_snapshot_data(resolved.snapshot_id)
    previous_str = previous.value if previous is not None else None

    # The key being present (even as YAML ~ / null) means we should assert.
    if field_was_set(model=resolved, name='expect_previous_status'):
        assert previous_str == resolved.expect_previous_status, (
            f'Expected previous status {resolved.expect_previous_status!r}, got {previous_str!r}'
        )


async def execute_wait_until_completed(*, agent: Agent, step: WaitUntilCompletedStep, captures: dict[str, Any]) -> None:
    resolved = resolve_step(step=step, captures=captures)
    assert isinstance(resolved, WaitUntilCompletedStep)
    assert resolved.snapshot_id, 'waitUntilCompleted step requires snapshotId'
    timeout_s = (resolved.timeout_ms or 5000) / 1000.0

    deadline = time.monotonic() + timeout_s
    snap = None
    while time.monotonic() < deadline:
        snap = await agent.get_snapshot_data(snapshot_id=resolved.snapshot_id)
        if snap is not None and snap.status is not None and snap.status.value in TERMINAL_STATUSES:
            break
        await asyncio.sleep(0.1)

    assert snap is not None, f'Snapshot {resolved.snapshot_id!r} not found after waiting'
    status = snap.status.value if snap.status is not None else None
    assert status in TERMINAL_STATUSES, (
        f'Snapshot {resolved.snapshot_id!r} did not reach terminal status within {timeout_s}s. Status: {status!r}'
    )

    if resolved.expect_snapshot is not None:
        assert_snapshot(snap=snap, expect=resolved.expect_snapshot)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


# Gated spec capabilities this runtime implements; see SpecTest.requires.
SUPPORTED_REQUIRES: frozenset[str] = frozenset()


@pytest.mark.asyncio
@pytest.mark.parametrize('spec_test', SPEC_TESTS, ids=[t.name for t in SPEC_TESTS])
async def test_agent_conformance(spec_test: SpecTest) -> None:
    unknown = [r for r in (spec_test.requires or []) if r not in KNOWN_REQUIRES]
    assert not unknown, (
        f"unknown capabilities: {', '.join(unknown)}; add them to the spec's capabilities list or fix the spelling"
    )
    unsupported = [r for r in (spec_test.requires or []) if r not in SUPPORTED_REQUIRES]
    if unsupported:
        pytest.skip(f'requires unsupported capabilities: {", ".join(unsupported)}')
    harness = setup_harness()
    agent = harness.agents.get(spec_test.agent)
    assert agent is not None, f'Unknown agent {spec_test.agent!r} in test {spec_test.name!r}'

    captures: dict[str, Any] = {}

    for i, step in enumerate(spec_test.steps):
        label = f'step[{i}] ({step.type})'
        try:
            if isinstance(step, SendStep):
                await execute_send(agent=agent, pm=harness.pm, step=step, captures=captures)
            elif isinstance(step, GetSnapshotDataStep):
                await execute_get_snapshot_data(agent=agent, step=step, captures=captures)
            elif isinstance(step, AbortStep):
                await execute_abort(agent=agent, step=step, captures=captures)
            elif isinstance(step, WaitUntilCompletedStep):
                await execute_wait_until_completed(agent=agent, step=step, captures=captures)
            else:
                raise AssertionError(f'Unknown step type: {type(step).__name__}')
        except AssertionError:
            raise
        except Exception as e:
            raise AssertionError(f'{label} in test {spec_test.name!r} failed: {e}') from e


def test_assert_contains_none_is_noop() -> None:
    """A null expected value means the spec did not pin that field."""
    assert_contains(actual={'x': 1}, expected=None)
    assert_contains(actual=None, expected=None)
    assert_contains(actual=[1, 2], expected=None)


def test_spec_suite_rejects_unknown_step_type() -> None:
    with pytest.raises(ValidationError):
        SpecSuite.model_validate({'tests': [{'name': 't', 'agent': 'a', 'steps': [{'type': 'nope'}]}]})
    with pytest.raises(ValidationError):
        SpecSuite.model_validate({'tests': [{'name': 't', 'agent': 'a', 'steps': ['send']}]})
    with pytest.raises(ValidationError):
        SpecSuite.model_validate({'tests': ['foo']})


def test_spec_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match='hasSnapshatId'):
        OutputAssertions.model_validate({'hasSnapshatId': True})


def test_expect_chunk_requires_exactly_one_payload() -> None:
    ExpectChunk.model_validate({'turnEnd': {'finishReason': 'stop'}})
    with pytest.raises(ValidationError, match='exactly one'):
        ExpectChunk.model_validate({})
    with pytest.raises(ValidationError, match='exactly one'):
        ExpectChunk.model_validate({'turnEnd': {}, 'modelChunk': {'role': 'model'}})


def test_spec_init_keeps_template_state() -> None:
    init = SpecInit.model_validate({'snapshotId': '{{snap1}}', 'state': '{{state1}}'})
    assert init.snapshot_id == '{{snap1}}'
    assert init.state == '{{state1}}'


def test_abort_keeps_explicit_null_previous_status() -> None:
    step = AbortStep.model_validate({'type': 'abort', 'snapshotId': 'x', 'expectPreviousStatus': None})
    assert field_was_set(model=step, name='expect_previous_status')
    assert step.expect_previous_status is None
    dumped = step.model_dump(by_alias=True, exclude_unset=True)
    assert 'expectPreviousStatus' in dumped
    assert dumped['expectPreviousStatus'] is None
    resolved = resolve_step(step=step, captures={})
    assert isinstance(resolved, AbortStep)
    assert field_was_set(model=resolved, name='expect_previous_status')
    assert resolved.expect_previous_status is None


def test_resolve_templates_inline_object_is_json() -> None:
    got = resolve_templates(
        value='seeded-{{state1}}',
        captures={'state1': {'sessionId': 'abc', 'custom': {'counter': 1}}},
    )
    assert got == 'seeded-{"sessionId":"abc","custom":{"counter":1}}'


def test_assert_expect_error_matches_status_prefixed_message() -> None:
    err = GenkitError(status='FAILED_PRECONDITION', message="Cannot send 'state' to agent")
    _assert_expect_error(
        thrown=err,
        expect_err=SendExpectError(status='FAILED_PRECONDITION', message="Cannot send 'state'"),
    )
    # The status name is in the string the caller sees, so a spec can search for it.
    _assert_expect_error(thrown=err, expect_err=SendExpectError(message='FAILED_PRECONDITION'))
    with pytest.raises(AssertionError, match='error.message'):
        _assert_expect_error(thrown=err, expect_err=SendExpectError(message='NOT_THIS'))


def test_send_expect_error_rejects_string() -> None:
    with pytest.raises(ValidationError):
        SendStep.model_validate({'type': 'send', 'expectError': "Cannot send 'state' to agent"})


def test_lookup_expect_error_rejects_mapping() -> None:
    with pytest.raises(ValidationError):
        GetSnapshotDataStep.model_validate({
            'type': 'getSnapshotData',
            'snapshotId': 's',
            'expectError': {'status': 'NOT_FOUND', 'message': 'not found'},
        })


def test_thrown_message_includes_status_prefix() -> None:
    err = GenkitError(status='NOT_FOUND', message='branching session')
    assert _thrown_message(thrown=err) == 'NOT_FOUND: branching session'
    wrapped = GenkitError(status='NOT_FOUND', message='branching session', cause=RuntimeError('inner'))
    assert _thrown_message(thrown=wrapped) == 'NOT_FOUND: branching session'
    assert 'inner' not in _thrown_message(thrown=wrapped)


def test_has_id_false_means_absent() -> None:
    assert_has_id(actual=None, want=False, path='output.snapshotId')
    with pytest.raises(AssertionError, match='absent'):
        assert_has_id(actual='snap-1', want=False, path='output.snapshotId')


def test_dump_keeps_explicit_null_and_drops_unset() -> None:
    pinned = SessionSnapshotSchema(snapshot_id='s', created_at='t', finish_reason=None)
    dumped = dump(model=pinned)
    assert 'finishReason' in dumped
    assert dumped['finishReason'] is None
    omitted = SessionSnapshotSchema(snapshot_id='s', created_at='t')
    assert 'finishReason' not in dump(model=omitted)


def test_snapshot_error_contains_is_exact_message() -> None:
    """Snapshot errorContains uses subset matching; a string field is exact."""

    def snap(*, message: str, status: str | None = None) -> SessionSnapshotSchema:
        return SessionSnapshotSchema(
            snapshot_id='s',
            created_at='t',
            error=GenkitRuntimeError(status=status, message=message),
        )

    wrapped = snap(message='background: intentional failure (wrapped)')
    expect_partial = SnapshotAssertions.model_validate({'errorContains': {'message': 'intentional failure'}})
    with pytest.raises(AssertionError, match='snapshot.error.message'):
        assert_snapshot(snap=wrapped, expect=expect_partial)
    assert_snapshot(
        snap=wrapped,
        expect=SnapshotAssertions.model_validate({
            'errorContains': {'message': 'background: intentional failure (wrapped)'}
        }),
    )
    assert_snapshot(
        snap=snap(message='intentional failure', status='INTERNAL'),
        expect=SnapshotAssertions.model_validate({'errorContains': {'message': 'intentional failure'}}),
    )
