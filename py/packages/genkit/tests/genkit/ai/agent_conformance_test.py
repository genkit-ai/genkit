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
harness requirements. Mirrors js/ai/tests/agents_spec_test.ts and
go/ai/exp/agents_conformance_test.go.
"""

from __future__ import annotations

import asyncio
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

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
    AgentResult,
    Artifact,
    MessageData,
    Part,
    Role,
    TextPart,
)
from genkit.agent import Agent

SPEC_PATH = pathlib.Path(__file__).parent / '../../../../../../tests/specs/agent.yaml'
TERMINAL_STATUSES = {'completed', 'failed', 'aborted'}


def load_spec() -> list[dict[str, Any]]:
    with SPEC_PATH.open() as f:
        suite = yaml.safe_load(f)
    tests = suite.get('tests')
    assert isinstance(tests, list) and tests, 'agent.yaml contains no tests'
    for t in tests:
        assert isinstance(t.get('name'), str), 'spec test missing name'
        assert isinstance(t.get('agent'), str), f'spec test {t.get("name")!r} missing agent'
        assert isinstance(t.get('steps'), list), f'spec test {t.get("name")!r} missing steps'
    return tests


SPEC_TESTS = load_spec()

# Known divergences between the Python implementation and the shared spec.
# Each entry marks the affected spec tests xfail(strict=True); the PR that
# fixes a divergence deletes its entries, turning those tests green in the
# same diff. Do not add entries: new divergences should be fixed, not listed.
KNOWN_DIVERGENCES: dict[str, str] = {
    'abort pending agent': 'abort returns resulting status instead of previous status',
    'abort already aborted agent': 'abort returns resulting status instead of previous status',
    'pure detach without payload': 'abort returns resulting status instead of previous status',
    'snapshotId and matching sessionId together resume': 'snapshotId+sessionId pair rejected instead of ownership-guarded',
    'snapshotId with mismatched sessionId rejected': 'snapshotId+sessionId pair rejected instead of ownership-guarded',
    'detached run emits no customPatch chunks': 'detached runs leak stream chunks (race)',
}


def _params() -> list[Any]:
    out = []
    for t in SPEC_TESTS:
        if t['name'] in KNOWN_DIVERGENCES:
            out.append(pytest.param(t, marks=pytest.mark.xfail(reason=KNOWN_DIVERGENCES[t['name']], strict=True)))
        else:
            out.append(pytest.param(t))
    return out


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------

_FULL_TEMPLATE = re.compile(r'^\{\{(\w+)\}\}$')
_INLINE_TEMPLATE = re.compile(r'\{\{(\w+)\}\}')


def resolve_templates(value: Any, captures: dict[str, Any]) -> Any:
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
            return v if isinstance(v, str) else str(v)

        return _INLINE_TEMPLATE.sub(sub, value)
    if isinstance(value, list):
        return [resolve_templates(item, captures) for item in value]
    if isinstance(value, dict):
        return {k: resolve_templates(v, captures) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# "Contains" assertion helpers
# ---------------------------------------------------------------------------


def assert_contains(actual: Any, expected: Any, path: str = '') -> None:
    """Assert that ``actual`` contains all fields specified in ``expected``.

    Dicts are matched key-by-key (extra keys in actual are allowed). Lists are
    matched as an in-order (not necessarily contiguous) subsequence. Scalars
    must match exactly.
    """
    if expected is None:
        # YAML `~` inside a contains-expectation asserts the field is null/absent.
        assert actual is None, f'Expected null at {path}, got {actual!r}'
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), f'Expected list at {path}, got {type(actual).__name__}: {actual!r}'
        assert_contains_subsequence(actual, expected, path)
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f'Expected dict at {path}, got {type(actual).__name__}: {actual!r}'
        for key, val in expected.items():
            assert_contains(actual.get(key), val, f'{path}.{key}')
        return

    assert actual == expected, f'Mismatch at {path}: expected {expected!r}, got {actual!r}'


def assert_contains_subsequence(actual: list[Any], expected: list[Any], path: str) -> None:
    """Assert all ``expected`` items appear in ``actual`` in the same relative order."""
    actual_idx = 0
    for i, exp_item in enumerate(expected):
        found = False
        while actual_idx < len(actual):
            try:
                assert_contains(actual[actual_idx], exp_item, f'{path}[{actual_idx}]')
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


def dump(model: BaseModel) -> dict[str, Any]:
    """Serialize a wire model to its camelCase JSON form for spec comparison."""
    return model.model_dump(by_alias=True, exclude_none=True, mode='json')


# ---------------------------------------------------------------------------
# Harness setup
# ---------------------------------------------------------------------------


def _model_text(text: str) -> MessageData:
    return MessageData(role=Role.MODEL, content=[Part(root=TextPart(text=text))])


class InterruptQuery(BaseModel):
    query: str


class RestartInput(BaseModel):
    action: str


class RestartOutput(BaseModel):
    result: str


@dataclass
class Harness:
    ai: Genkit
    pm: ProgrammableModel
    agents: dict[str, Agent] = field(default_factory=dict)


def setup_harness() -> Harness:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    h = Harness(ai=ai, pm=pm)

    # --- Tools ---

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

    # --- Prompt-backed agents ---

    h.agents['promptAgent'] = ai.define_agent(
        name='promptAgent',
        model='programmableModel',
        config={'temperature': 1},
    )
    h.agents['promptAgentWithStore'] = ai.define_agent(
        name='promptAgentWithStore',
        model='programmableModel',
        config={'temperature': 1},
        store=InMemorySessionStore(),
    )
    h.agents['promptAgentWithTools'] = ai.define_agent(
        name='promptAgentWithTools',
        model='programmableModel',
        config={'temperature': 1},
        tools=['testTool'],
    )
    h.agents['promptAgentWithInterrupt'] = ai.define_agent(
        name='promptAgentWithInterrupt',
        model='programmableModel',
        config={'temperature': 1},
        tools=['interruptTool'],
        store=InMemorySessionStore(),
    )
    h.agents['promptAgentWithRestartTool'] = ai.define_agent(
        name='promptAgentWithRestartTool',
        model='programmableModel',
        config={'temperature': 1},
        tools=['restartTool'],
        store=InMemorySessionStore(),
    )

    # --- Custom agents ---

    def run_turns(turn_body):  # noqa: ANN001, ANN202 - AgentFn factory
        """Wrap a per-turn body into the canonical custom AgentFn shape."""

        async def agent_fn(session_runner: SessionRunner, ctx: ActionRunContext) -> AgentResult:
            async def handle_turn(inp: AgentInput, turn_ctx: TurnContext) -> TurnResult | None:
                await turn_body(session_runner, ctx, inp, turn_ctx)
                return TurnResult(finish_reason=AgentFinishReason.STOP)

            await session_runner.run(handle_turn)
            return await session_runner.result()

        return agent_fn

    # customAgentBlocking: server-managed, blocks until the abort signal fires.
    async def blocking_turn(sr: SessionRunner, ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext) -> None:
        await ctx.abort_signal.wait()
        await sr.add_messages([_model_text('unblocked')])

    h.agents['customAgentBlocking'] = ai.define_custom_agent(
        name='customAgentBlocking',
        fn=run_turns(blocking_turn),
        store=InMemorySessionStore(),
    )

    # customAgentFailing: server-managed, throws during processing. The error
    # must propagate out of the agent fn (as in the JS/Go harnesses, where the
    # throw escapes sess.run) so a detached run finalizes its snapshot as
    # 'failed'. Python's SessionRunner.run swallows turn errors into
    # last_turn_error, so the fn re-raises it — the Python spelling of Go's
    # `if err := sess.Run(...); err != nil { return nil, err }`.
    async def failing_agent_fn(session_runner: SessionRunner, _ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(_inp: AgentInput, _tc: TurnContext) -> TurnResult | None:
            raise RuntimeError('intentional failure')

        await session_runner.run(handle_turn)
        if session_runner.last_turn_error is not None:
            raise RuntimeError(session_runner.last_turn_error.message or 'intentional failure')
        return await session_runner.result()

    h.agents['customAgentFailing'] = ai.define_custom_agent(
        name='customAgentFailing',
        fn=failing_agent_fn,
        store=InMemorySessionStore(),
    )

    # customAgentWithArtifacts: client-managed, adds and updates artifacts.
    async def artifacts_turn(sr: SessionRunner, _ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext) -> None:
        await sr.add_artifacts([Artifact(name='doc1', parts=[Part(root=TextPart(text='v1'))])])
        await sr.add_artifacts([Artifact(name='doc1', parts=[Part(root=TextPart(text='v2'))])])
        await sr.add_artifacts([Artifact(name='doc2', parts=[Part(root=TextPart(text='other'))])])
        await sr.add_messages([_model_text('done')])

    h.agents['customAgentWithArtifacts'] = ai.define_custom_agent(
        name='customAgentWithArtifacts',
        fn=run_turns(artifacts_turn),
    )

    # customAgentWithCustomState: client-managed, increments a counter per turn.
    async def counter_turn(sr: SessionRunner, _ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext) -> None:
        prev = await sr.get_custom() or {}
        counter = (prev.get('counter') or 0) + 1
        await sr.update_custom(lambda _prev: {'counter': counter})
        await sr.add_messages([_model_text('done')])

    h.agents['customAgentWithCustomState'] = ai.define_custom_agent(
        name='customAgentWithCustomState',
        fn=run_turns(counter_turn),
    )

    # customAgentWithMultiCustomState: several sequential custom-state updates
    # within one turn (first patch = whole-doc replace, then incremental diffs).
    async def multi_custom_turn(sr: SessionRunner, _ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext) -> None:
        await sr.update_custom(lambda _prev: {'counter': 1, 'status': 'working'})
        await sr.update_custom(lambda prev: {**(prev or {}), 'counter': 2})
        await sr.update_custom(lambda prev: {**(prev or {}), 'status': 'done'})
        await sr.add_messages([_model_text('done')])

    h.agents['customAgentWithMultiCustomState'] = ai.define_custom_agent(
        name='customAgentWithMultiCustomState',
        fn=run_turns(multi_custom_turn),
    )

    # customAgentWithArtifactsStore: server-managed, adds a numbered artifact
    # on each invocation.
    async def artifacts_store_turn(
        sr: SessionRunner, _ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext
    ) -> None:
        existing = await sr.get_artifacts()
        count = len(existing) + 1
        await sr.add_artifacts([Artifact(name=f'doc{count}', parts=[Part(root=TextPart(text=f'content{count}'))])])
        await sr.add_messages([_model_text('done')])

    h.agents['customAgentWithArtifactsStore'] = ai.define_custom_agent(
        name='customAgentWithArtifactsStore',
        fn=run_turns(artifacts_store_turn),
        store=InMemorySessionStore(),
    )

    # customAgentWithCustomStateStore: server-managed counter.
    h.agents['customAgentWithCustomStateStore'] = ai.define_custom_agent(
        name='customAgentWithCustomStateStore',
        fn=run_turns(counter_turn),
        store=InMemorySessionStore(),
    )

    return h


# ---------------------------------------------------------------------------
# Step executors
# ---------------------------------------------------------------------------


def program_model(pm: ProgrammableModel, step: dict[str, Any]) -> None:
    pm.reset()
    responses = step.get('modelResponses') or []
    pm.responses = [ModelResponse.model_validate(r) for r in responses]
    stream_chunks = step.get('streamChunks')
    if stream_chunks:
        pm.chunks = [[ModelResponseChunk.model_validate(c) for c in group] for group in stream_chunks]


def assert_chunks(actual_chunks: list[Any], expected_chunks: list[Any]) -> None:
    """Strict ordered chunk comparison per the spec's expectChunks contract."""
    actual = [dump(c) for c in actual_chunks]
    assert len(actual) == len(expected_chunks), (
        f'Expected {len(expected_chunks)} chunks, got {len(actual)}.\n  Actual: {actual!r}\n  Expected: {expected_chunks!r}'
    )
    for i, expected in enumerate(expected_chunks):
        got = actual[i]
        if 'turnEnd' in expected:
            # turnEnd carries a dynamic snapshotId; only assert presence, plus
            # finishReason exactly when the spec pins it.
            assert 'turnEnd' in got, f'Chunk {i}: expected turnEnd, got {got!r}'
            want_fr = expected['turnEnd'].get('finishReason') if isinstance(expected['turnEnd'], dict) else None
            if want_fr is not None:
                assert got['turnEnd'].get('finishReason') == want_fr, (
                    f"Chunk {i}: expected turnEnd.finishReason {want_fr!r}, got {got['turnEnd'].get('finishReason')!r}"
                )
        elif 'modelChunk' in expected:
            assert_contains(got.get('modelChunk'), expected['modelChunk'], f'chunk[{i}].modelChunk')
        elif 'artifact' in expected:
            assert_contains(got.get('artifact'), expected['artifact'], f'chunk[{i}].artifact')
        elif 'customPatch' in expected:
            assert_contains(got.get('customPatch'), expected['customPatch'], f'chunk[{i}].customPatch')
        else:
            assert_contains(got, expected, f'chunk[{i}]')


def assert_output(out: dict[str, Any], expect: dict[str, Any]) -> None:
    if expect.get('message') is not None:
        assert_contains(out.get('message'), expect['message'], 'output.message')

    if expect.get('hasSnapshotId'):
        assert isinstance(out.get('snapshotId'), str) and out['snapshotId'], (
            f'Expected output to have a snapshotId, got: {out.get("snapshotId")!r}'
        )

    if expect.get('hasSessionId'):
        state = out.get('state')
        assert state, 'Expected output to have state for sessionId check'
        assert isinstance(state.get('sessionId'), str) and state['sessionId'], (
            f'Expected output.state to have a sessionId, got: {state.get("sessionId")!r}'
        )

    if expect.get('stateContains') is not None:
        assert out.get('state') is not None, 'Expected output to have state'
        assert_contains(out['state'], expect['stateContains'], 'output.state')

    if expect.get('artifactsContain') is not None:
        artifacts = out.get('artifacts')
        assert artifacts is not None, 'Expected output to have artifacts'
        for expected_art in expect['artifactsContain']:
            found = next((a for a in artifacts if a.get('name') == expected_art.get('name')), None)
            assert found is not None, f'Expected artifact {expected_art.get("name")!r} not found in output'
            assert_contains(found, expected_art, f'artifact({expected_art.get("name")})')

    if 'finishReason' in expect:
        assert out.get('finishReason') == expect['finishReason'], (
            f'Expected output.finishReason {expect["finishReason"]!r}, got {out.get("finishReason")!r}'
        )

    if expect.get('errorContains') is not None:
        err = out.get('error')
        assert err, f'Expected output to have an error, got: {err!r}'
        want = expect['errorContains']
        if 'status' in want:
            assert err.get('status') == want['status'], (
                f'Expected output.error.status {want["status"]!r}, got {err.get("status")!r}'
            )
        if 'message' in want:
            assert want['message'] in (err.get('message') or ''), (
                f'Expected output.error.message to contain {want["message"]!r}, got: {err.get("message")!r}'
            )


def assert_snapshot(snap: dict[str, Any], expect: dict[str, Any]) -> None:
    if 'parentId' in expect:
        assert snap.get('parentId') == expect['parentId'], (
            f'Expected parentId {expect["parentId"]!r}, got {snap.get("parentId")!r}'
        )
    if 'status' in expect:
        assert snap.get('status') == expect['status'], (
            f'Expected status {expect["status"]!r}, got {snap.get("status")!r}'
        )
    if 'finishReason' in expect:
        assert snap.get('finishReason') == expect['finishReason'], (
            f'Expected snapshot.finishReason {expect["finishReason"]!r}, got {snap.get("finishReason")!r}'
        )
    if expect.get('hasSessionId'):
        state = snap.get('state') or {}
        assert isinstance(state.get('sessionId'), str) and state['sessionId'], (
            f'Expected snapshot.state to have a sessionId, got: {state.get("sessionId")!r}'
        )
    if expect.get('stateContains') is not None:
        assert_contains(snap.get('state'), expect['stateContains'], 'snapshot.state')
    if expect.get('errorContains') is not None:
        assert snap.get('error'), 'Expected snapshot to have error'
        assert_contains(snap['error'], expect['errorContains'], 'snapshot.error')


async def execute_send(agent: Agent, pm: ProgrammableModel, step: dict[str, Any], captures: dict[str, Any]) -> None:
    resolved = resolve_templates(step, captures)
    program_model(pm, resolved)

    conn = await agent.stream_bidi(AgentInit.model_validate(resolved.get('init') or {}))
    for inp in resolved.get('inputs') or []:
        await conn.send(AgentInput.model_validate(inp))
    await conn.close()

    # expectError: the turn throws (API misuse) rather than resolving with a
    # graceful finishReason='failed' output.
    if resolved.get('expectError'):
        expect_err = resolved['expectError']
        thrown: BaseException | None = None
        try:
            async for _chunk in conn.receive():
                pass
            await conn.output()
        except (GenkitError, Exception) as e:  # noqa: BLE001 - spec asserts on the raised error
            thrown = e
        assert thrown is not None, 'Expected the turn to throw an error, but it resolved successfully.'
        if 'status' in expect_err:
            status = getattr(thrown, 'status', None)
            assert status == expect_err['status'], (
                f'Expected thrown error.status {expect_err["status"]!r}, got {status!r} (message: {thrown})'
            )
        if 'message' in expect_err:
            assert expect_err['message'] in str(thrown), (
                f'Expected thrown error.message to contain {expect_err["message"]!r}, got: {thrown}'
            )
        return

    chunks = [c async for c in conn.receive()]
    output = await conn.output()
    out = dump(output)

    if resolved.get('expectChunks') is not None:
        assert_chunks(chunks, resolved['expectChunks'])

    if resolved.get('expectOutput') is not None:
        assert_output(out, resolved['expectOutput'])

    # Captures for subsequent steps (use the unresolved step so capture names
    # are never themselves template-substituted).
    if step.get('captureSnapshotId'):
        assert out.get('snapshotId'), (
            f'captureSnapshotId {step["captureSnapshotId"]!r} requested but output has no snapshotId'
        )
        captures[step['captureSnapshotId']] = out['snapshotId']
    if step.get('captureState'):
        assert out.get('state'), f'captureState {step["captureState"]!r} requested but output has no state'
        captures[step['captureState']] = out['state']
    if step.get('captureSessionId'):
        state = out.get('state') or {}
        assert state.get('sessionId'), (
            f'captureSessionId {step["captureSessionId"]!r} requested but output has no state.sessionId'
        )
        captures[step['captureSessionId']] = state['sessionId']


async def execute_get_snapshot_data(agent: Agent, step: dict[str, Any], captures: dict[str, Any]) -> None:
    resolved = resolve_templates(step, captures)
    snapshot_id = resolved.get('snapshotId')
    session_id = resolved.get('sessionId')
    assert bool(snapshot_id) != bool(session_id), (
        'getSnapshotData step requires exactly one of snapshotId or sessionId'
    )

    if resolved.get('expectError'):
        with pytest.raises(Exception, match=re.escape(resolved['expectError'])):
            await agent.get_snapshot_data(snapshot_id=snapshot_id, session_id=session_id)
        return

    snap = await agent.get_snapshot_data(snapshot_id=snapshot_id, session_id=session_id)
    assert snap is not None, f'Snapshot not found for snapshotId={snapshot_id!r} sessionId={session_id!r}'

    if resolved.get('expectSnapshot') is not None:
        assert_snapshot(dump(snap), resolved['expectSnapshot'])


async def execute_abort(agent: Agent, step: dict[str, Any], captures: dict[str, Any]) -> None:
    resolved = resolve_templates(step, captures)
    snapshot_id = resolved.get('snapshotId')
    assert snapshot_id, 'abort step requires snapshotId'

    previous = await agent.abort_snapshot_data(snapshot_id)
    previous_str = previous.value if previous is not None else None

    # The key being present (even as YAML ~ / null) means we should assert.
    if 'expectPreviousStatus' in resolved:
        expected = resolved['expectPreviousStatus']
        assert previous_str == expected, f'Expected previous status {expected!r}, got {previous_str!r}'


async def execute_wait_until_completed(agent: Agent, step: dict[str, Any], captures: dict[str, Any]) -> None:
    resolved = resolve_templates(step, captures)
    snapshot_id = resolved.get('snapshotId')
    assert snapshot_id, 'waitUntilCompleted step requires snapshotId'
    timeout_s = (resolved.get('timeoutMs') or 5000) / 1000.0

    deadline = time.monotonic() + timeout_s
    snap = None
    while time.monotonic() < deadline:
        snap = await agent.get_snapshot_data(snapshot_id=snapshot_id)
        if snap is not None and snap.status is not None and snap.status.value in TERMINAL_STATUSES:
            break
        await asyncio.sleep(0.1)

    assert snap is not None, f'Snapshot {snapshot_id!r} not found after waiting'
    status = snap.status.value if snap.status is not None else None
    assert status in TERMINAL_STATUSES, (
        f'Snapshot {snapshot_id!r} did not reach terminal status within {timeout_s}s. Status: {status!r}'
    )

    if resolved.get('expectSnapshot') is not None:
        assert_snapshot(dump(snap), resolved['expectSnapshot'])


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('spec_test', _params(), ids=[t['name'] for t in SPEC_TESTS])
async def test_agent_conformance(spec_test: dict[str, Any]) -> None:
    harness = setup_harness()
    agent = harness.agents.get(spec_test['agent'])
    assert agent is not None, f'Unknown agent {spec_test["agent"]!r} in test {spec_test["name"]!r}'

    captures: dict[str, Any] = {}

    for i, step in enumerate(spec_test['steps']):
        step_type = step.get('type')
        label = f'step[{i}] ({step_type})'
        try:
            if step_type == 'send':
                await execute_send(agent, harness.pm, step, captures)
            elif step_type == 'getSnapshotData':
                await execute_get_snapshot_data(agent, step, captures)
            elif step_type == 'abort':
                await execute_abort(agent, step, captures)
            elif step_type == 'waitUntilCompleted':
                await execute_wait_until_completed(agent, step, captures)
            else:
                raise AssertionError(f'Unknown step type: {step_type!r}')
        except AssertionError as e:
            raise AssertionError(f'{label} in test {spec_test["name"]!r} failed: {e}') from e
