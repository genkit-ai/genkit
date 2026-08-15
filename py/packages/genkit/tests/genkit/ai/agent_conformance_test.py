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
import contextlib
import json
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
    AgentStreamChunk,
    Artifact,
    MessageData,
    Part,
    Role,
    TextPart,
    TurnEnd,
)
from genkit.agent import Agent

SPEC_PATH = pathlib.Path(__file__).parent / '../../../../../../tests/specs/agent.yaml'
TERMINAL_STATUSES = {'completed', 'failed', 'aborted'}


def _tests_from_suite(*, suite: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    if not isinstance(suite, dict):
        raise AssertionError(f'agent.yaml must be a mapping, got {type(suite).__name__}')
    tests = suite.get('tests')
    assert isinstance(tests, list) and tests, 'agent.yaml contains no tests'
    for i, t in enumerate(tests):
        if not isinstance(t, dict):
            raise AssertionError(f'agent.yaml tests[{i}] must be a mapping, got {type(t).__name__}')
        assert isinstance(t.get('name'), str), f'spec test at tests[{i}] missing name'
        assert isinstance(t.get('agent'), str), f'spec test {t.get("name")!r} missing agent'
        steps = t.get('steps')
        assert isinstance(steps, list), f'spec test {t.get("name")!r} missing steps'
        for j, step in enumerate(steps):
            if not isinstance(step, dict):
                raise AssertionError(
                    f'agent.yaml tests[{i}] ({t["name"]!r}) steps[{j}] must be a mapping, got {type(step).__name__}'
                )
    return tests


def load_spec() -> list[dict[str, Any]]:
    with SPEC_PATH.open() as f:
        suite = yaml.safe_load(f)
    return _tests_from_suite(suite={} if suite is None else suite)


SPEC_TESTS = load_spec()


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
    """Serialize a wire model to its camelCase JSON form for spec comparison."""
    return model.model_dump(by_alias=True, exclude_none=True, mode='json')


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

    def run_turns(*, turn_body):  # noqa: ANN001, ANN202 - AgentFn factory
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
        await sr.add_messages([_model_text(text='unblocked')])

    h.agents['customAgentBlocking'] = ai.define_custom_agent(
        name='customAgentBlocking',
        fn=run_turns(turn_body=blocking_turn),
        store=InMemorySessionStore(),
    )

    # customAgentFailing: server-managed; the turn raises on purpose.
    # Raise inside the turn callback, then return session_runner.result().
    # run() records the failure on the session and returns, so a detached
    # client can still read status=failed and the error from the snapshot.
    # Re-raising after run() fails the invocation and skips that write.
    async def failing_agent_fn(session_runner: SessionRunner, _ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(_inp: AgentInput, _tc: TurnContext) -> TurnResult | None:
            raise RuntimeError('intentional failure')

        await session_runner.run(handle_turn)
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
        await sr.add_messages([_model_text(text='done')])

    h.agents['customAgentWithArtifacts'] = ai.define_custom_agent(
        name='customAgentWithArtifacts',
        fn=run_turns(turn_body=artifacts_turn),
    )

    # customAgentWithCustomState: client-managed, increments a counter per turn.
    async def counter_turn(sr: SessionRunner, _ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext) -> None:
        prev = await sr.get_custom() or {}
        counter = (prev.get('counter') or 0) + 1
        await sr.update_custom(lambda _prev: {'counter': counter})
        await sr.add_messages([_model_text(text='done')])

    h.agents['customAgentWithCustomState'] = ai.define_custom_agent(
        name='customAgentWithCustomState',
        fn=run_turns(turn_body=counter_turn),
    )

    # customAgentWithMultiCustomState: several sequential custom-state updates
    # within one turn (first patch = whole-doc replace, then incremental diffs).
    async def multi_custom_turn(sr: SessionRunner, _ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext) -> None:
        await sr.update_custom(lambda _prev: {'counter': 1, 'status': 'working'})
        await sr.update_custom(lambda prev: {**(prev or {}), 'counter': 2})
        await sr.update_custom(lambda prev: {**(prev or {}), 'status': 'done'})
        await sr.add_messages([_model_text(text='done')])

    h.agents['customAgentWithMultiCustomState'] = ai.define_custom_agent(
        name='customAgentWithMultiCustomState',
        fn=run_turns(turn_body=multi_custom_turn),
    )

    # customAgentWithArtifactsStore: server-managed, adds a numbered artifact
    # on each invocation.
    async def artifacts_store_turn(
        sr: SessionRunner, _ctx: ActionRunContext, _inp: AgentInput, _tc: TurnContext
    ) -> None:
        existing = await sr.get_artifacts()
        count = len(existing) + 1
        await sr.add_artifacts([Artifact(name=f'doc{count}', parts=[Part(root=TextPart(text=f'content{count}'))])])
        await sr.add_messages([_model_text(text='done')])

    h.agents['customAgentWithArtifactsStore'] = ai.define_custom_agent(
        name='customAgentWithArtifactsStore',
        fn=run_turns(turn_body=artifacts_store_turn),
        store=InMemorySessionStore(),
    )

    # customAgentWithCustomStateStore: server-managed counter.
    h.agents['customAgentWithCustomStateStore'] = ai.define_custom_agent(
        name='customAgentWithCustomStateStore',
        fn=run_turns(turn_body=counter_turn),
        store=InMemorySessionStore(),
    )

    return h


# ---------------------------------------------------------------------------
# Step executors
# ---------------------------------------------------------------------------


def program_model(*, pm: ProgrammableModel, step: dict[str, Any]) -> None:
    pm.reset()
    responses = step.get('modelResponses') or []
    pm.responses = [ModelResponse.model_validate(r) for r in responses]
    stream_chunks = step.get('streamChunks')
    if stream_chunks:
        pm.chunks = [[ModelResponseChunk.model_validate(c) for c in group] for group in stream_chunks]


def assert_chunks(*, actual_chunks: list[Any], expected_chunks: list[Any]) -> None:
    """Strict ordered chunk comparison per the spec's expectChunks contract."""
    actual = [dump(model=c) for c in actual_chunks]
    assert len(actual) == len(expected_chunks), (
        f'Expected {len(expected_chunks)} chunks, got {len(actual)}.\n  Actual: {actual!r}\n  Expected: {expected_chunks!r}'
    )
    for i, expected in enumerate(expected_chunks):
        got = actual[i]
        if 'turnEnd' in expected:
            # turnEnd carries a dynamic snapshotId; only assert presence, plus
            # finishReason exactly when the spec pins it (key present, including YAML ~).
            assert 'turnEnd' in got, f'Chunk {i}: expected turnEnd, got {got!r}'
            turn_end = expected['turnEnd']
            if isinstance(turn_end, dict) and 'finishReason' in turn_end:
                # Pydantic json dumps drop None even when the field was set, so
                # omitted vs null is model_fields_set, not exclude_unset.
                te_model = actual_chunks[i].turn_end
                want_fr = turn_end['finishReason']
                if want_fr is None:
                    set_null = (
                        te_model is not None
                        and 'finish_reason' in te_model.model_fields_set
                        and te_model.finish_reason is None
                    )
                    assert set_null, (
                        f'Chunk {i}: expected turnEnd.finishReason null, '
                        f'fields_set={getattr(te_model, "model_fields_set", None)!r} '
                        f'value={getattr(te_model, "finish_reason", None)!r}'
                    )
                else:
                    got_fr = te_model.finish_reason if te_model is not None else None
                    assert got_fr == want_fr, (
                        f'Chunk {i}: expected turnEnd.finishReason {want_fr!r}, got {got_fr!r}'
                    )
        elif 'modelChunk' in expected:
            assert_contains(actual=got.get('modelChunk'), expected=expected['modelChunk'], path=f'chunk[{i}].modelChunk')
        elif 'artifact' in expected:
            assert_contains(actual=got.get('artifact'), expected=expected['artifact'], path=f'chunk[{i}].artifact')
        elif 'customPatch' in expected:
            assert_contains(actual=got.get('customPatch'), expected=expected['customPatch'], path=f'chunk[{i}].customPatch')
        else:
            assert_contains(actual=got, expected=expected, path=f'chunk[{i}]')


def assert_output(*, out: dict[str, Any], expect: dict[str, Any]) -> None:
    if expect.get('message') is not None:
        assert_contains(actual=out.get('message'), expected=expect['message'], path='output.message')

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
        assert_contains(actual=out['state'], expected=expect['stateContains'], path='output.state')

    if expect.get('artifactsContain') is not None:
        artifacts = out.get('artifacts')
        assert artifacts is not None, 'Expected output to have artifacts'
        for expected_art in expect['artifactsContain']:
            found = next((a for a in artifacts if a.get('name') == expected_art.get('name')), None)
            assert found is not None, f'Expected artifact {expected_art.get("name")!r} not found in output'
            assert_contains(actual=found, expected=expected_art, path=f'artifact({expected_art.get("name")})')

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


def assert_snapshot(*, snap: dict[str, Any], expect: dict[str, Any]) -> None:
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
        assert_contains(actual=snap.get('state'), expected=expect['stateContains'], path='snapshot.state')
    if expect.get('errorContains') is not None:
        err = snap.get('error')
        assert err, 'Expected snapshot to have error'
        # Snapshot error is subset/contains (scalar fields exact), same as JS.
        # Output errorContains is different: status exact, message substring.
        assert_contains(actual=err, expected=expect['errorContains'], path='snapshot.error')


async def _close_quietly(*, conn: Any) -> None:  # noqa: ANN401
    if conn is None:
        return
    with contextlib.suppress(Exception):
        await conn.close()


def _require_send_expect_error(*, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if not isinstance(value, dict):
        raise AssertionError(
            f'send expectError must be a mapping with optional status/message, got {type(value).__name__}: {value!r}'
        )
    if not value:
        raise AssertionError('send expectError must include status and/or message, got {}')
    for key in ('status', 'message'):
        if key in value and not isinstance(value[key], str):
            raise AssertionError(
                f'send expectError.{key} must be a string, got {type(value[key]).__name__}: {value[key]!r}'
            )
    return value


def _require_lookup_expect_error(*, value: Any) -> str:  # noqa: ANN401
    if not isinstance(value, str) or not value:
        raise AssertionError(
            f'getSnapshotData expectError must be a non-empty string substring, got {type(value).__name__}: {value!r}'
        )
    return value


def _thrown_message(*, thrown: BaseException) -> str:
    """Return the error's message field, not ``str(exc)`` (which prefixes status)."""
    return getattr(thrown, 'original_message', None) or getattr(thrown, 'message', None) or str(thrown)


def _assert_expect_error(*, thrown: BaseException | None, expect_err: dict[str, Any]) -> None:
    assert thrown is not None, 'Expected the turn to throw an error, but it resolved successfully.'
    if 'status' in expect_err:
        status = getattr(thrown, 'status', None)
        assert status == expect_err['status'], (
            f'Expected thrown error.status {expect_err["status"]!r}, got {status!r} (message: {thrown})'
        )
    if 'message' in expect_err:
        thrown_msg = _thrown_message(thrown=thrown)
        assert expect_err['message'] in thrown_msg, (
            f'Expected thrown error.message to contain {expect_err["message"]!r}, got: {thrown_msg!r}'
        )


async def execute_send(*, agent: Agent, pm: ProgrammableModel, step: dict[str, Any], captures: dict[str, Any]) -> None:
    resolved = resolve_templates(value=step, captures=captures)
    program_model(pm=pm, step=resolved)

    # expectError: the turn throws (API misuse) rather than resolving with a
    # graceful finishReason='failed' output. Cover stream_bidi / send as well
    # as receive / output — an init rejection can surface before the stream.
    if 'expectError' in resolved:
        expect_err = _require_send_expect_error(value=resolved['expectError'])
        thrown: BaseException | None = None
        conn = None
        try:
            conn = await agent.stream_bidi(AgentInit.model_validate(resolved.get('init') or {}))
            for inp in resolved.get('inputs') or []:
                await conn.send(AgentInput.model_validate(inp))
            await conn.close()
            async for _chunk in conn.receive():
                pass
            await conn.output()
        except Exception as e:  # noqa: BLE001 - spec asserts on the raised error
            thrown = e
        finally:
            await _close_quietly(conn=conn)
        _assert_expect_error(thrown=thrown, expect_err=expect_err)
        return

    conn = None
    try:
        conn = await agent.stream_bidi(AgentInit.model_validate(resolved.get('init') or {}))
        for inp in resolved.get('inputs') or []:
            await conn.send(AgentInput.model_validate(inp))
        await conn.close()

        chunks = [c async for c in conn.receive()]
        output = await conn.output()
    finally:
        await _close_quietly(conn=conn)
    out = dump(model=output)

    if resolved.get('expectChunks') is not None:
        assert_chunks(actual_chunks=chunks, expected_chunks=resolved['expectChunks'])

    if resolved.get('expectOutput') is not None:
        assert_output(out=out, expect=resolved['expectOutput'])

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


async def execute_get_snapshot_data(*, agent: Agent, step: dict[str, Any], captures: dict[str, Any]) -> None:
    resolved = resolve_templates(value=step, captures=captures)
    snapshot_id = resolved.get('snapshotId')
    session_id = resolved.get('sessionId')
    assert bool(snapshot_id) != bool(session_id), (
        'getSnapshotData step requires exactly one of snapshotId or sessionId'
    )

    if 'expectError' in resolved:
        expect_err = _require_lookup_expect_error(value=resolved['expectError'])
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

    if resolved.get('expectSnapshot') is not None:
        assert_snapshot(snap=dump(model=snap), expect=resolved['expectSnapshot'])


async def execute_abort(*, agent: Agent, step: dict[str, Any], captures: dict[str, Any]) -> None:
    resolved = resolve_templates(value=step, captures=captures)
    snapshot_id = resolved.get('snapshotId')
    assert snapshot_id, 'abort step requires snapshotId'

    previous = await agent.abort_snapshot_data(snapshot_id)
    previous_str = previous.value if previous is not None else None

    # The key being present (even as YAML ~ / null) means we should assert.
    if 'expectPreviousStatus' in resolved:
        expected = resolved['expectPreviousStatus']
        assert previous_str == expected, f'Expected previous status {expected!r}, got {previous_str!r}'


async def execute_wait_until_completed(*, agent: Agent, step: dict[str, Any], captures: dict[str, Any]) -> None:
    resolved = resolve_templates(value=step, captures=captures)
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
        assert_snapshot(snap=dump(model=snap), expect=resolved['expectSnapshot'])


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('spec_test', SPEC_TESTS, ids=[t['name'] for t in SPEC_TESTS])
async def test_agent_conformance(spec_test: dict[str, Any]) -> None:
    harness = setup_harness()
    agent = harness.agents.get(spec_test['agent'])
    assert agent is not None, f'Unknown agent {spec_test["agent"]!r} in test {spec_test["name"]!r}'

    captures: dict[str, Any] = {}

    for i, step in enumerate(spec_test['steps']):
        label = f'step[{i}]'
        try:
            if not isinstance(step, dict):
                raise AssertionError(f'step must be a mapping, got {type(step).__name__}')
            step_type = step.get('type')
            label = f'step[{i}] ({step_type})'
            if step_type == 'send':
                await execute_send(agent=agent, pm=harness.pm, step=step, captures=captures)
            elif step_type == 'getSnapshotData':
                await execute_get_snapshot_data(agent=agent, step=step, captures=captures)
            elif step_type == 'abort':
                await execute_abort(agent=agent, step=step, captures=captures)
            elif step_type == 'waitUntilCompleted':
                await execute_wait_until_completed(agent=agent, step=step, captures=captures)
            else:
                raise AssertionError(f'Unknown step type: {step_type!r}')
        except Exception as e:
            raise AssertionError(f'{label} in test {spec_test["name"]!r} failed: {e}') from e


def test_assert_contains_none_is_noop() -> None:
    """A null expected value means "not specified", matching the other harnesses."""
    assert_contains(actual={'x': 1}, expected=None)
    assert_contains(actual=None, expected=None)
    assert_contains(actual=[1, 2], expected=None)


def test_tests_from_suite_rejects_non_mapping() -> None:
    with pytest.raises(AssertionError, match='must be a mapping, got list'):
        _tests_from_suite(suite=['foo'])
    with pytest.raises(AssertionError, match='must be a mapping, got bool'):
        _tests_from_suite(suite=False)
    with pytest.raises(AssertionError, match='contains no tests'):
        _tests_from_suite(suite={})
    with pytest.raises(AssertionError, match=r'tests\[0\] must be a mapping'):
        _tests_from_suite(suite={'tests': ['foo']})
    with pytest.raises(AssertionError, match=r'steps\[0\] must be a mapping'):
        _tests_from_suite(suite={'tests': [{'name': 't', 'agent': 'a', 'steps': ['send']}]})


def test_resolve_templates_inline_object_is_json() -> None:
    got = resolve_templates(
        value='seeded-{{state1}}',
        captures={'state1': {'sessionId': 'abc', 'custom': {'counter': 1}}},
    )
    assert got == 'seeded-{"sessionId":"abc","custom":{"counter":1}}'


def test_assert_expect_error_matches_message_not_status_prefix() -> None:
    err = GenkitError(status='FAILED_PRECONDITION', message="Cannot send 'state' to agent")
    _assert_expect_error(
        thrown=err,
        expect_err={'status': 'FAILED_PRECONDITION', 'message': "Cannot send 'state'"},
    )
    with pytest.raises(AssertionError, match='error.message'):
        _assert_expect_error(thrown=err, expect_err={'message': 'FAILED_PRECONDITION'})


def test_send_expect_error_rejects_string() -> None:
    with pytest.raises(AssertionError, match='must be a mapping'):
        _require_send_expect_error(value="Cannot send 'state' to agent")
    with pytest.raises(AssertionError, match='must include status and/or message'):
        _require_send_expect_error(value={})
    with pytest.raises(AssertionError, match='status must be a string'):
        _require_send_expect_error(value={'status': None})


def test_lookup_expect_error_rejects_mapping() -> None:
    with pytest.raises(AssertionError, match='must be a non-empty string'):
        _require_lookup_expect_error(value={'status': 'NOT_FOUND', 'message': 'not found'})
    with pytest.raises(AssertionError, match='must be a non-empty string'):
        _require_lookup_expect_error(value='')


def test_thrown_message_skips_status_prefix() -> None:
    err = GenkitError(status='NOT_FOUND', message='branching session')
    assert _thrown_message(thrown=err) == 'branching session'
    assert 'NOT_FOUND' not in _thrown_message(thrown=err)


def test_snapshot_error_contains_is_exact_message() -> None:
    """Snapshot errorContains uses subset matching; a string field is exact."""
    wrapped = {'error': {'message': 'background: intentional failure (wrapped)'}}
    with pytest.raises(AssertionError, match='snapshot.error.message'):
        assert_snapshot(snap=wrapped, expect={'errorContains': {'message': 'intentional failure'}})
    assert_snapshot(
        snap=wrapped,
        expect={'errorContains': {'message': 'background: intentional failure (wrapped)'}},
    )
    assert_snapshot(
        snap={'error': {'status': 'INTERNAL', 'message': 'intentional failure'}},
        expect={'errorContains': {'message': 'intentional failure'}},
    )


def test_assert_chunks_pins_null_finish_reason() -> None:
    chunks = [AgentStreamChunk(turn_end=TurnEnd(finish_reason=AgentFinishReason.STOP))]
    with pytest.raises(AssertionError, match='finishReason'):
        assert_chunks(actual_chunks=chunks, expected_chunks=[{'turnEnd': {'finishReason': None}}])
    assert_chunks(actual_chunks=chunks, expected_chunks=[{'turnEnd': {}}])

    omitted = [AgentStreamChunk(turn_end=TurnEnd())]
    with pytest.raises(AssertionError, match='finishReason null'):
        assert_chunks(actual_chunks=omitted, expected_chunks=[{'turnEnd': {'finishReason': None}}])

    explicit_null = [AgentStreamChunk(turn_end=TurnEnd.model_validate({'finishReason': None}))]
    assert_chunks(actual_chunks=explicit_null, expected_chunks=[{'turnEnd': {'finishReason': None}}])
