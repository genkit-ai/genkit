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

"""Core agent turn execution loop and runtime orchestrator."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar, cast
from uuid import uuid4

from pydantic import ValidationError

from genkit._ai._agents._session import (
    Session,
    SessionStore,
    SnapshotSubscriber,
    StateT,
    run_with_session,
)
from genkit._ai._agents._snapshot import walk_back_to_resumable
from genkit._ai._agents._types import ClientTransform, TurnResult
from genkit._ai._generate import generate_action
from genkit._ai._json_patch import diff_json
from genkit._core._action import ActionRunContext, StreamingCallback, get_current_context
from genkit._core._channel import CloseableQueue, QueueShutDown
from genkit._core._error import GenkitError
from genkit._core._model import GenerateActionOptions, Message, ModelResponse, ModelResponseChunk
from genkit._core._registry import Registry
from genkit._core._tracing import SpanMetadata, run_in_new_span
from genkit._core._typing import (
    AgentFinishReason,
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentResult,
    AgentStreamChunk,
    Artifact,
    FinishReason,
    GenkitRuntimeError,
    JsonPatch,
    JsonPatchOp,
    JsonPatchOperation,
    MessageData,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
    TurnEnd,
)

PREAMBLE_KEY = '_genkit_agent_preamble'

InT = TypeVar('InT')
OutT = TypeVar('OutT')
StreamOutT = TypeVar('StreamOutT')
StreamInT = TypeVar('StreamInT')


class SessionRunner(Generic[StateT]):
    """Per-turn input loop for one agent invocation.

    ``AgentFn`` calls ``session_runner.run(handle_turn)``; after each turn the runtime
    emits snapshots and ``turnEnd`` chunks via ``on_end_turn``.
    """

    def __init__(
        self,
        session: Session[StateT],
        turn_inputs: CloseableQueue[AgentInput],
        on_begin_turn: Callable[[], Awaitable[None]] | None = None,
        on_end_turn: Callable[[AgentFinishReason | None], Awaitable[None]] | None = None,
    ) -> None:
        self.session = session
        self.turn_inputs = turn_inputs
        self.on_begin_turn = on_begin_turn
        self.on_end_turn = on_end_turn
        self.turn_index: int = 0
        self.last_turn_finish_reason: AgentFinishReason | None = None
        self.last_turn_error: GenkitRuntimeError | None = None
        self.last_good_state: SessionState | None = None
        self.last_good_state_version: int | None = None
        self.last_good_finish_reason: AgentFinishReason | None = None

    async def seed_last_good_state(self) -> None:
        """Capture initial session state as the fallback for first-turn failures."""
        self.last_good_state = await self.session.state()
        self.last_good_state_version = self.session.version

    async def run(
        self,
        fn: Callable[[AgentInput], Awaitable[TurnResult | None]],
    ) -> None:
        """Consume inputs from the intake queue, calling fn for each turn.

        Each turn is wrapped in a trace span ``runTurn-N`` (1-based). Inbound
        messages are automatically added to the session before fn is called.
        After fn returns, on_end_turn is called (snapshot + chunk emission)
        and turn_index is incremented.

        Turn failures resolve gracefully: the invocation finishes with
        ``finish_reason=failed`` and ``error`` on ``AgentOutput``, and the
        turn loop stops.
        """
        async for inp in self.turn_inputs:
            # Auto-add inbound messages to session history
            if inp.message:
                await self.session.add_messages(inp.message)

            if self.on_begin_turn is not None:
                await self.on_begin_turn()

            span_meta = SpanMetadata(
                name=f'runTurn-{self.turn_index + 1}',
                type='flowStep',
                input=inp,
            )
            try:
                with run_in_new_span(span_meta):
                    turn_result = await fn(inp)
                    finish_reason = turn_result.finish_reason if turn_result else None
                    self.last_turn_finish_reason = finish_reason
                    self.last_turn_error = None

                    if self.on_end_turn is not None:
                        await self.on_end_turn(finish_reason)

                    span_meta.output = {'finishReason': finish_reason}

                self.last_good_state = await self.session.state()
                self.last_good_state_version = self.session.version
                self.last_good_finish_reason = self.last_turn_finish_reason
                self.turn_index += 1
            except BaseException as exc:
                self.last_turn_finish_reason = AgentFinishReason.FAILED
                self.last_turn_error = to_error_details(exc)

                if self.on_end_turn is not None:
                    await self.on_end_turn(AgentFinishReason.FAILED)

                break

    async def result(self) -> AgentResult:
        """Last message, artifacts, and finish reason from the current session."""
        state = await self.session.state()
        msg = state.messages[-1] if state.messages else None
        arts = list(state.artifacts) if state.artifacts else []
        return AgentResult(
            message=msg,
            artifacts=arts,
            finish_reason=self.last_turn_finish_reason,
        )

    # --- Session passthrough helpers ---

    async def get_messages(self) -> list[MessageData]:
        return await self.session.get_messages()

    async def set_messages(self, messages: list[MessageData]) -> None:
        await self.session.set_messages(messages)

    async def add_messages(self, *messages: MessageData) -> None:
        await self.session.add_messages(*messages)

    async def get_artifacts(self) -> list[Artifact]:
        return await self.session.get_artifacts()

    async def add_artifacts(self, *artifacts: Artifact) -> None:
        await self.session.add_artifacts(*artifacts)

    async def get_custom(self) -> StateT | None:
        return await self.session.get_custom()

    async def update_custom(self, fn: Callable[[StateT | None], StateT]) -> None:
        await self.session.update_custom(fn)


# AgentFn — custom agent entrypoint; receives SessionRunner + ActionRunContext.
AgentFn = Callable[
    [SessionRunner, ActionRunContext],
    Awaitable[AgentResult],
]

# BidiFunc — low-level bidi action handler (asyncio queues for in/out streams).
BidiFunc = Callable[
    [InT, asyncio.Queue[StreamInT], asyncio.Queue[StreamOutT]],
    Awaitable[OutT],
]


# ---------------------------------------------------------------------------
# AgentRuntime
# ---------------------------------------------------------------------------


def validate_custom_state(custom: Any, state_schema: type[Any] | None, agent_name: str) -> None:  # noqa: ANN401
    """Reject custom state that doesn't match the agent's declared shape.

    Runs at load time on the state about to seed a turn — whether it came from a
    snapshot or a client that shipped its own blob — so a malformed payload fails
    fast with a clear error instead of surfacing deep inside a tool. No-ops when
    no schema is declared, and skips a never-set state so a required field doesn't
    trip a session that simply hasn't written state yet.
    """
    if state_schema is None or custom is None or not hasattr(state_schema, 'model_validate'):
        return
    try:
        state_schema.model_validate(custom)
    except ValidationError as e:
        # Surface the per-field failures and the expected shape so the caller can
        # see exactly what was wrong, not just that something was.
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                f"Invalid custom state for agent '{agent_name}': {e.error_count()} schema validation error(s).\n{e}"
            ),
            details={
                'schema': state_schema.model_json_schema(),
                'errors': [{'loc': list(err['loc']), 'message': err['msg'], 'type': err['type']} for err in e.errors()],
            },
        ) from e


async def load_session(
    init: AgentInit,
    store: SessionStore | None,
    *,
    agent_name: str = '',
    state_schema: type[Any] | None = None,
) -> tuple[Session[Any], SessionSnapshot | None]:
    """Construct a Session from AgentInit payload.

    Server-managed (store set): resume via snapshot_id or session_id.
    Client-managed (no store): use init.state or start fresh.

    When ``state_schema`` is set the custom state loaded from a snapshot or the
    client is validated against it before the session is built.
    """
    name = agent_name or 'agent'

    if init.snapshot_id and init.session_id:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(f"Cannot send both 'snapshot_id' and 'session_id' to agent '{name}'. Provide exactly one."),
        )
    if (init.snapshot_id or init.session_id) and store is None:
        field = 'snapshot_id' if init.snapshot_id else 'session_id'
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=(
                f"Cannot use '{field}' with agent '{name}': this agent has no "
                "store configured (client-managed state). Send 'state' instead."
            ),
        )
    if init.state is not None and store is not None:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=(
                f"Cannot send 'state' to agent '{name}': this agent uses a "
                "server-managed store. Send 'snapshot_id' or 'session_id' instead."
            ),
        )

    if store is not None and init.snapshot_id:
        snap = await store.get_snapshot(snapshot_id=init.snapshot_id)
        if snap is None:
            raise GenkitError(
                status='NOT_FOUND',
                message=f'Snapshot {init.snapshot_id!r} not found',
            )
        # A failed/aborted/pending snapshot is kept for inspection but isn't a
        # valid place to continue a conversation from.
        if snap.status != SnapshotStatus.COMPLETED:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f'Snapshot {init.snapshot_id!r} is not resumable '
                    f'(status: {snap.status.value if snap.status else "unknown"}). '
                    "Only 'completed' snapshots can be resumed."
                ),
            )
        validate_custom_state(snap.state.custom if snap.state else None, state_schema, name)
        return Session(initial_state=snap.state), snap

    session_id = init.session_id
    if store is not None and not session_id:
        session_id = str(uuid4())

    if store is not None and session_id:
        # The latest leaf may be a failed/aborted/pending turn, which can't be
        # resumed — fall back to the last good snapshot behind it.
        snap = await walk_back_to_resumable(store, await store.get_snapshot(session_id=session_id))
        if snap is not None:
            validate_custom_state(snap.state.custom if snap.state else None, state_schema, name)
            return Session(initial_state=snap.state), snap
        return (
            Session(
                initial_state=SessionState(
                    session_id=session_id,
                    messages=[],
                    artifacts=[],
                )
            ),
            None,
        )

    if init.state is not None:
        validate_custom_state(init.state.custom, state_schema, name)
        return Session(initial_state=init.state), None

    return Session(), None


class AgentRuntime:
    """Drives the agent fn to completion; owns session, router, and intake."""

    def __init__(
        self,
        name: str,
        session: Session[Any],
        parent_snapshot: SessionSnapshot | None,
        store: SessionStore | None,
        client_transform: ClientTransform | None,
        session_outputs: CloseableQueue[AgentStreamChunk],
    ) -> None:
        self.name = name
        self.session = session
        self.store = store
        self.client_transform = client_transform
        self.last_snapshot: SessionSnapshot | None = parent_snapshot
        self.last_snapshot_version: int = self.session.version if parent_snapshot is not None else -1
        self.detached: bool = False
        self.first_custom_patch_in_turn: bool = True
        self.last_sent_custom: object | None = None  # Cache of last streamed custom state to compute JSON Patch deltas

        self.session_outputs = session_outputs

        # Separate turn inputs queue: runtime controls its lifecycle,
        # BidiAction's client_inputs is forwarded here by run().
        self.turn_inputs = CloseableQueue(maxsize=1)

        self.session_runner = SessionRunner(
            session,
            self.turn_inputs,
            on_begin_turn=self.reset_custom_patch_turn,
            on_end_turn=self.emit_turn_end,
        )

        session.on_custom_changed(self.emit_custom_patch)
        session.on_artifact_changed(self.emit_artifact)

    async def reset_custom_patch_turn(self) -> None:
        # Re-base clients that may not share the server's custom-state baseline.
        self.first_custom_patch_in_turn = True

    def transform_state(self, state: SessionState) -> SessionState | None:
        state_fn = self.client_transform.get('state') if self.client_transform else None
        if state_fn is None:
            return state
        return state_fn(state)

    def transform_chunk(self, chunk: AgentStreamChunk) -> AgentStreamChunk | None:
        chunk_fn = self.client_transform.get('chunk') if self.client_transform else None
        if chunk_fn is None:
            return chunk
        return chunk_fn(chunk)

    async def client_custom(self) -> object | None:
        state = await self.session.state()
        client_state = self.transform_state(state)
        return client_state.custom if client_state is not None else None

    async def emit_custom_patch(self) -> None:
        """Stream custom state updates to the client as JSON Patch deltas."""
        if self.detached:
            return

        transformed = await self.client_custom()
        if self.first_custom_patch_in_turn:
            # Send full state on the first patch of a turn to re-base the client.
            ops: list[JsonPatchOperation] = [
                JsonPatchOperation(op=JsonPatchOp.REPLACE, path='', value=copy.deepcopy(transformed))
            ]
            self.first_custom_patch_in_turn = False
        else:
            # Send only the diff against the last sent state on subsequent patches.
            ops = diff_json(self.last_sent_custom, transformed)

        self.last_sent_custom = copy.deepcopy(transformed)
        if not ops:
            return

        self.send_chunk(AgentStreamChunk(custom_patch=JsonPatch(root=ops)))

    async def emit_artifact(self, artifact: Artifact) -> None:
        if self.detached:
            return
        self.send_chunk(AgentStreamChunk(artifact=artifact))

    async def maybe_snapshot(
        self,
        *,
        finish_reason: AgentFinishReason | None = None,
        status: SnapshotStatus | None = None,
        error: GenkitRuntimeError | None = None,
        force: bool = False,
    ) -> str | None:
        """Persist a snapshot whenever a store is configured and state changed.

        With a store, every turn is persisted (no opt-out): the durable head
        always advances so a stateless resume never regresses to an older turn.
        """
        if self.store is None:
            return None
        if not force and self.last_snapshot is not None and self.session.version == self.last_snapshot_version:
            return self.last_snapshot.snapshot_id

        state = await self.session.state()

        parent_id = self.last_snapshot.snapshot_id if self.last_snapshot else None
        now = datetime.now(timezone.utc).isoformat()
        snap_status = status or SnapshotStatus.COMPLETED

        def make_snap(
            existing: SessionSnapshot | None,
        ) -> SessionSnapshot | None:
            if existing is not None and existing.status == SnapshotStatus.ABORTED:
                return None
            return SessionSnapshot(
                snapshot_id=existing.snapshot_id if existing else '',
                parent_id=parent_id or '',
                status=snap_status,
                state=state,
                created_at=existing.created_at if existing and existing.created_at else now,
                finish_reason=finish_reason,
                error=error,
            )

        snap = await self.store.save_snapshot(None, make_snap)
        if snap is not None:
            self.last_snapshot = snap
            self.last_snapshot_version = self.session.version
            return snap.snapshot_id
        return self.last_snapshot.snapshot_id if self.last_snapshot else None

    async def ensure_recovery_snapshot(self) -> str | None:
        """Persist last-good state after a failed turn when the callback skipped it."""
        if self.store is None or self.session_runner.last_good_state is None:
            return self.last_snapshot.snapshot_id if self.last_snapshot else None

        if (
            self.session_runner.last_good_state_version is not None
            and self.session_runner.last_good_state_version == self.last_snapshot_version
        ):
            return self.last_snapshot.snapshot_id if self.last_snapshot else None

        if self.session_runner.turn_index == 0:
            return None

        parent_id = self.last_snapshot.snapshot_id if self.last_snapshot else None
        now = datetime.now(timezone.utc).isoformat()
        last_good = self.session_runner.last_good_state

        def recovery(existing: SessionSnapshot | None) -> SessionSnapshot | None:
            if existing is not None and existing.status == SnapshotStatus.ABORTED:
                return None
            return SessionSnapshot(
                snapshot_id='',
                parent_id=parent_id or '',
                status=SnapshotStatus.COMPLETED,
                state=last_good,
                created_at=now,
                finish_reason=self.session_runner.last_good_finish_reason,
            )

        snap = await self.store.save_snapshot(None, recovery)
        if snap is not None:
            self.last_snapshot = snap
            if self.session_runner.last_good_state_version is not None:
                self.last_snapshot_version = self.session_runner.last_good_state_version
            return snap.snapshot_id
        return self.last_snapshot.snapshot_id if self.last_snapshot else None

    async def emit_turn_end(self, finish_reason: AgentFinishReason | None = None) -> None:
        """Called by SessionRunner after each turn: snapshot + TurnEnd chunk."""
        if self.detached:
            return
        is_failed = finish_reason == AgentFinishReason.FAILED
        snapshot_id = await self.maybe_snapshot(
            finish_reason=finish_reason,
            status=SnapshotStatus.FAILED if is_failed else SnapshotStatus.COMPLETED,
            error=self.session_runner.last_turn_error if is_failed else None,
            force=is_failed,
        )
        # turnEnd is just the boundary marker. The full session rides home on
        # the turn's AgentOutput, so a client never has to stitch state off a
        # mid-stream chunk.
        self.send_chunk(
            AgentStreamChunk(
                turn_end=TurnEnd(
                    snapshot_id=snapshot_id or None,
                    finish_reason=finish_reason,
                )
            )
        )

    async def failed_agent_output(self, result: AgentResult | None) -> AgentOutput:
        last_good = self.session_runner.last_good_state or await self.session.state()
        msgs = list(last_good.messages or [])
        out = AgentOutput(
            session_id=last_good.session_id,
            finish_reason=AgentFinishReason.FAILED,
            error=self.session_runner.last_turn_error,
            message=msgs[-1] if msgs else (result.message if result else None),
            artifacts=list(last_good.artifacts or []) if last_good.artifacts else (result.artifacts if result else []),
        )
        # Same split as a successful turn: client-managed gets the last-good state
        # inline, server-managed resumes by the last-good snapshot.
        if self.store is None:
            out.state = self.transform_state(last_good)
        else:
            out.snapshot_id = await self.ensure_recovery_snapshot()
        return out

    async def watch_snapshot_abort(self, snapshot_id: str, abort_signal: asyncio.Event) -> None:
        if self.store is None or not isinstance(self.store, SnapshotSubscriber):
            return
        q = await self.store.on_snapshot_status_change(snapshot_id)
        while True:
            status = await q.get()
            if status is None:
                return
            if status == SnapshotStatus.ABORTED:
                abort_signal.set()
                return

    async def finalize_detach(
        self,
        pending_snap: SessionSnapshot,
        fn_task: asyncio.Task,
        forward_task: asyncio.Task,
        err_holder: list[BaseException],
        result_holder: list[AgentResult],
    ) -> None:
        """Background task: wait for fn, then rewrite pending snapshot with final state."""
        await fn_task
        await forward_task

        state = await self.session.state()
        now = datetime.now(timezone.utc).isoformat()
        fn_err = err_holder[0] if err_holder else None
        if fn_err:
            finish_reason = AgentFinishReason.FAILED
        else:
            result = result_holder[0] if result_holder else None
            finish_reason = (
                result.finish_reason if result and result.finish_reason else self.session_runner.last_turn_finish_reason
            )

        def finalize(existing: SessionSnapshot | None) -> SessionSnapshot | None:
            # If already aborted by user, leave it.
            if existing is not None and existing.status == SnapshotStatus.ABORTED:
                return None
            return SessionSnapshot(
                snapshot_id=existing.snapshot_id if existing else '',
                parent_id=pending_snap.parent_id or '',
                status=SnapshotStatus.FAILED if fn_err else SnapshotStatus.COMPLETED,
                state=state,
                error=(to_error_details(fn_err) if fn_err else None),
                finish_reason=finish_reason,
                created_at=existing.created_at if existing else now,
            )

        try:
            await self.store.save_snapshot(pending_snap.snapshot_id, finalize)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001, S110
            pass  # best-effort; log in production

    async def run(self, fn: AgentFn, client_inputs: CloseableQueue[AgentInput]) -> AgentOutput:
        """Drive fn to completion, return AgentOutput.

        Two terminal paths (v1):
          1. fn completes normally -> invocation-end snapshot -> AgentOutput
          2. detach signal       -> pending snapshot -> AgentOutput(snapshot_id)
             background finalizer rewrites snapshot when fn finishes
        """
        detach_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        abort_signal = asyncio.Event()
        action_ctx = ActionRunContext(
            context=get_current_context(),
            streaming_callback=cast(StreamingCallback, self.send_chunk),
            abort_signal=abort_signal,
        )

        # Forward task: BidiAction client_inputs -> runtime intake.
        # Detects detach=True and signals via detach_future.
        async def forward_inbound_stream() -> None:
            is_detached = False
            try:
                async for item in client_inputs:
                    if getattr(item, 'detach', False):
                        is_detached = True
                        if not detach_future.done():
                            detach_future.set_result(None)

                        val = item

                        async def finish_detach_input(
                            payload: AgentInput | None = None,
                            *,
                            bound_item: AgentInput = val,
                        ) -> None:
                            p = payload if payload is not None else bound_item
                            if agent_input_has_payload(p):
                                try:
                                    await self.turn_inputs.put(p)
                                except QueueShutDown:
                                    pass
                            self.turn_inputs.close()

                        asyncio.create_task(finish_detach_input())
                        return
                    await self.turn_inputs.put(item)
            finally:
                # Synchronously close self.turn_inputs when client_inputs terminates normally,
                # ensuring the SessionRunner's active turn loop exits cleanly.
                # If we are detaching, the background task will close it after writing the payload.
                if not is_detached:
                    self.turn_inputs.close()

        forward_task = asyncio.create_task(forward_inbound_stream())

        result_holder: list[AgentResult] = []
        err_holder: list[BaseException] = []

        async def run_agent_loop() -> None:
            try:
                result = await run_with_session(
                    self.session,
                    fn(self.session_runner, action_ctx),
                )
                result_holder.append(result)
            except Exception as e:  # noqa: BLE001
                err_holder.append(e)
            finally:
                # Synchronously close self.turn_inputs to signal turn completion,
                # letting SessionRunner stop waiting for more inputs.
                self.turn_inputs.close()

        fn_task = asyncio.create_task(run_agent_loop())

        # Wait for fn completion OR detach signal, whichever comes first.
        done, _ = await asyncio.wait(
            {fn_task, asyncio.ensure_future(asyncio.shield(detach_future))},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # --- Detach path ---
        if detach_future.done():
            if self.store is None:
                # Detach without a store is a config error; signal abort and raise.
                abort_signal.set()
                await fn_task
                await forward_task
                raise ValueError(
                    f"Agent '{self.name}' received a detach request, but cannot proceed because detach "
                    'requires a session store. Please configure a session store to enable client-detached '
                    'background execution.'
                )

            parent_id = self.last_snapshot.snapshot_id if self.last_snapshot else None
            now = datetime.now(timezone.utc).isoformat()
            state = await self.session.state()

            def pending(_: SessionSnapshot | None) -> SessionSnapshot | None:
                return SessionSnapshot(
                    snapshot_id='',
                    parent_id=parent_id or '',
                    status=SnapshotStatus.PENDING,
                    state=state,
                    created_at=now,
                )

            pending_snap = await self.store.save_snapshot(None, pending)
            if pending_snap is None:
                raise ValueError(
                    f"Agent '{self.name}' failed to persist the initial 'PENDING' recovery snapshot "
                    'during the detach flow. The turn execution has been aborted.'
                )

            # Stop sending chunks to the (now-gone) client.
            # Background task finalizes snapshot when fn finishes.
            self.detached = True
            asyncio.create_task(self.watch_snapshot_abort(pending_snap.snapshot_id, abort_signal))
            asyncio.create_task(self.finalize_detach(pending_snap, fn_task, forward_task, err_holder, result_holder))
            return AgentOutput(
                snapshot_id=pending_snap.snapshot_id,
                finish_reason=AgentFinishReason.DETACHED,
            )

        # --- Normal completion path ---
        await fn_task
        if not forward_task.done():
            forward_task.cancel()
            try:
                await forward_task
            except asyncio.CancelledError:
                pass

        result = result_holder[0] if result_holder else None

        if (
            self.session_runner.last_turn_finish_reason == AgentFinishReason.FAILED
            and self.session_runner.last_turn_error
        ):
            return await self.failed_agent_output(result)

        if err_holder:
            raise err_holder[0]

        snapshot_id = await self.maybe_snapshot()
        if not snapshot_id and self.last_snapshot is not None:
            snapshot_id = self.last_snapshot.snapshot_id

        finish_reason = result.finish_reason if result else self.session_runner.last_turn_finish_reason
        state = await self.session.state()
        out = AgentOutput(
            session_id=state.session_id,
            snapshot_id=snapshot_id or None,
            message=result.message if result else None,
            artifacts=list(result.artifacts) if result and result.artifacts else [],
            finish_reason=finish_reason,
        )
        # Client-managed has no store, so the client is the source of truth: ship
        # the whole session and let it copy verbatim. Server-managed returns only
        # the snapshot id — the durable store is the real history, and the client
        # tracks a lightweight running view from the final reply.
        if self.store is None:
            out.state = self.transform_state(state)
        return out

    def send_chunk(self, chunk: AgentStreamChunk) -> None:
        transformed = self.transform_chunk(chunk)
        if transformed is not None:
            try:
                self.session_outputs.put_nowait(transformed)
            except QueueShutDown:
                pass


# ---------------------------------------------------------------------------
# Prompt Agent Orchestration Helper
# ---------------------------------------------------------------------------


async def generate_prompt_agent_turn(
    *,
    session_runner: SessionRunner,
    ctx: ActionRunContext,
    registry: Registry,
    gen_options: GenerateActionOptions,
    history: list[MessageData],
) -> TurnResult | None:
    """Run generate for one agent turn and persist session messages."""

    def on_chunk(chunk: ModelResponseChunk) -> None:
        ctx.send_chunk(AgentStreamChunk(model_chunk=chunk))

    response = await generate_action(
        registry,
        gen_options,
        on_chunk=on_chunk,
        abort_signal=ctx.abort_signal,
        context=ctx.context,
    )

    if response.finish_reason == FinishReason.INTERRUPTED:
        await persist_turn_messages(
            session_runner,
            history,
            response.message,
            response=response,
        )
        return TurnResult(finish_reason=AgentFinishReason.INTERRUPTED)

    if response.message:
        await persist_turn_messages(
            session_runner,
            history,
            response.message,
            response=response,
        )

    # Return the turn result wrapping the model finish reason
    return TurnResult(finish_reason=to_agent_finish_reason(response.finish_reason))


# ---------------------------------------------------------------------------
# Internal Helper Functions
# ---------------------------------------------------------------------------


def to_error_details(exc: BaseException) -> GenkitRuntimeError:
    status = getattr(exc, 'status', None) or 'INTERNAL'
    message = str(exc) or 'Internal failure'
    details = getattr(exc, 'detail', None) or getattr(exc, 'details', None)
    if details is None and not isinstance(exc, GenkitError):
        details = str(exc)
    return GenkitRuntimeError(status=str(status), message=message, details=details)


def to_agent_finish_reason(fr: FinishReason | None) -> AgentFinishReason | None:
    if fr is None:
        return None
    for reason in AgentFinishReason:
        if reason.value == fr.value:
            return reason
    return AgentFinishReason.UNKNOWN


def coerce_message(msg: MessageData) -> Message:
    return msg if isinstance(msg, Message) else Message.model_validate(msg.model_dump())


async def persist_turn_messages(
    session_runner: SessionRunner,
    history: list[MessageData],
    response_message: MessageData | Message | None,
    *,
    response: ModelResponse | None = None,
) -> None:
    if response is not None and response.request is not None and response.request.messages:
        clean: list[MessageData] = []
        for m in response.request.messages:
            meta = getattr(m, 'metadata', None) or {}
            if meta.get(PREAMBLE_KEY):
                continue
            clean.append(coerce_message(m))
        if response_message is not None:
            clean.append(coerce_message(response_message))
        await session_runner.set_messages(clean)
        return

    if response_message is None:
        return

    clean_history: list[MessageData] = [coerce_message(m) for m in history]
    clean_history = [m for m in clean_history if not (m.metadata or {}).get(PREAMBLE_KEY)]
    clean_history.append(coerce_message(response_message))
    await session_runner.set_messages(clean_history)


def agent_input_has_payload(inp: AgentInput) -> bool:
    """True when ``AgentInput`` carries turn data beyond a detach directive."""
    if inp.message:
        return True
    if inp.resume is not None:
        if inp.resume.restart:
            return True
        if inp.resume.respond:
            return True
    return False
