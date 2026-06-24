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
from typing import Any, Generic, TypeVar
from uuid import uuid4

from genkit._ai._agents._helpers import (
    agent_input_has_payload,
    emit_interrupt_tool_chunk,
    persist_turn_messages,
    to_agent_finish_reason,
    to_error_details,
)
from genkit._ai._agents._session import (
    Session,
    SessionStore,
    SnapshotAborter,
    SnapshotCallback,
    SnapshotContext,
    StateT,
    assert_valid_session_id,
    run_with_session,
)
from genkit._ai._agents._types import ClientTransform, TurnResult
from genkit._ai._generate import generate_action
from genkit._ai._json_patch import diff_json
from genkit._ai._model import ModelResponseChunk as StreamModelResponseChunk
from genkit._core._action import ActionRunContext, get_current_context
from genkit._core._channel import CloseableQueue, QueueShutDown
from genkit._core._error import GenkitError, StatusCodes
from genkit._core._model import GenerateActionOptions, ModelResponseChunk
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
    JsonPatch,
    JsonPatchOperation,
    MessageData,
    RuntimeError as GenkitRuntimeError,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
    TurnEnd,
)

StreamT = TypeVar('StreamT')
InT = TypeVar('InT')
OutT = TypeVar('OutT')
StreamOutT = TypeVar('StreamOutT')
StreamInT = TypeVar('StreamInT')

# IntakeQueueItem — incoming payload sent by client (message, resume, or detach).
IntakeQueueItem = AgentInput

# BidiInQueueItem — raw intake payload received on the low-level BidiAction queue.
BidiInQueueItem = AgentInput

# StreamQueueItem — outbound streaming chunk (metadata, custom state, message chunk).
StreamQueueItem = AgentStreamChunk


# ---------------------------------------------------------------------------
# SessionRunner
# ---------------------------------------------------------------------------


class SessionRunner(Generic[StateT]):
    """Per-turn input loop for one agent invocation.

    ``AgentFn`` calls ``sess.run(handle_turn)``; after each turn the runtime
    emits snapshots and ``turnEnd`` chunks via ``on_end_turn``.
    """

    def __init__(
        self,
        session: Session[StateT],
        turn_inputs: CloseableQueue[IntakeQueueItem],
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


async def load_session(
    init: AgentInit,
    store: SessionStore | None,
    *,
    agent_name: str = '',
) -> tuple[Session[Any], SessionSnapshot | None]:
    """Construct a Session from AgentInit payload.

    Server-managed (store set): resume via snapshot_id or session_id.
    Client-managed (no store): use init.state or start fresh.
    """
    name = agent_name or 'agent'

    if init.snapshot_id and init.session_id:
        raise GenkitError(
            status=StatusCodes.INVALID_ARGUMENT,
            message=(f"Cannot send both 'snapshot_id' and 'session_id' to agent '{name}'. Provide exactly one."),
        )
    if (init.snapshot_id or init.session_id) and store is None:
        field = 'snapshot_id' if init.snapshot_id else 'session_id'
        raise GenkitError(
            status=StatusCodes.FAILED_PRECONDITION,
            message=(
                f"Cannot use '{field}' with agent '{name}': this agent has no "
                "store configured (client-managed state). Send 'state' instead."
            ),
        )
    if init.state is not None and store is not None:
        raise GenkitError(
            status=StatusCodes.FAILED_PRECONDITION,
            message=(
                f"Cannot send 'state' to agent '{name}': this agent uses a "
                "server-managed store. Send 'snapshot_id' or 'session_id' instead."
            ),
        )

    if store is not None and init.snapshot_id:
        snap = await store.get_snapshot(snapshot_id=init.snapshot_id)
        if snap is None:
            raise GenkitError(
                status=StatusCodes.NOT_FOUND,
                message=f'Snapshot {init.snapshot_id!r} not found',
            )
        return Session(initial_state=snap.state), snap

    session_id = init.session_id
    if store is not None and not session_id:
        session_id = str(uuid4())

    if store is not None and session_id:
        assert_valid_session_id(session_id)
        snap = await store.get_snapshot(session_id=session_id)
        if snap is not None:
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
        snapshot_callback: SnapshotCallback | None,
        client_transform: ClientTransform | None,
        session_outputs: CloseableQueue[StreamQueueItem],
    ) -> None:
        self.name = name
        self.session = session
        self.store = store
        self.snapshot_callback = snapshot_callback
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

        self.sess = SessionRunner(
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
                JsonPatchOperation(op='replace', path='', value=copy.deepcopy(transformed))
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
    ) -> str:
        """Save snapshot if store configured + callback approves + state changed."""
        if self.store is None:
            return ''
        if not force and self.last_snapshot is not None and self.session.version == self.last_snapshot_version:
            return self.last_snapshot.snapshot_id

        state = await self.session.state()

        if self.snapshot_callback is not None and status != SnapshotStatus.FAILED:
            ctx = SnapshotContext(
                state=state,
                prev_state=self.last_snapshot.state if self.last_snapshot else None,
                turn_index=self.sess.turn_index,
            )
            if not self.snapshot_callback(ctx):
                return self.last_snapshot.snapshot_id if self.last_snapshot else ''

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
        return self.last_snapshot.snapshot_id if self.last_snapshot else ''

    async def ensure_recovery_snapshot(self) -> str | None:
        """Persist last-good state after a failed turn when the callback skipped it."""
        if self.store is None or self.sess.last_good_state is None:
            return self.last_snapshot.snapshot_id if self.last_snapshot else None

        if (
            self.sess.last_good_state_version is not None
            and self.sess.last_good_state_version == self.last_snapshot_version
        ):
            return self.last_snapshot.snapshot_id if self.last_snapshot else None

        if self.sess.turn_index == 0:
            return None

        parent_id = self.last_snapshot.snapshot_id if self.last_snapshot else None
        now = datetime.now(timezone.utc).isoformat()
        last_good = self.sess.last_good_state

        def recovery(existing: SessionSnapshot | None) -> SessionSnapshot | None:
            if existing is not None and existing.status == SnapshotStatus.ABORTED:
                return None
            return SessionSnapshot(
                snapshot_id='',
                parent_id=parent_id or '',
                status=SnapshotStatus.COMPLETED,
                state=last_good,
                created_at=now,
                finish_reason=self.sess.last_good_finish_reason,
            )

        snap = await self.store.save_snapshot(None, recovery)
        if snap is not None:
            self.last_snapshot = snap
            if self.sess.last_good_state_version is not None:
                self.last_snapshot_version = self.sess.last_good_state_version
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
            error=self.sess.last_turn_error if is_failed else None,
            force=is_failed,
        )
        self.send_chunk(
            AgentStreamChunk(
                turn_end=TurnEnd(
                    snapshot_id=snapshot_id or None,
                    finish_reason=finish_reason,
                )
            )
        )

    async def failed_agent_output(self, result: AgentResult | None) -> AgentOutput:
        last_good = self.sess.last_good_state or await self.session.state()
        msgs = list(last_good.messages or [])
        out = AgentOutput(
            finish_reason=AgentFinishReason.FAILED,
            error=self.sess.last_turn_error,
            message=msgs[-1] if msgs else (result.message if result else None),
            artifacts=list(last_good.artifacts or []) if last_good.artifacts else (result.artifacts if result else []),
        )
        if self.store is not None:
            out.snapshot_id = await self.ensure_recovery_snapshot()
        else:
            out.state = self.transform_state(last_good)
        return out

    async def watch_snapshot_abort(self, snapshot_id: str, abort_signal: asyncio.Event) -> None:
        if self.store is None or not isinstance(self.store, SnapshotAborter):
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
                result.finish_reason if result and result.finish_reason else self.sess.last_turn_finish_reason
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
                error=(GenkitRuntimeError(status='INTERNAL', message=str(fn_err)) if fn_err else None),
                finish_reason=finish_reason,
                created_at=existing.created_at if existing else now,
            )

        try:
            await self.store.save_snapshot(pending_snap.snapshot_id, finalize)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001, S110
            pass  # best-effort; log in production

    async def run(self, fn: AgentFn, client_inputs: CloseableQueue[BidiInQueueItem]) -> AgentOutput:
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
            streaming_callback=self.send_chunk,
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
                    fn(self.sess, action_ctx),
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

        if self.sess.last_turn_finish_reason == AgentFinishReason.FAILED and self.sess.last_turn_error:
            return await self.failed_agent_output(result)

        if err_holder:
            raise err_holder[0]

        snapshot_id = await self.maybe_snapshot()
        if not snapshot_id and self.last_snapshot is not None:
            snapshot_id = self.last_snapshot.snapshot_id

        finish_reason = result.finish_reason if result else self.sess.last_turn_finish_reason
        out = AgentOutput(
            snapshot_id=snapshot_id or None,
            message=result.message if result else None,
            artifacts=list(result.artifacts) if result and result.artifacts else [],
            finish_reason=finish_reason,
        )

        if self.store is None:
            state = await self.session.state()
            out.state = self.transform_state(state)

        return out

    def send_chunk(self, chunk: StreamQueueItem) -> None:
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
    sess: SessionRunner,
    ctx: ActionRunContext,
    registry: Registry,
    gen_options: GenerateActionOptions,
    history: list[MessageData],
) -> TurnResult | None:
    """Run generate for one agent turn and persist session messages."""

    def on_chunk(chunk: StreamModelResponseChunk) -> None:
        wire_chunk = (
            chunk if isinstance(chunk, ModelResponseChunk) else ModelResponseChunk.model_validate(chunk.model_dump())
        )
        ctx.send_chunk(AgentStreamChunk(model_chunk=wire_chunk))

    response = await generate_action(
        registry,
        gen_options,
        on_chunk=on_chunk,
        abort_signal=ctx.abort_signal,
        context=ctx.context,
    )

    if response.finish_reason == FinishReason.INTERRUPTED:
        emit_interrupt_tool_chunk(ctx, response)
        await persist_turn_messages(
            sess,
            history,
            response.message,
            response=response,
        )
        return TurnResult(finish_reason=AgentFinishReason.INTERRUPTED)

    if response.message:
        await persist_turn_messages(
            sess,
            history,
            response.message,
            response=response,
        )

    # Return the turn result wrapping the model finish reason
    return TurnResult(finish_reason=to_agent_finish_reason(response.finish_reason))
