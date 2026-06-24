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

"""Genkit agents: runtime, registration, and bidi connection API."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic, TypedDict, TypeVar
from uuid import uuid4

from opentelemetry import trace as trace_api

from genkit._ai._agents._client import AgentSession
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
from genkit._ai._agents._transports._inprocess import InProcessTransport
from genkit._ai._generate import generate_action
from genkit._ai._json_patch import _deep_equal, diff_json
from genkit._ai._model import ModelResponseChunk as StreamModelResponseChunk
from genkit._ai._prompt import (
    ExecutablePrompt,
    PromptGenerateOptions,
    _prepare,
    lookup_prompt,
    register_prompt_actions,
)
from genkit._ai._tools import Tool
from genkit._core._action import (
    Action,
    ActionKind,
    ActionRunContext,
    BidiAction,
    get_current_context,
)
from genkit._core._channel import CloseableQueue, QueueShutDown
from genkit._core._error import GenkitError, StatusCodes
from genkit._core._middleware import BaseMiddleware
from genkit._core._model import GenerateActionOptions, Message, ModelConfig, ModelResponse
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
    MiddlewareRef,
    ModelResponseChunk,
    Part,
    Resume,
    Role,
    RuntimeError as GenkitRuntimeError,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
    ToolRequestPart,
    TurnEnd,
)

# ---------------------------------------------------------------------------
# TypeVars
# ---------------------------------------------------------------------------

StreamT = TypeVar('StreamT')
InT = TypeVar('InT')
OutT = TypeVar('OutT')
StreamOutT = TypeVar('StreamOutT')
StreamInT = TypeVar('StreamInT')

# Bidi/intake queue items carry pure payloads; closure is signaled via exceptions.
BidiInQueueItem = AgentInput
StreamQueueItem = AgentStreamChunk
IntakeQueueItem = AgentInput


class ToolRequestRecord(TypedDict, total=False):
    """Normalized tool-request fields extracted from session history."""

    name: str | None
    ref: str | None
    input: object | None


# ---------------------------------------------------------------------------
# TurnResult (runtime-only — per-turn finish reason for SessionRunner)
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """Optional return from a per-turn handler; surfaces generate finish reason."""

    finish_reason: AgentFinishReason | None = None


def to_agent_finish_reason(fr: FinishReason | None) -> AgentFinishReason | None:
    if fr is None:
        return None
    for reason in AgentFinishReason:
        if reason.value == fr.value:
            return reason
    return AgentFinishReason.UNKNOWN


def tool_request_parts(message: MessageData | None) -> list[Part]:
    parts: list[Part] = []
    if message is None:
        return parts
    for part in message.content or []:
        p = part if isinstance(part, Part) else Part.model_validate(part)
        if isinstance(p.root, ToolRequestPart):
            parts.append(p)
    return parts


def to_error_details(exc: BaseException) -> GenkitRuntimeError:
    status = getattr(exc, 'status', None) or 'INTERNAL'
    message = str(exc) or 'Internal failure'
    details = getattr(exc, 'detail', None) or getattr(exc, 'details', None)
    if details is None and not isinstance(exc, GenkitError):
        details = str(exc)
    return GenkitRuntimeError(status=str(status), message=message, details=details)


def collect_tool_requests_from_history(history: list[MessageData]) -> list[ToolRequestRecord]:
    found: list[ToolRequestRecord] = []
    for msg in history:
        role = getattr(msg, 'role', None)
        if role not in (Role.MODEL, 'model'):
            continue
        for part in msg.content or []:
            root = getattr(part, 'root', part)
            tr = getattr(root, 'tool_request', None)
            if tr is None and isinstance(part, dict):
                tr = part.get('toolRequest') or part.get('tool_request')
            if tr is None:
                continue
            if hasattr(tr, 'model_dump'):
                tr_dict: ToolRequestRecord = tr.model_dump(by_alias=True, exclude_none=True)
            elif isinstance(tr, dict):
                tr_dict = tr  # type: ignore[assignment]
            else:
                tr_record: ToolRequestRecord = {
                    'name': getattr(tr, 'name', None),
                    'ref': getattr(tr, 'ref', None),
                    'input': getattr(tr, 'input', None),
                }
                tr_dict = tr_record
            found.append(tr_dict)
    return found


def validate_resume_against_history(resume: Resume, history: list[MessageData]) -> None:
    """Ensure resume entries reference tool requests present in session history."""
    all_tool_requests = collect_tool_requests_from_history(history)

    for restart in resume.restart or []:
        tr = restart.tool_request
        name = tr.name if tr else None
        ref = tr.ref if tr else None
        inp = tr.input if tr else None
        match = next((r for r in all_tool_requests if r.get('name') == name and r.get('ref') == ref), None)
        if match is None:
            ref_suffix = f' (ref: {ref})' if ref else ''
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f"resume.restart references tool '{name}'{ref_suffix} which was not found in session history.",
            )
        if not _deep_equal(inp, match.get('input')):
            ref_suffix = f' (ref: {ref})' if ref else ''
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f"resume.restart for tool '{name}'{ref_suffix} has modified inputs that do not match "
                    'the original tool request in session history.'
                ),
            )

    for respond in resume.respond or []:
        tr = respond.tool_response
        name = tr.name if tr else None
        ref = tr.ref if tr else None
        match = next((r for r in all_tool_requests if r.get('name') == name and r.get('ref') == ref), None)
        if match is None:
            ref_suffix = f' (ref: {ref})' if ref else ''
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f"resume.respond references tool '{name}'{ref_suffix} which was not found in session history.",
            )


def _emit_interrupt_tool_chunk(ctx: ActionRunContext, response: ModelResponse) -> None:
    parts = tool_request_parts(response.message)
    if not parts:
        return
    ctx.send_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(role=Role.TOOL, content=parts),
        )
    )


_HISTORY_TAG = '_genkit_history'
# Intra-turn scratch tag only — stripped before anything persists or
# crosses the wire, so it's free to be snake_case. Distinct from the
# chat/session path's `preamble` marker so the two preamble systems
# can't strip each other's messages.
_PREAMBLE_KEY = 'agent_preamble'


def _coerce_message(msg: MessageData) -> Message:
    return msg if isinstance(msg, Message) else Message.model_validate(msg.model_dump())


def _message_with_metadata(msg: MessageData, metadata: dict[str, object]) -> Message:
    base = _coerce_message(msg)
    merged = {**(base.metadata or {}), **metadata}
    return base.model_copy(update={'metadata': merged})


def _message_without_metadata_key(msg: MessageData, key: str) -> Message:
    base = _coerce_message(msg)
    if not base.metadata or key not in base.metadata:
        return base
    remaining = {k: v for k, v in base.metadata.items() if k != key}
    return base.model_copy(update={'metadata': remaining or None})


def tag_history_for_render(messages: list[MessageData]) -> list[Message]:
    """Mark session messages so prompt render can tell them apart from template output."""
    return [_message_with_metadata(m, {_HISTORY_TAG: True}) for m in messages]


def apply_preamble_tags(messages: Sequence[MessageData]) -> list[Message]:
    """After render: tag prompt-template messages and strip the internal history marker."""
    tagged: list[Message] = []
    for msg in messages:
        meta = getattr(msg, 'metadata', None) or {}
        if meta.get(_HISTORY_TAG):
            tagged.append(_message_without_metadata_key(msg, _HISTORY_TAG))
        else:
            tagged.append(_message_with_metadata(msg, {_PREAMBLE_KEY: True}))
    return tagged


async def _persist_turn_messages(
    sess: SessionRunner,
    history: list[MessageData],
    response_message: MessageData | Message | None,
    *,
    strip_preamble: bool = False,
    response: ModelResponse | None = None,
) -> None:
    if strip_preamble and response is not None and response.request is not None and response.request.messages:
        clean: list[MessageData] = []
        for m in response.request.messages:
            meta = getattr(m, 'metadata', None) or {}
            if meta.get(_PREAMBLE_KEY):
                continue
            clean.append(_coerce_message(m))
        if response_message is not None:
            clean.append(_coerce_message(response_message))
        await sess.set_messages(clean)
        return

    if response_message is None:
        return

    clean_history: list[MessageData] = [_coerce_message(m) for m in history]
    if strip_preamble:
        clean_history = [m for m in clean_history if not (m.metadata or {}).get(_PREAMBLE_KEY)]
    clean_history.append(_coerce_message(response_message))
    await sess.set_messages(clean_history)


# StateTransform — redact or reshape session state before it leaves the server.
StateTransform = Callable[[SessionState], SessionState | None]

# ChunkTransform — reshape or drop a stream chunk before it reaches the client.
ChunkTransform = Callable[[AgentStreamChunk], AgentStreamChunk | None]


class ClientTransform(TypedDict, total=False):
    """Project server-side agent data onto the client-visible view."""

    state: StateTransform
    chunk: ChunkTransform


def resolve_client_transform(
    *,
    client_transform: ClientTransform | None = None,
    transform: StateTransform | None = None,
) -> ClientTransform | None:
    """``transform`` is shorthand for ``client_transform={'state': transform}``."""
    if client_transform is not None:
        return client_transform
    if transform is not None:
        return {'state': transform}
    return None


# AgentFn — custom agent entrypoint; receives SessionRunner + ActionRunContext.
AgentFn = Callable[
    ['SessionRunner', ActionRunContext],
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
        self.last_sent_custom: object | None = None

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
        if self.detached:
            return

        transformed = await self.client_custom()
        if self.first_custom_patch_in_turn:
            ops: list[JsonPatchOperation] = [
                JsonPatchOperation(op='replace', path='', value=copy.deepcopy(transformed))
            ]
            self.first_custom_patch_in_turn = False
        else:
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

        def _make_snap(
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

        snap = await self.store.save_snapshot(None, _make_snap)
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

        def _recovery(existing: SessionSnapshot | None) -> SessionSnapshot | None:
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

        snap = await self.store.save_snapshot(None, _recovery)
        if snap is not None:
            self.last_snapshot = snap
            if self.sess.last_good_state_version is not None:
                self.last_snapshot_version = self.sess.last_good_state_version
            return snap.snapshot_id
        return self.last_snapshot.snapshot_id if self.last_snapshot else None

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

    def send_chunk(self, chunk: object) -> None:
        """Forward chunk to client."""
        if not isinstance(chunk, AgentStreamChunk):
            return
        if self.detached:
            return
        wire_chunk = self.transform_chunk(chunk)
        if wire_chunk is None:
            return
        self.session_outputs.put_nowait(wire_chunk)

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

        def _finalize(existing: SessionSnapshot | None) -> SessionSnapshot | None:
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
            await self.store.save_snapshot(pending_snap.snapshot_id, _finalize)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001, S110
            pass  # best-effort; log in production

    async def run(self, fn: AgentFn, client_inputs: CloseableQueue[BidiInQueueItem]) -> AgentOutput:
        """Drive fn to completion, return AgentOutput.

        Two terminal paths (v1):
          1. fn completes normally → invocation-end snapshot → AgentOutput
          2. detach signal       → pending snapshot → AgentOutput(snapshot_id)
             background finalizer rewrites snapshot when fn finishes
        """
        detach_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        abort_signal = asyncio.Event()
        action_ctx = ActionRunContext(
            context=get_current_context(),
            streaming_callback=self.send_chunk,
            abort_signal=abort_signal,
        )

        # Forward task: BidiAction client_inputs → runtime intake.
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

                        async def _finish_detach_input(
                            payload: AgentInput | None = None,
                            *,
                            bound_item: AgentInput = val,
                        ) -> None:
                            p = payload if payload is not None else bound_item
                            if _agent_input_has_payload(p):
                                try:
                                    await self.turn_inputs.put(p)
                                except QueueShutDown:
                                    pass
                            self.turn_inputs.close()

                        asyncio.create_task(_finish_detach_input())
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
                raise ValueError(f'agent {self.name!r}: detach requires a session store')

            parent_id = self.last_snapshot.snapshot_id if self.last_snapshot else None
            now = datetime.now(timezone.utc).isoformat()
            state = await self.session.state()

            def _pending(_: SessionSnapshot | None) -> SessionSnapshot | None:
                return SessionSnapshot(
                    snapshot_id='',
                    parent_id=parent_id or '',
                    status=SnapshotStatus.PENDING,
                    state=state,
                    created_at=now,
                )

            pending_snap = await self.store.save_snapshot(None, _pending)
            if pending_snap is None:
                raise ValueError('detach: failed to save pending snapshot')

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


def _agent_input_has_payload(inp: AgentInput) -> bool:
    """True when ``AgentInput`` carries turn data beyond a detach directive."""
    if inp.message:
        return True
    if inp.resume is not None:
        if inp.resume.restart:
            return True
        if inp.resume.respond:
            return True
    return False


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


# ---------------------------------------------------------------------------
# Agent — in-process registered agent; extends BidiAction + implements AgentAPI
# ---------------------------------------------------------------------------


class Agent(BidiAction, Generic[StateT, StreamT]):
    """In-process agent: registered in the registry, implements AgentAPI.

    Created by ``define_agent`` / ``define_custom_agent``. Extends BidiAction
    so it lives in the registry and can be served over HTTP. Also implements
    AgentAPI so it can be used as a client directly without a separate handle.
    """

    def __init__(
        self,
        *,
        name: str,
        bidi_fn: Callable[..., Awaitable[Any]],  # noqa: ANN401
        store: SessionStore | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Initialise Agent; transport is inferred from the action + store."""
        super().__init__(
            kind=ActionKind.AGENT,
            name=name,
            bidi_fn=bidi_fn,
            description=description,
            metadata={**(metadata or {}), 'agent': {'stateManagement': 'server' if store is not None else 'client'}},
        )
        self._transport = InProcessTransport(self, store)

    # ------------------------------------------------------------------
    # AgentAPI implementation
    # ------------------------------------------------------------------

    def chat(self, init: AgentInit | None = None) -> AgentSession[StateT, StreamT]:
        """Starts a new in-process session, or attaches to one via init."""
        session_transport = copy.copy(self._transport)
        return AgentSession(session_transport, init)

    async def load_chat(self, snapshot_id: str) -> AgentSession[StateT, StreamT]:
        """Loads a server snapshot and returns a session with history restored."""
        snapshot = await self._transport.get_snapshot(snapshot_id)
        if snapshot is None:
            raise ValueError(f'Snapshot {snapshot_id} not found.')
        session_transport = copy.copy(self._transport)
        session = AgentSession(session_transport)
        session.load_from_snapshot(snapshot)
        return session

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot | None:
        """Reads a snapshot without starting a session."""
        return await self._transport.get_snapshot(snapshot_id)

    async def abort(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts a running snapshot."""
        return await self._transport.abort_snapshot(snapshot_id)


# ---------------------------------------------------------------------------
# define_custom_agent + define_agent
# ---------------------------------------------------------------------------


def define_custom_agent(
    registry: Registry,
    name: str,
    fn: AgentFn,
    *,
    store: SessionStore | None = None,
    snapshot_callback: SnapshotCallback | None = None,
    client_transform: ClientTransform | None = None,
    transform: StateTransform | None = None,
    description: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Agent:
    """Register a custom agent; ``fn`` owns the turn loop via ``sess.run``."""
    resolved_transform = resolve_client_transform(
        client_transform=client_transform,
        transform=transform,
    )

    async def bidi_fn(
        init: AgentInit,
        in_queue: CloseableQueue[BidiInQueueItem],
        out_queue: CloseableQueue[StreamQueueItem],
    ) -> AgentOutput:
        session, parent = await load_session(init, store, agent_name=name)
        state = await session.state()
        if state.session_id:
            span = trace_api.get_current_span()
            if span.is_recording():
                span.set_attribute('genkit:metadata:agent:sessionId', state.session_id)

        rt = AgentRuntime(
            name=name,
            session=session,
            parent_snapshot=parent,
            store=store,
            snapshot_callback=snapshot_callback,
            client_transform=resolved_transform,
            session_outputs=out_queue,
        )
        await rt.sess.seed_last_good_state()
        return await rt.run(fn, in_queue)

    agent = Agent(
        name=name,
        bidi_fn=bidi_fn,
        description=description,
        metadata=metadata,
        store=store,
    )
    registry.register_action_from_instance(agent)

    if store is not None:

        async def snapshot_fn(input_dict: Any) -> SessionSnapshot:  # noqa: ANN401
            if isinstance(input_dict, dict):
                snapshot_id = input_dict.get('snapshotId') or input_dict.get('snapshot_id')
            else:
                snapshot_id = getattr(input_dict, 'snapshotId', None) or getattr(input_dict, 'snapshot_id', None)
            if not isinstance(snapshot_id, str):
                raise ValueError('snapshot_id required and must be a string')
            snap = await agent.get_snapshot(snapshot_id)
            if snap is None:
                raise ValueError(f'Snapshot {snapshot_id} not found')
            return snap

        snapshot_action = Action(
            kind=ActionKind.AGENT_SNAPSHOT,
            name=name,
            fn=snapshot_fn,
        )
        registry.register_action_from_instance(snapshot_action)

    return agent


async def _generate_prompt_agent_turn(
    sess: SessionRunner,
    ctx: ActionRunContext,
    *,
    child_registry: Registry,
    gen_options: GenerateActionOptions,
    history: list[MessageData],
    strip_preamble: bool,
) -> TurnResult | None:
    """Run generate for one agent turn and persist session messages."""

    def _on_chunk(chunk: StreamModelResponseChunk) -> None:
        wire_chunk = (
            chunk if isinstance(chunk, ModelResponseChunk) else ModelResponseChunk.model_validate(chunk.model_dump())
        )
        ctx.send_chunk(AgentStreamChunk(model_chunk=wire_chunk))

    response = await generate_action(
        child_registry,
        gen_options,
        on_chunk=_on_chunk,
        abort_signal=ctx.abort_signal,
        context=ctx.context,
    )

    if response.finish_reason == FinishReason.INTERRUPTED:
        _emit_interrupt_tool_chunk(ctx, response)
        await _persist_turn_messages(
            sess,
            history,
            response.message,
            strip_preamble=strip_preamble,
            response=response,
        )
        return TurnResult(finish_reason=AgentFinishReason.INTERRUPTED)

    if response.message:
        await _persist_turn_messages(
            sess,
            history,
            response.message,
            strip_preamble=strip_preamble,
            response=response,
        )

    # Return the turn result wrapping the model finish reason
    return TurnResult(finish_reason=to_agent_finish_reason(response.finish_reason))


def define_agent(
    registry: Registry,
    name: str,
    *,
    model: str | None = None,
    system: str | list[Part] | None = None,
    tools: Sequence[str | Tool] | None = None,
    use: Sequence[BaseMiddleware | MiddlewareRef] | None = None,
    config: dict[str, object] | ModelConfig | None = None,
    max_turns: int | None = None,
    description: str | None = None,
    metadata: dict[str, object] | None = None,
    store: SessionStore | None = None,
    snapshot_callback: SnapshotCallback | None = None,
    client_transform: ClientTransform | None = None,
    transform: StateTransform | None = None,
) -> Agent:
    """Register a prompt-backed agent.

    Conversation input arrives via ``AgentSession.send``;
    ``system`` is the only static preamble re-rendered each turn alongside
    session history. For template variables, few-shot messages, RAG docs, or
    prompt variants, use ``define_prompt`` + ``define_prompt_agent`` instead.
    """
    executable_prompt = ExecutablePrompt(
        registry,
        name=name,
        model=model,
        config=config,
        description=description,
        system=system,
        max_turns=max_turns,
        tools=tools,
        use=use,
    )
    register_prompt_actions(registry, executable_prompt, name, None)
    return define_prompt_agent(
        registry=registry,
        name=name,
        store=store,
        snapshot_callback=snapshot_callback,
        client_transform=client_transform,
        transform=transform,
        description=description,
        metadata=metadata,
    )


def define_prompt_agent(
    registry: Registry,
    name: str,
    *,
    store: SessionStore | None = None,
    snapshot_callback: SnapshotCallback | None = None,
    client_transform: ClientTransform | None = None,
    transform: StateTransform | None = None,
    description: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Agent:
    """Wire an already-registered prompt as an agent.

    Looks up the prompt named `name` from the registry and wires it as an
    agent. Use this when the prompt is defined separately via ai.define_prompt()
    or loaded from a .prompt file.

    The agent name and prompt name are the same string.
    """

    async def _agent_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput) -> TurnResult | None:
            history = await sess.get_messages()
            resume_respond = None
            resume_restart = None
            if inp.resume is not None:
                validate_resume_against_history(inp.resume, history)
                resume_respond = inp.resume.respond or None
                resume_restart = inp.resume.restart or None

            executable = await lookup_prompt(registry, name)
            call_opts: PromptGenerateOptions = {
                'messages': tag_history_for_render(history),
                'resume_respond': resume_respond,
                'resume_restart': resume_restart,
                'context': ctx.context,
            }
            child_registry, gen_options = await _prepare(executable, {}, call_opts)
            rendered_messages = list(gen_options.messages or [])
            gen_options = gen_options.model_copy(
                update={'messages': apply_preamble_tags(rendered_messages)},
            )

            return await _generate_prompt_agent_turn(
                sess,
                ctx,
                child_registry=child_registry,
                gen_options=gen_options,
                history=history,
                strip_preamble=True,
            )

        await sess.run(handle_turn)
        return await sess.result()

    return define_custom_agent(
        registry=registry,
        name=name,
        fn=_agent_fn,
        store=store,
        snapshot_callback=snapshot_callback,
        client_transform=client_transform,
        transform=transform,
        description=description,
        metadata=metadata,
    )
