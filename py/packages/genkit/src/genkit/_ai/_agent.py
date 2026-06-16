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
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic, TypedDict, TypeVar

from opentelemetry import trace as trace_api

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
from genkit._ai._session import (
    Session,
    SessionStore,
    SnapshotAborter,
    SnapshotCallback,
    SnapshotContext,
    StateT,
    _assert_valid_session_id,
    run_with_session,
)
from genkit._ai._tools import Tool
from genkit._core._action import (
    QUEUE_SENTINEL,
    ActionKind,
    ActionRunContext,
    BidiAction,
    BidiConnection,
    QueueSentinel,
    define_bidi_action,
)
from genkit._core._error import GenkitError
from genkit._core._middleware import BaseMiddleware
from genkit._core._model import Document, GenerateActionOptions, Message, ModelConfig, ModelResponse
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
    Error,
    FinishReason,
    JsonPatch,
    JsonPatchOperation,
    MessageData,
    MiddlewareRef,
    ModelResponseChunk,
    Part,
    Resume,
    Role,
    SessionSnapshot,
    SessionState,
    SnapshotEvent,
    SnapshotStatus,
    TextPart,
    ToolChoice,
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

# Bidi/intake queue items include a sentinel marking stream end.
_BidiInQueueItem = AgentInput | QueueSentinel
_StreamQueueItem = AgentStreamChunk | QueueSentinel
_IntakeQueueItem = AgentInput | QueueSentinel


class _ToolRequestRecord(TypedDict, total=False):
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


def _to_agent_finish_reason(fr: FinishReason | None) -> AgentFinishReason | None:
    if fr is None:
        return None
    for reason in AgentFinishReason:
        if reason.value == fr.value:
            return reason
    return AgentFinishReason.UNKNOWN


def _tool_request_parts(message: MessageData | None) -> list[Part]:
    parts: list[Part] = []
    if message is None:
        return parts
    for part in message.content or []:
        p = part if isinstance(part, Part) else Part.model_validate(part)
        if isinstance(p.root, ToolRequestPart):
            parts.append(p)
    return parts


def _to_error_details(exc: BaseException) -> Error:
    status = getattr(exc, 'status', None) or 'INTERNAL'
    message = str(exc) or 'Internal failure'
    details = getattr(exc, 'detail', None) or getattr(exc, 'details', None)
    if details is None and not isinstance(exc, GenkitError):
        details = str(exc)
    return Error(status=str(status), message=message, details=details)


def _collect_tool_requests_from_history(history: list[MessageData]) -> list[_ToolRequestRecord]:
    found: list[_ToolRequestRecord] = []
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
                tr_dict: _ToolRequestRecord = tr.model_dump(by_alias=True, exclude_none=True)
            elif isinstance(tr, dict):
                tr_dict = tr  # type: ignore[assignment]
            else:
                tr_record: _ToolRequestRecord = {
                    'name': getattr(tr, 'name', None),
                    'ref': getattr(tr, 'ref', None),
                    'input': getattr(tr, 'input', None),
                }
                tr_dict = tr_record
            found.append(tr_dict)
    return found


def validate_resume_against_history(resume: Resume, history: list[MessageData]) -> None:
    """Ensure resume entries reference tool requests present in session history."""
    all_tool_requests = _collect_tool_requests_from_history(history)

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
    parts = _tool_request_parts(response.message)
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


# ---------------------------------------------------------------------------
# AgentFnArg (runtime-only, not a wire type)
# ---------------------------------------------------------------------------


class AgentFnArg(Generic[StreamT]):
    """Second argument to AgentFn: stream sender + abort signal."""

    def __init__(
        self,
        send_chunk: Callable[[AgentStreamChunk], None],
        abort_signal: asyncio.Event,
    ) -> None:
        self.send_chunk = send_chunk
        self.abort_signal = abort_signal


# StateTransform — redact or reshape session state before it leaves the server.
StateTransform = Callable[[SessionState], SessionState | None]

# ChunkTransform — reshape or drop a stream chunk before it reaches the client.
ChunkTransform = Callable[[AgentStreamChunk], AgentStreamChunk | None]


class ClientTransform(TypedDict, total=False):
    """Project server-side agent data onto the client-visible view."""

    state: StateTransform
    chunk: ChunkTransform


def _resolve_client_transform(
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
# _AgentRuntime
# ---------------------------------------------------------------------------


async def _load_session(
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
            status='INVALID_ARGUMENT',
            message=(f"Cannot send both 'snapshotId' and 'sessionId' to agent '{name}'. Provide exactly one."),
        )
    if (init.snapshot_id or init.session_id) and store is None:
        field = 'snapshotId' if init.snapshot_id else 'sessionId'
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
                "server-managed store. Send 'snapshotId' or 'sessionId' instead."
            ),
        )

    if store is not None and init.snapshot_id:
        snap = await store.get_snapshot(snapshot_id=init.snapshot_id)
        if snap is None:
            raise GenkitError(
                status='NOT_FOUND',
                message=f'Snapshot {init.snapshot_id!r} not found',
            )
        return Session(initial_state=snap.state), snap

    if store is not None and init.session_id:
        _assert_valid_session_id(init.session_id)
        snap = await store.get_snapshot(session_id=init.session_id)
        if snap is not None:
            return Session(initial_state=snap.state), snap
        return (
            Session(
                initial_state=SessionState(
                    session_id=init.session_id,
                    messages=[],
                    artifacts=[],
                )
            ),
            None,
        )

    if init.state is not None:
        return Session(initial_state=init.state), None
    return Session(), None


class _AgentRuntime:
    """Drives the agent fn to completion; owns session, router, and intake."""

    def __init__(
        self,
        name: str,
        session: Session[Any],
        parent_snapshot: SessionSnapshot | None,
        store: SessionStore | None,
        snapshot_callback: SnapshotCallback | None,
        client_transform: ClientTransform | None,
        out_queue: asyncio.Queue[_StreamQueueItem],
    ) -> None:
        self._name = name
        self._session = session
        self._store = store
        self._snapshot_callback = snapshot_callback
        self._client_transform = client_transform
        self._last_snapshot: SessionSnapshot | None = parent_snapshot
        self._last_snapshot_version: int = self._session.version if parent_snapshot is not None else -1
        self._detached: bool = False
        self._first_custom_patch_in_turn: bool = True
        self._last_sent_custom: object | None = None

        self._out_queue = out_queue

        # Separate intake queue: runtime controls its lifecycle,
        # BidiAction's in_queue is forwarded here by run().
        self._intake: asyncio.Queue[_IntakeQueueItem] = asyncio.Queue(maxsize=1)

        self._sess = SessionRunner(
            session,
            self._intake,
            on_begin_turn=self._reset_custom_patch_turn,
            on_end_turn=self._emit_turn_end,
        )

        session.on_custom_changed(self._emit_custom_patch)
        session.on_artifact_changed(self._emit_artifact)

    async def _reset_custom_patch_turn(self) -> None:
        # Re-base clients that may not share the server's custom-state baseline.
        self._first_custom_patch_in_turn = True

    def _transform_state(self, state: SessionState) -> SessionState | None:
        state_fn = self._client_transform.get('state') if self._client_transform else None
        if state_fn is None:
            return state
        return state_fn(state)

    def _transform_chunk(self, chunk: AgentStreamChunk) -> AgentStreamChunk | None:
        chunk_fn = self._client_transform.get('chunk') if self._client_transform else None
        if chunk_fn is None:
            return chunk
        return chunk_fn(chunk)

    async def _client_custom(self) -> object | None:
        state = await self._session.state()
        client_state = self._transform_state(state)
        return client_state.custom if client_state is not None else None

    async def _emit_custom_patch(self) -> None:
        if self._detached:
            return

        transformed = await self._client_custom()
        if self._first_custom_patch_in_turn:
            ops: list[JsonPatchOperation] = [
                JsonPatchOperation(op='replace', path='', value=copy.deepcopy(transformed))
            ]
            self._first_custom_patch_in_turn = False
        else:
            ops = diff_json(self._last_sent_custom, transformed)

        self._last_sent_custom = copy.deepcopy(transformed)
        if not ops:
            return

        self._send_chunk(AgentStreamChunk(custom_patch=JsonPatch(root=ops)))

    async def _emit_artifact(self, artifact: Artifact) -> None:
        if self._detached:
            return
        self._send_chunk(AgentStreamChunk(artifact=artifact))

    async def _maybe_snapshot(
        self,
        event: SnapshotEvent,
        *,
        finish_reason: AgentFinishReason | None = None,
        status: SnapshotStatus | None = None,
        error: Error | None = None,
        force: bool = False,
    ) -> str:
        """Save snapshot if store configured + callback approves + state changed."""
        if self._store is None:
            return ''
        if not force and self._last_snapshot is not None and self._session.version == self._last_snapshot_version:
            return self._last_snapshot.snapshot_id

        state = await self._session.state()

        if self._snapshot_callback is not None and status != SnapshotStatus.FAILED:
            ctx = SnapshotContext(
                event=event,
                state=state,
                prev_state=self._last_snapshot.state if self._last_snapshot else None,
                turn_index=self._sess.turn_index,
            )
            if not self._snapshot_callback(ctx):
                return self._last_snapshot.snapshot_id if self._last_snapshot else ''

        parent_id = self._last_snapshot.snapshot_id if self._last_snapshot else None
        now = datetime.now(timezone.utc).isoformat()
        snap_status = status or SnapshotStatus.DONE

        def _make_snap(
            existing: SessionSnapshot | None,
        ) -> SessionSnapshot | None:
            if existing is not None and existing.status == SnapshotStatus.ABORTED:
                return None
            return SessionSnapshot(
                snapshot_id=existing.snapshot_id if existing else '',
                parent_id=parent_id or '',
                event=event,
                status=snap_status,
                state=state,
                created_at=existing.created_at if existing and existing.created_at else now,
                finish_reason=finish_reason,
                error=error,
            )

        snap = await self._store.save_snapshot(None, _make_snap)
        if snap is not None:
            self._last_snapshot = snap
            self._last_snapshot_version = self._session.version
            return snap.snapshot_id
        return self._last_snapshot.snapshot_id if self._last_snapshot else ''

    async def _ensure_recovery_snapshot(self) -> str | None:
        """Persist last-good state after a failed turn when the callback skipped it."""
        if self._store is None or self._sess.last_good_state is None:
            return self._last_snapshot.snapshot_id if self._last_snapshot else None

        if (
            self._sess.last_good_state_version is not None
            and self._sess.last_good_state_version == self._last_snapshot_version
        ):
            return self._last_snapshot.snapshot_id if self._last_snapshot else None

        if self._sess.turn_index == 0:
            return None

        parent_id = self._last_snapshot.snapshot_id if self._last_snapshot else None
        now = datetime.now(timezone.utc).isoformat()
        last_good = self._sess.last_good_state

        def _recovery(existing: SessionSnapshot | None) -> SessionSnapshot | None:
            if existing is not None and existing.status == SnapshotStatus.ABORTED:
                return None
            return SessionSnapshot(
                snapshot_id='',
                parent_id=parent_id or '',
                event=SnapshotEvent.TURNEND,
                status=SnapshotStatus.DONE,
                state=last_good,
                created_at=now,
                finish_reason=self._sess.last_good_finish_reason,
            )

        snap = await self._store.save_snapshot(None, _recovery)
        if snap is not None:
            self._last_snapshot = snap
            if self._sess.last_good_state_version is not None:
                self._last_snapshot_version = self._sess.last_good_state_version
            return snap.snapshot_id
        return self._last_snapshot.snapshot_id if self._last_snapshot else None

    async def _failed_agent_output(self, result: AgentResult | None) -> AgentOutput:
        last_good = self._sess.last_good_state or await self._session.state()
        msgs = list(last_good.messages or [])
        out = AgentOutput(
            finish_reason=AgentFinishReason.FAILED,
            error=self._sess.last_turn_error,
            message=msgs[-1] if msgs else (result.message if result else None),
            artifacts=list(last_good.artifacts or []) if last_good.artifacts else (result.artifacts if result else []),
        )
        if self._store is not None:
            out.snapshot_id = await self._ensure_recovery_snapshot()
        else:
            out.state = self._transform_state(last_good)
        return out

    async def _watch_snapshot_abort(self, snapshot_id: str, abort_signal: asyncio.Event) -> None:
        if self._store is None or not isinstance(self._store, SnapshotAborter):
            return
        q = await self._store.on_snapshot_status_change(snapshot_id)
        while True:
            status = await q.get()
            if status is None:
                return
            if status == SnapshotStatus.ABORTED:
                abort_signal.set()
                return

    def _send_chunk(self, chunk: object) -> None:
        """Forward chunk to client and apply artifact side effects inline.

        Post-detach: side effects still apply (artifacts land in session)
        but wire forwarding is suppressed (client is gone).
        """
        if not isinstance(chunk, AgentStreamChunk):
            return
        if chunk.artifact is not None:
            asyncio.get_event_loop().create_task(self._session.add_artifacts(chunk.artifact, _suppress_events=True))
        if self._detached:
            return
        wire_chunk = self._transform_chunk(chunk)
        if wire_chunk is None:
            return
        self._out_queue.put_nowait(wire_chunk)

    async def _emit_turn_end(self, finish_reason: AgentFinishReason | None = None) -> None:
        """Called by SessionRunner after each turn: snapshot + TurnEnd chunk."""
        if self._detached:
            return
        is_failed = finish_reason == AgentFinishReason.FAILED
        snapshot_id = await self._maybe_snapshot(
            SnapshotEvent.TURNEND,
            finish_reason=finish_reason,
            status=SnapshotStatus.FAILED if is_failed else SnapshotStatus.DONE,
            error=self._sess.last_turn_error if is_failed else None,
            force=is_failed,
        )
        self._send_chunk(
            AgentStreamChunk(
                turn_end=TurnEnd(
                    snapshot_id=snapshot_id or None,
                    finish_reason=finish_reason,
                )
            )
        )

    async def _finalize_detach(
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

        state = await self._session.state()
        now = datetime.now(timezone.utc).isoformat()
        fn_err = err_holder[0] if err_holder else None
        if fn_err:
            finish_reason = AgentFinishReason.FAILED
        else:
            result = result_holder[0] if result_holder else None
            finish_reason = (
                result.finish_reason if result and result.finish_reason else self._sess.last_turn_finish_reason
            )

        def _finalize(existing: SessionSnapshot | None) -> SessionSnapshot | None:
            # If already aborted by user, leave it.
            if existing is not None and existing.status == SnapshotStatus.ABORTED:
                return None
            return SessionSnapshot(
                snapshot_id=existing.snapshot_id if existing else '',
                parent_id=pending_snap.parent_id or '',
                event=SnapshotEvent.TURNEND,
                status=SnapshotStatus.FAILED if fn_err else SnapshotStatus.DONE,
                state=state,
                error=(Error(status='INTERNAL', message=str(fn_err)) if fn_err else None),
                finish_reason=finish_reason,
                created_at=existing.created_at if existing else now,
            )

        try:
            await self._store.save_snapshot(pending_snap.snapshot_id, _finalize)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001, S110
            pass  # best-effort; log in production

    async def run(self, fn: AgentFn, in_queue: asyncio.Queue[_BidiInQueueItem]) -> AgentOutput:
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
            streaming_callback=self._send_chunk,
            abort_signal=abort_signal,
        )

        # Forward task: BidiAction in_queue → runtime intake.
        # Detects detach=True and signals via detach_future.
        async def _forward() -> None:
            while True:
                item = await in_queue.get()
                if isinstance(item, QueueSentinel):
                    await self._intake.put(QUEUE_SENTINEL)
                    return
                if getattr(item, 'detach', False):
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
                            await self._intake.put(p)
                        await self._intake.put(QUEUE_SENTINEL)

                    asyncio.create_task(_finish_detach_input())
                    return
                await self._intake.put(item)

        forward_task = asyncio.create_task(_forward())

        result_holder: list[AgentResult] = []
        err_holder: list[BaseException] = []

        async def _run_fn() -> None:
            try:
                result = await run_with_session(
                    self._session,
                    fn(self._sess, action_ctx),
                )
                result_holder.append(result)
            except Exception as e:  # noqa: BLE001
                err_holder.append(e)
            finally:
                try:
                    self._intake.put_nowait(QUEUE_SENTINEL)
                except asyncio.QueueFull:
                    pass

        fn_task = asyncio.create_task(_run_fn())

        # Wait for fn completion OR detach signal, whichever comes first.
        done, _ = await asyncio.wait(
            {fn_task, asyncio.ensure_future(asyncio.shield(detach_future))},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # --- Detach path ---
        if detach_future.done() and fn_task not in done:
            if self._store is None:
                # Detach without a store is a config error; signal abort and raise.
                abort_signal.set()
                await fn_task
                await forward_task
                raise ValueError(f'agent {self._name!r}: detach requires a session store')

            parent_id = self._last_snapshot.snapshot_id if self._last_snapshot else None
            now = datetime.now(timezone.utc).isoformat()
            state = await self._session.state()

            def _pending(_: SessionSnapshot | None) -> SessionSnapshot | None:
                return SessionSnapshot(
                    snapshot_id='',
                    parent_id=parent_id or '',
                    event=SnapshotEvent.TURNEND,
                    status=SnapshotStatus.PENDING,
                    state=state,
                    created_at=now,
                )

            pending_snap = await self._store.save_snapshot(None, _pending)
            if pending_snap is None:
                raise ValueError('detach: failed to save pending snapshot')

            # Stop sending chunks to the (now-gone) client.
            # Background task finalizes snapshot when fn finishes.
            self._detached = True
            asyncio.create_task(self._watch_snapshot_abort(pending_snap.snapshot_id, abort_signal))
            asyncio.create_task(self._finalize_detach(pending_snap, fn_task, forward_task, err_holder, result_holder))
            return AgentOutput(
                snapshot_id=pending_snap.snapshot_id,
                finish_reason=AgentFinishReason.DETACHED,
            )

        # --- Normal completion path ---
        await fn_task
        await forward_task

        result = result_holder[0] if result_holder else None

        if self._sess.last_turn_finish_reason == AgentFinishReason.FAILED and self._sess.last_turn_error:
            return await self._failed_agent_output(result)

        if err_holder:
            raise err_holder[0]

        snapshot_id = await self._maybe_snapshot(SnapshotEvent.INVOCATIONEND)
        if not snapshot_id and self._last_snapshot is not None:
            snapshot_id = self._last_snapshot.snapshot_id

        finish_reason = result.finish_reason if result else self._sess.last_turn_finish_reason
        out = AgentOutput(
            snapshot_id=snapshot_id or None,
            message=result.message if result else None,
            artifacts=list(result.artifacts) if result and result.artifacts else [],
            finish_reason=finish_reason,
        )

        if self._store is None:
            state = await self._session.state()
            out.state = self._transform_state(state)

        return out


# ---------------------------------------------------------------------------
# AgentConnection
# ---------------------------------------------------------------------------


class AgentConnection(Generic[StreamT, StateT]):
    """Public handle for an active agent session.

    Wraps the bidi connection with send helpers so callers need not build
    ``AgentInput`` by hand.
    """

    def __init__(
        self,
        conn: BidiConnection[AgentInput, AgentStreamChunk, AgentOutput],
    ) -> None:
        self._conn = conn

    async def send(self, input: AgentInput) -> None:  # noqa: A002
        """Send a raw AgentInput."""
        await self._conn.send(input)

    async def send_text(self, text: str) -> None:
        """Send a user text message for one turn."""
        await self._conn.send(AgentInput(messages=[MessageData(role='user', content=[Part(root=TextPart(text=text))])]))

    async def send_resume(self, resume: Resume) -> None:
        """Send a resume payload to continue an interrupted tool call."""
        await self._conn.send(AgentInput(resume=resume))

    async def detach(self) -> None:
        """v2: ask the server to background the invocation and close the connection."""
        await self._conn.send(AgentInput(detach=True))

    async def close(self) -> None:
        """Signal no more inputs will be sent."""
        await self._conn.close()

    async def receive(self) -> AsyncIterator[AgentStreamChunk]:
        """Async iterator of AgentStreamChunk from the server."""
        async for chunk in self._conn.receive():
            yield chunk

    async def output(self) -> AgentOutput:
        """Await the final AgentOutput."""
        return await self._conn.output()


def _agent_input_has_payload(inp: AgentInput) -> bool:
    """True when ``AgentInput`` carries turn data beyond a detach directive."""
    if inp.messages:
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
        intake: asyncio.Queue[_IntakeQueueItem],
        on_begin_turn: Callable[[], Awaitable[None]] | None = None,
        on_end_turn: Callable[[AgentFinishReason | None], Awaitable[None]] | None = None,
    ) -> None:
        self._session = session
        self._intake = intake
        self._on_begin_turn = on_begin_turn
        self._on_end_turn = on_end_turn
        self.turn_index: int = 0
        self.last_turn_finish_reason: AgentFinishReason | None = None
        self.last_turn_error: Error | None = None
        self.last_good_state: SessionState | None = None
        self.last_good_state_version: int | None = None
        self.last_good_finish_reason: AgentFinishReason | None = None

    async def seed_last_good_state(self) -> None:
        """Capture initial session state as the fallback for first-turn failures."""
        self.last_good_state = await self._session.state()
        self.last_good_state_version = self._session.version

    async def run(
        self,
        fn: Callable[[AgentInput], Awaitable[TurnResult | None]],
    ) -> None:
        """Consume inputs from the intake queue, calling fn for each turn.

        Each turn is wrapped in a trace span ``runTurn-N`` (1-based). Inbound
        messages are automatically added to the session before fn is called.
        After fn returns, on_end_turn is called (snapshot + chunk emission)
        and turn_index is incremented.

        Turn failures resolve gracefully: ``last_turn_finish_reason`` becomes
        ``failed``, ``last_turn_error`` is populated, and the loop stops.
        """
        while True:
            inp = await self._intake.get()
            if isinstance(inp, QueueSentinel):
                return

            # Auto-add inbound messages to session history
            if inp.messages:
                await self._session.add_messages(*inp.messages)

            if self._on_begin_turn is not None:
                await self._on_begin_turn()

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

                    if self._on_end_turn is not None:
                        await self._on_end_turn(finish_reason)

                    span_meta.output = {'finishReason': finish_reason}

                self.last_good_state = await self._session.state()
                self.last_good_state_version = self._session.version
                self.last_good_finish_reason = self.last_turn_finish_reason
                self.turn_index += 1
            except Exception as exc:  # noqa: BLE001
                self.last_turn_finish_reason = AgentFinishReason.FAILED
                self.last_turn_error = _to_error_details(exc)

                if self._on_end_turn is not None:
                    await self._on_end_turn(AgentFinishReason.FAILED)

                break

    async def result(self) -> AgentResult:
        """Last message, artifacts, and finish reason from the current session."""
        state = await self._session.state()
        msg = state.messages[-1] if state.messages else None
        arts = list(state.artifacts) if state.artifacts else []
        return AgentResult(
            message=msg,
            artifacts=arts,
            finish_reason=self.last_turn_finish_reason,
        )

    # --- Session passthrough helpers ---

    async def get_messages(self) -> list[MessageData]:
        return await self._session.get_messages()

    async def set_messages(self, messages: list[MessageData]) -> None:
        await self._session.set_messages(messages)

    async def add_messages(self, *messages: MessageData) -> None:
        await self._session.add_messages(*messages)

    async def get_artifacts(self) -> list[Artifact]:
        return await self._session.get_artifacts()

    async def add_artifacts(self, *artifacts: Artifact) -> None:
        await self._session.add_artifacts(*artifacts)

    async def get_custom(self) -> StateT | None:
        return await self._session.get_custom()

    async def update_custom(self, fn: Callable[[StateT | None], StateT]) -> None:
        await self._session.update_custom(fn)


# ---------------------------------------------------------------------------
# Agent + define_custom_agent + define_agent
# ---------------------------------------------------------------------------


class Agent:
    """Registered bidi agent — call ``stream_bidi()`` for an ``AgentConnection``."""

    def __init__(self, bidi_action: BidiAction) -> None:
        self._action = bidi_action

    @property
    def name(self) -> str:
        return self._action.name

    async def stream_bidi(
        self,
        init: AgentInit | None = None,
        context: dict[str, object] | None = None,
    ) -> AgentConnection[Any, Any]:
        """Start a new agent session. Returns an AgentConnection."""
        conn = await self._action.stream_bidi(init or AgentInit(), context=context)
        return AgentConnection(conn)


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
    resolved_transform = _resolve_client_transform(
        client_transform=client_transform,
        transform=transform,
    )

    async def _bidi_fn(
        init: AgentInit,
        in_queue: asyncio.Queue[_BidiInQueueItem],
        out_queue: asyncio.Queue[_StreamQueueItem],
    ) -> AgentOutput:
        session, parent = await _load_session(init, store, agent_name=name)
        state = await session.state()
        if state.session_id:
            span = trace_api.get_current_span()
            if span.is_recording():
                span.set_attribute('genkit:metadata:agent:sessionId', state.session_id)

        rt = _AgentRuntime(
            name=name,
            session=session,
            parent_snapshot=parent,
            store=store,
            snapshot_callback=snapshot_callback,
            client_transform=resolved_transform,
            out_queue=out_queue,
        )
        await rt._sess.seed_last_good_state()
        return await rt.run(fn, in_queue)

    action = define_bidi_action(
        registry=registry,
        kind=ActionKind.AGENT,
        name=name,
        bidi_fn=_bidi_fn,
        description=description,
        metadata={**(metadata or {}), 'agent': True},
    )
    return Agent(action)


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

    return TurnResult(finish_reason=_to_agent_finish_reason(response.finish_reason))


def define_agent(
    registry: Registry,
    name: str,
    *,
    variant: str | None = None,
    model: str | None = None,
    config: dict[str, object] | ModelConfig | None = None,
    description: str | None = None,
    input_schema: type | dict[str, object] | str | None = None,
    system: str | list[Part] | None = None,
    prompt: str | list[Part] | None = None,
    messages: str | list[Message] | None = None,
    output_format: str | None = None,
    output_content_type: str | None = None,
    output_instructions: str | None = None,
    output_schema: type | dict[str, object] | str | None = None,
    output_constrained: bool | None = None,
    max_turns: int | None = None,
    return_tool_requests: bool | None = None,
    metadata: dict[str, object] | None = None,
    tools: Sequence[str | Tool] | None = None,
    tool_choice: ToolChoice | None = None,
    use: Sequence[BaseMiddleware | MiddlewareRef] | None = None,
    docs: list[Document] | None = None,
    resources: list[str] | None = None,
    store: SessionStore | None = None,
    snapshot_callback: SnapshotCallback | None = None,
    client_transform: ClientTransform | None = None,
    transform: StateTransform | None = None,
) -> Agent:
    """Register a prompt-backed agent (define_prompt + define_prompt_agent)."""
    executable_prompt = ExecutablePrompt(
        registry,
        variant=variant,
        name=name,
        model=model,
        config=config,
        description=description,
        input_schema=input_schema,
        system=system,
        prompt=prompt,
        messages=messages,
        output_format=output_format,
        output_content_type=output_content_type,
        output_instructions=output_instructions,
        output_schema=output_schema,
        output_constrained=output_constrained,
        max_turns=max_turns,
        return_tool_requests=return_tool_requests,
        metadata=metadata,
        tools=tools,
        tool_choice=tool_choice,
        use=use,
        docs=docs,
        resources=resources,
    )
    register_prompt_actions(registry, executable_prompt, name, variant)
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
