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

from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Generator, Iterator
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from genkit._ai._agents._snapshot import lookup_label
from genkit._ai._agents._types import StateManagement
from genkit._ai._json_patch import apply_json_patch
from genkit._core._channel import CloseableQueue
from genkit._core._model import Message
from genkit._core._typing import (
    AgentFinishReason,
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentStreamChunk,
    Artifact,
    MessageData,
    ModelResponseChunk,
    Part,
    Resume,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)

StateT = TypeVar('StateT')
StreamT = TypeVar('StreamT')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')
# The transport protocol only ever hands these types back out (in the sessions it
# returns), never takes them in, so they're covariant.
StateT_co = TypeVar('StateT_co', covariant=True)
StreamT_co = TypeVar('StreamT_co', covariant=True)


# ===========================================================================
# Client Transport Protocol
# ===========================================================================


@runtime_checkable
class AgentTransport(Protocol, Generic[StateT_co, StreamT_co]):
    """Interface implemented by the transport layer (local or websocket)."""

    # Declares server- vs client-managed state; must be set explicitly on the transport.
    state_management: StateManagement

    async def run_turn(
        self,
        agent_input: AgentInput,
        init: AgentInit,
        abort_event: asyncio.Event | None = None,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        """Run a single turn, returning a (stream, output) pair.

        The transport must drive the turn to completion on its own — the returned
        output awaitable has to resolve whether or not the caller consumes the
        stream. The stream is an optional live view of the same turn, never the
        thing that advances it, so ``await output`` works without draining chunks.
        Concretely: read your own transport (socket, queue, SSE) to the end in a
        background task; don't rely on the client to pull the stream for you.

        ``init`` describes how to resume the conversation for this turn. A stateful
        in-process transport reads it only when it opens its connection; a
        stateless transport (HTTP) replays it on every request. The session keeps
        it current via ``_wire_init`` so the transport never has to know how state
        is managed.
        """
        ...

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        """Retrieves a session snapshot from the server store."""
        ...

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts the specified snapshot on the server."""
        ...

    async def close(self) -> None:
        """Cleanly closes any persistent connections held by this transport."""
        ...


# ===========================================================================
# Client Return Types & Models
# ===========================================================================


@dataclass
class AgentChunk(Generic[StreamT]):
    """Represents a structured stream chunk yielded during a turn."""

    text: str | None = None
    reasoning: str | None = None
    tool_request: ToolRequestPart | None = None
    tool_response: ToolResponsePart | None = None
    artifact: Artifact | None = None
    custom: StreamT | None = None  # post-patch resolved custom state; set when custom_patch is present
    raw: AgentStreamChunk | None = None


class AgentInterrupt(Generic[InputT, OutputT]):
    """Represents a tool request interrupt that paused the turn."""

    def __init__(
        self,
        name: str,
        ref: str | None,
        input_data: InputT,
    ) -> None:
        self.name = name
        self.ref = ref
        self.input = input_data

    def respond_part(self, output: OutputT) -> ToolResponsePart:
        """Wire-shaped tool response for batching into ``session.resume(Resume(...))``."""
        return ToolResponsePart(
            tool_response=ToolResponse(
                name=self.name,
                ref=self.ref,
                output=output,
            )
        )

    def restart_part(
        self,
        *,
        resumed_metadata: dict[str, Any] | None = None,
        replace_input: Any | None = None,  # noqa: ANN401
    ) -> ToolRequestPart:
        """Wire-shaped restart request for batching into ``session.resume(Resume(...))``."""
        from genkit._ai._tools import restart_tool

        part = ToolRequestPart(
            tool_request=ToolRequest(
                name=self.name,
                ref=self.ref,
                input=self.input,
            )
        )
        if resumed_metadata is not None or replace_input is not None:
            return restart_tool(
                interrupt=part,
                resumed_metadata=resumed_metadata,
                replace_input=replace_input,
            )
        return part


@dataclass
class AgentResponse(Generic[StateT]):
    """Completed turn result — client-side wrapper around AgentOutput with rich accessors."""

    raw: AgentOutput
    messages: list[MessageData]
    custom: StateT | None = None

    @property
    def text(self) -> str:
        """Full text content of the response message."""
        if self.raw.message is None:
            return ''
        texts: list[str] = []
        for part in self.raw.message.content or []:
            p = part if isinstance(part, Part) else Part.model_validate(part)
            if isinstance(p.root, TextPart):
                texts.append(p.root.text)
        return ''.join(texts)

    @property
    def finish_reason(self) -> AgentFinishReason | None:
        """Why the turn ended."""
        return self.raw.finish_reason

    @property
    def snapshot_id(self) -> str | None:
        """Server snapshot id after this turn, if store-backed."""
        return self.raw.snapshot_id

    @property
    def artifacts(self) -> list[Artifact]:
        """Artifacts emitted during this turn."""
        return self.raw.artifacts or []

    @property
    def message(self) -> MessageData | None:
        """The response message (raw)."""
        return self.raw.message

    @property
    def tool_requests(self) -> list[ToolRequestPart]:
        """Tool requests in the response message."""
        if self.raw.message is None:
            return []
        result: list[ToolRequestPart] = []
        for part in self.raw.message.content or []:
            p = part if isinstance(part, Part) else Part.model_validate(part)
            if isinstance(p.root, ToolRequestPart):
                result.append(p.root)
        return result

    @property
    def interrupts(self) -> list[AgentInterrupt[Any, Any]]:
        """Tool requests that paused this turn."""
        return _agent_interrupts_from_message(self.raw.message)


def _agent_interrupts_from_message(message: MessageData | None) -> list[AgentInterrupt[Any, Any]]:
    if message is None:
        return []
    msg = message if isinstance(message, Message) else Message(message)
    return [
        AgentInterrupt(
            name=part.tool_request.name,
            ref=part.tool_request.ref,
            input_data=part.tool_request.input,
        )
        for part in msg.interrupts
    ]


class AgentTurn(Generic[StateT, StreamT]):
    """A single in-flight turn — iterate for chunks, await for the final response."""

    def __init__(
        self,
        stream: AsyncIterable[AgentChunk[StreamT]],
        output: Awaitable[AgentResponse[StateT]],
        abort_fn: Callable[[], Awaitable[None] | None] | None = None,
    ) -> None:
        self._stream = stream
        self._output = output
        self._abort_fn = abort_fn

    def __aiter__(self) -> AsyncIterator[AgentChunk[StreamT]]:
        return self._stream.__aiter__()

    def __await__(self) -> Generator[Any, None, AgentResponse[StateT]]:
        """Return the completed turn result. The turn runs whether or not you stream."""
        return self._output.__await__()

    async def abort(self) -> None:
        """Detaches the client from this turn: stops streaming and settles the result now.

        This is a client-side abort. The server turn keeps running to completion
        in the background so its work still lands; we just stop listening and
        resolve the awaited result as cancelled. To actually halt server-side
        work on a store-backed agent, use ``session.abort()``.
        """
        if self._abort_fn:
            res = self._abort_fn()
            if inspect.isawaitable(res):
                await res
        # Detaching cancelled the result. But the turn may have already settled on
        # its own a beat earlier (succeeded or failed), in which case we didn't
        # cancel it. Read that terminal state so a failed turn's exception isn't
        # logged as never-retrieved; .exception() does this without raising (unlike
        # awaiting), and returns None for a successful turn.
        if isinstance(self._output, asyncio.Future) and self._output.done() and not self._output.cancelled():
            self._output.exception()


# ===========================================================================
# Client APIs & Session Handles
# ===========================================================================


@runtime_checkable
class AgentAPI(Protocol, Generic[StateT, StreamT]):
    """Implemented by both Agent (in-process) and AgentClient (remote)."""

    def chat(self, init: AgentInit | None = None) -> AgentSession[StateT, StreamT]:
        """Starts a new session, or attaches to one via init."""
        ...

    async def load_chat(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> AgentSession[StateT, StreamT]:
        """Loads a server snapshot and returns a session with history restored."""
        ...

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        """Reads a snapshot without starting a session."""
        ...

    async def abort(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts a running snapshot."""
        ...


class AgentClient(Generic[StateT, StreamT]):
    """Remote/transport-backed agent — wraps any AgentTransport. Implements AgentAPI."""

    def __init__(self, transport: AgentTransport[StateT, StreamT]) -> None:
        self._transport = transport

    def chat(self, init: AgentInit | None = None) -> AgentSession[StateT, StreamT]:
        """Starts a new session, or attaches to one via init."""
        session_transport = copy.copy(self._transport)
        return AgentSession(session_transport, init)

    async def load_chat(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> AgentSession[StateT, StreamT]:
        """Loads a server snapshot and returns a session with history restored."""
        snapshot = await self._transport.get_snapshot(snapshot_id=snapshot_id, session_id=session_id)
        if snapshot is None:
            raise ValueError(f'Snapshot {lookup_label(snapshot_id=snapshot_id, session_id=session_id)!r} not found.')
        session_transport = copy.copy(self._transport)
        session_transport.state_management = 'server'
        session = AgentSession(session_transport)
        session._load_from_snapshot(snapshot)
        return session

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        """Reads a snapshot without starting a session."""
        return await self._transport.get_snapshot(snapshot_id=snapshot_id, session_id=session_id)

    async def abort(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts a running snapshot on the server."""
        return await self._transport.abort_snapshot(snapshot_id)


def _to_agent_input(input: str | AgentInput) -> AgentInput:  # noqa: A002
    """Wraps a plain string in an AgentInput; passes an AgentInput through unchanged."""
    if isinstance(input, str):
        return AgentInput(message=MessageData(role='user', content=[Part(root=TextPart(text=input))]))
    return input


def _extract_abort_signal(opts: Any) -> Any:  # noqa: ANN401
    """Pulls a caller-supplied abort_signal out of opts (dict or attribute), if any."""
    if isinstance(opts, dict):
        return opts.get('abort_signal')
    if opts is not None and hasattr(opts, 'abort_signal'):
        return opts.abort_signal
    return None


def _validate_init(init: AgentInit) -> None:
    """Ensures init specifies at most one resume handle."""
    provided = [
        name
        for name, present in (
            ('state', init.state is not None),
            ('snapshot_id', bool(init.snapshot_id)),
            ('session_id', bool(init.session_id)),
        )
        if present
    ]
    if len(provided) > 1:
        raise ValueError(
            f'AgentInit may specify at most one of state, snapshot_id, or session_id; got {", ".join(provided)}.'
        )


class _TurnDriver(Generic[StateT, StreamT]):
    """Runs one in-flight turn and exposes it as an AgentTurn.

    Everything a turn needs lives here so AgentSession.send stays small: it starts
    the transport, pumps chunks onto the caller's stream, resolves the final output,
    and handles abort. Keeping it in one object means a single cancellation path and
    no closures that reference each other before they exist.
    """

    def __init__(
        self,
        inp: AgentInput,
        init: AgentInit,
        *,
        run_turn: Callable[
            [AgentInput, AgentInit, asyncio.Event],
            Awaitable[tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]],
        ],
        commit_output: Callable[[AgentOutput], AgentResponse[StateT]],
        commit_custom_patch: Callable[[Any], StateT | None],
        rollback: Callable[[], None] | None = None,
        external_signal: Any = None,  # noqa: ANN401
    ) -> None:
        self.inp = inp
        self.init = init
        self.run_turn = run_turn
        self.commit_output = commit_output
        self.commit_custom_patch = commit_custom_patch
        self.rollback = rollback
        self.external_signal = external_signal
        self.aborted = asyncio.Event()
        self.output: asyncio.Future[AgentResponse[StateT]] = asyncio.get_event_loop().create_future()
        self.chunks: CloseableQueue[AgentChunk[StreamT] | BaseException] = CloseableQueue()
        self.run_task: asyncio.Task[None] | None = None
        self.signal_task: asyncio.Task[None] | None = None
        self.turn: AgentTurn[StateT, StreamT] = AgentTurn(
            stream=self.stream(),
            output=self.output,
            abort_fn=self.abort,
        )

    def start(self) -> AgentTurn[StateT, StreamT]:
        """Launches the turn in the background and returns its handle immediately."""
        self.output.add_done_callback(lambda _: self.cancel_signal_watcher())
        self.run_task = asyncio.create_task(self.run())
        if self.external_signal is not None:
            self.signal_task = asyncio.create_task(self.watch_external_signal())
        return self.turn

    async def run(self) -> None:
        """Pumps chunks to the caller's stream while a sibling task resolves the output.

        Per the transport contract, the ``output`` awaitable is the authoritative
        "turn is done" signal and settles on its own. We resolve ``self.output`` from a
        sibling task rather than awaiting ``output`` after the pump loop so the turn's
        *result* stays decoupled from the *stream's* health.

        - If ``emit`` chokes on a bad chunk after the transport has already produced a
          valid output, the resolver has set ``self.output`` from the real result, so the
          decode error only surfaces on the stream — a good turn isn't poisoned by a
          plumbing hiccup. Sequencing ``await output`` after the pump would instead dump
          that exception onto ``self.output``.
        - The pump and resolver can settle in either order; both guard on
          ``self.output.done()`` so whoever's first wins and the other is a no-op
          (including ``abort()`` cancelling ``self.output`` directly).
        - The done-callback cancels the resolver once ``self.output`` is settled by
          anyone, so the task never dangles.
        """
        try:
            stream, output = await self.run_turn(self.inp, self.init, self.aborted)
            resolve_task = asyncio.create_task(self.resolve_output(output))
            self.output.add_done_callback(lambda _: resolve_task.cancel())

            async for chunk in stream:
                self.emit(chunk)
                if chunk.turn_end:
                    break
        except BaseException as e:
            if not self.output.done():
                self.output.set_exception(e)
            self.chunks.put_nowait(e)
        finally:
            # Closing wakes the stream consumer once buffered chunks drain, so the
            # turn ends without a sentinel value threading through the queue.
            self.chunks.close()

    async def resolve_output(self, output: Awaitable[AgentOutput]) -> None:
        """Awaits the transport's final output and publishes it as the turn's result."""
        try:
            raw = await output
            response = self.commit_output(raw)
            if not self.output.done():
                self.output.set_result(response)
        except BaseException as e:
            if not self.output.done():
                self.output.set_exception(e)

    def emit(self, chunk: AgentStreamChunk) -> None:
        """Applies any state patch, transforms the wire chunk, and enqueues it."""
        custom = self.commit_custom_patch(chunk.custom_patch) if chunk.custom_patch else None

        agent_chunk: AgentChunk[Any] = AgentChunk(
            text=get_chunk_text(chunk.model_chunk),
            reasoning=getattr(chunk.model_chunk, 'reasoning', None) if chunk.model_chunk else None,
            tool_request=get_chunk_tool_request(chunk.model_chunk),
            tool_response=get_chunk_tool_response(chunk.model_chunk),
            artifact=getattr(chunk, 'artifact', None),
            custom=custom,
            raw=chunk,
        )

        self.chunks.put_nowait(agent_chunk)

    async def stream(self) -> AsyncIterator[AgentChunk[StreamT]]:
        """Yields transformed chunks until the turn ends, re-raising any failure."""
        async for item in self.chunks:
            if isinstance(item, BaseException):
                raise item
            yield item

    async def watch_external_signal(self) -> None:
        """Aborts the turn when the caller's external abort_signal fires."""
        try:
            if self.external_signal is not None and hasattr(self.external_signal, 'wait'):
                await self.external_signal.wait()
            self.abort()
        except asyncio.CancelledError:
            pass

    def cancel_signal_watcher(self) -> None:
        if self.signal_task is not None:
            self.signal_task.cancel()

    def abort(self) -> None:
        """Detaches the client from the turn, leaving the server turn to finish.

        We cancel the local pump and result so the caller stops streaming and
        ``await turn`` settles immediately. The transport keeps draining its
        in-flight turn in the background, so this is a client-side abort only.
        """
        self.aborted.set()
        if not self.output.done():
            # The turn hadn't settled, so its optimistic user message never got a
            # reply. Undo the push before cancelling so the session reads exactly
            # as it did pre-send and the next turn continues cleanly. If the turn
            # already settled we leave it alone — the reply landed.
            if self.rollback is not None:
                self.rollback()
            self.output.cancel()
        if self.run_task is not None and not self.run_task.done():
            self.run_task.cancel()


class AgentSession(Generic[StateT, StreamT]):
    """A stateful conversation session with an agent.

    Public surface: read ``snapshot_id``, ``state``, ``custom``, ``messages``, and
    ``artifacts``; call ``send``, ``resume``, ``detach``, ``abort``, and ``close``.
    Everything prefixed with ``_`` is internal wiring.

    ``state`` is the live ``SessionState`` blob. ``custom``/``messages``/``artifacts``
    are convenience views into it. ``snapshot_id`` is separate — the store resume handle.

    The session keeps ``_snapshot_id`` and ``_state`` (which carries the session id)
    in sync with each turn's output and rebuilds the per-turn resume payload from
    them via ``_wire_init``.
    """

    def __init__(
        self,
        transport: AgentTransport[StateT, StreamT],
        init: AgentInit | None = None,
    ) -> None:
        self._transport = transport
        self._snapshot_id: str | None = None
        self._state = SessionState()

        if init is not None:
            _validate_init(init)
            if init.state is not None:
                self._set_state(init.state)
            elif init.snapshot_id:
                self._snapshot_id = init.snapshot_id
            elif init.session_id:
                self._state.session_id = init.session_id

    @property
    def snapshot_id(self) -> str | None:
        """Store resume handle, kept in sync with the latest turn (store-backed only)."""
        return self._snapshot_id

    @property
    def state(self) -> SessionState:
        """Live session blob for client-managed resume via ``AgentInit(state=...)``."""
        return self._state

    @property
    def session_id(self) -> str | None:
        return self._state.session_id

    @property
    def custom(self) -> StateT | None:
        return cast(StateT | None, self._state.custom)

    @property
    def messages(self) -> list[MessageData]:
        """Running view of the conversation.

        Server-managed: this is a lightweight view, NOT the source of truth. It
        holds your inputs plus each turn's final reply, but not the intermediate
        tool-request/tool-response messages a turn may generate — only the
        snapshot in the store has those. Use ``get_snapshot()`` / ``load_chat``
        for the authoritative history. Client-managed: this is the full history.
        """
        if self._state.messages is None:
            self._state.messages = []
        return self._state.messages

    @property
    def artifacts(self) -> list[Artifact]:
        if self._state.artifacts is None:
            self._state.artifacts = []
        return self._state.artifacts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AgentSession[StateT, StreamT]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.close()

    def send(
        self,
        input: str | AgentInput,
        opts: Any = None,  # noqa: ANN401
    ) -> AgentTurn[StateT, StreamT]:
        """Sends a message to the agent and returns a handle to the in-flight turn."""
        inp = _to_agent_input(input)

        # Capture the resume payload from history *before* this turn's message.
        # The transport carries the new message as the turn input and the agent
        # records it there, so a client-managed payload that also bundled it would
        # land the same message in history twice.
        init = self._wire_init()
        # Optimistically append the user message so it shows immediately; remember
        # the prior length so a failed server-managed turn can roll it back. This
        # length-based rollback assumes turns are single-flight — overlapping
        # sends on one session would race on this list and corrupt the rollback.
        message_count_before = len(self.messages)
        if inp.message:
            self.messages.append(inp.message)

        driver = _TurnDriver(
            inp=inp,
            init=init,
            run_turn=self._transport.run_turn,
            commit_output=lambda raw: self._commit_output(raw, message_count_before),
            commit_custom_patch=self._commit_custom_patch,
            rollback=lambda: self._rollback_optimistic(message_count_before),
            external_signal=_extract_abort_signal(opts),
        )
        return driver.start()

    def resume(self, resume: Resume) -> AgentTurn[StateT, StreamT]:
        """Continues a conversation from an interrupt."""
        inp = AgentInput(resume=resume)
        return self.send(inp)

    async def get_snapshot(self) -> SessionSnapshot | None:
        """Reads the current snapshot from the server store, if store-backed."""
        if not self._snapshot_id:
            return None
        return await self._transport.get_snapshot(snapshot_id=self._snapshot_id)

    async def detach(self, input: str | AgentInput) -> DetachedTask[StateT]:  # noqa: A002
        """Runs a turn in the background on the server and returns a poll handle."""
        inp = _to_agent_input(input)
        inp.detach = True

        init = self._wire_init()
        # Optimistically append the user message so the local history reflects the
        # detached turn; the reply is retrieved later by polling the snapshot.
        message_count_before = len(self.messages)
        if inp.message:
            self.messages.append(inp.message)

        # The transport drives the turn to completion on its own (see
        # AgentTransport.run_turn), so the output resolves whether or not anyone
        # reads the stream. For detach we only care about the resulting handle.
        _stream, output_awaitable = await self._transport.run_turn(inp, init)
        raw_output = await output_awaitable

        self._update_from_output(raw_output, message_count_before)

        if not raw_output.snapshot_id:
            raise ValueError('detach did not return a snapshot_id.')
        return DetachedTask(raw_output.snapshot_id, self._transport)

    async def abort(self) -> SnapshotStatus | None:
        """Stops the session's server-side work by aborting its current snapshot.

        Raises:
            ValueError: if there's no snapshot to abort — the agent is
                client-managed (no store) or no turn has produced a snapshot yet.
                For a client-side stop that just detaches the caller, use
                ``turn.abort()`` instead.
        """
        if not self._snapshot_id:
            raise ValueError(
                'No active snapshot to abort. session.abort() stops server-side work and '
                'needs a store-backed agent with a snapshot (e.g. after detach() or a '
                'completed turn). For a client-side stop, use turn.abort().'
            )
        return await self._transport.abort_snapshot(self._snapshot_id)

    async def close(self) -> None:
        """Cleanly closes the underlying transport."""
        await self._transport.close()

    # ------------------------------------------------------------------
    # Internal (transport / runtime wiring)
    # ------------------------------------------------------------------

    def _load_from_snapshot(self, snapshot: SessionSnapshot) -> None:
        self._snapshot_id = snapshot.snapshot_id
        if snapshot.state is not None:
            self._set_state(snapshot.state)

    def _set_state(self, state: SessionState) -> None:
        self._state = state.model_copy(deep=True)

    def _wire_init(self) -> AgentInit:
        """Builds the resume payload for this turn from the live session state.

        The session doesn't hold onto the original init; it just keeps
        ``_snapshot_id`` and ``_state`` (which carries the session id) synced with
        each turn's output and reconstructs the resume handle every request.
        """
        if self._transport.state_management == 'client':
            # No server store, so the client is the source of truth: ship the
            # full live state every turn.
            return AgentInit(state=self._state.model_copy(deep=True))

        # Server store owns the state; point it at what to load. Prefer the
        # latest snapshot, fall back to the session id, else start fresh.
        if self._snapshot_id:
            return AgentInit(snapshot_id=self._snapshot_id)
        if self._state.session_id:
            return AgentInit(session_id=self._state.session_id)
        return AgentInit()

    def _apply_custom_patch(self, patch: Any) -> None:  # noqa: ANN401
        patch_list = patch.root if hasattr(patch, 'root') else patch
        self._state.custom = apply_json_patch(self._state.custom, patch_list)

    def _commit_output(self, raw: AgentOutput, message_count_before: int) -> AgentResponse[StateT]:
        """Folds a turn's final output into the session and builds the turn result."""
        self._update_from_output(raw, message_count_before)
        return AgentResponse(raw=raw, messages=list(self.messages), custom=self.custom)

    def _commit_custom_patch(self, patch: Any) -> StateT | None:  # noqa: ANN401
        """Applies a streamed custom-state patch and returns the new custom value."""
        self._apply_custom_patch(patch)
        return self.custom

    def _rollback_optimistic(self, message_count_before: int) -> None:
        """Drops the user message ``send`` optimistically pushed for an aborted turn.

        Inverse of the eager append in ``send``: trims the running view back to its
        pre-send length so an aborted turn doesn't strand an unanswered message and
        the next turn resumes from before it. Assumes single-flight turns, same as
        the failed-turn rollback in ``_update_from_output``.
        """
        del self.messages[message_count_before:]

    def _merge_artifacts(self, artifacts: list[Artifact]) -> None:
        """Merge a turn's artifacts into the running view, replacing by name."""
        for art in artifacts:
            name = getattr(art, 'name', None)
            idx = (
                next((i for i, x in enumerate(self.artifacts) if getattr(x, 'name', None) == name), -1) if name else -1
            )
            if idx >= 0:
                self.artifacts[idx] = art
            else:
                self.artifacts.append(art)

    def _update_from_output(self, raw: AgentOutput, message_count_before: int) -> None:
        # message_count_before is the history length captured before this turn's
        # optimistic user-message push, so a failed server-managed turn can roll
        # that push back (see send()).
        if raw.snapshot_id is not None:
            self._snapshot_id = raw.snapshot_id

        if raw.state is not None:
            # Client-managed: the whole session round-trips, so copy it verbatim
            # (``state`` carries the session id too). On a failed turn this is the
            # last-good state, which also discards the optimistic user message.
            self._set_state(raw.state)
            return

        # Server-managed: the durable store is the source of truth and only sends
        # back a snapshot id (+ the turn's final reply). We keep a lightweight
        # running view; custom state stays live via streamed patches.
        if raw.session_id is not None:
            self._state.session_id = raw.session_id

        if raw.finish_reason in (AgentFinishReason.FAILED, AgentFinishReason.ABORTED):
            # No durable reply this turn, so drop the optimistic user message
            # rather than strand it unanswered. The snapshot still holds the truth.
            self._rollback_optimistic(message_count_before)
            return

        if raw.message is not None:
            # Only the final reply comes back, not the turn's intermediate
            # tool-request/tool-response messages — so messages is a view, not the
            # source of truth (see the ``messages`` property).
            self.messages.append(raw.message)
        if raw.artifacts:
            self._merge_artifacts(raw.artifacts)


class DetachedTask(Generic[StateT]):
    """Represents a background agent task running on the server."""

    def __init__(self, snapshot_id: str, transport: AgentTransport[StateT, Any]) -> None:
        self.snapshot_id = snapshot_id
        self._transport = transport

    async def poll(self) -> SessionSnapshot | None:
        """Query the server for the current task status/snapshot."""
        return await self._transport.get_snapshot(snapshot_id=self.snapshot_id)

    async def abort(self) -> SnapshotStatus | None:
        """Aborts the detached task on the server."""
        return await self._transport.abort_snapshot(self.snapshot_id)


# ===========================================================================
# Internal Helper Functions
# ===========================================================================


def chunk_part_roots(model_chunk: ModelResponseChunk | None) -> Iterator[object]:
    """Yields the inner root of each part in a chunk, normalizing dicts to Part."""
    for part in model_chunk.content if model_chunk and model_chunk.content else []:
        p = part if isinstance(part, Part) else Part.model_validate(part)
        yield p.root


def get_chunk_text(model_chunk: ModelResponseChunk | None) -> str | None:
    texts = [root.text for root in chunk_part_roots(model_chunk) if isinstance(root, TextPart)]
    return ''.join(texts) if texts else None


def get_chunk_tool_request(model_chunk: ModelResponseChunk | None) -> ToolRequestPart | None:
    for part in get_chunk_tool_requests(model_chunk):
        return part
    return None


def get_chunk_tool_requests(model_chunk: ModelResponseChunk | None) -> list[ToolRequestPart]:
    return [root for root in chunk_part_roots(model_chunk) if isinstance(root, ToolRequestPart)]


def get_chunk_tool_response(model_chunk: ModelResponseChunk | None) -> ToolResponsePart | None:
    for root in chunk_part_roots(model_chunk):
        if isinstance(root, ToolResponsePart):
            return root
