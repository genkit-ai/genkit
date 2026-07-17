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
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, cast

from typing_extensions import TypeVar as TypeVarExt

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
    Media,
    MediaPart,
    MessageData,
    Part,
    ReasoningPart,
    Resume,
    Role,
    SessionSnapshot as SessionSnapshotSchema,
    SessionState as SessionStateSchema,
    SnapshotStatus,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)

StateT = TypeVarExt('StateT', default=Any)
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')
# The transport protocol only ever hands this type back out (in the sessions it
# returns), never takes it in, so it's covariant.
StateT_co = TypeVar('StateT_co', covariant=True)


class SessionState(SessionStateSchema, Generic[StateT]):
    """Session state generic over custom state."""

    custom: StateT | None = None


class SessionSnapshot(SessionSnapshotSchema, Generic[StateT]):
    """Session snapshot generic over custom state."""

    state: SessionState[StateT] | None = None


# ===========================================================================
# Client Transport Protocol
# ===========================================================================


class AgentTransport(Protocol, Generic[StateT_co]):
    """Interface implemented by the transport layer (local or websocket)."""

    # Declares server- vs client-managed state; must be set explicitly on the transport.
    state_management: StateManagement

    async def run_turn(
        self,
        *,
        agent_input: AgentInput,
        init: AgentInit,
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
    ) -> SessionSnapshotSchema | None:
        """Retrieves a session snapshot from the server store."""
        ...

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts the specified snapshot on the server."""
        ...


# ===========================================================================
# Client Return Types & Models
# ===========================================================================


@dataclass
class AgentChunk(Generic[StateT]):
    """Represents a structured stream chunk yielded during a turn."""

    text: str | None = None
    reasoning: str | None = None
    accumulated_text: str = ''  # this turn's text so far, including this chunk
    tool_requests: list[ToolRequestPart] = field(default_factory=list)
    data: Any | None = None  # structured output part, if the chunk carries one
    media: Media | None = None
    artifact: Artifact | None = None
    custom: StateT | None = None  # post-patch resolved custom state; set when custom_patch is present
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

    def respond(self, output: OutputT) -> ToolResponsePart:
        """Wire-shaped tool response for batching into ``chat.resume(respond=[...])``."""
        return ToolResponsePart(
            tool_response=ToolResponse(
                name=self.name,
                ref=self.ref,
                output=output,
            )
        )

    def restart(
        self,
        *,
        resumed_metadata: dict[str, Any] | None = None,
        replace_input: Any | None = None,  # noqa: ANN401
    ) -> ToolRequestPart:
        """Wire-shaped restart request for batching into ``chat.resume(restart=[...])``."""
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
    state: StateT | None = None

    @property
    def text(self) -> str:
        """Full text content of the response message."""
        return text_of(self.raw.message.content) if self.raw.message else ''

    @property
    def reasoning(self) -> str:
        """Concatenated reasoning the model exposed for this turn."""
        return reasoning_of(self.raw.message.content) if self.raw.message else ''

    @property
    def media(self) -> Media | None:
        """First media part of the response message, if any."""
        return first_media_of(self.raw.message.content) if self.raw.message else None

    @property
    def data(self) -> Any:  # noqa: ANN401
        """Structured-output value of the response message, if the model returned one."""
        return first_data_of(self.raw.message.content) if self.raw.message else None

    @property
    def finish_reason(self) -> AgentFinishReason | None:
        """Why the turn ended."""
        return self.raw.finish_reason

    @property
    def finish_message(self) -> str | None:
        """Human-readable detail when a turn ends abnormally (e.g. blocked or failed)."""
        return self.raw.error.message if self.raw.error else None

    @property
    def snapshot_id(self) -> str | None:
        """Server snapshot id after this turn, if store-backed."""
        return self.raw.snapshot_id

    @property
    def session_id(self) -> str | None:
        """Server session id this turn belongs to, if store-backed."""
        return self.raw.session_id

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
        return tool_requests_of(self.raw.message.content) if self.raw.message else []

    @property
    def interrupts(self) -> list[AgentInterrupt[Any, Any]]:
        """Tool requests that paused this turn."""
        return agent_interrupts_from_message(self.raw.message)

    def assert_valid(self) -> None:
        """Raises if the turn didn't produce a usable reply (blocked, or no message)."""
        if self.raw.finish_reason == AgentFinishReason.BLOCKED:
            detail = f': {self.finish_message}' if self.finish_message else ''
            raise ValueError(f'Generation blocked{detail}.')
        if self.raw.message is None:
            raise ValueError('Agent response has no message.')


def agent_interrupts_from_message(message: MessageData | None) -> list[AgentInterrupt[Any, Any]]:
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


class AgentTurn(Generic[StateT]):
    """A single in-flight turn — iterate for chunks, await for the final response."""

    def __init__(
        self,
        stream: AsyncIterable[AgentChunk[StateT]],
        output: Awaitable[AgentResponse[StateT]],
        abort_fn: Callable[[], Awaitable[None] | None] | None = None,
    ) -> None:
        self._stream = stream
        self._output = output
        self._abort_fn = abort_fn

    def __aiter__(self) -> AsyncIterator[AgentChunk[StateT]]:
        return self._stream_detaching_on_cancel()

    async def _stream_detaching_on_cancel(self) -> AsyncIterator[AgentChunk[StateT]]:
        """Yields chunks, detaching the turn if the consumer is cancelled.

        Wrapping the stream means ``async with asyncio.timeout(...)`` or a plain
        task cancel around the loop behaves like ``turn.abort()`` — the client
        stops listening and the server finishes in the background — then the
        cancellation propagates so the deadline/cancel still surfaces.
        """
        try:
            async for chunk in self._stream:
                yield chunk
        except asyncio.CancelledError:
            await self.abort()
            raise

    def __await__(self) -> Generator[Any, None, AgentResponse[StateT]]:
        """Return the completed turn result. The turn runs whether or not you stream."""
        return self._await_detaching_on_cancel().__await__()

    @property
    def stream(self) -> AsyncIterator[AgentChunk[StateT]]:
        """The turn's chunk stream — the same thing ``async for turn`` iterates.

        Offered so a turn reads like the SDK's other streaming handles, where you
        reach for ``.stream`` / ``.response`` when you want both halves explicitly.
        """
        return self._stream_detaching_on_cancel()

    @property
    def response(self) -> Awaitable[AgentResponse[StateT]]:
        """The turn's final result — the same thing ``await turn`` resolves to."""
        return self._await_detaching_on_cancel()

    async def _await_detaching_on_cancel(self) -> AgentResponse[StateT]:
        """Awaits the result, detaching the turn if the awaiter is cancelled.

        Lets ``async with asyncio.timeout(...): await turn`` (or any task cancel)
        detach the client the same way ``turn.abort()`` would, then re-raises so
        the deadline still surfaces as a TimeoutError/CancelledError.
        """
        try:
            return await self._output
        except asyncio.CancelledError:
            await self.abort()
            raise

    async def abort(self) -> None:
        """Detaches the client from this turn: stops streaming and settles the result now.

        This is a client-side abort. The server turn keeps running to completion
        in the background so its work still lands; we just stop listening and
        resolve the awaited result as cancelled. The prompt you sent stays in
        history — it was still asked — so the session reads like a turn that
        simply got no reply. To actually halt server-side work on a store-backed
        agent, use ``chat.abort()``.
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


class AgentAPI(Protocol, Generic[StateT]):
    """Implemented by both Agent (in-process) and AgentClient (remote)."""

    def chat(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
        messages: list[MessageData] | None = None,
        artifacts: list[Artifact] | None = None,
        state: StateT | None = None,
    ) -> AgentChat[StateT]:
        """Starts a new session, or attaches to one via a snapshot/session id or saved conversation state."""
        ...

    async def load_chat(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> AgentChat[StateT]:
        """Loads a server snapshot and returns a session with history restored."""
        ...

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshotSchema | None:
        """Reads a snapshot without starting a session."""
        ...

    async def abort(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts a running snapshot."""
        ...


class AgentClient(Generic[StateT]):
    """Transport-backed agent client — wraps any AgentTransport and implements AgentAPI.

    This is the one ergonomic surface (``chat``/``load_chat``/``get_snapshot``/
    ``abort``) for talking to an agent, whether it's remote (HTTP transport) or
    in-process (the local agent action). Point it at a transport and you get the
    same client either way.

    ``state_schema`` types the custom session state so the resulting chat hands
    back a validated model instead of a bare dict; leave it None for untyped state.
    """

    def __init__(
        self,
        transport: AgentTransport[StateT],
        *,
        state_schema: type[Any] | None = None,
    ) -> None:
        self._transport = transport
        self._state_schema = state_schema

    def chat(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
        messages: list[MessageData] | None = None,
        artifacts: list[Artifact] | None = None,
        state: StateT | None = None,
    ) -> AgentChat[StateT]:
        """Starts a new session, or attaches to one via a snapshot/session id or saved conversation state."""
        session_transport = copy.copy(self._transport)
        return AgentChat(
            session_transport,
            init_from(snapshot_id, session_id, messages, artifacts, state),
            state_schema=self._state_schema,
        )

    async def load_chat(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> AgentChat[StateT]:
        """Loads a server snapshot and returns a chat with history restored."""
        snapshot = await self._transport.get_snapshot(snapshot_id=snapshot_id, session_id=session_id)
        if snapshot is None:
            raise ValueError(f'Snapshot {lookup_label(snapshot_id=snapshot_id, session_id=session_id)!r} not found.')
        session_transport = copy.copy(self._transport)
        session_transport.state_management = 'server'
        chat = AgentChat(session_transport, state_schema=self._state_schema)
        chat._load_from_snapshot(snapshot)
        return chat

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshotSchema | None:
        """Reads a snapshot without starting a session."""
        return await self._transport.get_snapshot(snapshot_id=snapshot_id, session_id=session_id)

    async def abort(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts a running snapshot on the server."""
        return await self._transport.abort_snapshot(snapshot_id)


def to_agent_input(input: str | AgentInput) -> AgentInput:  # noqa: A002
    """Wraps a plain string in an AgentInput, or copies a passed-in one.

    Always returns a fresh object the caller doesn't own, so per-turn tweaks
    (e.g. flagging detach) never mutate an AgentInput the caller might reuse.
    """
    if isinstance(input, str):
        return AgentInput(message=MessageData(role='user', content=[Part(root=TextPart(text=input))]))
    return input.model_copy()


def init_from(
    snapshot_id: str | None,
    session_id: str | None,
    messages: list[MessageData] | None,
    artifacts: list[Artifact] | None,
    state: Any,  # noqa: ANN401
) -> AgentInit | None:
    """Bundles the chat-attach kwargs into the wire init, or None when all unset."""
    has_state = messages is not None or artifacts is not None or state is not None
    if snapshot_id is None and session_id is None and not has_state:
        return None
    session_state = SessionState(messages=messages, artifacts=artifacts, custom=state) if has_state else None
    return AgentInit(snapshot_id=snapshot_id, session_id=session_id, state=session_state)


def validate_init(init: AgentInit) -> None:
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


def as_part(part: Any) -> Part:  # noqa: ANN401
    return part if isinstance(part, Part) else Part.model_validate(part)


class StreamedMessageAccumulator:
    """Rebuilds a turn's messages from its chunk stream.

    The chunk stream is the one channel that carries a turn's intermediate
    tool-request/tool-response steps; nothing else does. We stitch them back the
    same way the store records them: consecutive model deltas (same role and
    message index) merge into one message; a ``tool`` chunk arrives whole.
    """

    def __init__(self) -> None:
        # Underscored to avoid colliding with the messages() accessor below.
        self._messages: list[MessageData] = []
        self.role: Role | str | None = None
        self.index: float | None = None
        self.parts: list[Part] = []

    def add(self, chunk: AgentStreamChunk) -> None:
        mc = chunk.model_chunk
        if mc is None:
            return
        role = mc.role if mc.role is not None else Role.MODEL
        if self.role is not None and (role != self.role or mc.index != self.index):
            self.flush()
        self.role = role
        self.index = mc.index
        for part in mc.content or []:
            self.parts.append(as_part(part))

    def flush(self) -> None:
        if self.role is None:
            return
        # Merge adjacent text deltas back into a single part while preserving the
        # order of tool/data/media parts as they streamed.
        merged: list[Part] = []
        text_buf: list[str] = []
        for p in self.parts:
            root = p.root
            if isinstance(root, TextPart) and root.text is not None:
                text_buf.append(root.text)
                continue
            if text_buf:
                merged.append(Part(root=TextPart(text=''.join(text_buf))))
                text_buf = []
            merged.append(p)
        if text_buf:
            merged.append(Part(root=TextPart(text=''.join(text_buf))))
        if merged:
            self._messages.append(MessageData(role=self.role, content=merged))
        self.role = None
        self.index = None
        self.parts = []

    def messages(self) -> list[MessageData]:
        """The reconstructed messages, finalizing any in-progress message first."""
        self.flush()
        return self._messages


class RunTurnFn(Protocol):
    """A bound ``AgentTransport.run_turn``, kept as a field so TurnDriver stays transport-agnostic."""

    async def __call__(
        self,
        *,
        agent_input: AgentInput,
        init: AgentInit,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]: ...


class TurnDriver(Generic[StateT]):
    """Runs one in-flight turn and exposes it as an AgentTurn.

    Everything a turn needs lives here so AgentChat.send stays small: it starts
    the transport, pumps chunks onto the caller's stream, resolves the final output,
    and handles abort. Keeping it in one object means a single cancellation path and
    no closures that reference each other before they exist.
    """

    def __init__(
        self,
        inp: AgentInput,
        init: AgentInit,
        *,
        run_turn: RunTurnFn,
        commit_output: Callable[[AgentOutput], AgentResponse[StateT]],
        commit_custom_patch: Callable[[Any], StateT | None],
        accumulate_chunk: Callable[[AgentStreamChunk], None] | None = None,
    ) -> None:
        self.inp = inp
        self.init = init
        self.run_turn = run_turn
        self.commit_output = commit_output
        self.commit_custom_patch = commit_custom_patch
        self.accumulate_chunk = accumulate_chunk
        self.accumulated_text = ''
        self.output: asyncio.Future[AgentResponse[StateT]] = asyncio.get_running_loop().create_future()
        self.chunks: CloseableQueue[AgentChunk[StateT] | BaseException] = CloseableQueue()
        self.run_task: asyncio.Task[None] | None = None
        self.turn: AgentTurn[StateT] = AgentTurn(
            stream=self.stream(),
            output=self.output,
            abort_fn=self.abort,
        )

    def start(self) -> AgentTurn[StateT]:
        """Launches the turn in the background and returns its handle immediately."""
        self.run_task = asyncio.create_task(self.run())
        return self.turn

    async def run(self) -> None:
        """Pumps every chunk to the caller's stream, then resolves the turn's result.

        The transport drives its turn to completion on its own; we drain its chunk
        stream — which also feeds the message accumulator and applies state patches —
        and only then read the authoritative output. Committing after the pump means a
        server-managed turn sees a complete message history (the intermediate messages
        are stitched from chunks, not carried on the output).

        ``self.output`` is otherwise only touched by ``abort()``, so the
        ``done()`` guards let abort win a race without the result being set twice.
        """
        try:
            stream, output = await self.run_turn(agent_input=self.inp, init=self.init)
            async for chunk in stream:
                self.emit(chunk)
                if chunk.turn_end:
                    break
            raw = await output
            if not self.output.done():
                self.output.set_result(self.commit_output(raw))
        except BaseException as e:
            if not self.output.done():
                self.output.set_exception(e)
            self.chunks.put_nowait(e)
        finally:
            # Closing wakes the stream consumer once buffered chunks drain, so the
            # turn ends without a sentinel value threading through the queue.
            self.chunks.close()

    def emit(self, chunk: AgentStreamChunk) -> None:
        """Applies any state patch, transforms the wire chunk, and enqueues it."""
        if self.accumulate_chunk is not None:
            self.accumulate_chunk(chunk)
        custom = self.commit_custom_patch(chunk.custom_patch) if chunk.custom_patch else None

        content = chunk.model_chunk.content if chunk.model_chunk else None
        text = text_of(content)
        self.accumulated_text += text

        agent_chunk: AgentChunk[Any] = AgentChunk(
            text=text or None,
            reasoning=reasoning_of(content) or None,
            accumulated_text=self.accumulated_text,
            tool_requests=tool_requests_of(content),
            data=first_data_of(content),
            media=first_media_of(content),
            artifact=getattr(chunk, 'artifact', None),
            custom=custom,
            raw=chunk,
        )

        self.chunks.put_nowait(agent_chunk)

    async def stream(self) -> AsyncIterator[AgentChunk[StateT]]:
        """Yields transformed chunks until the turn ends, re-raising any failure."""
        async for item in self.chunks:
            if isinstance(item, BaseException):
                raise item
            yield item

    def abort(self) -> None:
        """Detaches the client from the turn, leaving the server turn to finish.

        We cancel the local pump and result so the caller stops streaming and
        ``await turn`` settles immediately. The transport keeps draining its
        in-flight turn in the background, so this is a client-side abort only.
        The optimistic user message stays in history — the prompt was still
        asked, so the running view keeps it like any other turn.
        """
        if not self.output.done():
            self.output.cancel()
        if self.run_task is not None and not self.run_task.done():
            self.run_task.cancel()


class AgentChat(Generic[StateT]):
    """A stateful conversation session with an agent.

    Public surface: read ``snapshot_id``, ``session_id``, ``state``, ``messages``,
    and ``artifacts``; call ``send``, ``resume``, ``detach``, and ``abort``.
    Everything prefixed with ``_`` is internal wiring.

    ``state`` is the agent's custom session state; ``messages`` and ``artifacts``
    are the running conversation and files. The chat tracks all three directly,
    so a client-managed resume just hands them back (``chat(messages=chat.messages,
    state=chat.state, artifacts=chat.artifacts)``). ``snapshot_id`` and
    ``session_id`` are the server store's resume handles.

    The chat keeps these fields in sync with each turn's output and rebuilds the
    per-turn resume payload from them via ``_wire_init``.
    """

    def __init__(
        self,
        transport: AgentTransport[StateT],
        init: AgentInit | None = None,
        *,
        state_schema: type[Any] | None = None,
    ) -> None:
        self._transport = transport
        self._state_schema = state_schema
        self._snapshot_id: str | None = None
        # The snapshot the next turn branches from. Usually the same as
        # _snapshot_id, but a detached turn parks its (speculative) snapshot in
        # _snapshot_id for abort()/get_snapshot() while this stays on the last
        # completed turn — so if that background turn is aborted or fails, the
        # next send() still resumes from solid ground instead of a dead handle.
        self._resume_snapshot_id: str | None = None
        self._session_id: str | None = None
        self._messages: list[MessageData] = []
        self._artifacts: list[Artifact] = []
        # Held as the wire-shaped blob (plain JSON); reads validate it into the
        # declared state model on the way out via _coerce_custom.
        self._custom: Any | None = None
        # Rebuilds a server-managed turn's full message history (including
        # intermediate tool steps) from its chunk stream; created fresh per turn.
        self._turn_accumulator: StreamedMessageAccumulator | None = None

        if init is not None:
            validate_init(init)
            if init.state is not None:
                self._set_state(init.state)
            elif init.snapshot_id:
                self._snapshot_id = init.snapshot_id
                self._resume_snapshot_id = init.snapshot_id
            elif init.session_id:
                self._session_id = init.session_id

    @property
    def snapshot_id(self) -> str | None:
        """Store resume handle, kept in sync with the latest turn (store-backed only)."""
        return self._snapshot_id

    @property
    def state(self) -> StateT | None:
        """The agent's custom session state, kept live as the turn streams."""
        return self._coerce_custom(self._custom)

    def _coerce_custom(self, value: Any) -> StateT | None:  # noqa: ANN401
        """Validate the wire-shaped custom blob into the declared state model.

        Custom state rides the wire as plain JSON, so without a ``state_schema``
        the caller just gets that mapping back. With one, they get a real model
        instance — so reading ``state`` gives typed attribute access instead of a
        bare dict.
        """
        schema = self._state_schema
        if value is None or schema is None or isinstance(value, schema):
            return cast('StateT | None', value)
        return cast('StateT | None', schema.model_validate(value))

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def messages(self) -> list[MessageData]:
        """Running view of the conversation, built the same way in both modes.

        A turn's messages are stitched from its chunk stream — the only channel
        that carries the in-between tool-request/tool-response steps — with the
        final reply taken from the turn's output when it's there (it carries the
        metadata a resume needs). The chunks are identical whether state is server-
        or client-managed, so there's one path here. For server-managed sessions
        the durable snapshot in the store stays the source of truth — use
        ``get_snapshot()`` / ``load_chat`` when you need it (e.g. after a detached
        turn, whose chunks this session never saw).
        """
        return self._messages

    @property
    def artifacts(self) -> list[Artifact]:
        return self._artifacts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(
        self,
        input: str | AgentInput,
    ) -> AgentTurn[StateT]:
        """Sends a message to the agent and returns a handle to the in-flight turn.

        Stream it with ``async for`` or ``await`` it for the final result. To stop
        a turn, call ``turn.abort()`` for a client-side detach (the server finishes
        in the background), or wrap the await/stream in ``asyncio.timeout(...)`` /
        cancel the surrounding task — both detach the same way and then surface the
        deadline. Use ``chat.abort()`` to halt server-side work on a store-backed
        agent.
        """
        inp = to_agent_input(input)

        # Capture the resume payload from history *before* this turn's message.
        # The transport carries the new message as the turn input and the agent
        # records it there, so a client-managed payload that also bundled it would
        # land the same message in history twice.
        init = self._wire_init()
        # Optimistically append the user message so it shows immediately; remember
        # the prior length so a turn that lands no reply (server-side failed/aborted)
        # can roll it back. This length-based rollback assumes turns are single-flight
        # — overlapping sends on one session would race on this list and corrupt it.
        message_count_before = len(self.messages)
        if inp.message:
            self.messages.append(inp.message)

        self._turn_accumulator = StreamedMessageAccumulator()
        driver = TurnDriver(
            inp=inp,
            init=init,
            run_turn=self._transport.run_turn,
            commit_output=lambda raw: self._commit_output(raw, message_count_before),
            commit_custom_patch=self._commit_custom_patch,
            accumulate_chunk=self._turn_accumulator.add,
        )
        return driver.start()

    def resume(
        self,
        *,
        respond: list[ToolResponsePart] | None = None,
        restart: list[ToolRequestPart] | None = None,
    ) -> AgentTurn[StateT]:
        """Continues a conversation from an interrupt.

        ``respond`` answers paused tool requests; ``restart`` re-runs them with
        new metadata. Build each part with the interrupt's ``respond`` / ``restart``
        helpers. Sugar for ``send`` with a resume payload.
        """
        inp = AgentInput(resume=Resume(respond=respond, restart=restart))
        return self.send(inp)

    async def get_snapshot(self) -> SessionSnapshotSchema | None:
        """Reads the current snapshot from the server store, if store-backed."""
        if not self._snapshot_id:
            return None
        return await self._transport.get_snapshot(snapshot_id=self._snapshot_id)

    async def detach(self, input: str | AgentInput) -> DetachedTask[StateT]:  # noqa: A002
        """Runs a turn in the background on the server and returns a poll handle."""
        inp = to_agent_input(input)
        inp.detach = True

        init = self._wire_init()
        # Optimistically append the user message so the local history reflects the
        # detached turn; the reply is retrieved later by polling the snapshot.
        message_count_before = len(self.messages)
        if inp.message:
            self.messages.append(inp.message)

        # This session never sees a detached turn's chunks, so start from an empty
        # accumulator — otherwise a prior turn's leftover messages would be folded
        # in when the output settles.
        self._turn_accumulator = StreamedMessageAccumulator()

        # The transport drives the turn to completion on its own (see
        # AgentTransport.run_turn), so the output resolves whether or not anyone
        # reads the stream. For detach we only care about the resulting handle.
        _stream, output_awaitable = await self._transport.run_turn(agent_input=inp, init=init)
        raw_output = await output_awaitable

        # The detached turn's snapshot is still in flight: it may complete, but it
        # may also be aborted (task.abort()) or fail. Keep the resume handle on the
        # last completed turn so a follow-up send() doesn't try to branch off a
        # snapshot that never settled. _snapshot_id still tracks the detached turn
        # so abort()/get_snapshot() act on it.
        resume_before_detach = self._resume_snapshot_id
        self._update_from_output(raw_output, message_count_before)
        self._resume_snapshot_id = resume_before_detach

        if not raw_output.snapshot_id:
            raise ValueError('detach did not return a snapshot_id.')
        # If the background turn is aborted, the prompt we optimistically pushed
        # was never really answered, so drop it the same way a failed turn does —
        # leaving the chat as if the detach never happened.
        return DetachedTask(
            snapshot_id=raw_output.snapshot_id,
            transport=self._transport,
            state_schema=self._state_schema,
            on_abort_rollback=lambda: self._rollback_optimistic(message_count_before),
        )

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

    # ------------------------------------------------------------------
    # Internal (transport / runtime wiring)
    # ------------------------------------------------------------------

    def _load_from_snapshot(self, snapshot: SessionSnapshotSchema) -> None:
        self._snapshot_id = snapshot.snapshot_id
        self._resume_snapshot_id = snapshot.snapshot_id
        if snapshot.state is not None:
            self._set_state(snapshot.state)

    def _set_state(self, state: SessionStateSchema) -> None:
        snapshot = state.model_copy(deep=True)
        self._session_id = snapshot.session_id
        self._messages = list(snapshot.messages or [])
        self._artifacts = list(snapshot.artifacts or [])
        self._custom = snapshot.custom

    def _session_state(self) -> SessionState:
        """Assembles the wire-shaped state blob from the chat's tracked fields."""
        return SessionState(
            session_id=self._session_id,
            messages=self._messages,
            custom=self._custom,
            artifacts=self._artifacts,
        ).model_copy(deep=True)

    def _wire_init(self) -> AgentInit:
        """Builds the resume payload for this turn from the live session state.

        The session doesn't hold onto the original init; it just keeps the
        tracked fields (and ``_snapshot_id``) synced with each turn's output and
        reconstructs the resume handle every request.
        """
        if self._transport.state_management == 'client':
            # No server store, so the client is the source of truth: ship the
            # full live state every turn.
            return AgentInit(state=self._session_state())

        # Server store owns the state; point it at what to load. Prefer the
        # last completed snapshot, fall back to the session id, else start fresh.
        if self._resume_snapshot_id:
            return AgentInit(snapshot_id=self._resume_snapshot_id)
        if self._session_id:
            return AgentInit(session_id=self._session_id)
        return AgentInit()

    def _apply_custom_patch(self, patch: Any) -> None:  # noqa: ANN401
        patch_list = patch.root if hasattr(patch, 'root') else patch
        self._custom = apply_json_patch(doc=self._custom, patch=patch_list)

    def _commit_output(self, raw: AgentOutput, message_count_before: int) -> AgentResponse[StateT]:
        """Folds a turn's final output into the session and builds the turn result."""
        self._update_from_output(raw, message_count_before)
        return AgentResponse(raw=raw, messages=list(self.messages), state=self.state)

    def _commit_custom_patch(self, patch: Any) -> StateT | None:  # noqa: ANN401
        """Applies a streamed custom-state patch and returns the new custom state."""
        self._apply_custom_patch(patch)
        return self.state

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
        # optimistic user-message push, so a turn that lands no reply can roll
        # that push back (see send()).
        if raw.snapshot_id is not None:
            self._snapshot_id = raw.snapshot_id
            self._resume_snapshot_id = raw.snapshot_id

        if raw.finish_reason in (AgentFinishReason.FAILED, AgentFinishReason.ABORTED):
            # No reply landed this turn, so drop the optimistic user message
            # rather than strand it unanswered; the next turn resumes from before
            # it. The durable snapshot still holds the truth.
            self._rollback_optimistic(message_count_before)
        else:
            self._append_turn_messages(raw)

        self._sync_nonmessage_state(raw)

    def _append_turn_messages(self, raw: AgentOutput) -> None:
        """Extend the running view with this turn's messages.

        Intermediate tool-request/tool-response steps only exist on the chunk
        stream, so they're always taken from the accumulator. The final reply is
        taken from the turn's output when present — it's the copy that carries the
        interrupt and output-format metadata a resume depends on, which the model
        chunks don't.
        """
        streamed = self._turn_accumulator.messages() if self._turn_accumulator is not None else []

        # No output reply this turn (e.g. a long-lived socket that only resolves
        # at close): the streamed model group is the reply, so keep it whole.
        if raw.message is None:
            self.messages.extend(streamed)
            return

        # Prefer the output's reply over the streamed copy of it, so drop that
        # trailing streamed model group before appending the authoritative one.
        if streamed and streamed[-1].role == Role.MODEL:
            streamed = streamed[:-1]
        self.messages.extend(streamed)
        self.messages.append(raw.message)

    def _sync_nonmessage_state(self, raw: AgentOutput) -> None:
        """Refresh the non-message state a turn carries back.

        This is the one place the two modes diverge, and they have to: a
        client-managed session round-trips the whole blob, so the output is
        authoritative for the session id, custom state, and artifacts (the
        last-good values on a failed turn). A server-managed session never puts
        full state on the wire — custom stays live from streamed patches, and we
        only fold in whatever session id / artifacts the output reports.
        """
        if raw.state is not None:
            # Client-managed: the whole session round-trips, so the output is
            # authoritative for the custom state and artifacts. There's no server
            # store to key a session on, so session_id has no meaning here and is
            # always None.
            self._session_id = None
            self._custom = copy.deepcopy(raw.state.custom)
            self._artifacts = [a.model_copy(deep=True) for a in raw.state.artifacts] if raw.state.artifacts else []
            return

        # Server-managed: the store assigns and owns the session id, so adopt it
        # (and any artifacts) from the output.
        if raw.session_id is not None:
            self._session_id = raw.session_id
        if raw.artifacts:
            self._merge_artifacts(raw.artifacts)


TERMINAL_SNAPSHOT_STATUSES = frozenset({
    SnapshotStatus.COMPLETED,
    SnapshotStatus.FAILED,
    SnapshotStatus.ABORTED,
    SnapshotStatus.EXPIRED,
})


class DetachedTask(Generic[StateT]):
    """A handle to a background (detached) turn running on the server."""

    def __init__(
        self,
        *,
        snapshot_id: str,
        transport: AgentTransport[StateT],
        state_schema: type[StateT] | None = None,
        on_abort_rollback: Callable[[], None] | None = None,
    ) -> None:
        self.snapshot_id = snapshot_id
        self._transport = transport
        self._state_schema = state_schema
        self._on_abort_rollback = on_abort_rollback

    def _parse_snapshot(self, raw: SessionSnapshotSchema | None) -> SessionSnapshot[StateT] | None:
        if raw is None:
            return None
        snap = SessionSnapshot[StateT].model_validate(raw.model_dump(by_alias=True))
        if (
            snap.state is not None
            and snap.state.custom is not None
            and self._state_schema is not None
            and hasattr(self._state_schema, 'model_validate')
        ):
            try:
                snap.state.custom = cast(Any, self._state_schema).model_validate(snap.state.custom)
            except Exception:
                if snap.status is not None and snap.status in TERMINAL_SNAPSHOT_STATUSES:
                    raise
        return snap

    async def poll(self, interval: float = 1.0) -> AsyncIterator[SessionSnapshot[StateT]]:
        """Yields the task's snapshot every ``interval`` seconds until it settles.

        Re-reads the server snapshot on a fixed cadence and stops once it reaches
        a terminal status (completed, failed, aborted, or expired), so a caller
        can drive a live status UI with a plain ``async for``. For just the final
        result, await ``wait`` instead.
        """
        while True:
            raw = await self._transport.get_snapshot(snapshot_id=self.snapshot_id)
            snap = self._parse_snapshot(raw)
            if snap is not None:
                yield snap
                if snap.status is not None and snap.status in TERMINAL_SNAPSHOT_STATUSES:
                    return
            await asyncio.sleep(interval)

    async def wait(self, interval: float = 1.0) -> SessionSnapshot[StateT]:
        """Polls until the task settles and returns its final snapshot."""
        last: SessionSnapshot[StateT] | None = None
        async for snapshot in self.poll(interval):
            last = snapshot
        if last is None:
            raise ValueError(f'Detached task {self.snapshot_id} produced no snapshot.')
        return last

    async def abort(self) -> SnapshotStatus | None:
        """Aborts the detached task on the server.

        If the turn was actually aborted (and not already finished by the time the
        abort lands), the originating chat drops the prompt it optimistically held
        for this turn, so its view doesn't strand an unanswered message.
        """
        status = await self._transport.abort_snapshot(self.snapshot_id)
        if status == SnapshotStatus.ABORTED and self._on_abort_rollback is not None:
            self._on_abort_rollback()
        return status


# ===========================================================================
# Internal Helper Functions
# ===========================================================================


def part_roots(content: list[Part] | None) -> Iterator[object]:
    """Yields the inner root of each content part, normalizing dicts to Part."""
    for part in content or []:
        p = part if isinstance(part, Part) else Part.model_validate(part)
        yield p.root


def text_of(content: list[Part] | None) -> str:
    """All text parts concatenated."""
    return ''.join(r.text for r in part_roots(content) if isinstance(r, TextPart) and r.text)


def reasoning_of(content: list[Part] | None) -> str:
    """All reasoning parts concatenated."""
    return ''.join(r.reasoning for r in part_roots(content) if isinstance(r, ReasoningPart) and r.reasoning)


def first_media_of(content: list[Part] | None) -> Media | None:
    """The first media part, if any."""
    for r in part_roots(content):
        if isinstance(r, MediaPart):
            return r.media
    return None


def first_data_of(content: list[Part] | None) -> Any:  # noqa: ANN401
    """The first structured-data part value, if any."""
    for r in part_roots(content):
        data = getattr(r, 'data', None)
        if data is not None:
            return data
    return None


def tool_requests_of(content: list[Part] | None) -> list[ToolRequestPart]:
    """All tool-request parts."""
    return [r for r in part_roots(content) if isinstance(r, ToolRequestPart)]
