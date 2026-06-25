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
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

from genkit._ai._json_patch import apply_json_patch
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
        """
        ...

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot | None:
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


class AgentInterrupt(Generic[InputT, OutputT, StateT, StreamT]):
    """Represents a tool request interrupt that paused the turn."""

    def __init__(
        self,
        name: str,
        ref: str | None,
        input_data: InputT,
        session: AgentSession[StateT, StreamT],
    ) -> None:
        self.name = name
        self.ref = ref
        self.input = input_data
        self._session = session

    def respond(self, output: OutputT) -> AgentTurn[StateT, StreamT]:
        """Builds a resume turn to answer the interrupt and continue the session."""
        resume_payload = Resume(
            respond=[
                ToolResponsePart(
                    tool_response=ToolResponse(
                        name=self.name,
                        ref=self.ref,
                        output=output,
                    )
                )
            ]
        )
        return self._session.resume(resume_payload)

    def restart(self) -> AgentTurn[StateT, StreamT]:
        """Builds a resume turn that re-issues the tool call unchanged."""
        resume_payload = Resume(
            restart=[
                ToolRequestPart(
                    tool_request=ToolRequest(
                        name=self.name,
                        ref=self.ref,
                        input=self.input,
                    )
                )
            ]
        )
        return self._session.resume(resume_payload)


@dataclass
class AgentResponse(Generic[StateT]):
    """Completed turn result — client-side wrapper around AgentOutput with rich accessors."""

    raw: AgentOutput
    messages: list[MessageData]
    state: StateT | None = None

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


class AgentTurn(Generic[StateT, StreamT]):
    """Represents a single active in-flight turn."""

    def __init__(
        self,
        stream: AsyncIterable[AgentChunk[StreamT]],
        output: Awaitable[AgentResponse[StateT]],
        abort_fn: Callable[[], Awaitable[None] | None] | None = None,
    ) -> None:
        self._stream = stream
        self._output = output
        self._abort_fn = abort_fn
        self._interrupt: AgentInterrupt[Any, Any, StateT, StreamT] | None = None

    @property
    def stream(self) -> AsyncIterable[AgentChunk[StreamT]]:
        return self._stream

    def __aiter__(self) -> AsyncIterator[AgentChunk[StreamT]]:
        """Allow iterating directly over the turn to stream chunks."""
        return self._stream.__aiter__()

    def __await__(self) -> Generator[Any, None, AgentResponse[StateT]]:
        """Allow ``await turn`` to run the turn and return its final response.

        Streaming the chunks is optional — the turn runs in the background either
        way — so awaiting the turn directly is the same as awaiting ``turn.output``.
        """
        return self._output.__await__()

    @property
    def output(self) -> Awaitable[AgentResponse[StateT]]:
        return self._output

    @property
    def interrupt(self) -> AgentInterrupt[Any, Any, StateT, StreamT] | None:
        """Returns the interrupt if the turn paused on one, otherwise None."""
        return self._interrupt

    async def abort(self) -> None:
        """Aborts this turn and blocks until all server/client tasks are fully terminated."""
        if self._abort_fn:
            res = self._abort_fn()
            if inspect.isawaitable(res):
                await res
        try:
            await self._output
        except (asyncio.CancelledError, Exception):  # noqa: S110
            pass


# ===========================================================================
# Client APIs & Session Handles
# ===========================================================================


@runtime_checkable
class AgentAPI(Protocol, Generic[StateT, StreamT]):
    """Implemented by both Agent (in-process) and AgentClient (remote)."""

    def chat(self, init: AgentInit | None = None) -> AgentSession[StateT, StreamT]:
        """Starts a new session, or attaches to one via init."""
        ...

    async def load_chat(self, snapshot_id: str) -> AgentSession[StateT, StreamT]:
        """Loads a server snapshot and returns a session with history restored."""
        ...

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot | None:
        """Reads a snapshot without starting a session."""
        ...

    async def abort(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts a running snapshot."""
        ...


class AgentClient(Generic[StateT, StreamT]):
    """Remote/transport-backed agent — wraps any AgentTransport. Implements AgentAPI."""

    def __init__(self, transport: AgentTransport[StateT, StreamT], info: object = None) -> None:
        self._transport = transport
        self.info = info

    def chat(self, init: AgentInit | None = None) -> AgentSession[StateT, StreamT]:
        """Starts a new session, or attaches to one via init."""
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


class _TurnDriver(Generic[StateT, StreamT]):
    """Runs one in-flight turn and exposes it as an AgentTurn.

    Everything a turn needs lives here so AgentSession.send stays small: it starts
    the transport, pumps chunks onto the caller's stream, resolves the final output,
    surfaces interrupts, and handles abort. Keeping it in one object means a single
    cancellation path and no closures that reference each other before they exist.
    """

    def __init__(
        self,
        session: AgentSession[StateT, StreamT],
        inp: AgentInput,
        init: AgentInit,
        external_signal: Any = None,  # noqa: ANN401
    ) -> None:
        self._session = session
        self._inp = inp
        self._init = init
        self._external_signal = external_signal
        self._aborted = asyncio.Event()
        self._output: asyncio.Future[AgentResponse[StateT]] = asyncio.get_event_loop().create_future()
        self._chunks: asyncio.Queue[AgentChunk[StreamT] | BaseException | None] = asyncio.Queue()
        self._run_task: asyncio.Task[None] | None = None
        self._signal_task: asyncio.Task[None] | None = None
        # Built up front so the pump can record interrupts on it without a forward ref.
        self._turn: AgentTurn[StateT, StreamT] = AgentTurn(
            stream=self._stream(),
            output=self._output,
            abort_fn=self.abort,
        )

    def start(self) -> AgentTurn[StateT, StreamT]:
        """Launches the turn in the background and returns its handle immediately."""
        self._output.add_done_callback(lambda _: self._cancel_signal_watcher())
        self._run_task = asyncio.create_task(self._run())
        if self._external_signal is not None:
            self._signal_task = asyncio.create_task(self._watch_external_signal())
        return self._turn

    async def _run(self) -> None:
        """Pumps chunks to the caller's stream while a sibling task resolves the output.

        Per the transport contract, the ``output`` awaitable is the authoritative
        "turn is done" signal and settles on its own. We resolve ``self._output`` from a
        sibling task rather than awaiting ``output`` after the pump loop so the turn's
        *result* stays decoupled from the *stream's* health.

        - If ``_emit`` chokes on a bad chunk after the transport has already produced a
          valid output, the resolver has set ``self._output`` from the real result, so the
          decode error only surfaces on the stream — a good turn isn't poisoned by a
          plumbing hiccup. Sequencing ``await output`` after the pump would instead dump
          that exception onto ``self._output``.
        - The pump and resolver can settle in either order; both guard on
          ``self._output.done()`` so whoever's first wins and the other is a no-op
          (including ``abort()`` cancelling ``self._output`` directly).
        - The done-callback cancels the resolver once ``self._output`` is settled by
          anyone, so the task never dangles.
        """
        try:
            stream, output = await self._session._transport.run_turn(self._inp, self._init, abort_event=self._aborted)
            resolve_task = asyncio.create_task(self._resolve_output(output))
            self._output.add_done_callback(lambda _: resolve_task.cancel())

            async for chunk in stream:
                self._emit(chunk)
                if chunk.turn_end:
                    break
        except BaseException as e:
            if not self._output.done():
                self._output.set_exception(e)
            self._chunks.put_nowait(e)
        finally:
            self._chunks.put_nowait(None)

    async def _resolve_output(self, output: Awaitable[AgentOutput]) -> None:
        """Awaits the transport's final output and publishes it as the turn's result."""
        try:
            raw = await output
            self._session.update_from_output(raw)
            if not self._output.done():
                self._output.set_result(
                    AgentResponse(raw=raw, messages=list(self._session.messages), state=self._session.state)
                )
        except BaseException as e:
            if not self._output.done():
                self._output.set_exception(e)

    def _emit(self, chunk: AgentStreamChunk) -> None:
        """Applies any state patch, transforms the wire chunk, and records an interrupt."""
        if chunk.custom_patch:
            self._session.apply_custom_patch(chunk.custom_patch)

        agent_chunk: AgentChunk[Any] = AgentChunk(
            text=get_chunk_text(chunk.model_chunk),
            reasoning=getattr(chunk.model_chunk, 'reasoning', None) if chunk.model_chunk else None,
            tool_request=get_chunk_tool_request(chunk.model_chunk),
            tool_response=get_chunk_tool_response(chunk.model_chunk),
            artifact=getattr(chunk, 'artifact', None),
            custom=self._session.state if chunk.custom_patch else None,
            raw=chunk,
        )

        if agent_chunk.tool_request:
            request = agent_chunk.tool_request.tool_request
            self._turn._interrupt = AgentInterrupt(
                name=request.name,
                ref=request.ref,
                input_data=request.input,
                session=self._session,
            )

        self._chunks.put_nowait(agent_chunk)

    async def _stream(self) -> AsyncIterator[AgentChunk[StreamT]]:
        """Yields transformed chunks until the turn ends, re-raising any failure."""
        while True:
            item = await self._chunks.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    async def _watch_external_signal(self) -> None:
        """Aborts the turn when the caller's external abort_signal fires."""
        try:
            if self._external_signal is not None and hasattr(self._external_signal, 'wait'):
                await self._external_signal.wait()
            self.abort()
        except asyncio.CancelledError:
            pass

    def _cancel_signal_watcher(self) -> None:
        if self._signal_task is not None:
            self._signal_task.cancel()

    def abort(self) -> None:
        """Tells the transport to stop and cancels the in-flight turn."""
        self._aborted.set()
        if not self._output.done():
            self._output.cancel()
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()


class AgentSession(Generic[StateT, StreamT]):
    """A stateful conversation session with an agent."""

    def __init__(
        self,
        transport: AgentTransport[StateT, StreamT],
        connect_init: AgentInit | None = None,
    ) -> None:
        self._transport = transport
        self._connect_init = connect_init
        self.snapshot_id: str | None = None
        self.session_id: str | None = (connect_init.session_id if connect_init else None) or (
            connect_init.state.session_id if connect_init and connect_init.state else None
        )
        self.state: StateT | None = None
        self.messages: list[MessageData] = []
        self.artifacts: list[Artifact] = []

        if connect_init:
            if connect_init.snapshot_id:
                self.snapshot_id = connect_init.snapshot_id
            if connect_init.state:
                self.hydrate_from_state(connect_init.state)

    async def __aenter__(self) -> AgentSession[StateT, StreamT]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.close()

    def hydrate_from_state(self, state: SessionState) -> None:
        self.state = state.custom
        if state.session_id:
            self.session_id = state.session_id
        self.messages = list(state.messages) if state.messages else []
        self.artifacts = list(state.artifacts) if state.artifacts else []

    def load_from_snapshot(self, snapshot: SessionSnapshot) -> None:
        self.snapshot_id = snapshot.snapshot_id
        if snapshot.state is not None:
            self.hydrate_from_state(snapshot.state)

    def build_init(self) -> AgentInit:
        if self.snapshot_id:
            return AgentInit(snapshot_id=self.snapshot_id)
        if self._client_managed():
            if self.session_id is None:
                self.session_id = str(uuid4())
            return AgentInit(
                state=SessionState(
                    session_id=self.session_id,
                    messages=self.messages,
                    custom=self.state,
                    artifacts=self.artifacts,
                )
            )
        if self.state is not None:
            if self.session_id is None:
                self.session_id = str(uuid4())
            return AgentInit(
                state=SessionState(
                    session_id=self.session_id,
                    messages=self.messages,
                    custom=self.state,
                    artifacts=self.artifacts,
                )
            )
        return self._connect_init or AgentInit()

    def _client_managed(self) -> bool:
        # In-process transport exposes store=None when the agent has no session store.
        # Other transports (e.g. HTTP) omit the attribute — server owns identity there.
        return getattr(self._transport, 'store', ...) is None

    def send(
        self,
        input: str | AgentInput,
        opts: Any = None,  # noqa: ANN401
    ) -> AgentTurn[StateT, StreamT]:
        """Sends a message to the agent and returns a handle to the in-flight turn."""
        inp = _to_agent_input(input)
        init = self.build_init()
        if inp.message:
            self.messages.append(inp.message)

        driver = _TurnDriver(
            session=self,
            inp=inp,
            init=init,
            external_signal=_extract_abort_signal(opts),
        )
        return driver.start()

    def resume(self, resume: Resume) -> AgentTurn[StateT, StreamT]:
        """Continues a conversation from an interrupt."""
        inp = AgentInput(resume=resume)
        return self.send(inp)

    async def get_snapshot(self) -> SessionSnapshot | None:
        if not self.snapshot_id:
            return None
        return await self._transport.get_snapshot(self.snapshot_id)

    async def detach(self, input: str | AgentInput) -> DetachedTask[StateT]:  # noqa: A002
        inp = _to_agent_input(input)
        inp.detach = True
        init = self.build_init()

        # The transport drives the turn to completion on its own (see
        # AgentTransport.run_turn), so the output resolves whether or not anyone
        # reads the stream. For detach we only care about the resulting handle.
        _stream, output_awaitable = await self._transport.run_turn(inp, init)
        raw_output = await output_awaitable

        self.update_from_output(raw_output)

        if not raw_output.snapshot_id:
            raise ValueError('detach did not return a snapshot_id.')
        return DetachedTask(raw_output.snapshot_id, self._transport)

    async def abort(self) -> SnapshotStatus | None:
        """Aborts the active snapshot on the server."""
        if not self.snapshot_id:
            return None
        return await self._transport.abort_snapshot(self.snapshot_id)

    async def close(self) -> None:
        """Cleanly closes the underlying transport."""
        await self._transport.close()
        # For in-process client-managed agents the full state (preamble stripped)
        # is only available after the invocation completes.  Capture it here.
        # TODO: Investigate avoiding getattr here as it is a leaky abstraction.
        # e.g. transport.close() could return the final output or None.
        final = getattr(self._transport, 'final_output', None)
        if final is not None and final.state is not None:
            self.update_from_output(final)

    def apply_custom_patch(self, patch: Any) -> None:  # noqa: ANN401
        patch_list = patch.root if hasattr(patch, 'root') else patch
        self.state = apply_json_patch(self.state, patch_list)

    def update_from_output(self, raw: AgentOutput) -> None:
        # session_id and snapshot_id are the output's identity envelope and always
        # live at the top level; state is just the payload (custom/messages/artifacts).
        if raw.snapshot_id is not None:
            self.snapshot_id = raw.snapshot_id
        if raw.session_id is not None:
            self.session_id = raw.session_id
        if raw.state is not None:
            self.state = raw.state.custom
            if raw.state.messages:
                self.messages = list(raw.state.messages)
            if raw.state.artifacts:
                self.artifacts = list(raw.state.artifacts)
        else:
            if raw.message:
                self.messages.append(raw.message)
            if raw.artifacts:
                self.artifacts.extend(raw.artifacts)


class DetachedTask(Generic[StateT]):
    """Represents a background agent task running on the server."""

    def __init__(self, snapshot_id: str, transport: AgentTransport[StateT, Any]) -> None:
        self.snapshot_id = snapshot_id
        self._transport = transport

    async def poll(self) -> SessionSnapshot | None:
        """Query the server for the current task status/snapshot."""
        return await self._transport.get_snapshot(self.snapshot_id)

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
    for root in chunk_part_roots(model_chunk):
        if isinstance(root, ToolRequestPart):
            return root


def get_chunk_tool_response(model_chunk: ModelResponseChunk | None) -> ToolResponsePart | None:
    for root in chunk_part_roots(model_chunk):
        if isinstance(root, ToolResponsePart):
            return root
