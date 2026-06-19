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

"""Agent Session Client APIs for stateful turn-by-turn interactions."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterable, AsyncIterator, Callable, Awaitable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import websockets

from genkit._ai._agent import Agent, AgentConnection
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

StateT = TypeVar("StateT")
StreamT = TypeVar("StreamT")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


# ---------------------------------------------------------------------------
# Structured Helper Types
# ---------------------------------------------------------------------------

@dataclass
class AgentChunk(Generic[StreamT]):
    """Represents a structured stream chunk yielded during a turn."""

    text: str | None = None
    reasoning: str | None = None
    accumulated_text: str | None = None
    tool_request: ToolRequestPart | None = None
    tool_response: ToolResponsePart | None = None
    artifact: Artifact | None = None
    status: StreamT | None = None
    raw: AgentStreamChunk | None = None


class AgentInterrupt(Generic[InputT, OutputT]):
    """Represents a tool request interrupt that paused the turn."""

    def __init__(
        self,
        name: str,
        ref: str | None,
        input_data: InputT,
        session: AgentSession[Any, Any],
    ) -> None:
        self.name = name
        self.ref = ref
        self.input = input_data
        self._session = session

    def respond(self, output: OutputT) -> AgentTurn[Any, Any]:
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

    def restart(self) -> AgentTurn[Any, Any]:
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



# ---------------------------------------------------------------------------
# AgentTurn representation
# ---------------------------------------------------------------------------

class AgentTurn(Generic[StateT, StreamT]):
    """Represents a single active in-flight turn."""

    def __init__(
        self,
        stream: AsyncIterable[AgentChunk[StreamT]],
        output: Awaitable[AgentOutput[StateT]],
        abort_fn: Callable[[], None] | None = None,
    ) -> None:
        self._stream = stream
        self._output = output
        self._abort_fn = abort_fn
        self._interrupt: AgentInterrupt | None = None

    @property
    def stream(self) -> AsyncIterable[AgentChunk[StreamT]]:
        return self._stream

    @property
    def output(self) -> Awaitable[AgentOutput[StateT]]:
        return self._output

    @property
    def interrupt(self) -> AgentInterrupt | None:
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
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# Transport Agnostic Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentTransport(Protocol, Generic[StateT, StreamT]):
    """Protocol for executing agent interactions over a specific transport."""

    async def run_turn(
        self,
        input: AgentInput,
        init: AgentInit[StateT],
        abort_event: asyncio.Event | None = None,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput[StateT]]]:
        """Runs a single turn and returns the stream and output awaitables."""
        ...

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot[StateT] | None:
        """Retrieves a session snapshot from the server store."""
        ...

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts the specified snapshot on the server."""
        ...

    async def close(self) -> None:
        """Cleanly closes any persistent connections held by this transport."""
        ...


# ---------------------------------------------------------------------------
# Stateful AgentSession Client
# ---------------------------------------------------------------------------

class AgentSession(Generic[StateT, StreamT]):
    """A stateful conversation session with an agent."""

    def __init__(
        self,
        transport: AgentTransport[StateT, StreamT],
        connect_init: AgentInit[StateT] | None = None,
    ) -> None:
        self._transport = transport
        self._connect_init = connect_init
        self.snapshot_id: str | None = None
        self.state: StateT | None = None
        self.messages: list[MessageData] = []
        self.artifacts: list[Artifact] = []

        if connect_init:
            if connect_init.snapshot_id:
                self.snapshot_id = connect_init.snapshot_id
            if connect_init.state:
                self._hydrate_from_state(connect_init.state)

    def _hydrate_from_state(self, state: SessionState[StateT]) -> None:
        self.state = state.custom
        self.messages = list(state.messages) if state.messages else []
        self.artifacts = list(state.artifacts) if state.artifacts else []

    def _load_from_snapshot(self, snapshot: SessionSnapshot[StateT]) -> None:
        self.snapshot_id = snapshot.snapshot_id
        self._hydrate_from_state(snapshot.state)

    def _build_init(self) -> AgentInit[StateT]:
        if self.snapshot_id:
            return AgentInit(snapshot_id=self.snapshot_id)
        if self.state is not None:
            return AgentInit(
                state=SessionState(
                    session_id=self.snapshot_id,
                    messages=self.messages,
                    custom=self.state,
                    artifacts=self.artifacts,
                )
            )
        return self._connect_init or AgentInit()

    def send(
        self,
        input: str | AgentInput,
        opts: Any = None,
    ) -> AgentTurn[StateT, StreamT]:
        """Sends a message to the agent for a new turn."""
        agent_input = _to_agent_input(input)
        if agent_input.messages:
            self.messages.extend(agent_input.messages)

        output_future = asyncio.get_event_loop().create_future()
        abort_event = asyncio.Event()

        async def turn_stream_generator() -> AsyncIterator[AgentChunk[StreamT]]:
            try:
                init = self._build_init()
                raw_stream, raw_output = await self._transport.run_turn(
                    agent_input, init, abort_event=abort_event
                )

                accumulated_text = ""
                accumulated_artifacts = []
                async for raw_chunk in raw_stream:
                    text = _get_chunk_text(raw_chunk.model_chunk)
                    if text:
                        accumulated_text += text
                    if raw_chunk.artifact:
                        accumulated_artifacts.append(raw_chunk.artifact)

                    tool_req = _get_chunk_tool_request(raw_chunk.model_chunk)
                    tool_resp = _get_chunk_tool_response(raw_chunk.model_chunk)

                    if raw_chunk.custom_patch:
                        self._apply_custom_patch(raw_chunk.custom_patch)

                    chunk = AgentChunk(
                        text=text,
                        reasoning=getattr(raw_chunk.model_chunk, "reasoning", None) if raw_chunk.model_chunk else None,
                        accumulated_text=accumulated_text if text else None,
                        tool_request=tool_req,
                        tool_response=tool_resp,
                        artifact=raw_chunk.artifact,
                        status=getattr(raw_chunk, "status", None),
                        raw=raw_chunk,
                    )

                    if tool_req:
                        turn._interrupt = AgentInterrupt(
                            name=tool_req.tool_request.name,
                            ref=tool_req.tool_request.ref,
                            input_data=tool_req.tool_request.input,
                            session=self,
                        )

                    yield chunk

                    if raw_chunk.turn_end:
                        break
                        
                # Wait for output and apply updates
                raw_output_res = await raw_output
                
                # Reconstruct final message/artifacts if transport omitted them (e.g. persistent in-process connection)
                if not raw_output_res.message and accumulated_text:
                    raw_output_res.message = MessageData(
                        role="model",
                        content=[Part(root=TextPart(text=accumulated_text))]
                    )
                if not raw_output_res.artifacts and accumulated_artifacts:
                    raw_output_res.artifacts = accumulated_artifacts
                    
                self._apply_output(raw_output_res)
                if not output_future.done():
                    output_future.set_result(raw_output_res)
            except (Exception, asyncio.CancelledError) as e:
                if not output_future.done():
                    output_future.set_exception(e)
                raise e

        turn = AgentTurn(
            stream=turn_stream_generator(),
            output=output_future,
            abort_fn=lambda: abort_event.set(),
        )
        return turn

    def resume(
        self,
        resume: Resume,
        opts: Any = None,
    ) -> AgentTurn[StateT, StreamT]:
        """Resumes a turn after an interrupt."""
        return self.send(AgentInput(resume=resume), opts)

    async def detach(self, input: str | AgentInput) -> Any:
        """Detaches the session connection for background execution."""
        agent_input = {**_to_agent_input(input).model_dump(by_alias=True), "detach": True}
        agent_input_parsed = AgentInput.model_validate(agent_input)

        if agent_input_parsed.messages:
            self.messages.extend(agent_input_parsed.messages)

        init = self._build_init()
        _, raw_output_awaitable = await self._transport.run_turn(agent_input_parsed, init)
        raw_output = await raw_output_awaitable
        self._apply_output(raw_output)

        if not raw_output.snapshot_id:
            raise ValueError("detach did not return a snapshot_id.")
        return DetachedTask(raw_output.snapshot_id, self._transport)

    async def abort(self) -> SnapshotStatus | None:
        """Aborts the active snapshot on the server."""
        if not self.snapshot_id:
            return None
        return await self._transport.abort_snapshot(self.snapshot_id)

    async def close(self) -> None:
        """Cleanly closes the underlying transport."""
        await self._transport.close()

    def _apply_custom_patch(self, patch: Any) -> None:
        patch_list = patch.root if hasattr(patch, "root") else patch
        self.state = apply_json_patch(self.state, patch_list)

    def _apply_output(self, raw: AgentOutput[StateT]) -> None:
        if raw.snapshot_id is not None:
            self.snapshot_id = raw.snapshot_id
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

    async def __aenter__(self) -> AgentSession[StateT, StreamT]:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Background Task Detached Task
# ---------------------------------------------------------------------------

class DetachedTask(Generic[StateT]):
    """Represents a background running task."""

    def __init__(self, snapshot_id: str, transport: AgentTransport[StateT, Any]) -> None:
        self.snapshot_id = snapshot_id
        self._transport = transport

    async def poll(self, interval_ms: int = 2000) -> AsyncIterator[SessionSnapshot[StateT]]:
        """Periodically polls server for session state until terminal state is reached."""
        while True:
            snap = await self._transport.get_snapshot(self.snapshot_id)
            if snap is None:
                raise ValueError(f"Snapshot {self.snapshot_id} not found during polling.")
            yield snap
            if snap.status in (SnapshotStatus.DONE, SnapshotStatus.FAILED, SnapshotStatus.ABORTED):
                return
            await asyncio.sleep(interval_ms / 1000.0)

    async def wait(self, interval_ms: int = 2000) -> SessionSnapshot[StateT]:
        """Blocks until the background task reaches a terminal state."""
        last_snap = None
        async for snap in self.poll(interval_ms=interval_ms):
            last_snap = snap
        if last_snap is None:
            raise ValueError(f"Task {self.snapshot_id} did not yield any snapshots.")
        return last_snap

    async def abort(self) -> SnapshotStatus | None:
        """Aborts the background task."""
        return await self._transport.abort_snapshot(self.snapshot_id)


# ---------------------------------------------------------------------------
# Unified AgentAPI Interface
# ---------------------------------------------------------------------------

class AgentAPI(Generic[StateT, StreamT]):
    """API Client representing a stateful connector to an agent."""

    def __init__(self, transport: AgentTransport[StateT, StreamT], info: Any = None) -> None:
        self._transport = transport
        self.info = info

    def connect(self, init: AgentInit[StateT] | None = None) -> AgentSession[StateT, StreamT]:
        """Starts a new session, or attaches to one via init."""
        return AgentSession(self._transport, init)

    async def resume(self, snapshot_id: str) -> AgentSession[StateT, StreamT]:
        """Loads a server snapshot and returns a session with history restored."""
        snapshot = await self._transport.get_snapshot(snapshot_id)
        if snapshot is None:
            raise ValueError(f"Snapshot {snapshot_id} not found.")
        session = AgentSession(self._transport)
        session._load_from_snapshot(snapshot)
        return session

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot[StateT] | None:
        """Reads a snapshot without starting a session."""
        return await self._transport.get_snapshot(snapshot_id)

    async def abort(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts a running snapshot on the server."""
        return await self._transport.abort_snapshot(snapshot_id)


# ---------------------------------------------------------------------------
# Core Helpers
# ---------------------------------------------------------------------------

def _to_agent_input(input_val: str | AgentInput) -> AgentInput:
    if isinstance(input_val, str):
        return AgentInput(
            messages=[
                MessageData(
                    role="user",
                    content=[Part(root=TextPart(text=input_val))],
                )
            ]
        )
    return input_val


def _get_chunk_text(model_chunk: ModelResponseChunk | None) -> str | None:
    if not model_chunk or not model_chunk.content:
        return None
    texts = []
    for part in model_chunk.content:
        p = part if isinstance(part, Part) else Part.model_validate(part)
        if isinstance(p.root, TextPart):
            texts.append(p.root.text)
    return "".join(texts) if texts else None


def _get_chunk_tool_request(model_chunk: ModelResponseChunk | None) -> ToolRequestPart | None:
    if not model_chunk or not model_chunk.content:
        return None
    for part in model_chunk.content:
        p = part if isinstance(part, Part) else Part.model_validate(part)
        if isinstance(p.root, ToolRequestPart):
            return p.root
    return None


def _get_chunk_tool_response(model_chunk: ModelResponseChunk | None) -> ToolResponsePart | None:
    if not model_chunk or not model_chunk.content:
        return None
    for part in model_chunk.content:
        p = part if isinstance(part, Part) else Part.model_validate(part)
        if isinstance(p.root, ToolResponsePart):
            return p.root
    return None


class InProcessAgentTransport(AgentTransport[StateT, StreamT]):
    """Local, in-process agent transport that executes the agent action directly."""

    def __init__(self, agent: Agent, store_configured: bool) -> None:
        self._agent = agent
        self.state_management = "server" if store_configured else "client"
        self._conn: AgentConnection | None = None

    async def run_turn(
        self,
        input: AgentInput,
        init: AgentInit[StateT],
        abort_event: asyncio.Event | None = None,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput[StateT]]]:
        if self._conn is None:
            self._conn = await self._agent.stream_bidi(init=init)
        conn = self._conn

        await conn.send(input)

        output_future = asyncio.get_event_loop().create_future()

        async def watch_abort() -> None:
            if abort_event is not None:
                await abort_event.wait()
                await conn.close()
                self._conn = None
                try:
                    await conn.output()
                except asyncio.CancelledError:
                    pass
                if not output_future.done():
                    output_future.set_exception(asyncio.CancelledError())

        abort_task = asyncio.create_task(watch_abort())
        output_future.add_done_callback(lambda _: abort_task.cancel())

        async def stream_generator() -> AsyncIterator[AgentStreamChunk]:
            try:
                async for chunk in conn.receive():
                    if chunk.turn_end:
                        snapshot_id = chunk.turn_end.snapshot_id
                        state = None
                        message = None
                        artifacts = []
                        if snapshot_id:
                            snap = await self.get_snapshot(snapshot_id)
                            if snap and snap.state:
                                state = snap.state
                                if snap.state.messages:
                                    message = snap.state.messages[-1]
                                if snap.state.artifacts:
                                    artifacts = list(snap.state.artifacts)

                        output = AgentOutput(
                            snapshot_id=snapshot_id,
                            finish_reason=chunk.turn_end.finish_reason,
                            error=getattr(chunk.turn_end, "error", None),
                            state=state,
                            message=message,
                            artifacts=artifacts,
                        )
                        if not output_future.done():
                            output_future.set_result(output)
                    yield chunk
                    if chunk.turn_end:
                        break
            except Exception as e:
                if not output_future.done():
                    output_future.set_exception(e)
                raise e

        return stream_generator(), output_future

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot[StateT] | None:
        store = getattr(self._agent, "_store", None)
        if store is None:
            return None
        return await store.get_snapshot(snapshot_id=snapshot_id)

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        store = getattr(self._agent, "_store", None)
        if store is None or not hasattr(store, "abort_snapshot"):
            return None
        return await store.abort_snapshot(snapshot_id)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


class WebSocketAgentTransport(AgentTransport[StateT, StreamT]):
    """Client-side agent transport that talks to a remote agent over WebSockets."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.state_management = "server"
        self._ws: Any = None
        self._receive_task: asyncio.Task | None = None
        self._active_stream_queue: asyncio.Queue[AgentStreamChunk] | None = None
        self._active_output_future: asyncio.Future[AgentOutput[StateT]] | None = None

    async def _read_loop(self) -> None:
        try:
            ws = self._ws
            if ws is None:
                return
            async for message in ws:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                data = json.loads(message)
                
                try:
                    chunk = AgentStreamChunk.model_validate(data)
                    if self._active_stream_queue:
                        await self._active_stream_queue.put(chunk)
                    
                    if chunk.turn_end:
                        output = AgentOutput(
                            snapshot_id=chunk.turn_end.snapshot_id,
                            finish_reason=chunk.turn_end.finish_reason,
                            error=getattr(chunk.turn_end, "error", None)
                        )
                        if self._active_output_future and not self._active_output_future.done():
                            self._active_output_future.set_result(output)
                except Exception:
                    if "output" in data:
                        output = AgentOutput.model_validate(data["output"])
                        if self._active_output_future and not self._active_output_future.done():
                            self._active_output_future.set_result(output)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._active_output_future and not self._active_output_future.done():
                self._active_output_future.set_exception(e)
        finally:
            self._ws = None

    async def run_turn(
        self,
        input: AgentInput,
        init: AgentInit[StateT],
        abort_event: asyncio.Event | None = None,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput[StateT]]]:
        ws = self._ws
        if ws is None:
            ws = await websockets.connect(self.url)
            self._ws = ws
            self._receive_task = asyncio.create_task(self._read_loop())

        queue = asyncio.Queue()
        self._active_stream_queue = queue
        self._active_output_future = asyncio.get_event_loop().create_future()

        frame = {
            "input": input.model_dump(by_alias=True, exclude_none=True),
            "init": init.model_dump(by_alias=True, exclude_none=True),
        }
        await ws.send(json.dumps(frame))

        async def stream_generator() -> AsyncIterator[AgentStreamChunk]:
            while True:
                chunk = await queue.get()
                yield chunk
                if chunk.turn_end:
                    break

        return stream_generator(), self._active_output_future

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot[StateT] | None:
        raise NotImplementedError("Snapshot lookup not implemented over WebSocket.")

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        raise NotImplementedError("Snapshot abort not implemented over WebSocket.")

    async def close(self) -> None:
        if self._ws:
            ws = self._ws
            self._ws = None
            await ws.close()
        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None

