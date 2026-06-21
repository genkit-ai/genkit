from __future__ import annotations
import asyncio
import inspect
from collections.abc import AsyncIterable, AsyncIterator, Callable, Awaitable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable
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
    Role,
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


@runtime_checkable
class AgentTransport(Protocol, Generic[StateT, StreamT]):
    """Protocol for executing agent interactions over a specific transport."""

    async def run_turn(
        self,
        input: AgentInput,
        init: AgentInit[StateT],
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


@runtime_checkable
class AgentAPI(Protocol, Generic[StateT, StreamT]):
    """Unified Protocol representing any client or local factory for interacting with an agent."""

    def connect(self, init: AgentInit[StateT] | None = None) -> AgentSession[StateT, StreamT]:
        """Starts a new session, or attaches to one via init."""
        ...

    async def resume(self, snapshot_id: str) -> AgentSession[StateT, StreamT]:
        """Loads a server snapshot and returns a session with history restored."""
        ...

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot[StateT] | None:
        """Reads a snapshot without starting a session."""
        ...

    async def abort(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts a running snapshot on the server."""
        ...


class AgentClient(Generic[StateT, StreamT]):
    """Concrete implementation of the AgentAPI protocol wrapping a transport."""

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

    async def __aenter__(self) -> AgentSession[StateT, StreamT]:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

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
        if isinstance(input, str):
            inp = AgentInput(
                messages=[
                    MessageData(
                        role='user',
                        content=[Part(root=TextPart(text=input))],
                    )
                ]
            )
        else:
            inp = input

        if inp.messages:
            self.messages.extend(inp.messages)

        init = self._build_init()
        cancelled_event = asyncio.Event()

        # Check for external abort_signal in opts
        external_signal = None
        if isinstance(opts, dict):
            external_signal = opts.get("abort_signal")
        elif opts and hasattr(opts, "abort_signal"):
            external_signal = getattr(opts, "abort_signal")

        turn_output_future = asyncio.get_event_loop().create_future()
        accumulated_text = ""
        accumulated_artifacts: list[Artifact] = []

        watch_sig_task = None
        if external_signal is not None:
            async def _watch_external():
                try:
                    await external_signal.wait()
                    cancelled_event.set()
                except asyncio.CancelledError:
                    pass
            watch_sig_task = asyncio.create_task(_watch_external())

        async def _run() -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput[StateT]]]:
            return await self._transport.run_turn(
                inp,
                init,
            )

        run_task = asyncio.create_task(_run())

        def cleanup_sig_task(_):
            if watch_sig_task:
                watch_sig_task.cancel()
        turn_output_future.add_done_callback(cleanup_sig_task)

        async def stream_generator() -> AsyncIterator[AgentChunk[StreamT]]:
            try:
                stream, output_awaitable = await run_task
            except BaseException as e:
                if not turn_output_future.done():
                    turn_output_future.set_exception(e)
                if watch_sig_task:
                    watch_sig_task.cancel()
                raise e

            async def watch_output() -> None:
                try:
                    res = await output_awaitable
                    if not res.message and accumulated_text:
                        res.message = MessageData(
                            role="model",
                            content=[Part(root=TextPart(text=accumulated_text))]
                        )
                    if not res.artifacts and accumulated_artifacts:
                        res.artifacts = list(accumulated_artifacts)
                    self._apply_output(res)
                    if not turn_output_future.done():
                        turn_output_future.set_result(res)
                except BaseException as e:
                    if not turn_output_future.done():
                        turn_output_future.set_exception(e)
                    raise

            watch_output_task = asyncio.create_task(watch_output())

            try:
                nonlocal accumulated_text, accumulated_artifacts
                stream_iter = stream.__aiter__()

                while not cancelled_event.is_set():
                    # Wait for either the next chunk OR the cancelled_event to fire
                    next_chunk_task = asyncio.create_task(stream_iter.__anext__())
                    wait_cancelled_task = asyncio.create_task(cancelled_event.wait())

                    done, pending = await asyncio.wait(
                        {next_chunk_task, wait_cancelled_task},
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # Cancel all pending tasks to prevent resource leaks
                    for t in pending:
                        t.cancel()

                    if cancelled_event.is_set():
                        next_chunk_task.cancel()
                        if not turn_output_future.done():
                            turn_output_future.cancel()
                        break

                    if next_chunk_task in done:
                        try:
                            chunk = next_chunk_task.result()

                            text = _get_chunk_text(chunk.model_chunk)
                            if text:
                                accumulated_text += text
                            if chunk.artifact:
                                accumulated_artifacts.append(chunk.artifact)
                            if chunk.custom_patch:
                                self._apply_custom_patch(chunk.custom_patch)

                            reasoning = getattr(chunk.model_chunk, "reasoning", None) if chunk.model_chunk else None
                            tool_request = _get_chunk_tool_request(chunk.model_chunk)
                            tool_response = _get_chunk_tool_response(chunk.model_chunk)
                            artifact = getattr(chunk, "artifact", None)
                            status = getattr(chunk, "status", None)

                            c = AgentChunk(
                                text=text,
                                reasoning=reasoning,
                                tool_request=tool_request,
                                tool_response=tool_response,
                                artifact=artifact,
                                status=status,
                                raw=chunk,
                            )

                            if tool_request:
                                turn._interrupt = AgentInterrupt(
                                    name=tool_request.tool_request.name,
                                    ref=tool_request.tool_request.ref,
                                    input_data=tool_request.tool_request.input,
                                    session=self,
                                )

                            yield c
                            if chunk.turn_end:
                                break
                        except StopAsyncIteration:
                            break
                        except BaseException as e:
                            if not turn_output_future.done():
                                turn_output_future.set_exception(e)
                            raise e
            finally:
                if cancelled_event.is_set():
                    watch_output_task.cancel()

        def do_abort():
            cancelled_event.set()
            if not turn_output_future.done():
                turn_output_future.cancel()
            if not run_task.done():
                run_task.cancel()

        turn = AgentTurn(
            stream=stream_generator(),
            output=turn_output_future,
            abort_fn=do_abort,
        )
        return turn

    def resume(self, resume: Resume) -> AgentTurn[StateT, StreamT]:
        """Continues a conversation from an interrupt."""
        inp = AgentInput(resume=resume)
        return self.send(inp)

    async def get_snapshot(self) -> SessionSnapshot[StateT] | None:
        if not self.snapshot_id:
            return None
        return await self._transport.get_snapshot(self.snapshot_id)

    async def run_detached(self, input: str | AgentInput) -> DetachedTask[StateT]:
        if isinstance(input, str):
            inp = AgentInput(
                messages=[
                    MessageData(
                        role='user',
                        content=[Part(root=TextPart(text=input))],
                    )
                ]
            )
        else:
            inp = input

        inp.detach = True
        init = self._build_init()

        _, output_awaitable = await self._transport.run_turn(inp, init)
        raw_output = await output_awaitable
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


class DetachedTask(Generic[StateT]):
    """Represents a background agent task running on the server."""

    def __init__(self, snapshot_id: str, transport: AgentTransport[StateT, Any]) -> None:
        self.snapshot_id = snapshot_id
        self._transport = transport

    async def poll(self) -> SessionSnapshot[StateT] | None:
        """Query the server for the current task status/snapshot."""
        return await self._transport.get_snapshot(self.snapshot_id)


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
