"""In-process agent transport: executes the agent action directly in the same process."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable
from typing import TypeVar

from genkit._ai._agents._client import AgentTransport
from genkit._ai._session import SessionStore
from genkit._core._action import BidiAction, BidiConnection
from genkit._core._typing import (
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentStreamChunk,
    SessionSnapshot,
    SnapshotStatus,
)

StateT = TypeVar('StateT')
StreamT = TypeVar('StreamT')


class InProcessAgentTransport(AgentTransport[StateT, StreamT]):
    """Local, in-process agent transport that executes the agent action directly."""

    def __init__(self, action: BidiAction, store_configured: bool, store: SessionStore | None = None) -> None:
        """Initialise with a BidiAction and optional session store."""
        self._action = action
        self._store = store
        self.state_management = 'server' if store_configured else 'client'
        self._conn: BidiConnection[AgentInput, AgentStreamChunk, AgentOutput] | None = None
        self._final_output: AgentOutput | None = None

    async def run_turn(
        self,
        input: AgentInput,
        init: AgentInit[StateT],
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput[StateT]]]:
        """Run a single turn and return the stream and output awaitables."""
        if self._conn is None:
            self._conn = await self._action.stream_bidi(init)
        conn = self._conn  # noqa: SIM108 (keep local var for clarity)

        await conn.send(input)

        output_future = asyncio.get_event_loop().create_future()

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
                            error=getattr(chunk.turn_end, 'error', None),
                            state=state,
                            message=message,
                            artifacts=artifacts,
                        )
                        if not output_future.done():
                            output_future.set_result(output)
                    yield chunk
                    if chunk.turn_end:
                        break
            except BaseException as e:
                if not output_future.done():
                    output_future.set_exception(e)
                raise e

        return stream_generator(), output_future

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot[StateT] | None:
        """Retrieve a session snapshot from the store."""
        if self._store is None:
            return None
        return await self._store.get_snapshot(snapshot_id=snapshot_id)

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        """Abort the specified snapshot on the server."""
        if self._store is None or not hasattr(self._store, 'abort_snapshot'):
            return None
        return await self._store.abort_snapshot(snapshot_id)

    async def close(self) -> None:
        """Close the underlying bidi connection."""
        if self._conn is not None:
            await self._conn.close()
            try:
                self._final_output = await self._conn.output()
            except Exception:  # noqa: BLE001
                self._final_output = None
            self._conn = None
