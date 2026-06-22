from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator, Awaitable
from typing import TypeVar

import websockets

from genkit._ai._agents._client import AgentTransport
from genkit._core._typing import AgentInit, AgentInput, AgentOutput, AgentStreamChunk, SessionSnapshot, SnapshotStatus

StateT = TypeVar('StateT')
StreamT = TypeVar('StreamT')


class WebSocketAgentTransport(AgentTransport[StateT, StreamT]):
    """Client-side agent transport that talks to a remote agent over WebSockets."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._ws: websockets.WebSocketClientProtocol | None = None

    async def run_turn(
        self,
        input: AgentInput,
        init: AgentInit[StateT],
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput[StateT]]]:
        if self._ws is None:
            self._ws = await websockets.connect(self.url)
        ws = self._ws

        await ws.send(
            json.dumps({
                'init': init.model_dump(by_alias=True),
                'input': input.model_dump(by_alias=True),
            })
        )

        output_future = asyncio.get_event_loop().create_future()

        async def stream_generator() -> AsyncIterator[AgentStreamChunk]:
            try:
                async for message in ws:
                    data = json.loads(message)
                    if 'chunk' in data:
                        chunk = AgentStreamChunk.model_validate(data['chunk'])
                        yield chunk
                    if 'output' in data:
                        output = AgentOutput.model_validate(data['output'])
                        if not output_future.done():
                            output_future.set_result(output)
                        break
            except Exception as e:
                if not output_future.done():
                    output_future.set_exception(e)
                raise e

        return stream_generator(), output_future

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot[StateT] | None:
        return None

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        return None

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
