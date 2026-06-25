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

"""In-process agent transport factory: executes the agent action directly in the same process."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable
from typing import TypeVar

from genkit._ai._agents._session import SessionStore, SnapshotAborter
from genkit._core._action import BidiAction, BidiConnection
from genkit._core._channel import CloseableQueue
from genkit._core._typing import (
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentStreamChunk,
    Artifact,
    MessageData,
    ModelResponseChunk,
    Part,
    Role,
    SessionSnapshot,
    SnapshotStatus,
    TextPart,
)

StateT = TypeVar('StateT')
StreamT = TypeVar('StreamT')


def _message_from_model_chunks(chunks: list[ModelResponseChunk]) -> MessageData | None:
    """Assemble a model message from streamed chunks when there's no snapshot to read."""
    if not chunks:
        return None
    role = chunks[0].role or Role.MODEL
    content: list[Part] = []
    text_parts: list[str] = []

    def flush_text() -> None:
        if text_parts:
            content.append(Part(root=TextPart(text=''.join(text_parts))))
            text_parts.clear()

    for chunk in chunks:
        for part in chunk.content or []:
            p = part if isinstance(part, Part) else Part.model_validate(part)
            if isinstance(p.root, TextPart):
                text_parts.append(p.root.text or '')
            else:
                flush_text()
                content.append(p)
    flush_text()
    if not content:
        return None
    return MessageData(role=role, content=content)


class InProcessTransport:
    """In-process transport representing the trivial case where there is no physical transport.

    This runs the agent's bidirectional action directly in the same process and interacts
    directly with the session store that the agent writes to. The store is encapsulated
    and not exposed as a public field.
    """

    def __init__(
        self,
        action: BidiAction,
        store: SessionStore | None,
    ) -> None:
        """Initialise; store is captured privately and not exposed as a public field."""
        self.action = action
        self.conn: BidiConnection[AgentInput, AgentStreamChunk, AgentOutput] | None = None
        self.final_output: AgentOutput | None = None
        self.store = store

    async def run_turn(
        self,
        agent_input: AgentInput,
        init: AgentInit,
        abort_event: asyncio.Event | None = None,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        """Run a single turn and return the stream and output awaitables."""
        if self.conn is None:
            self.conn = await self.action.stream_bidi(init)
        conn = self.conn

        await conn.send(agent_input)

        output_future: asyncio.Future[AgentOutput] = asyncio.Future()
        stream_queue = CloseableQueue[AgentStreamChunk | BaseException]()

        async def watch_abort() -> None:
            if abort_event is not None:
                await abort_event.wait()
                self.conn = None
                await conn.close()
                try:
                    await conn.output()
                except asyncio.CancelledError:
                    pass
                if not output_future.done():
                    output_future.set_exception(asyncio.CancelledError())

        abort_task = asyncio.create_task(watch_abort())
        output_future.add_done_callback(lambda _: abort_task.cancel())

        async def drain_connection() -> None:
            model_chunks: list[ModelResponseChunk] = []
            try:
                async for chunk in conn.receive():
                    if chunk.model_chunk is not None:
                        model_chunks.append(chunk.model_chunk)
                    if chunk.turn_end:
                        snapshot_id = chunk.turn_end.snapshot_id
                        snap = None
                        state = None
                        message = None
                        artifacts: list[Artifact] = []
                        if snapshot_id:
                            snap = await self.get_snapshot(snapshot_id)
                            if snap and snap.state:
                                state = snap.state
                                if snap.state.messages:
                                    message = snap.state.messages[-1]
                                if snap.state.artifacts:
                                    artifacts = list(snap.state.artifacts)
                        if message is None:
                            message = _message_from_model_chunks(model_chunks)

                        output = AgentOutput(
                            session_id=snap.session_id if snap else None,
                            snapshot_id=snapshot_id,
                            finish_reason=chunk.turn_end.finish_reason,
                            error=getattr(chunk.turn_end, 'error', None),
                            state=state,
                            message=message,
                            artifacts=artifacts,
                        )
                        if not output_future.done():
                            output_future.set_result(output)

                    stream_queue.put_nowait(chunk)
                    if chunk.turn_end:
                        break

                if not output_future.done():
                    out = await conn.output()
                    output_future.set_result(out)
            except BaseException as e:
                if not output_future.done():
                    output_future.set_exception(e)
                stream_queue.put_nowait(e)
            finally:
                stream_queue.close()

        asyncio.create_task(drain_connection())

        async def stream_generator() -> AsyncIterator[AgentStreamChunk]:
            async for chunk in stream_queue:
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk

        return stream_generator(), output_future

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot | None:
        """Retrieve a session snapshot via the store captured at construction."""
        if self.store is None:
            return None
        return await self.store.get_snapshot(snapshot_id=snapshot_id)

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        """Abort a snapshot via the store captured at construction."""
        if not isinstance(self.store, SnapshotAborter):
            return None
        return await self.store.abort_snapshot(snapshot_id)

    async def close(self) -> None:
        """Close the underlying bidi connection."""
        if self.conn is not None:
            await self.conn.close()
            try:
                self.final_output = await self.conn.output()
            except Exception:  # noqa: BLE001
                self.final_output = None
            self.conn = None
