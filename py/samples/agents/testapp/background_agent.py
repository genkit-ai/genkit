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

"""Kick off a long task, get a handle back immediately, poll for the result.

For work that outlives a request — a big research report here — the client sends
``detach`` and the server keeps running after returning a snapshot id. The client
polls that id until it settles (or aborts it). A store is required: it's where the
server parks the result for the client to pick up later.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from typing import Any

from _ai import ai

from genkit import ActionRunContext
from genkit.agent import InMemorySessionStore

# The store is what makes detach possible — the background turn writes its result
# there under the snapshot id, and the client reads it back when it's ready.
background_agent = ai.define_agent(
    name='backgroundAgent',
    system=(
        'You are a senior research analyst. Given a topic, produce a comprehensive markdown '
        'report with an executive summary, analysis, and recommendations.'
    ),
    store=InMemorySessionStore(),
)


@ai.flow()
async def test_background_agent(text: str, ctx: ActionRunContext) -> dict[str, Any]:
    """Detach a report, poll to completion, and return the settled status."""
    chat = background_agent.chat()
    # detach returns right away with a handle; the server keeps working.
    task = await chat.detach(text or 'Write a report on renewable energy trends')
    ctx.send_chunk(f'[detached] snapshotId={task.snapshot_id}')

    # Poll the store until the task reaches a terminal state.
    snapshot = await task.wait(interval=2.0)
    msgs = snapshot.state.messages if snapshot and snapshot.state else []
    preview = ''
    if msgs:
        parts = msgs[-1].content or []
        preview = ''.join(getattr(p.root, 'text', '') or '' for p in parts)[:200]
    return {'snapshot_id': task.snapshot_id, 'status': str(snapshot.status if snapshot else None), 'preview': preview}


if __name__ == '__main__':
    import asyncio

    ai.run_main(asyncio.sleep(0))
