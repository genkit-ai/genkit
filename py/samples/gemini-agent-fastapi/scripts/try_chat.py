# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Talk to the local agent with the ``AgentChat`` client.

A second way to drive the same ``/api/chat`` routes the curl quickstart hits —
but as a first-class Genkit client: ``chat.send()`` streams a turn and keeps the
session, so there's no NDJSON parsing or session-id juggling.

Run the server first, then in another terminal (from the sample root)::

    uv run python scripts/try_chat.py
"""

from __future__ import annotations

import asyncio
import os

import httpx

from genkit._core._http_client import get_cached_client
from genkit.agent import remote_agent

BASE = os.environ.get('AGENT_BASE_URL', 'http://localhost:8000')
DEMO_EMAIL = 'demo@example.com'
DEMO_PASSWORD = 'demo1234'  # noqa: S105


async def _login() -> str:
    async with httpx.AsyncClient() as http:
        resp = await http.post(
            f'{BASE}/api/auth/login',
            json={'email': DEMO_EMAIL, 'password': DEMO_PASSWORD},
        )
        resp.raise_for_status()
        return resp.json()['access_token']


async def main() -> None:
    token = await _login()

    # The routes are auth-protected and the transport reuses one httpx client per
    # event loop, so seed that client with the bearer header before the first call.
    get_cached_client('agent_transport', headers={'Authorization': f'Bearer {token}'})

    # The server owns the session store, so state is server-managed.
    agent = remote_agent(f'{BASE}/api/chat', state_management='server')
    chat = agent.chat()

    # Turn 1 — stream chunks live, then read the settled reply.
    print('> What plan am I on and what are my open orders?')
    turn = chat.send('What plan am I on and what are my open orders?')
    async for chunk in turn:
        for call in chunk.tool_requests:
            print(f'  → tool: {call.tool_request.name}')
        if chunk.text:
            print(chunk.accumulated_text, end='\r', flush=True)
    print((await turn).text)
    print(f'\n[session_id={chat.session_id}]')

    # Turn 2 — same session, so the agent still remembers turn 1.
    print('\n> Tell me more about the first one.')
    res = await chat.send('Tell me more about the first one.')
    print(res.text)

    await chat.close()


if __name__ == '__main__':
    asyncio.run(main())
