# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Minimal eval runner for the agent.

Runs each case in ``dataset.json`` through the real agent — same entry point the
HTTP transport uses — and checks the reply against loose substring expectations.
It's a starting point, not a full eval suite: swap the substring checks for an
LLM-as-judge or richer metrics as your agent's behavior gets more nuanced.

Usage (from the sample root, with GEMINI_API_KEY set)::

    uv run python evals/run_evals.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Make the sample's ``app`` package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.agents.copilot import copilot_agent  # noqa: E402
from app.core.identity import uid_from_email  # noqa: E402

from genkit import Message, Part, Role, TextPart  # noqa: E402
from genkit.agent import AgentInit, AgentInput  # noqa: E402

_DATASET = Path(__file__).resolve().parent / 'dataset.json'
_DEMO_EMAIL = 'demo@example.com'


def _context() -> dict[str, object]:
    uid = uid_from_email(_DEMO_EMAIL)
    return {
        'auth': {'uid': uid, 'email': _DEMO_EMAIL},
        'uid': uid,
        'session_id': None,
    }


def _reply_text(message: Message | None) -> str:
    if message is None:
        return ''
    parts: list[str] = []
    for part in message.content:
        text = getattr(part.root, 'text', None)
        if text:
            parts.append(str(text))
    return ' '.join(parts)


async def _run_case(message_text: str) -> str:
    agent_input = AgentInput(
        message=Message(role=Role.USER, content=[Part(root=TextPart(text=message_text))]),
    )
    conn = await copilot_agent.stream_bidi(AgentInit(), context=_context())
    await conn.send(agent_input)
    await conn.close()
    async for _chunk in conn.receive():
        pass
    output = await conn.output()
    return _reply_text(output.message)


def _grade(reply: str, case: dict) -> tuple[bool, str]:
    lowered = reply.lower()

    required = case.get('expect_substrings') or []
    missing = [s for s in required if s.lower() not in lowered]
    if missing:
        return False, f'missing required: {missing}'

    any_of = case.get('expect_any') or []
    if any_of and not any(s.lower() in lowered for s in any_of):
        return False, f'expected any of: {any_of}'

    return True, 'ok'


async def main() -> int:
    cases = json.loads(_DATASET.read_text())['cases']
    passed = 0

    for case in cases:
        name = case['name']
        try:
            reply = await _run_case(case['message'])
        except Exception as exc:  # noqa: BLE001
            print(f'ERROR  {name}: {exc}')
            continue

        ok, detail = _grade(reply, case)
        if ok:
            passed += 1
            print(f'PASS   {name}')
        else:
            print(f'FAIL   {name}: {detail}')
            print(f'       reply: {reply[:200]}')

    total = len(cases)
    print(f'\n{passed}/{total} passed')
    return 0 if passed == total else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
