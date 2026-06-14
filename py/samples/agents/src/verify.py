#!/usr/bin/env python3
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

"""Verify weather + banking endpoints — inline streamFlow client."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

import httpx


async def verify_weather(base: str, *, verbose: bool = False) -> None:
    url = f'{base.rstrip("/")}/api/chat/weather'
    session_id = str(uuid.uuid4())

    print('  weather: 2 POSTs, same sessionId')

    async with httpx.AsyncClient() as client:
        body1 = {
            'data': {'messages': [{'role': 'user', 'content': [{'text': 'Weather in Paris?'}]}]},
            'init': {'sessionId': session_id},
        }
        if verbose:
            print(json.dumps(body1, indent=2))

        stream1: list[dict] = []
        async with client.stream(
            'POST', url, json=body1, headers={'Accept': 'text/event-stream'}, timeout=180.0
        ) as resp:
            resp.raise_for_status()
            buf = ''
            async for raw in resp.aiter_text():
                buf += raw
                while '\n\n' in buf:
                    block, buf = buf.split('\n\n', 1)
                    if not block.strip().startswith('data: '):
                        continue
                    wire = json.loads(block.strip().removeprefix('data: '))
                    if verbose:
                        print('SSE ←', json.dumps(wire, separators=(',', ':')))
                    if 'message' in wire:
                        stream1.append(wire['message'])

        tools = [
            p['toolRequest']
            for m in stream1
            for p in (m.get('modelChunk') or {}).get('content') or []
            if p.get('toolRequest')
        ]
        if not any(t.get('name') == 'getWeather' for t in tools):
            raise AssertionError(f'expected getWeather, got {tools!r}')

        body2 = {
            'data': {'messages': [{'role': 'user', 'content': [{'text': 'What city did I ask about? One word.'}]}]},
            'init': {'sessionId': session_id},
        }
        stream2: list[dict] = []
        async with client.stream(
            'POST', url, json=body2, headers={'Accept': 'text/event-stream'}, timeout=180.0
        ) as resp:
            resp.raise_for_status()
            buf = ''
            async for raw in resp.aiter_text():
                buf += raw
                while '\n\n' in buf:
                    block, buf = buf.split('\n\n', 1)
                    if not block.strip().startswith('data: '):
                        continue
                    wire = json.loads(block.strip().removeprefix('data: '))
                    if 'message' in wire:
                        stream2.append(wire['message'])

        reply = ''.join(p.get('text', '') for m in stream2 for p in (m.get('modelChunk') or {}).get('content') or [])
        if 'paris' not in reply.lower():
            raise AssertionError(f'expected Paris recall, got {reply!r}')
        print(f'    turn 2: {reply!r}')


async def verify_banking(base: str, *, verbose: bool = False) -> None:
    url = f'{base.rstrip("/")}/api/chat/banking'
    session_id = str(uuid.uuid4())

    print('  banking: interrupt POST + resume POST')

    async with httpx.AsyncClient() as client:
        body1 = {
            'data': {'messages': [{'role': 'user', 'content': [{'text': 'Transfer $500 to account 12345 for rent.'}]}]},
            'init': {'sessionId': session_id},
        }
        stream1: list[dict] = []
        result1: dict = {}
        async with client.stream(
            'POST', url, json=body1, headers={'Accept': 'text/event-stream'}, timeout=180.0
        ) as resp:
            resp.raise_for_status()
            buf = ''
            async for raw in resp.aiter_text():
                buf += raw
                while '\n\n' in buf:
                    block, buf = buf.split('\n\n', 1)
                    if not block.strip().startswith('data: '):
                        continue
                    wire = json.loads(block.strip().removeprefix('data: '))
                    if verbose:
                        print('SSE ←', json.dumps(wire, separators=(',', ':')))
                    if 'message' in wire:
                        stream1.append(wire['message'])
                    if 'result' in wire:
                        result1 = wire['result']

        approval = None
        for m in stream1:
            for p in (m.get('modelChunk') or {}).get('content') or []:
                if (tr := p.get('toolRequest')) and tr.get('name') == 'userApproval':
                    approval = tr
        if approval is None:
            raise AssertionError('expected userApproval')
        if result1.get('finishReason') != 'interrupted':
            raise AssertionError(f'expected interrupted, got {result1.get("finishReason")!r}')

        body2 = {
            'data': {
                'resume': {
                    'respond': [
                        {
                            'toolResponse': {
                                'name': approval['name'],
                                'ref': approval['ref'],
                                'output': {'approved': True, 'feedback': 'Looks good'},
                            }
                        }
                    ]
                }
            },
            'init': {'sessionId': session_id},
        }
        result2: dict = {}
        async with client.stream(
            'POST', url, json=body2, headers={'Accept': 'text/event-stream'}, timeout=180.0
        ) as resp:
            resp.raise_for_status()
            buf = ''
            async for raw in resp.aiter_text():
                buf += raw
                while '\n\n' in buf:
                    block, buf = buf.split('\n\n', 1)
                    if not block.strip().startswith('data: '):
                        continue
                    wire = json.loads(block.strip().removeprefix('data: '))
                    if 'result' in wire:
                        result2 = wire['result']

        msg = result2.get('message') or {}
        reply = ''.join(p.get('text', '') for p in msg.get('content') or [])
        if not reply.strip():
            raise AssertionError('expected reply after resume')
        print(f'    after approve: {reply[:80]!r}')


async def run_checks(base_url: str, *, verbose: bool = False) -> int:
    print(f'Base URL: {base_url.rstrip("/")}\n')
    failed = 0
    for label, fn in [
        ('Weather agent', lambda: verify_weather(base_url, verbose=verbose)),
        ('Banking interrupt', lambda: verify_banking(base_url, verbose=verbose)),
    ]:
        print(label)
        try:
            await fn()
            print('  ✓ pass\n')
        except Exception as exc:
            failed += 1
            print(f'  ✗ fail: {exc!r}\n')
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://localhost:8080')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    sys.exit(asyncio.run(run_checks(args.base_url, verbose=args.verbose)))


if __name__ == '__main__':
    main()
