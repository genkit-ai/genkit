#!/usr/bin/env python3
# Copyright 2025 Google LLC
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

"""Flow fundamentals — exact Python parity with `js/testapps/flow-sample1` edge cases.

No model is used; these flows test and stress the framework itself: traced steps,
streaming, context propagation, error handling (caught and uncaught), ~1MB payloads,
and long-running multi-minute broadcasts.

Run directly:
    uv run src/main.py
Or inspect live execution and traces in Dev UI:
    genkit start -- uv run src/main.py
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from pydantic import BaseModel

from genkit import ActionRunContext, Genkit

ai = Genkit()


# --- 1. Schemas for Traced Steps, Streaming, and Long Broadcasts ---
class StreamChunk(BaseModel):
    """One unit emitted by `streamy` / `streamyThrowy`."""

    count: int


class WithInputSchemaInput(BaseModel):
    """Input shape for `withInputSchema` — mirrors the JS object input."""

    subject: str


class WithContextInput(BaseModel):
    """Input shape for `withContext`."""

    subject: str


class TimelineEntry(BaseModel):
    """One row of the long-broadcast timeline."""

    step: int
    timestamp: str
    elapsed_ms: int


class LongBroadcastInput(BaseModel):
    """Knobs for `test-long-broadcast`."""

    steps: int = 10
    step_delay_ms: int = 15_000


class LongBroadcastOutput(BaseModel):
    """Result returned by `test-long-broadcast`."""

    total_duration_ms: int
    steps_completed: int
    timeline: list[TimelineEntry]


# --- 2. Basic & Nested Traced Steps (`ai.run`) ---
@ai.flow(name='basic')
async def basic(subject: str) -> str:
    """Two traced steps that just shuffle the input string around."""

    async def call_llm() -> str:
        return f'subject: {subject}'

    foo = await ai.run(name='call-llm', fn=call_llm)

    async def call_llm1() -> str:
        return f'foo: {foo}'

    # => Dev UI records two sequential child spans: `call-llm` and `call-llm1`
    return await ai.run(name='call-llm1', fn=call_llm1)


@ai.flow(name='parent')
async def parent() -> str:
    """Calls `basic` and returns its output as a string. Demonstrates flow-from-flow span nesting."""
    # => Dev UI records `basic` as a child span nested cleanly under `parent`
    return await basic('foo')


@ai.flow(name='withInputSchema')
async def with_input_schema(input: WithInputSchemaInput) -> str:
    """Same as `basic` but the input is a typed object instead of a bare string."""

    async def call_llm() -> str:
        return f'subject: {input.subject}'

    foo = await ai.run(name='call-llm', fn=call_llm)

    async def call_llm1() -> str:
        return f'foo: {foo}'

    return await ai.run(name='call-llm1', fn=call_llm1)


@ai.flow(name='withContext')
async def with_context(input: WithContextInput, ctx: ActionRunContext) -> str:
    """Echoes the request context so you can confirm it's flowing through."""
    return f'subject: {input.subject}, context: {ctx.context}'


# --- 3. Streaming & Partial Error Streams (`streamyThrowy`) ---
@ai.flow(name='streamy', chunk_type=StreamChunk)
async def streamy(count: int, ctx: ActionRunContext) -> str:
    """Stream `count` chunks at one-second intervals, then return a summary."""
    i = 0
    while i < count:
        await asyncio.sleep(1)
        ctx.send_chunk(StreamChunk(count=i))
        i += 1
    return f'done: {count}, streamed: {i} times'


@ai.flow(name='streamyThrowy', chunk_type=StreamChunk)
async def streamy_throwy(count: int, ctx: ActionRunContext) -> str:
    """Stream a few chunks, then raise mid-stream so you can see partial output + error in Dev UI."""
    i = 0
    while i < count:
        if i == 3:
            # => Emits chunks 0, 1, 2, then throws mid-stream without hanging the stream pipe
            raise RuntimeError('whoops')
        await asyncio.sleep(1)
        ctx.send_chunk(StreamChunk(count=i))
        i += 1
    return f'done: {count}, streamed: {i} times'


# --- 4. Uncaught and Caught Error Tracing (`throwy*`) ---
@ai.flow(name='throwy')
async def throwy(subject: str) -> str:
    """Run a step, then raise. Tests that completed step spans are preserved when the flow crashes."""

    async def call_llm() -> str:
        return f'subject: {subject}'

    await ai.run(name='call-llm', fn=call_llm)
    if subject:
        # => Dev UI shows completed `call-llm` span right before `throwy` flow crashes
        raise RuntimeError(subject)

    async def call_llm_again() -> str:
        return 'unreachable'

    return await ai.run(name='call-llm', fn=call_llm_again)


@ai.flow(name='throwy2')
async def throwy2(subject: str) -> str:
    """Raise from inside a traced step. Tests that the child span status captures the step error."""

    async def call_llm() -> str:
        if subject:
            # => Dev UI marks the `call-llm` child span itself with status=ERROR
            raise RuntimeError(subject)
        return f'subject: {subject}'

    foo = await ai.run(name='call-llm', fn=call_llm)

    async def call_llm_again() -> str:
        return f'foo: {foo}'

    return await ai.run(name='call-llm', fn=call_llm_again)


@ai.flow(name='flowMultiStepCaughtError')
async def flow_multi_step_caught_error(input: str) -> str:
    """Catch an error from the middle step so the overall flow completes despite inner step failure."""
    counter = {'i': 1}

    async def step1() -> str:
        out = f'{input} {counter["i"]},'
        counter['i'] += 1
        return out

    result1 = await ai.run(name='step1', fn=step1)

    async def step2() -> str:
        if result1:
            raise RuntimeError('Got an error!')
        out = f'{result1} {counter["i"]},'
        counter['i'] += 1
        return out

    result2 = ''
    try:
        result2 = await ai.run(name='step2', fn=step2)
    except RuntimeError:
        pass

    async def step3() -> str:
        return f'{result2} {counter["i"]}'

    # => Dev UI timeline: step1 (PASS) -> step2 (FAIL) -> step3 (PASS) -> overall flow (PASS)
    return await ai.run(name='step3', fn=step3)


# --- 5. Stress Testing: Multi-Step & Large Payloads (`multiSteps`, `largeSteps`) ---
@ai.flow(name='multiSteps')
async def multi_steps(input: str) -> int:
    """Several traced steps with intermediate string and list transforms; returns a fixed int."""

    async def step1() -> str:
        return f'Hello, {input}! step 1'

    out1 = await ai.run(name='step1', fn=step1)

    async def step1_again() -> str:
        return f'Hello2222, {input}! step 1'

    await ai.run(name='step1', fn=step1_again)

    async def step2() -> str:
        return f'{out1} Faf '

    out2 = await ai.run(name='step2', fn=step2)

    async def step3_array() -> list[str]:
        return [out2, out2]

    out3 = await ai.run(name='step3-array', fn=step3_array)

    async def step4_num() -> str:
        return '-()-'.join(out3)

    await ai.run(name='step4-num', fn=step4_num)
    return 42


_LOREM = ('lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 'elit')


def _generate_string(length: int) -> str:
    """Build a roughly `length` byte string of repeating lorem-ipsum tokens."""
    parts: list[str] = []
    total = 0
    while total < length:
        word = random.choice(_LOREM)
        parts.append(word)
        parts.append(' ')
        total += len(word) + 1
    return ''.join(parts)[:length]


@ai.flow(name='largeSteps')
async def large_steps() -> str:
    """Steps that produce ~1MB string outputs — useful for stressing the trace pipe and chunk accumulator."""

    async def large_step1() -> str:
        return _generate_string(100_000)

    async def large_step2() -> str:
        return _generate_string(800_000)

    async def large_step3() -> str:
        return _generate_string(900_000)

    async def large_step4() -> str:
        return _generate_string(999_000)

    await ai.run(name='large-step1', fn=large_step1)
    await ai.run(name='large-step2', fn=large_step2)
    await ai.run(name='large-step3', fn=large_step3)
    await ai.run(name='large-step4', fn=large_step4)
    # => Confirms ~2.8MB total trace data streams across the reflection server without dropping
    return 'something...'


# --- 6. Stress Testing: Long-Running Broadcast (`test-long-broadcast`) ---
@ai.flow(name='test-long-broadcast')
async def test_long_broadcast(input: LongBroadcastInput | None = None) -> LongBroadcastOutput:
    """Multi-minute flow with nested spans for stress-testing trace broadcast."""
    cfg = input or LongBroadcastInput()
    start = time.monotonic()
    timeline: list[TimelineEntry] = []

    print(
        f'Starting long broadcast test: {cfg.steps} steps x {cfg.step_delay_ms / 1000}s'
        f' = ~{(cfg.steps * cfg.step_delay_ms) / 60_000:.1f} minutes'
    )

    third = cfg.step_delay_ms / 3 / 1000

    for i in range(1, cfg.steps + 1):
        step_start = time.monotonic()

        async def _do_step(step_idx: int = i) -> str:
            print(f'Step {step_idx}/{cfg.steps} starting...')

            async def fetch() -> str:
                await asyncio.sleep(third)
                return f'fetch-{step_idx}'

            async def process() -> str:
                await asyncio.sleep(third)
                return f'process-{step_idx}'

            async def save() -> str:
                await asyncio.sleep(third)
                return f'save-{step_idx}'

            await ai.run(name=f'step-{step_idx}-fetch', fn=fetch)
            await ai.run(name=f'step-{step_idx}-process', fn=process)
            await ai.run(name=f'step-{step_idx}-save', fn=save)

            return f'Step {step_idx} complete'

        await ai.run(name=f'step-{i}', fn=_do_step)

        elapsed_ms = int((time.monotonic() - step_start) * 1000)
        timeline.append(
            TimelineEntry(
                step=i,
                timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                elapsed_ms=elapsed_ms,
            )
        )

    return LongBroadcastOutput(
        total_duration_ms=int((time.monotonic() - start) * 1000),
        steps_completed=cfg.steps,
        timeline=timeline,
    )


# --- 7. Smoke Test Entrypoint ---
async def main() -> None:
    """Run core flows once from the CLI (skips slow broadcast and throwy crash tests)."""

    async def _show(label: str, value: Any) -> None:
        print(f'\n[{label}]\n  {value}')

    await _show('basic', await basic('hello'))
    await _show('parent', await parent())
    await _show('withInputSchema', await with_input_schema(WithInputSchemaInput(subject='world')))
    await _show('multiSteps', await multi_steps('world'))
    await _show('flowMultiStepCaughtError', await flow_multi_step_caught_error('hi'))


if __name__ == '__main__':
    ai.run_main(main())
