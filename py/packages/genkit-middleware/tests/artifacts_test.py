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

"""Tests for Artifacts middleware."""

from __future__ import annotations

import pytest
from genkit_middleware import Artifacts
from genkit_middleware._artifacts import (
    ARTIFACTS_LISTING_MARKER,
    build_artifact_listing,
    extract_artifact_text,
)

from genkit import ModelResponse
from genkit._ai._agents._session import Session, run_with_session
from genkit._core._model import GenerateActionOptions
from genkit._core._typing import Artifact, Part, Role, SessionState, TextPart
from genkit.middleware import GenerateHookParams, GenerateMiddlewareContext


def _make_params(options: GenerateActionOptions | None = None) -> GenerateHookParams:
    opts = options or GenerateActionOptions(messages=[])
    return GenerateHookParams(
        options=opts,
        iteration=0,
    )


def _listing_parts(messages) -> list[TextPart]:
    parts: list[TextPart] = []
    for msg in messages:
        if msg.role != Role.SYSTEM:
            continue
        for part in msg.content:
            root = part.root
            if isinstance(root, TextPart) and isinstance(root.metadata, dict):
                if root.metadata.get(ARTIFACTS_LISTING_MARKER):
                    parts.append(root)
    return parts


def test_build_artifact_listing_empty() -> None:
    listing = build_artifact_listing([])
    assert 'No artifacts are currently available' in listing
    assert listing.startswith('<artifacts>')


def test_extract_artifact_text() -> None:
    art = Artifact(name='a.txt', parts=[Part(TextPart(text='line1')), Part(TextPart(text='line2'))])
    assert extract_artifact_text(art) == 'line1\nline2'


@pytest.mark.asyncio
async def test_write_artifact_uses_current_session(ctx: GenerateMiddlewareContext) -> None:
    mw = Artifacts()
    session = Session(SessionState())

    async def check() -> None:
        tools = {t.name: t for t in mw.tools(ctx)}
        assert set(tools) == {'read_artifact', 'write_artifact'}

        write = tools['write_artifact']
        result = await write.run(input={'name': 'poem.txt', 'content': 'roses are red'})
        assert result.response.status == 'Artifact "poem.txt" saved successfully.'
        arts = await session.get_artifacts()
        assert len(arts) == 1
        assert arts[0].name == 'poem.txt'
        assert arts[0].parts[0].root.text == 'roses are red'

    await run_with_session(session=session, coro=check())


@pytest.mark.asyncio
async def test_read_artifact_returns_found(ctx: GenerateMiddlewareContext) -> None:
    mw = Artifacts()
    session = Session(SessionState())
    await session.add_artifacts(Artifact(name='notes.txt', parts=[Part(TextPart(text='hello'))]))

    async def check() -> None:
        read = next(t for t in mw.tools(ctx) if t.name == 'read_artifact')

        result = await read.run(input={'name': 'notes.txt'})
        assert result.response.name == 'notes.txt'
        assert result.response.content == 'hello'
        assert result.response.found is True

    await run_with_session(session=session, coro=check())


@pytest.mark.asyncio
async def test_read_artifact_without_session(ctx: GenerateMiddlewareContext) -> None:
    mw = Artifacts()
    read = next(t for t in mw.tools(ctx) if t.name == 'read_artifact')
    result = await read.run(input={'name': 'missing.txt'})
    assert result.response.name == 'missing.txt'
    assert 'no active agent session' in result.response.content.lower()
    assert result.response.found is False


@pytest.mark.asyncio
async def test_readonly_excludes_write_tool(ctx: GenerateMiddlewareContext) -> None:
    mw = Artifacts(readonly=True)
    names = {t.name for t in mw.tools(ctx)}
    assert names == {'read_artifact'}


@pytest.mark.asyncio
async def test_wrap_generate_injects_listing(ctx: GenerateMiddlewareContext) -> None:
    mw = Artifacts()
    session = Session(
        SessionState(artifacts=[Artifact(name='poem.txt', parts=[Part(TextPart(text='abc'))])]),
    )

    captured: list[GenerateActionOptions] = []

    async def next_fn(params, _ctx):
        captured.append(params.options)
        return ModelResponse(message=None)

    async def check() -> None:
        await mw.wrap_generate(_make_params(), ctx, next_fn)

        assert len(captured) == 1
        system_msgs = [m for m in captured[0].messages if m.role == Role.SYSTEM]
        assert len(system_msgs) == 1
        listing_parts = [
            p
            for p in system_msgs[0].content
            if isinstance(p.root, TextPart)
            and isinstance(p.root.metadata, dict)
            and p.root.metadata.get(ARTIFACTS_LISTING_MARKER)
        ]
        assert len(listing_parts) == 1
        assert 'poem.txt' in (listing_parts[0].root.text or '')
        assert '(3 chars)' in (listing_parts[0].root.text or '')

    await run_with_session(session=session, coro=check())


@pytest.mark.asyncio
async def test_wrap_generate_refreshes_listing(ctx: GenerateMiddlewareContext) -> None:
    mw = Artifacts()
    session = Session(SessionState())
    envelope = GenerateActionOptions(messages=[])

    seen: list[str] = []

    async def next_fn(params, _ctx):
        for part in _listing_parts(params.options.messages):
            seen.append(part.text or '')
        return ModelResponse(message=None)

    async def check() -> None:
        await mw.wrap_generate(_make_params(envelope), ctx, next_fn)
        await session.add_artifacts(Artifact(name='b.txt', parts=[Part(TextPart(text='x'))]))
        await mw.wrap_generate(_make_params(envelope), ctx, next_fn)

        assert len(seen) == 2
        assert 'No artifacts are currently available' in seen[0]
        assert 'b.txt' in seen[1]
        assert len(_listing_parts(envelope.messages)) == 0

    await run_with_session(session=session, coro=check())


@pytest.mark.asyncio
async def test_wrap_generate_does_not_mutate_envelope(ctx: GenerateMiddlewareContext) -> None:
    mw = Artifacts()
    envelope = GenerateActionOptions(messages=[])
    session = Session(
        SessionState(artifacts=[Artifact(name='a.txt', parts=[Part(TextPart(text='hi'))])]),
    )

    captured_request: list[GenerateActionOptions] = []

    async def next_fn(params, _ctx):
        captured_request.append(params.options)
        return ModelResponse(message=None)

    async def check() -> None:
        await mw.wrap_generate(_make_params(envelope), ctx, next_fn)

        assert len(_listing_parts(envelope.messages)) == 0
        assert len(_listing_parts(captured_request[0].messages)) == 1
        assert 'a.txt' in _listing_parts(captured_request[0].messages)[0].text

    await run_with_session(session=session, coro=check())
