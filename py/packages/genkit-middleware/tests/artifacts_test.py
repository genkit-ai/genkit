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

import subprocess  # noqa: S404
import sys

import pytest
from genkit_middleware import Middleware
from genkit_middleware._artifacts import (
    ARTIFACTS_LISTING_MARKER,
    build_artifact_listing,
    extract_artifact_text,
)
from genkit_middleware.exp import Artifacts

from genkit import GenkitError, Message, ModelResponse, Part
from genkit._ai._agents._session import Session, run_with_session
from genkit._core._model import Artifact, GenerateActionOptions, SessionState
from genkit._core._typing import FinishReason, Role
from genkit.exp import Genkit
from genkit.middleware import GenerateHookParams, GenerateMiddlewareContext, MiddlewareRef
from genkit.model import ToolRequest
from genkit.testing import define_scripted_model


def _make_params(options: GenerateActionOptions | None = None) -> GenerateHookParams:
    opts = options or GenerateActionOptions(messages=[])
    return GenerateHookParams(
        options=opts,
        iteration=0,
    )


def _listing_parts(messages) -> list[Part]:
    parts: list[Part] = []
    for msg in messages:
        if msg.role != Role.SYSTEM:
            continue
        for part in msg.content:
            if part.text is not None and isinstance(part.metadata, dict):
                if part.metadata.get(ARTIFACTS_LISTING_MARKER):
                    parts.append(part)
    return parts


def test_build_artifact_listing_empty() -> None:
    listing = build_artifact_listing([])
    assert 'No artifacts are currently available' in listing
    assert listing.startswith('<artifacts>')


def test_extract_artifact_text() -> None:
    art = Artifact(name='a.txt', parts=[Part.from_text('line1'), Part.from_text('line2')])
    assert extract_artifact_text(art) == 'line1\nline2'


@pytest.mark.asyncio
async def test_write_artifact_uses_current_session(ctx: GenerateMiddlewareContext) -> None:
    mw = Artifacts()
    session = Session(SessionState())

    async def check() -> None:
        tools = {t.name: t for t in mw.tools(ctx)}
        assert set(tools) == {'read_artifact', 'write_artifact'}

        write = tools['write_artifact']
        result = await write.action().run(input={'name': 'poem.txt', 'content': 'roses are red'})
        assert result.response.output['status'] == 'Artifact "poem.txt" saved successfully.'
        arts = await session.get_artifacts()
        assert len(arts) == 1
        assert arts[0].name == 'poem.txt'
        assert arts[0].parts[0].text == 'roses are red'

    await run_with_session(session=session, coro=check())


@pytest.mark.asyncio
async def test_read_artifact_returns_found(ctx: GenerateMiddlewareContext) -> None:
    mw = Artifacts()
    session = Session(SessionState())
    await session.add_artifacts([Artifact(name='notes.txt', parts=[Part.from_text('hello')])])

    async def check() -> None:
        read = next(t for t in mw.tools(ctx) if t.name == 'read_artifact')

        result = await read.action().run(input={'name': 'notes.txt'})
        assert result.response.output['name'] == 'notes.txt'
        assert result.response.output['content'] == 'hello'
        assert result.response.output['found'] is True

    await run_with_session(session=session, coro=check())


@pytest.mark.asyncio
async def test_read_artifact_without_session(ctx: GenerateMiddlewareContext) -> None:
    mw = Artifacts()
    read = next(t for t in mw.tools(ctx) if t.name == 'read_artifact')
    result = await read.action().run(input={'name': 'missing.txt'})
    assert result.response.output['name'] == 'missing.txt'
    assert 'no active agent session' in result.response.output['content'].lower()
    assert result.response.output['found'] is False


@pytest.mark.asyncio
async def test_readonly_excludes_write_tool(ctx: GenerateMiddlewareContext) -> None:
    mw = Artifacts(readonly=True)
    names = {t.name for t in mw.tools(ctx)}
    assert names == {'read_artifact'}


@pytest.mark.asyncio
async def test_wrap_generate_injects_listing(ctx: GenerateMiddlewareContext) -> None:
    mw = Artifacts()
    session = Session(
        SessionState(artifacts=[Artifact(name='poem.txt', parts=[Part.from_text('abc')])]),
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
            if p.text is not None and isinstance(p.metadata, dict) and p.metadata.get(ARTIFACTS_LISTING_MARKER)
        ]
        assert len(listing_parts) == 1
        assert 'poem.txt' in (listing_parts[0].text or '')
        assert '(3 chars)' in (listing_parts[0].text or '')

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
        await session.add_artifacts([Artifact(name='b.txt', parts=[Part.from_text('x')])])
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
        SessionState(artifacts=[Artifact(name='a.txt', parts=[Part.from_text('hi')])]),
    )

    captured_request: list[GenerateActionOptions] = []

    async def next_fn(params, _ctx):
        captured_request.append(params.options)
        return ModelResponse(message=None)

    async def check() -> None:
        await mw.wrap_generate(_make_params(envelope), ctx, next_fn)

        assert len(_listing_parts(envelope.messages)) == 0
        listing_parts = _listing_parts(captured_request[0].messages)
        assert len(listing_parts) == 1
        listing = listing_parts[0].text
        assert listing is not None
        assert 'a.txt' in listing

    await run_with_session(session=session, coro=check())


@pytest.mark.asyncio
async def test_agent_using_artifacts_without_middleware_plugin_saves_written_file_to_chat() -> None:
    """use=[Artifacts()] on an agent works with no Middleware() plugin: write_artifact lands on chat.artifacts."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(
                        tool_request=ToolRequest(
                            name='write_artifact',
                            input={'name': 'poem.txt', 'content': 'roses are red'},
                            ref='w1',
                        )
                    )
                ],
            ),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('saved')]),
        ),
    ]
    agent = ai.define_agent(name='workspaceAgent', model='scriptedModel', use=[Artifacts()])

    chat = agent.chat()
    out = await chat.send('Write poem.txt')

    assert out.text == 'saved'
    assert [(a.name, extract_artifact_text(a)) for a in chat.artifacts] == [('poem.txt', 'roses are red')]


@pytest.mark.asyncio
async def test_generate_naming_artifacts_middleware_with_middleware_plugin_raises_not_found() -> None:
    """With Middleware() registered, use=[MiddlewareRef(name='artifacts')] (a .prompt's use:) raises NOT_FOUND."""
    ai = Genkit(plugins=[Middleware()])
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        )
    ]

    with pytest.raises(GenkitError) as err:
        await ai.generate(model='scriptedModel', prompt='hi', use=[MiddlewareRef(name='artifacts')])

    assert err.value.status == 'NOT_FOUND'
    assert '"artifacts"' in str(err.value)


@pytest.mark.asyncio
async def test_generate_naming_retry_middleware_with_middleware_plugin_runs() -> None:
    """With Middleware() registered, use=[MiddlewareRef(name='retry')] still resolves and the call returns."""
    ai = Genkit(plugins=[Middleware()])
    pm, _ = define_scripted_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        )
    ]

    res = await ai.generate(model='scriptedModel', prompt='hi', use=[MiddlewareRef(name='retry')])

    assert res.text == 'ok'


def test_import_genkit_middleware_does_not_load_genkit_exp() -> None:
    """In a fresh interpreter, import genkit_middleware leaves genkit.exp out of sys.modules."""
    out = subprocess.run(  # noqa: S603
        [sys.executable, '-c', "import sys, genkit_middleware; print('genkit.exp' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert out.stdout.strip() == 'False'
