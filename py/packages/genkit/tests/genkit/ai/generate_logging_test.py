# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the debug-log records emitted by genkit._ai._generate."""

from collections.abc import Iterator

import pytest
import structlog
from structlog.testing import capture_logs

from genkit import Genkit, Message, ModelResponse
from genkit._ai._generate import generate_action
from genkit._ai._model import resolve_model_arg
from genkit._ai._testing import define_programmable_model
from genkit._ai._tools import Interrupt, restart_tool
from genkit._core._environment import GENKIT_ENV
from genkit._core._error import GenkitError
from genkit._core._logger import GENKIT_LOG, get_logger
from genkit._core._model import GenerateActionOptions
from genkit._core._registry import Registry
from genkit._core._typing import (
    FinishReason,
    GenerateActionOutputConfig,
    Part,
    Resume,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)

BLOB = 'A' * 1_000_000


@pytest.fixture(autouse=True)
def _restore_structlog() -> Iterator[None]:
    """Restore structlog's global configuration around each test."""
    saved = structlog.get_config().copy()
    was_configured = structlog.is_configured()
    yield
    structlog.reset_defaults()
    if was_configured:
        structlog.configure(**saved)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test with the Genkit env vars unset unless it sets them."""
    monkeypatch.delenv(GENKIT_LOG, raising=False)
    monkeypatch.delenv(GENKIT_ENV, raising=False)


async def _generate_once() -> None:
    """Run one generate call against a model returning a blob in raw and custom."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    pm.responses = [
        ModelResponse(
            message=Message(role=Role.MODEL, content=[TextPart(text='hello there')]),
            custom={'audio': BLOB},
            raw={'audio': BLOB},
        )
    ]
    _ = await ai.generate(prompt='hi')


@pytest.mark.asyncio
async def test_generate_logs_nothing_at_default_level() -> None:
    """A generate call under the default configuration emits no named records."""
    structlog.reset_defaults()

    with capture_logs() as entries:
        get_logger('genkit.test').info('capture probe')
        await _generate_once()

    events = [entry['event'] for entry in entries]
    assert 'capture probe' in events
    assert 'generate response' not in events


@pytest.mark.asyncio
async def test_generate_logs_named_records_when_debug_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With debug on, generate emits the named request/turn records."""
    structlog.reset_defaults()
    monkeypatch.setenv(GENKIT_LOG, 'debug')

    with capture_logs() as entries:
        await _generate_once()

    events = [entry['event'] for entry in entries]
    assert 'generate request resolved' in events
    assert 'calling model' in events
    responded = [entry for entry in entries if entry['event'] == 'model responded']
    assert len(responded) == 1
    assert responded[0]['tool_requests'] == 0
    assert 'response' not in responded[0]


@pytest.mark.asyncio
async def test_blocked_finish_still_logs_model_responded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal still gets a model-responded record so the log panel shows why."""
    structlog.reset_defaults()
    monkeypatch.setenv(GENKIT_LOG, 'debug')

    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.BLOCKED,
            finish_message='safety',
            message=Message(role=Role.MODEL, content=[TextPart(text='nope')]),
        )
    ]

    with capture_logs() as entries:
        response = await ai.generate(prompt='hi')
    assert response.finish_reason == FinishReason.BLOCKED
    assert response.text == 'nope'

    responded = [entry for entry in entries if entry['event'] == 'model responded']
    assert len(responded) == 1
    assert responded[0]['finish_reason'] == FinishReason.BLOCKED
    assert 'response' not in responded[0]


@pytest.mark.asyncio
async def test_leftover_logs_failed_finish_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """The breadcrumb uses the stamped finish, not the model's original stop."""
    structlog.reset_defaults()
    monkeypatch.setenv(GENKIT_LOG, 'debug')

    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[TextPart(text='not json')]),
        )
    ]

    with capture_logs() as entries:
        response = await ai.generate(prompt='hi', output_schema={'type': 'object'})
    assert response.finish_reason == FinishReason.FAILED

    responded = [entry for entry in entries if entry['event'] == 'model responded']
    assert len(responded) == 1
    assert responded[0]['finish_reason'] == FinishReason.FAILED


@pytest.mark.asyncio
async def test_response_is_not_serialized_when_debug_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The payload is not built at all when the event would be dropped."""
    structlog.reset_defaults()
    calls: list[int] = []
    original = ModelResponse.model_dump

    def counting_model_dump(self: ModelResponse, **kwargs: object) -> dict[str, object]:
        calls.append(1)
        return original(self, **kwargs)

    monkeypatch.setattr(ModelResponse, 'model_dump', counting_model_dump)

    await _generate_once()

    assert calls == []


def test_default_model_fallback_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting model= leaves a breadcrumb with the constructor default name."""
    structlog.reset_defaults()
    monkeypatch.setenv(GENKIT_LOG, 'debug')
    registry = Registry()
    registry.register_value('defaultModel', 'defaultModel', 'echo-model')

    with capture_logs() as entries:
        resolve_model_arg(model=None, registry=registry)

    events = [entry for entry in entries if entry['event'] == 'no model specified, using default model']
    assert len(events) == 1
    assert events[0]['model'] == 'echo-model'


@pytest.mark.asyncio
async def test_abnormal_finish_skips_output_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blocked model with a formatter warns instead of parsing."""
    structlog.reset_defaults()
    monkeypatch.setenv(GENKIT_LOG, 'debug')
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    pm.responses = [
        ModelResponse(
            message=Message(role=Role.MODEL, content=[TextPart(text='nope')]),
            finish_reason=FinishReason.BLOCKED,
            finish_message='safety',
        )
    ]

    with capture_logs() as entries:
        response = await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='programmableModel',
                messages=[Message(role=Role.USER, content=[TextPart(text='hi')])],
                output=GenerateActionOutputConfig(format='json'),
            ),
        )
    assert response.finish_reason == FinishReason.BLOCKED

    warned = [e for e in entries if e['event'] == 'model finished abnormally, skipping output parsing']
    assert len(warned) == 1
    assert warned[0]['finishReason'] == FinishReason.BLOCKED
    assert warned[0]['finishMessage'] == 'safety'
    assert 'model output does not match the expected schema' not in [e['event'] for e in entries]


@pytest.mark.asyncio
async def test_other_finish_does_not_warn_as_abnormal(monkeypatch: pytest.MonkeyPatch) -> None:
    """OTHER is an unmapped stop reason, not a refusal — do not warn at info."""
    structlog.reset_defaults()
    monkeypatch.setenv(GENKIT_LOG, 'info')
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    pm.responses = [
        ModelResponse(
            message=Message(role=Role.MODEL, content=[TextPart(text='{"ok": true}')]),
            finish_reason=FinishReason.OTHER,
        )
    ]

    with capture_logs() as entries:
        await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='programmableModel',
                messages=[Message(role=Role.USER, content=[TextPart(text='hi')])],
                output=GenerateActionOutputConfig(format='json'),
            ),
        )

    warned = [e for e in entries if e['event'] == 'model finished abnormally, skipping output parsing']
    assert warned == []


@pytest.mark.asyncio
async def test_schema_mismatch_logs_when_debug_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unparseable JSON is a debug breadcrumb; generate still returns."""
    structlog.reset_defaults()
    monkeypatch.setenv(GENKIT_LOG, 'debug')
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    pm.responses = [
        ModelResponse(
            message=Message(role=Role.MODEL, content=[TextPart(text='not json')]),
            finish_reason=FinishReason.STOP,
        )
    ]

    with capture_logs() as entries:
        response = await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='programmableModel',
                messages=[Message(role=Role.USER, content=[TextPart(text='hi')])],
                output=GenerateActionOutputConfig(format='json'),
            ),
        )

    assert response.message is not None
    mismatched = [e for e in entries if e['event'] == 'model output does not match the expected schema']
    assert len(mismatched) == 1
    assert mismatched[0]['model'] == 'programmableModel'


@pytest.mark.asyncio
async def test_tool_interrupt_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupting tool is named on the debug record."""
    structlog.reset_defaults()
    monkeypatch.setenv(GENKIT_LOG, 'debug')
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='hold')
    async def hold(_: dict) -> str:  # noqa: ARG001
        raise Interrupt({'hold': True})

    pm.responses = [
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='hold', input={}, ref='1')))],
            ),
            finish_reason=FinishReason.STOP,
        )
    ]

    with capture_logs() as entries:
        response = await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='programmableModel',
                messages=[Message(role=Role.USER, content=[TextPart(text='hi')])],
                tools=['hold'],
            ),
        )
    assert response.finish_reason == FinishReason.INTERRUPTED

    interrupted = [e for e in entries if e['event'] == 'tool triggered an interrupt']
    assert len(interrupted) == 1
    assert interrupted[0]['tool'] == 'hold'
    assert 'generation paused by tool interrupts' in [e['event'] for e in entries]
    responded = [e for e in entries if e['event'] == 'model responded']
    assert len(responded) == 1
    assert responded[0]['finish_reason'] == FinishReason.INTERRUPTED


@pytest.mark.asyncio
async def test_restarted_tool_interrupt_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool that interrupts again on restart leaves the same breadcrumb Go does."""
    structlog.reset_defaults()
    monkeypatch.setenv(GENKIT_LOG, 'debug')
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='hold')
    async def hold(_: dict) -> str:  # noqa: ARG001
        raise Interrupt({'hold': True})

    pm.responses = [
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='hold', input={}, ref='1')))],
            ),
            finish_reason=FinishReason.STOP,
        )
    ]
    first = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message(role=Role.USER, content=[TextPart(text='hi')])],
            tools=['hold'],
        ),
    )

    with capture_logs() as entries, pytest.raises(GenkitError, match='interrupted again'):
        await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='programmableModel',
                messages=list(first.messages),
                tools=['hold'],
                resume=Resume(restart=[restart_tool(interrupt=first.interrupts[0])]),
            ),
        )

    restarted = [e for e in entries if e['event'] == 'restarted tool triggered an interrupt']
    assert len(restarted) == 1
    assert restarted[0]['tool'] == 'hold'


@pytest.mark.asyncio
async def test_tool_stream_callback_failure_fails_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sinking callback fails generate the same way a model-chunk sink does."""
    structlog.reset_defaults()
    monkeypatch.setenv(GENKIT_LOG, 'debug')
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='echo')
    async def echo(_: dict) -> str:  # noqa: ARG001
        return 'ok'

    pm.responses = [
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='echo', input={}, ref='1')))],
            ),
            finish_reason=FinishReason.STOP,
        ),
        ModelResponse(
            message=Message(role=Role.MODEL, content=[TextPart(text='done')]),
            finish_reason=FinishReason.STOP,
        ),
    ]

    def on_chunk(chunk: object) -> None:
        if getattr(chunk, 'role', None) == Role.TOOL:
            raise RuntimeError('sink closed')

    with pytest.raises(RuntimeError, match='sink closed'):
        await generate_action(
            ai.registry,
            GenerateActionOptions(
                model='programmableModel',
                messages=[Message(role=Role.USER, content=[TextPart(text='hi')])],
                tools=['echo'],
            ),
            on_chunk=on_chunk,
        )
