# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the debug-log records emitted by genkit._ai._generate."""

from collections.abc import Iterator

import pytest
import structlog
from structlog.testing import capture_logs

from genkit import FinishReason, GenerationBlockedError, Genkit, Message, ModelResponse
from genkit._ai._testing import define_programmable_model
from genkit._core._environment import GENKIT_ENV
from genkit._core._logger import GENKIT_LOG
from genkit._core._typing import Role, TextPart

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
        await _generate_once()

    events = [entry['event'] for entry in entries]
    assert events == []


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
        with pytest.raises(GenerationBlockedError, match='Generation blocked: safety'):
            await ai.generate(prompt='hi')

    responded = [entry for entry in entries if entry['event'] == 'model responded']
    assert len(responded) == 1
    assert responded[0]['finish_reason'] == FinishReason.BLOCKED
    assert 'response' not in responded[0]


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
