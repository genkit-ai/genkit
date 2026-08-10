# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the debug-log payload built by genkit._ai._generate."""

from collections.abc import Iterator

import pytest
import structlog
from structlog.testing import capture_logs

from genkit import Genkit, Message, ModelResponse
from genkit._ai._generate import (
    _MAX_LOGGED_LIST_LEN,
    _MAX_LOGGED_STR_LEN,
    _PROVIDER_STR_LEN,
    _loggable_response,
    _redact_large_values,
)
from genkit._ai._testing import define_programmable_model
from genkit._core._environment import GENKIT_ENV
from genkit._core._logger import GENKIT_LOG
from genkit._core._typing import CustomPart, GenerationUsage, Media, MediaPart, Role, TextPart

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


def test_short_values_pass_through() -> None:
    """Values under the limits are untouched."""
    payload = {'text': 'hello', 'items': [1, None, True, 'short'], 'nested': {'k': 'v'}}

    assert _redact_large_values(payload) == payload


def test_data_uris_keep_their_media_type() -> None:
    """Data URIs report the dropped payload size and keep the media type."""
    uri = 'data:audio/L16;codec=pcm;rate=24000;base64,' + 'B' * 500

    assert _redact_large_values(uri) == 'data:audio/L16;codec=pcm;rate=24000;base64,...<500 chars>'


def test_bare_long_strings_are_truncated() -> None:
    """A long string with no data URI prefix keeps its leading characters."""
    blob = 'C' * (_MAX_LOGGED_STR_LEN + 4096)

    redacted = _redact_large_values(blob)

    assert redacted.startswith('C' * _MAX_LOGGED_STR_LEN)
    assert redacted.endswith('...<4096 chars>')


def test_binary_values_collapse_to_their_size() -> None:
    """bytes survive model_dump in python mode, so they are reported by length."""
    assert _redact_large_values(b'\x00' * 200_000) == '<200000 bytes>'
    assert _redact_large_values(bytearray(b'ab')) == '<2 bytes>'
    assert _redact_large_values(memoryview(b'abc')) == '<3 bytes>'


def test_long_lists_are_truncated() -> None:
    """An over-long list keeps its leading items and reports the remainder."""
    redacted = _redact_large_values(list(range(_MAX_LOGGED_LIST_LEN + 250)))

    assert len(redacted) == _MAX_LOGGED_LIST_LEN + 1
    assert redacted[:3] == [0, 1, 2]
    assert redacted[-1] == '...<250 more items>'


def test_redaction_recurses_into_containers() -> None:
    """Oversized values nested in dicts and lists are shrunk too."""
    blob = 'D' * (_MAX_LOGGED_STR_LEN + 1)

    redacted = _redact_large_values({'a': [{'b': blob, 'c': b'xy'}]})

    assert redacted == {'a': [{'b': 'D' * _MAX_LOGGED_STR_LEN + '...<1 chars>', 'c': '<2 bytes>'}]}


def test_loggable_response_keeps_provider_payloads_bounded() -> None:
    """raw and custom stay in the log for provider debugging, but bounded in size."""
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[TextPart(text='hi')]),
        custom={'audio': BLOB},
        raw={'audio': BLOB},
        usage=GenerationUsage(input_tokens=1, output_tokens=2),
    )

    logged = _loggable_response(response)

    assert logged['message']['content'][0]['text'] == 'hi'
    assert logged['usage']['inputTokens'] == 1
    assert logged['raw']['audio'].endswith(f'...<{1_000_000 - _PROVIDER_STR_LEN} chars>')
    assert logged['custom']['audio'] == logged['raw']['audio']
    assert len(str(logged)) < len(str(response.model_dump())) / 100


def test_loggable_response_truncates_nested_part_payloads() -> None:
    """Oversized values on a part, not just on the response, are shrunk."""
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[CustomPart(custom={'blob': BLOB})]),
    )

    logged = _loggable_response(response)

    assert logged['message']['content'][0]['custom']['blob'].endswith(f'...<{1_000_000 - _PROVIDER_STR_LEN} chars>')


def test_model_output_gets_a_longer_limit_than_provider_payloads() -> None:
    """Model text survives to _MAX_LOGGED_STR_LEN while raw is held to _PROVIDER_STR_LEN."""
    prose = 'W' * (_MAX_LOGGED_STR_LEN * 2)
    response = ModelResponse(
        message=Message(role=Role.MODEL, content=[TextPart(text=prose)]),
        raw={'echo': prose},
    )

    logged = _loggable_response(response)

    assert logged['message']['content'][0]['text'].endswith(f'...<{_MAX_LOGGED_STR_LEN} chars>')
    assert logged['raw']['echo'].endswith(f'...<{_MAX_LOGGED_STR_LEN * 2 - _PROVIDER_STR_LEN} chars>')


def test_loggable_response_truncates_media_in_message() -> None:
    """Inline media that survives into the message is truncated rather than dumped."""
    response = ModelResponse(
        message=Message(
            role=Role.MODEL,
            content=[MediaPart(media=Media(content_type='image/png', url='data:image/png;base64,' + 'E' * 900))],
        ),
    )

    logged = _loggable_response(response)

    assert logged['message']['content'][0]['media']['url'] == 'data:image/png;base64,...<900 chars>'


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
    """A generate call under the default configuration emits no response dump."""
    structlog.reset_defaults()

    with capture_logs() as entries:
        await _generate_once()

    events = [entry['event'] for entry in entries]
    assert events == []


@pytest.mark.asyncio
async def test_generate_logs_bounded_response_when_debug_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With debug on, the response is logged but the blob is truncated."""
    structlog.reset_defaults()
    monkeypatch.setenv(GENKIT_LOG, 'debug')

    with capture_logs() as entries:
        await _generate_once()

    logged = [entry for entry in entries if entry['event'] == 'generate response']
    assert len(logged) == 1
    assert len(str(logged[0]['response'])) < 3 * _MAX_LOGGED_STR_LEN


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
