# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for A2UI generate pins."""

from __future__ import annotations

import json
from typing import Any

from genkit_a2ui import A2UI_MIME_TYPE

from genkit import Genkit, Message, ModelResponse
from genkit._ai._testing import ProgrammableModel, define_programmable_model
from genkit._core._typing import DataPart, FinishReason, Part, Role, TextPart

BASIC_CATALOG_ID = 'https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json'
A2UI_FENCE = '```a2ui'
WEATHER_PROMPT = 'Show me the weather in Tokyo'


def fence_block(envelopes: list[dict[str, Any]]) -> str:
    return f'{A2UI_FENCE}\n{json.dumps(envelopes, indent=2)}\n```\n'


def weather_surface(*, catalog_id: str = BASIC_CATALOG_ID, text: str = 'hi') -> list[dict[str, Any]]:
    return [
        {'createSurface': {'surfaceId': 'SURFACE_ID', 'catalogId': catalog_id}},
        {
            'updateComponents': {
                'surfaceId': 'SURFACE_ID',
                'components': [{'id': 'root', 'component': 'Text', 'text': text}],
            }
        },
    ]


def weather_fence(*, catalog_id: str = BASIC_CATALOG_ID, text: str = 'hi') -> str:
    return 'Here is the weather:\n' + fence_block(weather_surface(catalog_id=catalog_id, text=text))


def fence_only(*, catalog_id: str = BASIC_CATALOG_ID) -> str:
    return fence_block(weather_surface(catalog_id=catalog_id))


def bad_component_fence() -> str:
    return 'oops:\n' + fence_block([
        {
            'updateComponents': {
                'surfaceId': 'SURFACE_ID',
                'components': [{'id': 'root', 'component': 'NotAThing'}],
            }
        }
    ])


def broken_fence() -> str:
    return 'partial ```a2ui\n[{"createSurface": bad'


def no_root_fence(*, catalog_id: str = BASIC_CATALOG_ID) -> str:
    return fence_block([
        {'createSurface': {'surfaceId': 'SURFACE_ID', 'catalogId': catalog_id}},
        {
            'updateComponents': {
                'surfaceId': 'SURFACE_ID',
                'components': [{'id': 'leaf', 'component': 'Text', 'text': 'hi'}],
            }
        },
    ])


def a2ui_data_part(envelopes: list[dict[str, Any]]) -> Part:
    return Part(DataPart(data={'envelopes': envelopes}, metadata={'mimeType': A2UI_MIME_TYPE}))


def text_part(text: str) -> Part:
    return Part(TextPart(text=text))


def model_ok(text: str = 'ok') -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[text_part(text)]),
    )


def setup() -> tuple[Genkit, ProgrammableModel]:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    return ai, pm


def roots(content: list[Part]) -> list[object]:
    return [part.root for part in content]


def a2ui_parts(content: list[Part]) -> list[DataPart]:
    out: list[DataPart] = []
    for root in roots(content):
        if not isinstance(root, DataPart):
            continue
        metadata = root.metadata or {}
        if metadata.get('mimeType') == A2UI_MIME_TYPE:
            out.append(root)
    return out


def joined_text(content: list[Part]) -> str:
    bits: list[str] = []
    for root in roots(content):
        if isinstance(root, TextPart) and root.text:
            bits.append(root.text)
    return ''.join(bits)


def envelopes(content: list[Part]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in a2ui_parts(content):
        data = part.data
        if not isinstance(data, dict):
            continue
        raw = data.get('envelopes')
        if isinstance(raw, list):
            out.extend(env for env in raw if isinstance(env, dict))
    return out


def create_surface_ids(content: list[Part]) -> list[str]:
    ids: list[str] = []
    for env in envelopes(content):
        payload = env.get('createSurface')
        if isinstance(payload, dict) and isinstance(payload.get('surfaceId'), str):
            ids.append(payload['surfaceId'])
    return ids


def assert_finished_message(
    response: ModelResponse,
    *,
    finish_reason: FinishReason = FinishReason.STOP,
) -> Message:
    assert response.finish_reason == finish_reason
    assert response.message is not None
    assert response.messages[-1] == response.message
    return response.message


def assert_no_a2ui_parts(content: list[Part]) -> None:
    assert a2ui_parts(content) == []


def assert_no_fence_in_text(content: list[Part]) -> None:
    assert A2UI_FENCE not in joined_text(content)


def request_messages(pm: ProgrammableModel) -> list[Message]:
    assert pm.last_request is not None
    return list(pm.last_request.messages)


def request_has_a2ui_part(pm: ProgrammableModel) -> bool:
    return any(a2ui_parts(message.content) for message in request_messages(pm))


def request_history_messages(pm: ProgrammableModel) -> list[Message]:
    """Conversation leftover the model sees — not the injected catalog prompt."""
    return [message for message in request_messages(pm) if message.role != Role.SYSTEM]


def request_joined_text(pm: ProgrammableModel) -> str:
    return '\n'.join(joined_text(message.content) for message in request_history_messages(pm))


def request_system_text(pm: ProgrammableModel) -> str:
    return '\n'.join(joined_text(message.content) for message in request_messages(pm) if message.role == Role.SYSTEM)
