# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""The A2UI part's wire shape: written as ``messages``, read as either key."""

from __future__ import annotations

from genkit_a2ui import A2UI_MIME_TYPE, a2ui_part, envelopes_from_parts, is_a2ui_part

from genkit import Part

CREATE = {'version': 'v0.9', 'createSurface': {'surfaceId': 's', 'catalogId': 'c'}}
DELETE = {'version': 'v0.9', 'deleteSurface': {'surfaceId': 's'}}


def _part(data: dict[str, object]) -> Part:
    return Part.from_data(data, metadata={'mimeType': A2UI_MIME_TYPE})


def test_a2ui_part_writes_the_spec_messages_key() -> None:
    assert a2ui_part([CREATE]).data == {'messages': [CREATE]}


def test_reads_the_legacy_envelopes_key() -> None:
    legacy = _part({'envelopes': [CREATE]})
    assert is_a2ui_part(legacy)
    assert envelopes_from_parts([legacy]) == [CREATE]


def test_messages_wins_when_both_keys_are_present() -> None:
    assert envelopes_from_parts([_part({'messages': [CREATE], 'envelopes': [DELETE]})]) == [CREATE]


def test_falls_back_to_envelopes_when_messages_is_none() -> None:
    part = _part({'messages': None, 'envelopes': [CREATE]})
    assert is_a2ui_part(part)
    assert envelopes_from_parts([part]) == [CREATE]


def test_rejects_a_non_list_message_list() -> None:
    part = _part({'messages': 'oops'})
    assert not is_a2ui_part(part)
    assert envelopes_from_parts([part]) == []
