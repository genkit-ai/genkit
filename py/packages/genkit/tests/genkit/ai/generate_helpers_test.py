# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for private helpers in genkit._ai._generate (interrupt / resume)."""

from genkit._ai._generate import (
    _find_corresponding_restart,
    _find_corresponding_tool_response,
    _interrupt_from_tool_exc,
    _to_pending_response,
)
from genkit._ai._tools import Interrupt
from genkit._core._error import GenkitError
from genkit._core._typing import ToolRequest, ToolRequestPart, ToolResponse, ToolResponsePart


def test_find_corresponding_restart_matches_name_and_ref() -> None:
    """``_find_corresponding_restart`` picks the resume TRP whose name+ref match the pending TRP; else None."""
    pending = ToolRequestPart(
        tool_request=ToolRequest(name='t', ref='r1', input={}),
    )
    match = ToolRequestPart(
        tool_request=ToolRequest(name='t', ref='r1', input={'new': True}),
        metadata={'resumed': True},
    )
    other_ref = ToolRequestPart(
        tool_request=ToolRequest(name='t', ref='r2', input={}),
        metadata={'resumed': True},
    )
    other_name = ToolRequestPart(
        tool_request=ToolRequest(name='u', ref='r1', input={}),
        metadata={'resumed': True},
    )

    assert _find_corresponding_restart([match], pending) is match
    assert _find_corresponding_restart([other_ref, match], pending) is match
    assert _find_corresponding_restart([other_ref], pending) is None
    assert _find_corresponding_restart([other_name], pending) is None
    assert _find_corresponding_restart(None, pending) is None
    assert _find_corresponding_restart([], pending) is None


def test_find_corresponding_tool_response_matches_name_and_ref() -> None:
    """``_find_corresponding_tool_response`` matches ``ToolResponsePart`` to pending TRP by name+ref."""
    pending = ToolRequestPart(
        tool_request=ToolRequest(name='t', ref='r1', input={}),
    )
    trp = ToolResponsePart(
        tool_response=ToolResponse(name='t', ref='r1', output=42),
    )
    other = ToolResponsePart(
        tool_response=ToolResponse(name='t', ref='r2', output=0),
    )

    got = _find_corresponding_tool_response([trp], pending)
    assert got is not None
    assert got == trp

    assert _find_corresponding_tool_response([other], pending) is None
    assert _find_corresponding_tool_response([], pending) is None


def test_interrupt_from_tool_exc() -> None:
    """``_interrupt_from_tool_exc`` unwraps bare ``Interrupt`` or ``GenkitError.cause``; else None."""
    intr = Interrupt({'x': 1})
    assert _interrupt_from_tool_exc(intr) is intr

    wrapped = GenkitError(message='x', cause=intr)
    assert _interrupt_from_tool_exc(wrapped) is intr

    assert _interrupt_from_tool_exc(ValueError('x')) is None


def test_to_pending_response_sets_pending_output() -> None:
    """``_to_pending_response`` merges prior TRP metadata with ``pendingOutput`` from the tool response."""
    req = ToolRequestPart(
        tool_request=ToolRequest(name='t', ref='r1', input={'a': 1}),
        metadata={'interrupt': {'old': True}},
    )
    resp = ToolResponsePart(
        tool_response=ToolResponse(name='t', ref='r1', output={'out': 2}),
    )
    part = _to_pending_response(req, resp)
    root = part.root
    assert isinstance(root, ToolRequestPart)
    assert root.tool_request.name == 't'
    assert root.tool_request.ref == 'r1'
    assert root.metadata is not None
    assert root.metadata.get('pendingOutput') == {'out': 2}
    assert root.metadata.get('interrupt') == {'old': True}
