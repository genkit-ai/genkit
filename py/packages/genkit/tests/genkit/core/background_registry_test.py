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

"""Plugins hand the registry a BackgroundAction; the registry files all three keys."""

from typing import Any

import pytest

from genkit import Genkit, Message, ModelResponse, Operation, Part
from genkit._core._typing import ModelInfo, Role
from genkit.model import BackgroundModel, ModelContext, ModelRequest
from genkit.plugin_api import Action, ActionKind, BackgroundAction, Plugin


async def _start(request: ModelRequest, ctx: ModelContext) -> Operation:
    return Operation(id='video-1', done=False)


async def _check(op: Operation, ctx: ModelContext) -> Operation:
    return Operation(
        id=op.id, done=True, output=ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('plated')]))
    )


async def _cancel(op: Operation, ctx: ModelContext) -> Operation:
    return Operation(id=op.id, done=True)


def _bistro_video(name: str = 'bistro-video') -> BackgroundModel:
    return BackgroundModel(
        name,
        start=_start,
        check=_check,
        cancel=_cancel,
        info=ModelInfo(label='Bistro Video'),
    )


class ResolvingKitchen(Plugin):
    """Resolves the background model only for its start key."""

    name = 'kitchen'

    def __init__(self) -> None:
        self.calls: list[tuple[ActionKind, str]] = []

    async def init(self) -> list[Action | BackgroundAction[Any, Any]]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | BackgroundAction[Any, Any] | None:
        self.calls.append((action_type, name))
        if action_type == ActionKind.BACKGROUND_MODEL and name == 'bistro-video':
            return _bistro_video()
        return None

    async def list_actions(self) -> list[Action | BackgroundAction[Any, Any]]:
        return [_bistro_video('kitchen/bistro-video')]


class EagerKitchen(Plugin):
    """Returns the background model from init."""

    name = 'eager'

    async def init(self) -> list[Action | BackgroundAction[Any, Any]]:
        return [_bistro_video()]

    async def resolve(self, action_type: ActionKind, name: str) -> Action | BackgroundAction[Any, Any] | None:
        return None

    async def list_actions(self) -> list[Action | BackgroundAction[Any, Any]]:
        return []


@pytest.mark.parametrize(
    ('key', 'member'),
    [
        ('/check-operation/kitchen/bistro-video/check', 'check_action'),
        ('/cancel-operation/kitchen/bistro-video/cancel', 'cancel_action'),
        ('/background-model/kitchen/bistro-video', 'start_action'),
    ],
)
@pytest.mark.asyncio
async def test_resolve_any_key_registers_all_three(key: str, member: str) -> None:
    """A check or cancel key reaches the start-only resolve; all siblings land."""
    plugin = ResolvingKitchen()
    ai = Genkit(plugins=[plugin])

    action = await ai._registry.resolve_action_by_key(key)

    assert action is not None
    assert action.name == key.split('/', 2)[2]
    for sibling in (
        '/background-model/kitchen/bistro-video',
        '/check-operation/kitchen/bistro-video/check',
        '/cancel-operation/kitchen/bistro-video/cancel',
    ):
        assert ai._registry.registered_action(*_split(sibling)) is not None, sibling
    looked_up = await BackgroundModel.lookup(ai._registry, '/background-model/kitchen/bistro-video')
    assert looked_up is not None
    assert getattr(looked_up, member).name == action.name


def _split(key: str) -> tuple[ActionKind, str]:
    _, kind, name = key.split('/', 2)
    return ActionKind(kind), name


@pytest.mark.asyncio
async def test_check_operation_through_a_resolving_plugin() -> None:
    """generate_operation + check_operation work with no hand-written check routing."""
    ai = Genkit(plugins=[ResolvingKitchen()])

    op = await ai.generate_operation(model='kitchen/bistro-video', prompt='A chef plating risotto.')
    assert op.action == '/background-model/kitchen/bistro-video'

    done = await ai.check_operation(op)

    assert done.output is not None
    assert done.output.text == 'plated'


@pytest.mark.asyncio
async def test_init_returning_a_background_model_registers_all_three() -> None:
    """init() can return the composite; no .actions flattening needed."""
    ai = Genkit(plugins=[EagerKitchen()])

    start = await ai._registry.resolve_action_by_key('/background-model/eager/bistro-video')
    check = await ai._registry.resolve_action_by_key('/check-operation/eager/bistro-video/check')
    cancel = await ai._registry.resolve_action_by_key('/cancel-operation/eager/bistro-video/cancel')

    assert start is not None and start.name == 'eager/bistro-video'
    assert check is not None and check.name == 'eager/bistro-video/check'
    assert cancel is not None and cancel.name == 'eager/bistro-video/cancel'


@pytest.mark.asyncio
async def test_list_actions_lists_a_background_model_once() -> None:
    """One catalog row, under the start key, with the model metadata."""
    ai = Genkit(plugins=[ResolvingKitchen()])

    catalog = await ai._registry.list_actions()

    rows = [key for key in catalog if 'bistro-video' in key]
    assert rows == ['/background-model/kitchen/bistro-video']
    meta = catalog['/background-model/kitchen/bistro-video']
    assert meta.metadata is not None
    assert meta.metadata['model']['label'] == 'Bistro Video'
    assert meta.metadata['model']['supports']['longRunning'] is True


@pytest.mark.asyncio
async def test_bare_action_from_resolve_still_works() -> None:
    """A plugin that routes check keys itself keeps working."""
    video = _bistro_video()

    class RoutingKitchen(Plugin):
        name = 'routing'

        async def init(self) -> list[Action | BackgroundAction[Any, Any]]:
            return []

        async def resolve(self, action_type: ActionKind, name: str) -> Action | BackgroundAction[Any, Any] | None:
            if action_type == ActionKind.CHECK_OPERATION:
                return video.check_action
            return None

        async def list_actions(self) -> list[Action | BackgroundAction[Any, Any]]:
            return []

    ai = Genkit(plugins=[RoutingKitchen()])

    check = await ai._registry.resolve_action_by_key('/check-operation/routing/bistro-video/check')

    assert check is video.check_action
    assert ai._registry.registered_action(ActionKind.BACKGROUND_MODEL, 'routing/bistro-video') is None
