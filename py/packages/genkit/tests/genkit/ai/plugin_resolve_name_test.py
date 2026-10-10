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

"""What a plugin's resolve receives, and what name its action is registered under.

``ai.generate(model='myplug/fast-model')`` calls the ``myplug`` plugin's
``resolve`` with ``fast-model``. The returned action is registered and named
under the id the caller asked for (``myplug/fast-model``).
"""

from collections.abc import Callable

import pytest

from genkit import ActionRunContext, Genkit, GenkitError, Message, ModelResponse, Operation, Part, Role
from genkit.embedder import EmbedRequest, EmbedResponse, embedder
from genkit.model import ModelRequest, background_model, model
from genkit.plugin_api import Action, ActionKind, ActionMetadata, Plugin


def _answering_model(name: str, text: str) -> Action:
    async def fn(request: ModelRequest) -> ModelResponse:
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(text)]))

    return model(name, fn)


class RecordingPlugin(Plugin):
    """Serves the ids in ``serves``; ``build`` decides the returned action's name."""

    name = 'myplug'

    def __init__(
        self,
        *,
        serves: set[str],
        build: Callable[[str], str] = lambda requested: requested,
        init_names: list[str] | None = None,
    ) -> None:
        self.serves = serves
        self.build = build
        self.init_names = init_names or []
        self.seen: list[tuple[ActionKind, str]] = []

    async def init(self) -> list[Action]:
        return [_answering_model(n, f'init {n}') for n in self.init_names]

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        self.seen.append((action_type, name))
        if action_type != ActionKind.MODEL or name not in self.serves:
            return None
        return _answering_model(self.build(name), f'resolved {name}')

    async def list_actions(self) -> list[ActionMetadata]:
        return []


class VideoPlugin(Plugin):
    """Builds a background model from the bare id, the way a plugin author would."""

    name = 'myplug'

    def __init__(self) -> None:
        self.seen: list[tuple[ActionKind, str]] = []

    def _video(self, model_id: str):  # noqa: ANN202
        async def start(request: ModelRequest, ctx: ActionRunContext) -> Operation:
            return Operation(id='job-1', done=False)

        async def check(op: Operation, ctx: ActionRunContext) -> Operation:
            return Operation(id=op.id, done=True)

        async def cancel(op: Operation, ctx: ActionRunContext) -> Operation:
            return Operation(id=op.id, done=True, metadata={'cancelled': True})

        return background_model(model_id, start=start, check=check, cancel=cancel)

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        self.seen.append((action_type, name))
        if action_type == ActionKind.BACKGROUND_MODEL and name == 'veo-x':
            return self._video(name).start_action
        if action_type == ActionKind.CHECK_OPERATION and name == 'veo-x/check':
            return self._video('veo-x').check_action
        if action_type == ActionKind.CANCEL_OPERATION and name == 'veo-x/cancel':
            return self._video('veo-x').cancel_action
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        return []


@pytest.mark.asyncio
async def test_generate_plugin_resolve_receives_id_without_plugin_prefix() -> None:
    """`resolve` sees `fast-model` for `generate(model='myplug/fast-model')`."""
    plugin = RecordingPlugin(serves={'fast-model'})
    ai = Genkit(plugins=[plugin])

    response = await ai.generate(model='myplug/fast-model', prompt='hi')

    assert response.text == 'resolved fast-model'
    assert (ActionKind.MODEL, 'fast-model') in plugin.seen
    assert all(not name.startswith('myplug/') for _, name in plugin.seen)


@pytest.mark.asyncio
async def test_generate_plugin_slash_id_keeps_full_path() -> None:
    """`myplug/anthropic/claude` resolves with `anthropic/claude` and registers under the full path."""
    plugin = RecordingPlugin(serves={'anthropic/claude'})
    ai = Genkit(plugins=[plugin])

    response = await ai.generate(model='myplug/anthropic/claude', prompt='hi')

    assert response.text == 'resolved anthropic/claude'
    assert (ActionKind.MODEL, 'anthropic/claude') in plugin.seen
    action = await ai.registry.resolve_action(ActionKind.MODEL, 'myplug/anthropic/claude')
    assert action is not None
    assert action.name == 'myplug/anthropic/claude'
    assert await ai.registry.resolve_action(ActionKind.MODEL, 'myplug/claude') is None


@pytest.mark.asyncio
async def test_generate_plugin_resolve_returning_prefixed_name_registers_once() -> None:
    """An action already named `myplug/x` stays `myplug/x`, not `myplug/myplug/x`."""
    plugin = RecordingPlugin(serves={'x'}, build=lambda requested: f'myplug/{requested}')
    ai = Genkit(plugins=[plugin])

    response = await ai.generate(model='myplug/x', prompt='hi')

    assert response.text == 'resolved x'
    action = await ai.registry.resolve_action(ActionKind.MODEL, 'myplug/x')
    assert action is not None
    assert action.name == 'myplug/x'


@pytest.mark.asyncio
async def test_generate_plugin_resolve_returning_bare_name_gets_prefix() -> None:
    """An action named `x` registers as `myplug/x`."""
    plugin = RecordingPlugin(serves={'x'})
    ai = Genkit(plugins=[plugin])

    response = await ai.generate(model='myplug/x', prompt='hi')

    assert response.text == 'resolved x'
    action = await ai.registry.resolve_action(ActionKind.MODEL, 'myplug/x')
    assert action is not None
    assert action.name == 'myplug/x'


@pytest.mark.asyncio
async def test_generate_plugin_resolve_returning_other_name_is_found_under_requested_id() -> None:
    """An action named `other/x` for a request of `x` is stored as `myplug/x`."""
    plugin = RecordingPlugin(serves={'x'}, build=lambda requested: f'other/{requested}')
    ai = Genkit(plugins=[plugin])

    response = await ai.generate(model='myplug/x', prompt='hi')

    assert response.text == 'resolved x'
    action = await ai.registry.resolve_action(ActionKind.MODEL, 'myplug/x')
    assert action is not None
    assert action.name == 'myplug/x'


@pytest.mark.asyncio
async def test_plugin_init_bare_names_get_plugin_prefix() -> None:
    """Actions from `init()` follow the same rule."""
    plugin = RecordingPlugin(serves=set(), init_names=['init-model'])
    ai = Genkit(plugins=[plugin])

    response = await ai.generate(model='myplug/init-model', prompt='hi')

    assert response.text == 'init init-model'
    assert plugin.seen == []


@pytest.mark.asyncio
async def test_generate_plugin_id_starting_with_plugin_name_is_found() -> None:
    """`myplug/myplug/x` is found when the plugin serves `myplug/x` and returns it bare."""
    plugin = RecordingPlugin(serves={'myplug/x'})
    ai = Genkit(plugins=[plugin])

    response = await ai.generate(model='myplug/myplug/x', prompt='hi')

    assert response.text == 'resolved myplug/x'
    assert (ActionKind.MODEL, 'myplug/x') in plugin.seen
    action = await ai.registry.resolve_action(ActionKind.MODEL, 'myplug/myplug/x')
    assert action is not None
    assert action.name == 'myplug/myplug/x'


@pytest.mark.asyncio
async def test_generate_plugin_id_starting_with_plugin_name_does_not_take_shorter_id() -> None:
    """After resolving `myplug/myplug/x`, `myplug/x` still asks resolve for `x`."""
    plugin = RecordingPlugin(serves={'myplug/x', 'x'})
    ai = Genkit(plugins=[plugin])

    await ai.generate(model='myplug/myplug/x', prompt='hi')
    plugin.seen.clear()
    response = await ai.generate(model='myplug/x', prompt='hi')

    assert response.text == 'resolved x'
    assert (ActionKind.MODEL, 'x') in plugin.seen
    assert (ActionKind.MODEL, 'myplug/x') not in plugin.seen


@pytest.mark.asyncio
async def test_plugin_init_name_with_other_prefix_keeps_full_path() -> None:
    """`init()` returning `other/x` registers `myplug/other/x`; `myplug/x` is NOT_FOUND."""
    plugin = RecordingPlugin(serves=set(), init_names=['other/x'])
    ai = Genkit(plugins=[plugin])

    response = await ai.generate(model='myplug/other/x', prompt='hi')

    assert response.text == 'init other/x'
    assert await ai.registry.resolve_action(ActionKind.MODEL, 'myplug/x') is None
    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model='myplug/x', prompt='hi')
    assert exc_info.value.status == 'NOT_FOUND'


@pytest.mark.asyncio
async def test_plugin_init_slash_names_keep_full_path() -> None:
    """`init()` returning `endpoints/1` registers `myplug/endpoints/1`."""
    plugin = RecordingPlugin(serves=set(), init_names=['endpoints/1'])
    ai = Genkit(plugins=[plugin])

    response = await ai.generate(model='myplug/endpoints/1', prompt='hi')

    assert response.text == 'init endpoints/1'
    assert plugin.seen == []
    assert await ai.registry.resolve_action(ActionKind.MODEL, 'myplug/1') is None


@pytest.mark.asyncio
async def test_plugin_model_helper_returns_action_named_with_plugin_prefix() -> None:
    """`await plugin.model('fast-model')` is named `myplug/fast-model`."""
    plugin = RecordingPlugin(serves={'fast-model'})

    action = await plugin.model('fast-model')

    assert action is not None
    assert action.name == 'myplug/fast-model'


@pytest.mark.asyncio
async def test_plugin_embedder_helper_returns_action_named_with_plugin_prefix() -> None:
    """`await plugin.embedder('embed-1')` is named `myplug/embed-1`."""

    class EmbedPlugin(Plugin):
        name = 'myplug'

        async def init(self) -> list[Action]:
            return []

        async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
            if action_type != ActionKind.EMBEDDER or name != 'embed-1':
                return None

            async def fn(request: EmbedRequest) -> EmbedResponse:
                return EmbedResponse(embeddings=[])

            return embedder(name, fn)

        async def list_actions(self) -> list[ActionMetadata]:
            return []

    action = await EmbedPlugin().embedder('embed-1')

    assert action is not None
    assert action.name == 'myplug/embed-1'


@pytest.mark.asyncio
async def test_plugin_model_helper_passes_bare_id() -> None:
    """`await plugin.model('myplug/fast-model')` calls `resolve` with `fast-model`."""
    plugin = RecordingPlugin(serves={'fast-model'})

    with_prefix = await plugin.model('myplug/fast-model')
    without_prefix = await plugin.model('fast-model')

    assert with_prefix is not None
    assert without_prefix is not None
    assert plugin.seen == [(ActionKind.MODEL, 'fast-model'), (ActionKind.MODEL, 'fast-model')]


@pytest.mark.asyncio
async def test_plugin_embedder_helper_passes_bare_id() -> None:
    """`await plugin.embedder('myplug/embed-1')` calls `resolve` with `embed-1`."""
    plugin = RecordingPlugin(serves=set())

    await plugin.embedder('myplug/embed-1')

    assert plugin.seen == [(ActionKind.EMBEDDER, 'embed-1')]


@pytest.mark.asyncio
async def test_generate_unknown_plugin_id_returns_not_found() -> None:
    """An id `resolve` declines is still NOT_FOUND (control)."""
    plugin = RecordingPlugin(serves={'fast-model'})
    ai = Genkit(plugins=[plugin])

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model='myplug/slow-model', prompt='hi')

    assert exc_info.value.status == 'NOT_FOUND'
    assert (ActionKind.MODEL, 'slow-model') in plugin.seen


@pytest.mark.asyncio
async def test_generate_operation_plugin_background_model_can_be_checked() -> None:
    """A background model built with a bare id hands out a handle `check_operation` resolves."""
    ai = Genkit(plugins=[VideoPlugin()])

    operation = await ai.generate_operation(model='myplug/veo-x', prompt='a cat')
    checked = await ai.check_operation(operation)

    assert operation.action == '/background-model/myplug/veo-x'
    assert operation.done is False
    assert checked.id == 'job-1'
    assert checked.done is True
    assert checked.action == '/background-model/myplug/veo-x'


@pytest.mark.asyncio
async def test_cancel_operation_plugin_background_model_keeps_handle_key() -> None:
    """Cancelling that handle returns one that still points at `myplug/veo-x`."""
    ai = Genkit(plugins=[VideoPlugin()])

    operation = await ai.generate_operation(model='myplug/veo-x', prompt='a cat')
    cancelled = await ai.cancel_operation(operation)

    assert cancelled.id == 'job-1'
    assert cancelled.done is True
    assert cancelled.metadata == {'cancelled': True}
    assert cancelled.action == '/background-model/myplug/veo-x'


@pytest.mark.asyncio
async def test_check_operation_resolve_receives_bare_check_name() -> None:
    """The check action is resolved as `veo-x/check`, not `myplug/veo-x/check`."""
    plugin = VideoPlugin()
    ai = Genkit(plugins=[plugin])

    operation = await ai.generate_operation(model='myplug/veo-x', prompt='a cat')
    await ai.check_operation(operation)

    assert (ActionKind.BACKGROUND_MODEL, 'veo-x') in plugin.seen
    assert (ActionKind.CHECK_OPERATION, 'veo-x/check') in plugin.seen
    assert all(not name.startswith('myplug/') for _, name in plugin.seen)


@pytest.mark.asyncio
async def test_check_operation_saved_handle_checks_on_a_fresh_app() -> None:
    """A handle saved with `/background-model/myplug/veo-x` checks on a new `Genkit` that never started it."""
    saved = Operation(id='job-1', done=False, action='/background-model/myplug/veo-x')
    ai = Genkit(plugins=[VideoPlugin()])

    checked = await ai.check_operation(saved)

    assert checked.id == 'job-1'
    assert checked.done is True
    assert checked.action == '/background-model/myplug/veo-x'
