# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Every action takes at most one input; the context goes to the parameter annotated ActionRunContext."""

import sys
import types
from collections.abc import Awaitable, Callable
from typing import Optional, cast

import pytest
from pydantic import TypeAdapter

from genkit import ActionRunContext, Document, Genkit, GenkitError, Message, ModelResponse, Operation, Part
from genkit._core._action import Action, ActionKind
from genkit._core._background import CheckModelOpFn, StartModelOpFn
from genkit._core._typing import ActionMetadata, Role
from genkit.model import ModelRequest
from genkit.plugin_api import Plugin

_REPLY = ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('Smoked Salmon Tartine')]))


@pytest.mark.asyncio
async def test_model_with_context_first_gets_request_and_context() -> None:
    """A model fn `(ctx, request)` gets each by annotation, not by position."""
    seen: list[object] = []

    async def chef(ctx: ActionRunContext, request: ModelRequest) -> ModelResponse:
        seen.extend([type(request), ctx.context])
        return _REPLY

    action = Action(ActionKind.MODEL, 'chef', chef)
    result = await action.run(ModelRequest(messages=[]), context={'table': 4})

    assert result.response.text == 'Smoked Salmon Tartine'
    assert seen == [ModelRequest, {'table': 4}]


@pytest.mark.asyncio
async def test_action_with_optional_context_union_gets_context() -> None:
    """`ctx: ActionRunContext | None` and `Optional[ActionRunContext]` both mark the context."""

    async def by_pipe(order: str, ctx: ActionRunContext | None = None) -> str:
        return f'{order} for table {ctx.context["table"] if ctx else "?"}'

    async def by_optional(ctx: Optional[ActionRunContext], order: str) -> str:  # noqa: UP045 - the old spelling is the case under test
        return f'{order} for table {ctx.context["table"] if ctx else "?"}'

    for fn in (by_pipe, by_optional):
        action = Action(ActionKind.CUSTOM, fn.__name__, fn)
        result = await action.run('soup', context={'table': 4})
        assert result.response == 'soup for table 4'


@pytest.mark.asyncio
async def test_action_with_postponed_context_annotations_gets_context() -> None:
    """Postponed annotations mark the context, parameterized or not."""
    seen: list[object] = []

    # Postponed annotations are stored as these strings.
    async def plain(order: str, ctx: 'ActionRunContext') -> None:
        seen.append(type(ctx))

    async def parameterized(ctx: 'ActionRunContext[str]', order: str) -> None:
        seen.append(type(ctx))

    for fn in (plain, parameterized):
        await Action(ActionKind.CUSTOM, fn.__name__, fn).run('soup')

    assert seen == [ActionRunContext, ActionRunContext]


# A user module that imports its context type only for type checkers, so the
# annotation stays the string 'ToolRunContext | None' at runtime.
_TYPE_CHECKING_CONTEXT_MODULE = """
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genkit import ToolRunContext


async def plate(order: str, ctx: ToolRunContext | None) -> str:
    return f'{order} for table {ctx.context["table"]}'
"""


@pytest.mark.asyncio
async def test_action_with_unresolvable_context_subclass_union_gets_context() -> None:
    """An unresolvable `'ToolRunContext | None'` still marks the context, by class name."""
    module_globals: dict[str, object] = {}
    exec(_TYPE_CHECKING_CONTEXT_MODULE, module_globals)  # noqa: S102 - builds a module with postponed annotations
    plate = cast(Callable[[str, ActionRunContext], Awaitable[str]], module_globals['plate'])

    result = await Action(ActionKind.CUSTOM, 'plate', plate).run('soup', context={'table': 4})

    assert result.response == 'soup for table 4'


@pytest.mark.skipif(sys.version_info < (3, 14), reason='annotations are evaluated lazily from Python 3.14')
@pytest.mark.asyncio
async def test_action_with_type_checking_only_context_runs_without_future_import() -> None:
    """On 3.14, a `TYPE_CHECKING`-only `ToolRunContext` without the `__future__` import still runs."""
    module_globals: dict[str, object] = {}
    source = _TYPE_CHECKING_CONTEXT_MODULE.replace('from __future__ import annotations\n', '')
    exec(source, module_globals)  # noqa: S102 - builds a module whose annotations are evaluated lazily
    plate = cast(Callable[[str, ActionRunContext], Awaitable[str]], module_globals['plate'])

    result = await Action(ActionKind.CUSTOM, 'plate', plate).run('soup', context={'table': 4})

    assert result.response == 'soup for table 4'


@pytest.mark.asyncio
async def test_action_with_defaulted_input_after_context_uses_default() -> None:
    """Running without an input leaves a defaulted input out, even when it comes after the context."""

    async def special(ctx: ActionRunContext, dish: str = 'risotto') -> str:
        return dish

    assert (await Action(ActionKind.CUSTOM, 'special', special).run()).response == 'risotto'


@pytest.mark.asyncio
async def test_custom_action_allows_unannotated_input() -> None:
    """Only tools and flows require an input annotation; other actions accept anything."""

    async def echo(order, ctx: ActionRunContext) -> object:  # noqa: ANN001
        return order

    action = Action(ActionKind.CUSTOM, 'echo', echo)

    assert action.input_schema == {}
    assert (await action.run({'dish': 'soup'})).response == {'dish': 'soup'}


def test_action_with_unannotated_second_parameter_raises_type_error() -> None:
    """`(request, ctx)` with no ctx annotation raises at definition, saying to annotate ActionRunContext."""

    async def chef(request: ModelRequest, ctx) -> ModelResponse:  # noqa: ANN001
        return _REPLY

    with pytest.raises(TypeError, match="model 'chef' takes one input, but 'ctx' is a second parameter") as exc:
        Action(ActionKind.MODEL, 'chef', chef)
    assert "annotate 'ctx' as ActionRunContext" in str(exc.value)


def test_action_with_positional_only_parameter_raises_type_error() -> None:
    """A positional-only parameter can't be passed by name, so it raises at definition."""

    async def chef(request: ModelRequest, /, ctx: ActionRunContext) -> ModelResponse:
        return _REPLY

    with pytest.raises(TypeError, match="model 'chef' parameter 'request' is positional-only"):
        Action(ActionKind.MODEL, 'chef', chef)


@pytest.mark.asyncio
async def test_background_model_with_context_first_start_gets_both() -> None:
    """`start(ctx, request)` gets each by name, and the handle has the start action key."""
    ai = Genkit()
    seen: list[object] = []

    async def start(ctx: ActionRunContext, request: ModelRequest) -> Operation:
        seen.extend([type(request), ctx.context])
        return Operation(id='render-1', done=False)

    async def check(op: Operation, ctx: ActionRunContext) -> Operation:
        return op

    # StartModelOpFn types (request, ctx) positionally; the runtime takes either order.
    action = ai.define_background_model(name='menu-video', start=cast(StartModelOpFn, start), check=check)
    result = await action.start_action.run(ModelRequest(messages=[]), context={'table': 4})

    assert seen == [ModelRequest, {'table': 4}]
    assert result.response.id == 'render-1'
    assert result.response.action == '/background-model/menu-video'


@pytest.mark.asyncio
async def test_background_model_with_context_first_check_gets_operation() -> None:
    """`check(ctx, op)` gets each by name, same as start."""
    ai = Genkit()
    seen: list[object] = []

    async def start(request: ModelRequest, ctx: ActionRunContext) -> Operation:
        return Operation(id='render-1', done=False)

    async def check(ctx: ActionRunContext, op: Operation) -> Operation:
        seen.extend([op.id, ctx.context])
        return op

    action = ai.define_background_model(name='menu-video', start=start, check=cast(CheckModelOpFn, check))
    started = await action.start_action.run(ModelRequest(messages=[]), context={'table': 4})
    result = await action.check_action.run(started.response, context={'table': 4})

    assert seen == ['render-1', {'table': 4}]
    assert result.response.action == '/background-model/menu-video'


def test_background_model_actions_lists_start_check_and_cancel() -> None:
    """`actions` is every action a background model registers; cancel only when given."""
    ai = Genkit()

    async def start(request: ModelRequest, ctx: ActionRunContext) -> Operation:
        return Operation(id='render-1')

    async def check(op: Operation, ctx: ActionRunContext) -> Operation:
        return op

    video = ai.define_background_model(name='menu-video', start=start, check=check, cancel=check)
    photo = ai.define_background_model(name='menu-photo', start=start, check=check)

    assert [a.name for a in video.actions] == ['menu-video', 'menu-video/check', 'menu-video/cancel']
    assert [a.name for a in photo.actions] == ['menu-photo', 'menu-photo/check']


def test_action_params_names_input_and_context() -> None:
    """`Action.params` exposes which parameter gets the input and which gets the context."""

    async def chef(ctx: ActionRunContext, request: ModelRequest) -> ModelResponse:
        return _REPLY

    params = Action(ActionKind.MODEL, 'chef', chef).params

    assert params.input is not None and params.input.name == 'request'
    assert params.context is not None and params.context.name == 'ctx'
    assert not params.input_optional


# -----------------------------------------------------------------------------
# Request and response types imported only under TYPE_CHECKING
#
# Ruff's TC rules move `ModelRequest`, `ModelResponse`, `EmbedRequest`, and
# `Operation` under `if TYPE_CHECKING:` when a handler only annotates them.
# Genkit fills in those names for the kinds that always use them.
# -----------------------------------------------------------------------------


def _load(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    exec(source, module.__dict__)  # noqa: S102 - builds a module with postponed annotations
    return module


_TYPE_CHECKING_MODEL = """
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from genkit import Message, ModelResponse as _ModelResponse, Part, Role

if TYPE_CHECKING:
    from genkit import ActionRunContext, ModelResponse
    from genkit.model import ModelRequest

    class AllergyRequest(ModelRequest):
        pass

seen: list[object] = []


class KitchenConfig(BaseModel):
    temperature: float | None = None


def _plate(text: str) -> _ModelResponse:
    return _ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(text)]))


async def chef(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
    seen.append(request)
    return _plate('Smoked Salmon Tartine')


async def tuned_chef(request: ModelRequest[KitchenConfig]) -> ModelResponse:
    seen.append(request.config.temperature if request.config else None)
    return _plate('Smoked Salmon Tartine')


async def misspelled_chef(request: ModelReqeust) -> ModelResponse:
    return _plate('never')


async def subclass_chef(request: AllergyRequest) -> ModelResponse:
    return _plate('never')


async def dict_chef(request: dict[str, object]) -> object:
    return {}
"""


_TYPE_CHECKING_EMBEDDER = """
from __future__ import annotations

from typing import TYPE_CHECKING

from genkit._core._typing import Embedding, EmbedResponse as _EmbedResponse

if TYPE_CHECKING:
    from genkit.embedder import EmbedRequest, EmbedResponse

seen: list[object] = []


async def menu_vectors(request: EmbedRequest) -> EmbedResponse:
    seen.extend(request.input)
    return _EmbedResponse(embeddings=[Embedding(embedding=[1.0, 2.0])])
"""


_TYPE_CHECKING_BACKGROUND_MODEL = """
from __future__ import annotations

from typing import TYPE_CHECKING

from genkit import Operation as _Operation

if TYPE_CHECKING:
    from genkit import Operation
    from genkit.model import ModelRequest


async def start(request: ModelRequest) -> Operation:
    return _Operation(id='render-1', done=False)


async def check(op: Operation) -> Operation:
    return op.model_copy(update={'done': True})
"""


_TYPE_CHECKING_FLOW = """
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    class OrderInput:
        pass


async def take_order(order: OrderInput) -> str:
    return 'x'
"""


class _KitchenPlugin(Plugin):
    name = 'kitchen'

    def __init__(self, fn: Callable[..., Awaitable[object]]) -> None:
        self._fn = fn

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        if action_type == ActionKind.MODEL and name == 'chef':
            return Action(kind=ActionKind.MODEL, name=name, fn=self._fn)
        if action_type == ActionKind.EMBEDDER and name == 'menu-vectors':
            return Action(kind=ActionKind.EMBEDDER, name=name, fn=self._fn)
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        return []


@pytest.mark.asyncio
async def test_generate_plugin_model_with_type_checking_request_and_response_returns_reply() -> None:
    """`request: ModelRequest` and `-> ModelResponse`, both TYPE_CHECKING-only, register and run."""
    module = _load('tc_chef', _TYPE_CHECKING_MODEL)
    ai = Genkit(plugins=[_KitchenPlugin(module.chef)])

    resp = await ai.generate(model='kitchen/chef', prompt='Suggest a dish.')

    assert resp.text == 'Smoked Salmon Tartine'
    assert [type(r) for r in module.seen] == [ModelRequest]


def test_define_model_with_type_checking_request_annotation_registers() -> None:
    """define_model runs _check_request_annotation before building the Action; that check lets the name through."""
    module = _load('tc_defined_chef', _TYPE_CHECKING_MODEL)

    action = Genkit().define_model(name='chef', fn=module.chef)

    assert action.input_schema == TypeAdapter(ModelRequest).json_schema()


@pytest.mark.asyncio
async def test_model_with_type_checking_parameterized_request_validates_config() -> None:
    """`ModelRequest[KitchenConfig]` keeps its config type: config is parsed and published."""
    module = _load('tc_tuned_chef', _TYPE_CHECKING_MODEL)
    action = Action(ActionKind.MODEL, 'chef', module.tuned_chef)

    await action.run({'messages': [], 'config': {'temperature': 0.5}})

    assert module.seen == [0.5]
    assert action.input_class is ModelRequest[module.KitchenConfig]
    assert action.input_schema == TypeAdapter(ModelRequest[module.KitchenConfig]).json_schema()


@pytest.mark.parametrize(
    ('handler', 'missing'), [('misspelled_chef', 'ModelReqeust'), ('subclass_chef', 'AllergyRequest')]
)
def test_model_with_unknown_type_checking_request_name_still_raises(handler: str, missing: str) -> None:
    """Only names Genkit knows are filled in; a typo or a TYPE_CHECKING-only subclass still raises."""
    fn = getattr(_load(f'tc_{handler}', _TYPE_CHECKING_MODEL), handler)

    with pytest.raises(TypeError, match=f"has type '{missing}', which can't be found"):
        Action(ActionKind.MODEL, 'chef', fn)


def test_model_with_runtime_resolvable_request_annotation_keeps_its_own_type() -> None:
    """A handler annotation that resolves is used as written."""
    fn = _load('tc_dict_chef', _TYPE_CHECKING_MODEL).dict_chef

    action = Action(ActionKind.MODEL, 'chef', fn)

    assert action.input_schema == TypeAdapter(dict[str, object]).json_schema()


@pytest.mark.asyncio
async def test_type_checking_request_annotation_validates_raw_json_as_model_request() -> None:
    """JSON from the Dev UI or reflection API is parsed into a ModelRequest, and bad JSON is rejected."""
    module = _load('tc_raw_json_chef', _TYPE_CHECKING_MODEL)
    action = Action(ActionKind.MODEL, 'chef', module.chef)

    await action.run({'messages': [{'role': 'user', 'content': [{'text': 'hi'}]}]})
    assert [type(r) for r in module.seen] == [ModelRequest]

    with pytest.raises(GenkitError, match='INVALID_ARGUMENT'):
        await action.run({'messages': 'hi'})


@pytest.mark.asyncio
async def test_embed_plugin_embedder_with_type_checking_request_and_response_returns_embeddings() -> None:
    """`request: EmbedRequest` and `-> EmbedResponse`, both TYPE_CHECKING-only, register and run."""
    module = _load('tc_menu_vectors', _TYPE_CHECKING_EMBEDDER)
    ai = Genkit(plugins=[_KitchenPlugin(module.menu_vectors)])

    embeddings = await ai.embed(embedder='kitchen/menu-vectors', content=Document.from_text('Smoked Salmon Tartine'))

    assert embeddings[0].embedding == [1.0, 2.0]
    # genkit.embedder.EmbedRequest declares list[Document], so that's what the handler gets.
    assert [type(d) for d in module.seen] == [Document]


@pytest.mark.asyncio
async def test_background_model_with_type_checking_operation_checks_operation() -> None:
    """start, check, and cancel annotated with TYPE_CHECKING-only `Operation` all register and run."""
    module = _load('tc_menu_video', _TYPE_CHECKING_BACKGROUND_MODEL)
    ai = Genkit()

    action = ai.define_background_model(name='menu-video', start=module.start, check=module.check, cancel=module.check)
    started = await action.start_action.run(ModelRequest(messages=[]))
    checked = await action.check_action.run(started.response)

    assert checked.response.done is True
    assert action.cancel_action is not None
    assert action.cancel_action.input_class is Operation


@pytest.mark.parametrize('define', ['flow', 'tool'])
def test_define_flow_or_tool_with_type_checking_input_annotation_still_raises(define: str) -> None:
    """Flows and tools have no fixed input type, so a missing name still raises."""
    fn = _load(f'tc_{define}', _TYPE_CHECKING_FLOW).take_order
    ai = Genkit()

    with pytest.raises(TypeError, match='OrderInput'):
        if define == 'flow':
            ai.flow()(fn)
        else:
            ai.tool()(fn)


@pytest.mark.skipif(sys.version_info < (3, 14), reason='annotations are evaluated lazily from Python 3.14')
@pytest.mark.asyncio
async def test_model_with_type_checking_parameterized_request_runs_without_future_import() -> None:
    """On 3.14 the annotation is read as a string too, so `ModelRequest[KitchenConfig]` resolves the same way."""
    source = _TYPE_CHECKING_MODEL.replace('from __future__ import annotations\n', '')
    # Annotations are evaluated lazily, so the undefined names in the other handlers never run.
    module = _load('tc_lazy_chef', source)
    action = Action(ActionKind.MODEL, 'chef', module.tuned_chef)

    await action.run({'messages': [], 'config': {'temperature': 0.5}})

    assert module.seen == [0.5]
    assert action.output_schema == TypeAdapter(ModelResponse).json_schema()
