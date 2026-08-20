#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""define_model accepts ModelRequest / ModelRequest[Cfg] and rejects the rest."""

from typing import Annotated, Any, Optional

import pytest
from pydantic import BaseModel

from genkit import Genkit
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import Message, ModelRequest, ModelResponse
from genkit._core._typing import Part, Role, TextPart


class Cfg(BaseModel):
    """Sample typed plugin config."""

    temperature: float | None = None


OK = ModelResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='ok'))]))


@pytest.fixture
def ai() -> Genkit:
    return Genkit()


# --- allowed shapes -----------------------------------------------------------


def test_typed_annotation_allowed(ai: Genkit) -> None:
    """define_model(fn with ModelRequest[Cfg]) registers."""

    async def fn(request: ModelRequest[Cfg], ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai.define_model(name='typed', fn=fn)


def test_bare_annotation_allowed(ai: Genkit) -> None:
    """define_model(fn with ModelRequest) registers. Config stays a dict."""

    async def fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai.define_model(name='bare', fn=fn)


def test_unannotated_allowed(ai: Genkit) -> None:
    """define_model(fn with no request annotation) still registers."""

    async def fn(request, ctx: ActionRunContext) -> ModelResponse:  # noqa: ANN001
        return OK

    ai.define_model(name='unannotated', fn=fn)


def test_annotated_wrapper_unwrapped_and_allowed(ai: Genkit) -> None:
    """Annotated[ModelRequest[Cfg], ...] is treated as ModelRequest[Cfg]."""

    async def fn(request: Annotated[ModelRequest[Cfg], 'doc'], ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai.define_model(name='annotated', fn=fn)


def test_dict_and_any_parametrizations_allowed_unblessed(ai: Genkit) -> None:
    """ModelRequest[dict] and ModelRequest[Any] still register; they do not validate a schema."""

    async def fn_d(request: ModelRequest[dict], ctx: ActionRunContext) -> ModelResponse:
        return OK

    async def fn_a(request: ModelRequest[Any], ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai.define_model(name='param_dict', fn=fn_d)
    ai.define_model(name='param_any', fn=fn_a)


# --- rejected antipatterns ----------------------------------------------------


def test_union_with_none_rejected(ai: Genkit) -> None:
    """ModelRequest[Cfg] | None is rejected — generate never passes None."""

    async def fn(request: ModelRequest[Cfg] | None, ctx: ActionRunContext) -> ModelResponse:
        return OK

    with pytest.raises(GenkitError, match='must be annotated as ModelRequest'):
        ai.define_model(name='union', fn=fn)


def test_optional_spelling_rejected(ai: Genkit) -> None:
    """Optional[ModelRequest[Cfg]] is the same reject as ModelRequest[Cfg] | None."""

    async def fn(request: Optional[ModelRequest[Cfg]], ctx: ActionRunContext) -> ModelResponse:  # noqa: UP045
        return OK

    with pytest.raises(GenkitError, match='must be annotated as ModelRequest'):
        ai.define_model(name='optional', fn=fn)


def test_dict_annotation_rejected(ai: Genkit) -> None:
    """A handler annotated dict is rejected. The request is a ModelRequest."""

    async def fn(request: dict, ctx: ActionRunContext) -> ModelResponse:
        return OK

    with pytest.raises(GenkitError, match='must be annotated as ModelRequest'):
        ai.define_model(name='rawdict', fn=fn)


def test_arbitrary_class_rejected(ai: Genkit) -> None:
    """A handler annotated with some other class is rejected. The request is a ModelRequest."""

    class NotARequest(BaseModel):
        pass

    async def fn(request: NotARequest, ctx: ActionRunContext) -> ModelResponse:
        return OK

    with pytest.raises(GenkitError, match='must be annotated as ModelRequest'):
        ai.define_model(name='arbitrary', fn=fn)
