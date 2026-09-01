#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""define_embedder accepts EmbedRequest / EmbedRequest[Cfg] and rejects the rest."""

from typing import Annotated, Any, Optional

import pytest
from pydantic import BaseModel

from genkit import Genkit
from genkit._core._error import GenkitError
from genkit._core._model import EmbedRequest
from genkit._core._typing import Embedding, EmbedResponse


class Cfg(BaseModel):
    """Sample typed plugin options."""

    task_type: str | None = None


OK = EmbedResponse(embeddings=[Embedding(embedding=[1.0])])


@pytest.fixture
def ai() -> Genkit:
    return Genkit()


def test_typed_annotation_allowed(ai: Genkit) -> None:
    """define_embedder(fn with EmbedRequest[Cfg]) registers."""

    async def fn(request: EmbedRequest[Cfg]) -> EmbedResponse:
        return OK

    ai.define_embedder(name='typed', fn=fn)


def test_bare_annotation_allowed(ai: Genkit) -> None:
    """define_embedder(fn with EmbedRequest) registers. Options stay a dict."""

    async def fn(request: EmbedRequest) -> EmbedResponse:
        return OK

    ai.define_embedder(name='bare', fn=fn)


def test_unannotated_allowed(ai: Genkit) -> None:
    """define_embedder(fn with no request annotation) still registers."""

    async def fn(request) -> EmbedResponse:  # noqa: ANN001
        return OK

    ai.define_embedder(name='unannotated', fn=fn)


def test_annotated_wrapper_unwrapped_and_allowed(ai: Genkit) -> None:
    """Annotated[EmbedRequest[Cfg], ...] is treated as EmbedRequest[Cfg]."""

    async def fn(request: Annotated[EmbedRequest[Cfg], 'doc']) -> EmbedResponse:
        return OK

    ai.define_embedder(name='annotated', fn=fn)


def test_dict_and_any_parametrizations_allowed_unblessed(ai: Genkit) -> None:
    """EmbedRequest[dict] and EmbedRequest[Any] still register; they do not validate a schema."""

    async def fn_d(request: EmbedRequest[dict]) -> EmbedResponse:
        return OK

    async def fn_a(request: EmbedRequest[Any]) -> EmbedResponse:
        return OK

    ai.define_embedder(name='param_dict', fn=fn_d)
    ai.define_embedder(name='param_any', fn=fn_a)


def test_union_with_none_rejected(ai: Genkit) -> None:
    """EmbedRequest[Cfg] | None is rejected — embed never passes None."""

    async def fn(request: EmbedRequest[Cfg] | None) -> EmbedResponse:
        return OK

    with pytest.raises(GenkitError, match='must be annotated as EmbedRequest'):
        ai.define_embedder(name='union', fn=fn)


def test_optional_spelling_rejected(ai: Genkit) -> None:
    """Optional[EmbedRequest[Cfg]] is the same reject as EmbedRequest[Cfg] | None."""

    async def fn(request: Optional[EmbedRequest[Cfg]]) -> EmbedResponse:  # noqa: UP045
        return OK

    with pytest.raises(GenkitError, match='must be annotated as EmbedRequest'):
        ai.define_embedder(name='optional', fn=fn)


def test_dict_annotation_rejected(ai: Genkit) -> None:
    """A handler annotated dict is rejected. The request is an EmbedRequest."""

    async def fn(request: dict) -> EmbedResponse:
        return OK

    with pytest.raises(GenkitError, match='must be annotated as EmbedRequest'):
        ai.define_embedder(name='rawdict', fn=fn)


def test_arbitrary_class_rejected(ai: Genkit) -> None:
    """A handler annotated with some other class is rejected."""

    class NotARequest(BaseModel):
        pass

    async def fn(request: NotARequest) -> EmbedResponse:
        return OK

    with pytest.raises(GenkitError, match='must be annotated as EmbedRequest'):
        ai.define_embedder(name='arbitrary', fn=fn)


def test_method_and_classmethod_annotations_allowed(ai: Genkit) -> None:
    """Methods on classes (unbound, bound, or @classmethod) inspect the request parameter, not self/cls."""

    class EmbedderHost:
        async def unbound_method(self, request: EmbedRequest[Cfg]) -> EmbedResponse:
            return OK

        async def annotated_self_method(self: 'EmbedderHost', request: EmbedRequest[Cfg]) -> EmbedResponse:
            return OK

        @classmethod
        async def class_method(cls, request: EmbedRequest[Cfg]) -> EmbedResponse:
            return OK

    host = EmbedderHost()
    ai.define_embedder(name='bound', fn=host.unbound_method)
    ai.define_embedder(name='unbound', fn=EmbedderHost.unbound_method)
    ai.define_embedder(name='annotated_self', fn=EmbedderHost.annotated_self_method)
    ai.define_embedder(name='classmethod', fn=EmbedderHost.class_method)


def test_unbound_method_wrong_annotation_rejected(ai: Genkit) -> None:
    """An unbound method with an invalid request annotation is still rejected."""

    class EmbedderHost:
        async def bad_method(self, request: dict) -> EmbedResponse:
            return OK

    with pytest.raises(GenkitError, match='must be annotated as EmbedRequest'):
        ai.define_embedder(name='bad_unbound', fn=EmbedderHost.bad_method)
