#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Middleware config keys: what a .prompt file, the Dev UI and a trace may send.

App middleware may use any pydantic BaseModel; its keys are its field names.
A GenkitConfig takes either spelling and writes camelCase. Either way a key
the class doesn't declare raises instead of being dropped.
"""

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from genkit import Genkit, GenkitError
from genkit._ai._generate import register_middleware, resolve_middleware_from_use
from genkit._core._middleware import BaseMiddleware, GenerateMiddleware
from genkit._core._typing import MiddlewareRef
from genkit.plugin_api import GenkitConfig


class AllergyGuardConfig(BaseModel):
    """App middleware config: a plain BaseModel."""

    blocked_allergens: list[str] = []
    max_warnings: int = 3


class AllergyGuard(BaseMiddleware[AllergyGuardConfig]):
    """Flags dishes containing a blocked allergen."""


class LenientConfig(BaseModel):
    """A class that opts into extra keys."""

    model_config = ConfigDict(extra='allow')

    region: str | None = None


class Lenient(BaseMiddleware[LenientConfig]):
    """Takes keys it doesn't declare."""


class BudgetConfig(GenkitConfig):
    """Shipped-middleware style config: camelCase on the wire."""

    max_spend_cents: int = 500


class Budget(BaseMiddleware[BudgetConfig]):
    """Caps spend per call."""


def _definition(cls: type[BaseMiddleware], name: str) -> GenerateMiddleware:
    return GenerateMiddleware(cls=cls, name=name)


def test_plain_basemodel_config_keys_are_field_names() -> None:
    """A .prompt `config` with the field names builds the config as written."""
    guard = _definition(AllergyGuard, 'allergy_guard').instantiate({'blocked_allergens': ['peanut'], 'max_warnings': 1})

    assert isinstance(guard, AllergyGuard)
    assert guard.config == AllergyGuardConfig(blocked_allergens=['peanut'], max_warnings=1)


def test_unknown_ref_key_names_middleware_and_accepted_keys() -> None:
    """`maxWarnings` against a `max_warnings` field raises instead of running with the default."""
    with pytest.raises(GenkitError) as raised:
        _definition(AllergyGuard, 'allergy_guard').instantiate({'maxWarnings': 0})

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert raised.value.original_message == (
        'Middleware "allergy_guard" got unknown config key(s) maxWarnings; '
        'accepted keys: blocked_allergens, max_warnings'
    )


def test_unknown_kwarg_raises_extra_forbidden() -> None:
    """A misspelled keyword raises the same extra_forbidden error a GenkitConfig would."""
    with pytest.raises(ValidationError) as raised:
        AllergyGuard(max_warning=1)

    assert [(e['type'], e['loc']) for e in raised.value.errors()] == [('extra_forbidden', ('max_warning',))]


def test_extra_allow_config_keeps_undeclared_keys() -> None:
    """A class that sets extra='allow' asked for undeclared keys; they reach the config."""
    lenient = _definition(Lenient, 'lenient').instantiate({'region': 'eu', 'tier': 'gold'})

    assert lenient.config.model_dump() == {'region': 'eu', 'tier': 'gold'}


@pytest.mark.parametrize('config', [{'maxSpendCents': 100}, {'max_spend_cents': 100}], ids=['camel', 'snake'])
def test_genkit_config_ref_takes_either_spelling(config: dict[str, Any]) -> None:
    """A GenkitConfig middleware takes the JS/Go camelCase key and the Python field name."""
    budget = _definition(Budget, 'budget').instantiate(config)

    assert budget.config == BudgetConfig(max_spend_cents=100)


def test_genkit_config_schema_and_ref_use_one_spelling() -> None:
    """The Dev UI schema and the ref written for an inline instance both say maxSpendCents."""
    ai = Genkit()
    ai.define_middleware(Budget, name='budget')
    registry: Any = ai._registry

    [ref] = register_middleware(registry, [Budget(max_spend_cents=100)]) or []
    schema = _definition(Budget, 'budget').config_schema
    assert schema is not None

    assert set(schema['properties']) == {'maxSpendCents'}
    assert ref == MiddlewareRef(name='budget', config={'maxSpendCents': 100})
    [rebuilt] = resolve_middleware_from_use(registry, [ref])
    assert rebuilt.config == BudgetConfig(max_spend_cents=100)


def test_inline_middleware_with_serialization_alias_and_computed_field_round_trips() -> None:
    """Output-only serialization_alias and @computed_field do not break inline use=[Mw(...)]."""
    from pydantic import Field, computed_field

    class Cfg(BaseModel):
        max_warnings: int = Field(3, serialization_alias='maxWarnings')

        @computed_field  # type: ignore[prop-decorator]
        @property
        def doubled(self) -> int:
            return self.max_warnings * 2

    class Mw(BaseMiddleware[Cfg]):
        pass

    ai = Genkit()
    registry: Any = ai._registry
    [ref] = register_middleware(registry, [Mw(max_warnings=1)]) or []
    assert ref.config == {'max_warnings': 1}
    [rebuilt] = resolve_middleware_from_use(registry, [ref])
    assert rebuilt.config.max_warnings == 1


def test_alias_without_validate_by_name_rejects_field_name_in_ref() -> None:
    """When a BaseModel field has alias='maxWarnings' and no validate_by_name, 'max_warnings' is rejected."""
    from pydantic import Field

    class Cfg(BaseModel):
        max_warnings: int = Field(3, alias='maxWarnings')

    class Mw(BaseMiddleware[Cfg]):
        pass

    with pytest.raises(GenkitError) as raised:
        _definition(Mw, 'mw').instantiate({'max_warnings': 1})
    assert raised.value.status == 'INVALID_ARGUMENT'
    assert 'accepted keys: maxWarnings' in raised.value.original_message
    assert _definition(Mw, 'mw').instantiate({'maxWarnings': 1}).config.max_warnings == 1
