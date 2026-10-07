#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""ModelRequest[Cfg] with a typing.TypedDict validates the same on every Python.

Pydantic rejects typing.TypedDict before 3.12. ModelRequest swaps in a
typing_extensions copy, so these assertions hold on 3.10 through 3.14.
"""

import sys
from typing import TypedDict

import pytest
from pydantic import TypeAdapter, ValidationError

from genkit._core._compat import pydantic_safe_typeddicts
from genkit._core._model import ModelRequest, declared_config_type


class Safety(TypedDict):
    """A nested setting, also a typing.TypedDict."""

    level: int


class _Required(TypedDict):
    model: str


class CallConfig(_Required, total=False):
    """``model`` required (from the base), the rest optional."""

    temperature: float
    safety: list[Safety] | None


def test_parametrized_request_is_cached() -> None:
    """Writing ModelRequest[CallConfig] twice gives one class, as Pydantic's cache does for any type."""
    assert ModelRequest[CallConfig] is ModelRequest[CallConfig]


def test_valid_config_passes_as_dict() -> None:
    """The model gets the dict it was sent."""
    request = ModelRequest[CallConfig].model_validate({
        'messages': [],
        'config': {'model': 'm', 'temperature': 0.2, 'safety': [{'level': 1}]},
    })

    assert request.config == {'model': 'm', 'temperature': 0.2, 'safety': [{'level': 1}]}


def test_required_key_from_base_is_enforced() -> None:
    """``model`` stays required after the swap; ``temperature`` stays optional."""
    with pytest.raises(ValidationError, match='model'):
        ModelRequest[CallConfig].model_validate({'messages': [], 'config': {'temperature': 0.2}})


@pytest.mark.parametrize(
    'config',
    [
        {'model': 'm', 'temperature': 'hot'},
        {'model': 'm', 'safety': [{'level': 'high'}]},
    ],
    ids=['top-level', 'nested'],
)
def test_bad_values_are_rejected(config: dict[str, object]) -> None:
    """Field types are checked, including inside the nested TypedDict."""
    with pytest.raises(ValidationError):
        ModelRequest[CallConfig].model_validate({'messages': [], 'config': config})


def test_json_schema_keeps_the_user_class_name() -> None:
    """The Dev UI shows the config as CallConfig, not a generated name."""
    schema = TypeAdapter(pydantic_safe_typeddicts(CallConfig)).json_schema()

    assert schema['title'] == 'CallConfig'
    assert schema['required'] == ['model']


def test_declared_config_type_is_a_typeddict() -> None:
    """generate sees a TypedDict, not a pydantic class, and passes the dict through."""
    declared = declared_config_type(ModelRequest[CallConfig])

    assert declared is not None
    assert declared.__name__ == 'CallConfig'
    assert declared.__module__ == CallConfig.__module__


def test_nothing_to_swap_returns_the_same_object() -> None:
    """Annotations without a typing.TypedDict come back untouched."""
    hint = list[int] | None

    assert pydantic_safe_typeddicts(hint) is hint


@pytest.mark.skipif(sys.version_info < (3, 12), reason='Swap only runs before 3.12')
def test_no_swap_on_312_and_later() -> None:
    """Pydantic takes typing.TypedDict from 3.12, so the class is used as written."""
    assert pydantic_safe_typeddicts(CallConfig) is CallConfig
