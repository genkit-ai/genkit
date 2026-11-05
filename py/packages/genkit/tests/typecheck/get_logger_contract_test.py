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

# pyright: reportUnnecessaryTypeIgnoreComment=true

"""Typed contract of ``genkit.get_logger``.

pyright, pyrefly and ty must accept the documented call style with no
suppressions. ``bind`` and structlog's async ``a*`` twins are outside the
contract, so calling them must be a type error in all three. Each checker
fails on an unneeded ignore here (pyright through the pragma above, pyrefly
and ty through py/pyproject.toml), so adding either to the protocol breaks CI.
"""

import logging
from collections.abc import Iterator

import pytest
import structlog
from structlog.testing import capture_logs
from typing_extensions import assert_type

from genkit import GenkitLogger, get_logger


@pytest.fixture(autouse=True)
def _debug_level() -> Iterator[None]:
    """Let DEBUG through regardless of the level an earlier test left behind."""
    saved = structlog.get_config().copy()
    was_configured = structlog.is_configured()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG))
    yield
    structlog.reset_defaults()
    if was_configured:
        structlog.configure(**saved)


def test_documented_call_style_type_checks() -> None:
    """Event string, printf args and keyword fields type-check on every level."""
    logger = get_logger(__name__)
    assert_type(logger, GenkitLogger)

    with capture_logs() as entries:
        logger.debug('menu cache miss', restaurant_id='r-42')
        logger.info('order placed', table=4, dish='salmon')
        logger.warning('allergen flagged: %s', 'peanut', table=4)
        logger.error('payment declined', order_id='o-17')
        try:
            raise ValueError('kitchen on fire')
        except ValueError:
            logger.exception('order failed', order_id='o-17')

    assert [e['event'] for e in entries] == [
        'menu cache miss',
        'order placed',
        'allergen flagged: peanut',
        'payment declined',
        'order failed',
    ]


def test_bind_is_outside_the_contract() -> None:
    """``bind`` still runs (the backend is structlog), but it does not type-check."""
    logger = get_logger(__name__)

    bound = logger.bind(table=4)  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-attribute]  # ty: ignore[unresolved-attribute]

    with capture_logs() as entries:
        bound.info('table seated')
    assert entries[0]['table'] == 4


@pytest.mark.asyncio
async def test_async_twins_are_outside_the_contract() -> None:
    """``ainfo`` still runs (the backend is structlog), but stdlib has no async twins so it does not type-check."""
    logger = get_logger(__name__)

    with capture_logs() as entries:
        await logger.ainfo('order placed', table=4)  # pyright: ignore[reportAttributeAccessIssue]  # pyrefly: ignore[missing-attribute]  # ty: ignore[unresolved-attribute]

    assert entries[0]['event'] == 'order placed'
    assert entries[0]['table'] == 4
