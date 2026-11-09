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

"""What the plugin assumes about the installed ``anthropic`` SDK.

anthropic 1.0 dropped ``temperature``/``top_p``/``top_k`` from
``messages.create()`` and moved to httpx2. These tests fail if the declared
``anthropic`` range admits such a release before the plugin is adapted.
"""

import inspect
import re
from collections.abc import Callable
from importlib.metadata import requires

import httpx
import pytest
from anthropic import DefaultAsyncHttpxClient
from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
from anthropic.resources.messages import AsyncMessages
from genkit_anthropic import AnthropicConfig


@pytest.mark.parametrize('create', [AsyncMessages.create, AsyncBetaMessages.create], ids=['messages', 'beta'])
def test_sdk_create_accepts_every_forwarded_kwarg(create: Callable[..., object]) -> None:
    """Every ``CREATE_KWARGS`` key is a named parameter of ``create()``."""
    missing = sorted(AnthropicConfig.CREATE_KWARGS - inspect.signature(create).parameters.keys())
    assert not missing, f'{create.__module__}.create() no longer accepts {missing}'


def test_sdk_transport_is_httpx() -> None:
    """The ``except httpx.*`` handlers in ``_models.py`` only fire on an httpx transport."""
    assert issubclass(DefaultAsyncHttpxClient, httpx.AsyncClient)


def test_declared_anthropic_range_excludes_1x() -> None:
    """``genkit-anthropic`` caps ``anthropic`` below 1.0 until it supports 1.x."""
    reqs = [r for r in requires('genkit-anthropic') or [] if re.match(r'anthropic\b', r)]
    assert len(reqs) == 1, reqs
    assert '<1' in reqs[0].replace(' ', ''), reqs[0]
