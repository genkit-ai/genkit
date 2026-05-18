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

"""Tests for the Genkit Resource API via the Genkit class (Veneer).

This test file verifies that `ai.define_resource` works correctly, mirroring the
JS SDK's `ai.defineResource`.
"""

from typing import Any, cast

import pytest

from genkit import ActionRunContext, Genkit
from genkit._ai._resource import ResourceInput
from genkit._core._action import ActionKind
from genkit._core._typing import Part, TextPart


@pytest.mark.asyncio
async def test_define_resource_veneer() -> None:
    """Verifies ai.define_resource registers a resource correctly."""
    ai = Genkit(plugins=[])

    async def my_resource_fn(input: ResourceInput, ctx: ActionRunContext) -> dict[str, list[Part]]:
        return {'content': [Part(root=TextPart(text=f'Content for {input.uri}'))]}

    act = ai.define_resource(fn=my_resource_fn, uri='http://example.com/foo')

    assert act.name == 'http://example.com/foo'
    assert act.metadata is not None
    metadata = cast(dict[str, Any], act.metadata)
    resource_meta = cast(dict[str, Any], metadata['resource'])
    assert resource_meta['uri'] == 'http://example.com/foo'

    # Verify lookup via global registry (contained in ai.registry)
    looked_up = await ai.registry.resolve_action(ActionKind.RESOURCE, 'http://example.com/foo')
    assert looked_up == act

    # Verify execution
    output = await act.run({'uri': 'http://example.com/foo'})
    assert 'Content for http://example.com/foo' in output.response['content'][0]['text']
