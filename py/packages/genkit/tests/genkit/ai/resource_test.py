# Copyright 2025 Google LLC
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

"""Tests for the Genkit Resource API.

This module verifies the functionality of defining, registering, and resolving resources
in the Genkit framework. It covers static resources, template-based resources,
dynamic resource matching, and metadata handling.
"""

from typing import Any, cast

import pytest

from genkit.blocks.resource import (
    ResourceInput,
    define_resource,
    find_matching_resource,
    is_dynamic_resource_action,
    resolve_resources,
    resource,
)
from genkit.core.action import ActionRunContext
from genkit.core.action.types import ActionKind
from genkit.core.registry import Registry
from genkit.core.typing import Metadata, Part, TextPart


@pytest.mark.asyncio
async def test_define_resource() -> None:
    """Verifies that a resource can be defined and registered correctly.

    Checks:
    - Resource name matches property.
    - Resource is retrievable from the registry by name.
    """
    registry = Registry()

    async def my_resource_fn(input: ResourceInput, ctx: ActionRunContext) -> dict[str, object]:
        return {'content': [Part(TextPart(text=f'Content for {input.uri}'))]}

    act = define_resource(registry, {'uri': 'http://example.com/foo'}, my_resource_fn)

    assert act.name == 'http://example.com/foo'
    assert act.metadata is not None
    metadata = cast(dict[str, Any], act.metadata)
    resource_meta = cast(dict[str, Any], metadata['resource'])
    assert resource_meta['uri'] == 'http://example.com/foo'

    # Verify that the action can be resolved from the registry
    # Registry lookup for resources usually prepends /resource/ etc.
    # but define_resource registers it with name=uri

    looked_up = await registry.resolve_action(ActionKind.RESOURCE, 'http://example.com/foo')
    assert looked_up == act


@pytest.mark.asyncio
async def test_resolve_resources() -> None:
    """Verifies resolving resource references into Action objects.

    Checks:
    - Resolving by string name works.
    - Resolving by Action object passes through.
    """
    registry = Registry()

    async def my_resource_fn(input: ResourceInput, ctx: ActionRunContext) -> dict[str, object]:
        return {'content': [Part(TextPart(text=f'Content for {input.uri}'))]}

    act = define_resource(registry, {'name': 'my-resource', 'uri': 'http://example.com/foo'}, my_resource_fn)

    resolved = await resolve_resources(registry, ['my-resource'])
    assert len(resolved) == 1
    assert resolved[0] == act

    resolved_obj = await resolve_resources(registry, [act])
    assert len(resolved_obj) == 1
    assert resolved_obj[0] == act


@pytest.mark.asyncio
async def test_find_matching_resource() -> None:
    """Verifies the logic for finding a matching resource given an input URI.

    Checks:
    - Exact match against registered static resources.
    - Template match against registered template resources.
    - Matching against a provided list of dynamic resource actions for override/adhoc usage.
    - Returns None when no match is found.
    """
    registry = Registry()

    # Static resource
    async def static_fn(input: ResourceInput, ctx: ActionRunContext) -> dict[str, object]:
        return {'content': []}

    static_res = define_resource(registry, {'uri': 'bar://baz', 'name': 'staticRes'}, static_fn)

    # Template resource
    async def template_fn(input: ResourceInput, ctx: ActionRunContext) -> dict[str, object]:
        return {'content': []}

    template_res = define_resource(registry, {'template': 'foo://bar/{baz}', 'name': 'templateRes'}, template_fn)

    # Dynamic resource list
    async def dynamic_fn(input: ResourceInput, ctx: ActionRunContext) -> dict[str, object]:
        return {'content': []}

    dynamic_res = resource({'uri': 'baz://qux'}, dynamic_fn)

    # Match static from registry
    res = await find_matching_resource(registry, [], ResourceInput(uri='bar://baz'))
    assert res == static_res

    # Match template from registry
    res = await find_matching_resource(registry, [], ResourceInput(uri='foo://bar/something'))
    assert res == template_res

    # Match dynamic from list
    res = await find_matching_resource(registry, [dynamic_res], ResourceInput(uri='baz://qux'))
    assert res == dynamic_res

    # No match
    res = await find_matching_resource(registry, [], ResourceInput(uri='unknown://uri'))
    assert res is None


def test_is_dynamic_resource_action() -> None:
    """Verifies identifying dynamic vs registered resource actions.

    Checks:
    - Unregistered resources created with `resource()` are dynamic.
    - Registered resources created with `define_resource()` are not dynamic.
    """

    async def fn(input: ResourceInput, ctx: ActionRunContext) -> dict[str, object]:
        return {'content': []}

    dynamic = resource({'uri': 'bar://baz'}, fn)
    assert is_dynamic_resource_action(dynamic)

    # Registered action (define_resource sets dynamic=False)
    async def static_fn(input: ResourceInput, ctx: ActionRunContext) -> dict[str, object]:
        return {'content': []}

    static = define_resource(Registry(), {'uri': 'foo://bar'}, static_fn)
    assert not is_dynamic_resource_action(static)


@pytest.mark.asyncio
async def test_parent_metadata() -> None:
    """Verifies that parent metadata is correctly attached to output items.

    When a resource is resolved via a template (e.g. `file://{id}`), the output parts
    should contain metadata referencing the parent resource URI and template.
    Checks:
    - Parent URI and template presence in output part metadata.
    """
    registry = Registry()

    async def fn(input: ResourceInput, ctx: ActionRunContext) -> dict[str, object]:
        return {
            'content': [
                Part(TextPart(text='sub1', metadata=Metadata(root={'resource': {'uri': f'{input.uri}/sub1.txt'}})))
            ]
        }

    res = define_resource(registry, {'template': 'file://{id}'}, fn)

    output = await res.arun({'uri': 'file://dir'})
    # output is ActionResponse
    # content is in output.response['content'] because wrapped_fn ensures serialization

    part = output.response['content'][0]
    # Check metadata
    assert part['metadata']['resource']['parent']['uri'] == 'file://dir'
    assert part['metadata']['resource']['parent']['template'] == 'file://{id}'
    assert part['metadata']['resource']['uri'] == 'file://dir/sub1.txt'


def test_dynamic_resource_matching() -> None:
    """Verifies the matching logic for a simple static URI dynamic resource."""

    async def my_resource_fn(input: ResourceInput, ctx: ActionRunContext) -> dict[str, object]:
        return {'content': [Part(TextPart(text='Match'))]}

    res = resource({'uri': 'http://example.com/foo'}, my_resource_fn)
    assert res.matches is not None

    assert res.matches(ResourceInput(uri='http://example.com/foo'))
    assert not res.matches(ResourceInput(uri='http://example.com/bar'))


def test_template_matching() -> None:
    """Verifies URI template pattern matching.

    Checks:
    - Matches correct URI structure.
    - Fails on paths extending beyond the template structure (strict matching).
    """

    async def my_resource_fn(input: ResourceInput, ctx: ActionRunContext) -> dict[str, object]:
        return {'content': []}

    res = resource({'template': 'http://example.com/items/{id}'}, my_resource_fn)
    assert res.matches is not None

    assert res.matches(ResourceInput(uri='http://example.com/items/123'))

    # Should not match because of strict end anchor or slash handling in our regex
    assert not res.matches(ResourceInput(uri='http://example.com/items/123/details'))


def test_reserved_expansion_matching() -> None:
    """Verifies RFC 6570 reserved expansion {+var} pattern matching.

    Checks:
    - Matches correct URI structure with slashes (reserved chars).
    """

    async def my_resource_fn(input: ResourceInput, ctx: ActionRunContext) -> dict[str, object]:
        return {'content': []}

    # Template with reserved expansion {+path} (matches slashes)
    res = resource({'template': 'http://example.com/files/{+path}'}, my_resource_fn)
    assert res.matches is not None

    assert res.matches(ResourceInput(uri='http://example.com/files/foo/bar/baz.txt'))

    # Regular template {path} regex ([^/]+) should NOT match slashes
    res_simple = resource({'template': 'http://example.com/items/{id}'}, my_resource_fn)
    assert res_simple.matches is not None

    assert not res_simple.matches(ResourceInput(uri='http://example.com/items/foo/bar'))
