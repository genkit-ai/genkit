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

"""Resource module for defining and managing resources."""

import inspect
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict, cast

from pydantic import BaseModel

from genkit._core._action import Action, ActionKind, ActionRunContext
from genkit._core._registry import Registry
from genkit._core._typing import Part


class ResourceOptions(TypedDict, total=False):
    """Options for defining a resource (name, uri/template, description, metadata)."""

    name: str
    uri: str
    template: str
    description: str
    metadata: dict[str, Any]


class ResourceInput(BaseModel):
    """Input for a resource request containing the URI to resolve."""

    uri: str


class ResourceOutput(BaseModel):
    """Output from a resource resolution containing content parts."""

    content: list[Part]


ResourcePayload = ResourceOutput | dict[str, Any]

ResourceFn = Callable[..., Awaitable[ResourcePayload]]


ResourceArgument = Action | str


async def resolve_resources(registry: Registry, resources: list[ResourceArgument] | None = None) -> list[Action]:
    """Resolve resource names/actions to Action objects."""
    if not resources:
        return []

    resolved_actions = []
    for ref in resources:
        if isinstance(ref, str):
            resolved_actions.append(await lookup_resource_by_name(registry, ref))
        elif isinstance(ref, Action):  # pyright: ignore[reportUnnecessaryIsInstance]
            resolved_actions.append(ref)
        else:
            raise ValueError('Resources must be strings or actions')
    return resolved_actions


async def lookup_resource_by_name(registry: Registry, name: str) -> Action:
    """Look up a resource action by name, trying common prefixes."""
    resource = (
        await registry.resolve_action(ActionKind.RESOURCE, name)
        or await registry.resolve_action(ActionKind.RESOURCE, f'/resource/{name}')
        or await registry.resolve_action(ActionKind.RESOURCE, f'/dynamic-action-provider/{name}')
    )
    if not resource:
        raise ValueError(f'Resource {name} not found')
    return resource


def define_resource(registry: Registry, opts: ResourceOptions, fn: ResourceFn) -> Action:
    """Register a resource action for a specific URI or template."""
    action = dynamic_resource(opts, fn)

    action.matches = create_matcher(opts.get('uri'), opts.get('template'))

    # Mark as not dynamic since it's being registered
    action.metadata['dynamic'] = False

    registry.register_action_from_instance(action)

    return action


def resource(opts: ResourceOptions, fn: ResourceFn) -> Action:
    """Create a dynamic resource action (alias for dynamic_resource)."""
    return dynamic_resource(opts, fn)


def dynamic_resource(opts: ResourceOptions, fn: ResourceFn) -> Action:
    """Create a resource Action that matches URIs and executes the given function."""
    if not inspect.iscoroutinefunction(fn):
        raise TypeError('fn must be an async function')

    uri = opts.get('uri') or opts.get('template')
    if not uri:
        raise ValueError('must specify either uri or template options')

    matcher = create_matcher(opts.get('uri'), opts.get('template'))

    async def wrapped_fn(input_data: ResourceInput, ctx: ActionRunContext) -> ResourcePayload:
        if isinstance(input_data, dict):
            input_data = ResourceInput(**input_data)

        try:
            template_match = matcher(input_data)
            if not template_match:
                raise ValueError(f'input {input_data} did not match template {uri}')

            sig = inspect.signature(fn)
            n_params = len(sig.parameters)

            if n_params == 0:
                parts = await fn()
            elif n_params == 1:
                parts = await fn(input_data)
            else:
                parts = await fn(input_data, ctx)

            # Post-processing parts to add metadata
            content_list = parts.content if isinstance(parts, ResourceOutput) else parts.get('content', [])

            for p in content_list:
                if isinstance(p, Part):
                    p = p.root

                if hasattr(p, 'metadata'):
                    if p.metadata is None:
                        # Different Part types have different metadata types (Metadata or dict)
                        # dict works for both types at runtime
                        # pyrefly:ignore[bad-assignment]
                        p.metadata = {}  # pyright: ignore[reportAttributeAccessIssue]
                    if isinstance(p.metadata, dict):
                        p_metadata = p.metadata
                    elif isinstance(p.metadata, dict):
                        p_metadata = p.metadata
                    else:
                        # dict works for both Part types at runtime
                        # pyrefly:ignore[bad-assignment]
                        p.metadata = {}  # pyright: ignore[reportAttributeAccessIssue]
                        p_metadata = p.metadata

                    template = opts.get('template')
                    # p_metadata is guaranteed to be dict here due to isinstance checks above,
                    # but type checkers can't narrow the union type. Use cast to inform them.
                    p_metadata = cast(dict[str, Any], p_metadata)

                    if 'resource' in p_metadata:
                        if 'parent' not in p_metadata['resource']:
                            p_metadata['resource']['parent'] = {'uri': input_data.uri}
                            if template:
                                p_metadata['resource']['parent']['template'] = template
                    else:
                        p_metadata['resource'] = {'uri': input_data.uri}
                        if template:
                            p_metadata['resource']['template'] = template
                elif isinstance(p, dict):
                    if 'metadata' not in p or p['metadata'] is None:
                        p['metadata'] = {}
                    p_metadata = p['metadata']
                else:
                    continue
            # Ensure we return a serializable dict (handling Pydantic models in list)
            if isinstance(parts, BaseModel):
                return parts.model_dump()
            elif isinstance(parts, dict):
                # Verify content items are dicts, if not dump them
                if 'content' in parts:
                    parts['content'] = [p.model_dump() if isinstance(p, BaseModel) else p for p in parts['content']]
                return parts
            return parts
        except Exception:
            raise

    name = opts.get('name') or uri

    act = Action(
        name=name,
        kind=ActionKind.RESOURCE,
        fn=wrapped_fn,
        metadata={
            'resource': {
                'uri': opts.get('uri'),
                'template': opts.get('template'),
            },
            'dynamic': True,
        },
        description=opts.get('description'),
        span_metadata={'genkit:metadata:resource:uri': uri},
    )
    act.matches = matcher
    return act


def create_matcher(uri: str | None, template: str | None) -> Callable[[object], bool]:
    """Create a matcher function for URI or template matching."""

    def matcher(input_data: object) -> bool:
        if not isinstance(input_data, ResourceInput):
            return False
        if uri:
            return input_data.uri == uri
        if template:
            return matches_uri_template(template, input_data.uri) is not None
        return False

    return matcher


def is_dynamic_resource_action(action: Action) -> bool:
    """Check if an action is a dynamic (unregistered) resource."""
    return action.kind == ActionKind.RESOURCE and bool(action.metadata.get('dynamic', True))


def matches_uri_template(template: str, uri: str) -> dict[str, str] | None:
    """Match URI against template, returning extracted params or None."""
    # Split template into parts: text and {param} placeholders
    parts = re.split(r'(\{[\w\+]+\})', template)
    pattern_parts = []
    for part in parts:
        if part.startswith('{') and part.endswith('}'):
            param_name = part[1:-1]
            if param_name.startswith('+'):
                # Reserved expansion: {+var} matches reserved chars like /
                param_name = param_name[1:]
                pattern_parts.append(f'(?P<{param_name}>.+)')
            else:
                # Basic expansion: {var} does not match /
                pattern_parts.append(f'(?P<{param_name}>[^/]+)')
        else:
            pattern_parts.append(re.escape(part))

    pattern = f'^{"".join(pattern_parts)}$'

    match = re.search(pattern, uri)
    if match:
        return match.groupdict()
    return None


async def find_matching_resource(
    registry: Registry, dynamic_resources: list[Action] | None, input_data: ResourceInput
) -> Action | None:
    """Find a matching resource action from dynamic resources or registry."""
    if dynamic_resources:
        for action in dynamic_resources:
            if hasattr(action, 'matches') and callable(action.matches) and action.matches(input_data):
                return action

    # Try exact match in registry
    resource = await registry.resolve_action(ActionKind.RESOURCE, input_data.uri)
    if resource:
        return resource

    # Iterate all resources to check for matches (e.g. templates)
    # This is less efficient but necessary for template matching if not optimized
    resources = await registry.resolve_actions_by_kind(ActionKind.RESOURCE)

    for action in resources.values():
        if hasattr(action, 'matches') and callable(action.matches) and action.matches(input_data):
            return action

    return None
