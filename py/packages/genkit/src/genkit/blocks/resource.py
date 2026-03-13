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

"""Resource module for defining and managing resources.

Resources in Genkit represent addressable content or data processing units containing
unstructured data (Post, PDF, etc.) that can be retrieved or generated. They are
identified by URIs (e.g. `file://`, `http://`, `gs://`) and can be static (fixed URI)
or dynamic (using URI templates).
This module provides tools to define resource actions that can resolve these URIs
and return content (`ResourceOutput`) containing `Part`s.
"""

import inspect
import re
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypedDict, cast

from pydantic import BaseModel

from genkit.aio import ensure_async
from genkit.core.action import Action, ActionRunContext
from genkit.core.action.types import ActionKind
from genkit.core.registry import Registry
from genkit.core.typing import Metadata, Part


class ResourceOptions(TypedDict, total=False):
    """Options for defining a resource.

    Attributes:
        name: Resource name. If not specified, uri or template will be used as name.
        uri: The URI of the resource. Can contain template variables for simple matches,
             but `template` is preferred for pattern matching.
        template: The URI template (ex. `my://resource/{id}`). See RFC6570 for specification.
                  Used for matching variable resources.
        description: A description of the resource, used for documentation and discovery.
        metadata: Arbitrary metadata to attach to the resource action.
    """

    name: str
    uri: str
    template: str
    description: str
    metadata: dict[str, Any]


class ResourceInput(BaseModel):
    """Input structure for a resource request.

    Attributes:
        uri: The full URI being requested/resolved.
    """

    uri: str


class ResourceOutput(BaseModel):
    """Output structure from a resource resolution.

    Attributes:
        content: A list of `Part` objects representing the resource content.
    """

    content: list[Part]


class ResourceFn(Protocol):
    """A function that returns parts for a given resource.

    The function receives the resolved input (including the URI) and context,
    and should return a `ResourceOutput` containing the content parts.
    """

    def __call__(self, input: ResourceInput, ctx: ActionRunContext) -> Awaitable[ResourceOutput]:
        """Call the resource function."""
        ...


ResourcePayload = ResourceOutput | dict[str, Any]

# We need a flexible type because the runtime supports various signatures (0-2 args, sync/async, dict return)
# but we also want to support the strict Protocol for those who want it.
# Note: Callable[..., T] is used for flexible args because accurate variable arg Union logic is complex/verbose.
FlexibleResourceFn = ResourceFn | Callable[..., Awaitable[ResourcePayload] | ResourcePayload]


class MatchableAction(Protocol):
    """Protocol for actions that have a matches method."""

    matches: Callable[[object], bool]


ResourceArgument = Action | str


async def resolve_resources(registry: Registry, resources: list[ResourceArgument] | None = None) -> list[Action]:
    """Resolves a list of resource names or actions into a list of Action objects.

    Args:
        registry: The registry to lookup resources in.
        resources: A list of resource references, which can be either direct `Action`
                   objects or strings (names/URIs).

    Returns:
        A list of resolved `Action` objects.

    Raises:
        ValueError: If a resource reference is invalid or cannot be found.
    """
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
    """Looks up a resource action by name in the registry.

    Tries to resolve the name directly, or with common prefixes like `/resource/`
    or `/dynamic-action-provider/`.

    Args:
        registry: The registry to search.
        name: The name or URI of the resource to lookup.

    Returns:
        The found `Action`.

    Raises:
        ValueError: If the resource cannot be found.
    """
    resource = (
        await registry.resolve_action(cast(ActionKind, ActionKind.RESOURCE), name)
        or await registry.resolve_action(cast(ActionKind, ActionKind.RESOURCE), f'/resource/{name}')
        or await registry.resolve_action(cast(ActionKind, ActionKind.RESOURCE), f'/dynamic-action-provider/{name}')
    )
    if not resource:
        raise ValueError(f'Resource {name} not found')
    return resource


def define_resource(registry: Registry, opts: ResourceOptions, fn: FlexibleResourceFn) -> Action:
    """Defines a resource and registers it with the given registry.

    This creates a resource action that can handle requests for a specific URI
    or URI template.

    Args:
        registry: The registry to register the resource with.
        opts: Options defining the resource (name, uri, template, etc.).
        fn: The function that implements resource content retrieval.

    Returns:
        The registered `Action` for the resource.
    """
    action = dynamic_resource(opts, fn)

    cast(MatchableAction, cast(object, action)).matches = create_matcher(opts.get('uri'), opts.get('template'))

    # Mark as not dynamic since it's being registered
    action.metadata['dynamic'] = False

    registry.register_action_from_instance(action)

    return action


def resource(opts: ResourceOptions, fn: FlexibleResourceFn) -> Action:
    """Defines a dynamic resource action without immediate registration.

    This is an alias for `dynamic_resource`. Useful for defining resources that
    might be registered later or used as standalone actions.

    Args:
        opts: Options defining the resource.
        fn: The resource implementation function.

    Returns:
        The created `Action`.
    """
    return dynamic_resource(opts, fn)


def dynamic_resource(opts: ResourceOptions, fn: FlexibleResourceFn) -> Action:
    """Defines a dynamic resource action.

    Creates an `Action` of kind `RESOURCE` that wraps the provided function.
    The wrapper handles:
    1. Input validation and matching against the URI/Template.
    2. Execution of the resource function.
    3. Post-processing of output to attach metadata (like parent resource info).

    Args:
        opts: Options including `uri` or `template` for matching.
        fn: The function performing the resource retrieval.

    Returns:
        An `Action` configured as a resource.

    Raises:
        ValueError: If neither `uri` nor `template` is provided in options.
    """
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
            afn = ensure_async(fn)
            n_params = len(sig.parameters)

            if n_params == 0:
                parts = await afn()
            elif n_params == 1:
                parts = await afn(input_data)
            else:
                parts = await afn(input_data, ctx)

            # Post-processing parts to add metadata
            content_list = parts.content if hasattr(parts, 'content') else parts.get('content', [])

            for p in content_list:
                if isinstance(p, Part):
                    p = p.root

                if hasattr(p, 'metadata'):
                    if p.metadata is None:
                        # Different Part types have different metadata types (Metadata or dict)
                        # dict works for both types at runtime
                        p.metadata = {}  # pyright: ignore[reportAttributeAccessIssue]
                    if isinstance(p.metadata, Metadata):
                        p_metadata = p.metadata.root
                    elif isinstance(p.metadata, dict):
                        p_metadata = p.metadata
                    else:
                        # dict works for both Part types at runtime
                        p.metadata = {}  # pyright: ignore[reportAttributeAccessIssue]
                        p_metadata = p.metadata

                    template = opts.get('template')
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
        kind=cast(ActionKind, ActionKind.RESOURCE),
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
    """Creates a matching function for resource validation.

    Args:
        uri: Optional fixed URI string.
        template: Optional URI template string.

    Returns:
        A callable that takes an object (expected to be ResourceInput) and returns True if it matches.
    """

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
    """Checks if an action is a dynamic resource (not registered).

    Args:
        action: The action to check.

    Returns:
        True if the action is a dynamic resource, False otherwise.
    """
    return action.kind == ActionKind.RESOURCE and bool(action.metadata.get('dynamic', True))


def matches_uri_template(template: str, uri: str) -> dict[str, str] | None:
    """Check if a URI matches a template and extract parameters.

    Args:
        template: URI template with {param} placeholders (e.g., "file://{path}").
        uri: The URI to match against the template.

    Returns:
        Dictionary of extracted parameters if match, None otherwise.

    Examples:
        >>> matches_uri_template('file://{path}', 'file:///home/user/doc.txt')
        {'path': '/home/user/doc.txt'}
        >>> matches_uri_template('user://{id}/profile', 'user://123/profile')
        {'id': '123'}
    """
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
    """Finds a matching resource action.

    Checks dynamic resources first, then the registry.

    Args:
        registry: The registry to search.
        dynamic_resources: Optional list of dynamic resource actions to check first.
        input_data: The resource input containing the URI matched against.

    Returns:
        The matching Action or None.
    """
    if dynamic_resources:
        for action in dynamic_resources:
            if (
                hasattr(action, 'matches')
                and callable(action.matches)
                and cast(MatchableAction, cast(object, action)).matches(input_data)
            ):
                return action

    # Try exact match in registry
    resource = await registry.resolve_action(cast(ActionKind, ActionKind.RESOURCE), input_data.uri)
    if resource:
        return resource

    # Iterate all resources to check for matches (e.g. templates)
    # This is less efficient but necessary for template matching if not optimized
    resources = await registry.resolve_actions_by_kind(cast(ActionKind, ActionKind.RESOURCE))

    for action in resources.values():
        if (
            hasattr(action, 'matches')
            and callable(action.matches)
            and cast(MatchableAction, cast(object, action)).matches(input_data)
        ):
            return action

    return None
