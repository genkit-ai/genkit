/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  action,
  Action,
  ActionContext,
  GenkitError,
  isAction,
  z,
} from '@genkit-ai/core';
import { Registry } from '@genkit-ai/core/registry';
import uriTemplate from 'uri-templates';
import { PartSchema } from './model-types.js';

/**
 * Options for defining a resource.
 */
export interface ResourceOptions {
  /**
   * Resource name. If not specified, uri or template will be used as name.
   */
  name?: string;

  /**
   * The URI of the resource. Can contain template variables.
   */
  uri?: string;

  /**
   * The URI template (ex. `my://resource/{id}`). See RFC6570 for specification.
   */
  template?: string;

  /**
   * A description of the resource.
   */
  description?: string;

  /**
   * Resource metadata.
   */
  metadata?: Record<string, any>;
}

export const ResourceInputSchema = z.object({
  uri: z.string(),
});

export type ResourceInput = z.infer<typeof ResourceInputSchema>;

export const ResourceOutputSchema = z.object({
  content: z.array(PartSchema),
});

export type ResourceOutput = z.infer<typeof ResourceOutputSchema>;

/**
 * A function that returns parts for a given resource.
 */
export type ResourceFn = (
  input: ResourceInput,
  ctx: ActionContext
) => ResourceOutput | Promise<ResourceOutput>;

/**
 * A resource action.
 */
export interface ResourceAction
  extends Action<typeof ResourceInputSchema, typeof ResourceOutputSchema> {
  matches(input: ResourceInput): boolean;
}

/**
 * A reference to a resource in the form of a name or a ResourceAction.
 */
export type ResourceArgument = ResourceAction | string;

export async function resolveResources(
  registry: Registry,
  resources?: ResourceArgument[]
): Promise<ResourceAction[]> {
  if (!resources || resources.length === 0) {
    return [];
  }

  return await Promise.all(
    resources.map(async (ref): Promise<ResourceAction> => {
      if (typeof ref === 'string') {
        return await lookupResourceByName(registry, ref);
      } else if (isAction(ref)) {
        return ref;
      }
      throw new Error('Resources must be strings, or actions');
    })
  );
}

export async function lookupResourceByName(
  registry: Registry,
  name: string
): Promise<ResourceAction> {
  const resource =
    (await registry.lookupAction(name)) ||
    (await registry.lookupAction(`/resource/${name}`)) ||
    (await registry.lookupAction(`/dynamic-action-provider/${name}`));
  if (!resource) {
    throw new Error(`Resource ${name} not found`);
  }
  return resource as ResourceAction;
}

/**
 * Defines a resource.
 *
 * @param registry The registry to register the resource with.
 * @param opts The resource options.
 * @param fn The resource function.
 * @returns The resource action.
 */
export function defineResource(
  registry: Registry,
  opts: ResourceOptions,
  fn: ResourceFn
): ResourceAction {
  const action = dynamicResource(opts, fn);
  action.matches = createMatcher(opts.uri, opts.template);
  action.__action.metadata.dynamic = false;
  registry.registerAction('resource', action);
  return action;
}

/**
 * A dynamic action with a `resource` type. Dynamic resources are detached actions -- not associated with any registry.
 */
export type DynamicResourceAction = ResourceAction & {
  __action: {
    metadata: {
      type: 'resource';
    };
  };
  /** @deprecated no-op, for backwards compatibility only. */
  attach(registry: Registry): ResourceAction;
  matches(input: ResourceInput): boolean;
};

/**
 * Finds a matching resource in the registry. If not found returns undefined.
 */
export async function findMatchingResource(
  registry: Registry,
  resources: ResourceAction[],
  input: ResourceInput
): Promise<ResourceAction | undefined> {
  // First look in any resources explicitly listed in the generate request
  for (const res of resources) {
    if (res.matches(input)) {
      return res;
    }
  }

  // Then search the registry
  for (const registryKey of Object.keys(
    await registry.listResolvableActions()
  )) {
    // We decided not to look in DAP actions because they might be slow.
    // DAP actions with resources will only be found if they are listed in the
    // resources section, and then they will be found above.
    if (registryKey.startsWith('/resource/')) {
      const resource = (await registry.lookupAction(
        registryKey
      )) as ResourceAction;

      if (resource.matches(input)) {
        return resource;
      }
    }
  }
  return undefined;
}

/** Checks whether provided object is a dynamic resource. */
export function isDynamicResourceAction(t: unknown): t is ResourceAction {
  return isAction(t) && t.__action?.metadata?.dynamic === true;
}

/**
 * Defines a dynamic resource. Dynamic resources are just like regular resources but will not be
 * registered in the Genkit registry and can be defined dynamically at runtime.
 */
export function resource(
  opts: ResourceOptions,
  fn: ResourceFn
): ResourceAction {
  return dynamicResource(opts, fn);
}

/**
 * Defines a dynamic resource. Dynamic resources are just like regular resources but will not be
 * registered in the Genkit registry and can be defined dynamically at runtime.
 *
 * @deprecated renamed to {@link resource}.
 */
export function dynamicResource(
  opts: ResourceOptions,
  fn: ResourceFn
): DynamicResourceAction {
  const uri = opts.uri ?? opts.template;
  if (!uri) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `must specify either url or template options`,
    });
  }
  const matcher = createMatcher(opts.uri, opts.template);

  const act = action(
    {
      actionType: 'resource',
      name: opts.name ?? uri,
      description: opts.description,
      inputSchema: ResourceInputSchema,
      outputSchema: ResourceOutputSchema,
      metadata: {
        resource: {
          uri: opts.uri,
          template: opts.template,
        },
        ...opts.metadata,
        type: 'resource',
        dynamic: true,
      },
    },
    async (input, ctx) => {
      const templateMatch = matcher(input);
      if (!templateMatch) {
        throw new GenkitError({
          status: 'INVALID_ARGUMENT',
          message: `input ${input} did not match template ${uri}`,
        });
      }
      const parts = await fn(input, ctx);
      parts.content.map((p) => {
        if (!p.metadata) {
          p.metadata = {};
        }
        if (p.metadata?.resource) {
          if (!(p.metadata as any).resource.parent) {
            (p.metadata as any).resource.parent = {
              uri: input.uri,
            };
            if (opts.template) {
              (p.metadata as any).resource.parent.template = opts.template;
            }
          }
        } else {
          (p.metadata as any).resource = {
            uri: input.uri,
          };
          if (opts.template) {
            (p.metadata as any).resource.template = opts.template;
          }
        }
        return p;
      });
      return parts;
    }
  ) as DynamicResourceAction;

  act.matches = matcher;
  act.attach = (_: Registry) => act;
  return act;
}

function createMatcher(
  uriOpt: string | undefined,
  templateOpt: string | undefined
): (input: ResourceInput) => boolean {
  // TODO: normalize resource URI during comparisons
  // foo://bar?baz=1&qux=2 and foo://bar?qux=2&baz=1 are equivalent URIs but would not match.
  if (uriOpt) {
    return (input: ResourceInput) => input.uri === uriOpt;
  }

  if (templateOpt) {
    const template = uriTemplate(templateOpt);
    return (input: ResourceInput) => template!.fromUri(input.uri) !== undefined;
  }

  throw new GenkitError({
    status: 'INVALID_ARGUMENT',
    message: 'must specify either url or template options',
  });
}
