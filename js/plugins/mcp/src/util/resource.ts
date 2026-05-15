/**
 * Copyright 2024 Google LLC
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

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import {
  ReadResourceResult,
  Resource,
  ResourceTemplate,
} from '@modelcontextprotocol/sdk/types.js';
import {
  GenkitError,
  dynamicResource,
  type DynamicResourceAction,
  type Genkit,
  type Part,
} from 'genkit';
import { logger } from 'genkit/logging';

function createDynamicResource(
  client: Client,
  resource: Resource,
  params: { serverName: string; name: string }
): DynamicResourceAction {
  return dynamicResource(
    {
      name: `${params.serverName}/${resource.name}`,
      description: resource.description || undefined,
      metadata: { mcp: { _meta: resource._meta || {} } },
      uri: resource.uri,
    },
    async (args, { context }) => {
      logger.debug(
        `[MCP] calling resource '${params.serverName}/${resource.name}' in host '${params.name}'`
      );
      const result = await client.readResource({
        uri: args.uri,
        _meta: context?.mcp?._meta,
      });
      return {
        content: result.contents.map((p) => fromMcpResourcePart(p)),
      };
    }
  );
}

function createDynamicResourceTemplate(
  client: Client,
  template: ResourceTemplate,
  params: { serverName: string; name: string }
): DynamicResourceAction {
  return dynamicResource(
    {
      name: `${params.serverName}/${template.name}`,
      description: template.description || undefined,
      metadata: { mcp: { _meta: template._meta || {} } },
      template: template.uriTemplate,
    },
    async (args, { context }) => {
      logger.debug(
        `[MCP] calling resource template '${params.serverName}/${template.name}' in host '${params.name}'`
      );
      const result = await client.readResource({
        uri: args.uri,
        _meta: context?.mcp?._meta,
      });
      return {
        content: result.contents.map((p) => fromMcpResourcePart(p)),
        metadata: result._meta,
      };
    }
  );
}

type ArrayElement<ArrayType extends readonly unknown[]> =
  ArrayType extends readonly (infer ElementType)[] ? ElementType : never;

function fromMcpResourcePart(
  part: ArrayElement<ReadResourceResult['contents']>
): Part {
  if ('text' in part && part.text) {
    return { text: part.text as string, metadata: part._meta };
  }
  if ('blob' in part && part.blob) {
    return {
      media: {
        contentType: part.mimeType,
        url: `data:${part.mimeType};base64,${part.blob}`,
      },
      metadata: part._meta,
    };
  }
  throw new GenkitError({
    status: 'UNIMPLEMENTED',
    message: `Part type ${'type' in part ? part.type : 'unknown'} is not currently supported.`,
  });
}

/**
 * Lookup all resources available in the server and fetches as a Genkit dynamic resource.
 */
export async function fetchDynamicResources(
  ai: Genkit,
  client: Client,
  params: { name: string; serverName: string }
): Promise<DynamicResourceAction[]> {
  let cursor: string | undefined;
  let allResources: DynamicResourceAction[] = [];
  while (true) {
    const { nextCursor, resources } = await client.listResources({ cursor });
    allResources.push(
      ...resources.map((r) => createDynamicResource(client, r, params))
    );
    cursor = nextCursor;
    if (!cursor) break;
  }
  while (true) {
    const { nextCursor, resourceTemplates } =
      await client.listResourceTemplates({ cursor });
    allResources.push(
      ...resourceTemplates.map((r) =>
        createDynamicResourceTemplate(client, r, params)
      )
    );
    cursor = nextCursor;
    if (!cursor) break;
  }
  return allResources;
}
