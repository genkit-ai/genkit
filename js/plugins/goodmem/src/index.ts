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

import * as fs from 'fs';
import { z, type Genkit } from 'genkit';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import * as path from 'path';

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

export interface GoodMemPluginParams {
  /** Base URL of the GoodMem API server (e.g., "https://api.goodmem.ai" or "http://localhost:8080"). */
  baseUrl: string;
  /** GoodMem API key for authentication (X-API-Key header). */
  apiKey: string;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function normalizeBaseUrl(url: string): string {
  return url.replace(/\/$/, '');
}

function apiHeaders(apiKey: string): Record<string, string> {
  return {
    'X-API-Key': apiKey,
    'Content-Type': 'application/json',
    Accept: 'application/json',
  };
}

async function apiFetch(
  url: string,
  apiKey: string,
  init: RequestInit = {}
): Promise<Response> {
  const headers = {
    ...apiHeaders(apiKey),
    ...(init.headers as Record<string, string> | undefined),
  };
  return fetch(url, { ...init, headers });
}

async function apiJson<T = any>(
  url: string,
  apiKey: string,
  init: RequestInit = {}
): Promise<T> {
  const res = await apiFetch(url, apiKey, init);
  if (!res.ok) {
    let detail: string;
    try {
      const body = await res.json();
      detail = body.message || body.error || JSON.stringify(body);
    } catch {
      detail = await res.text().catch(() => res.statusText);
    }
    throw new Error(`GoodMem API error (${res.status}): ${detail}`);
  }
  return (await res.json()) as T;
}

function getMimeType(ext: string): string | null {
  const map: Record<string, string> = {
    pdf: 'application/pdf',
    png: 'image/png',
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    gif: 'image/gif',
    webp: 'image/webp',
    txt: 'text/plain',
    html: 'text/html',
    md: 'text/markdown',
    csv: 'text/csv',
    json: 'application/json',
    xml: 'application/xml',
    doc: 'application/msword',
    docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    xls: 'application/vnd.ms-excel',
    xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    ppt: 'application/vnd.ms-powerpoint',
    pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
  };
  return map[ext.toLowerCase().replace('.', '')] || null;
}

// ---------------------------------------------------------------------------
// Exported helper functions (also wrapped as tools below)
// ---------------------------------------------------------------------------

/**
 * List all spaces in the GoodMem instance.
 * Available for programmatic use (also exposed as the `goodmem/list_spaces` tool).
 */
export async function listSpaces(params: GoodMemPluginParams) {
  const baseUrl = normalizeBaseUrl(params.baseUrl);
  const body = await apiJson(`${baseUrl}/v1/spaces`, params.apiKey);
  const spaces = Array.isArray(body) ? body : body?.spaces || [];
  return spaces;
}

/**
 * List all embedders available in the GoodMem instance.
 * Available for programmatic use (also exposed as the `goodmem/list_embedders` tool).
 */
export async function listEmbedders(params: GoodMemPluginParams) {
  const baseUrl = normalizeBaseUrl(params.baseUrl);
  const body = await apiJson(`${baseUrl}/v1/embedders`, params.apiKey);
  const embedders = Array.isArray(body) ? body : body?.embedders || [];
  return embedders;
}

// ---------------------------------------------------------------------------
// Zod schemas: spaces
// ---------------------------------------------------------------------------

const ListEmbeddersInputSchema = z.object({}).optional();
const ListEmbeddersOutputSchema = z.object({
  success: z.boolean(),
  embedders: z.array(z.any()).optional(),
  totalResults: z.number().optional(),
  error: z.string().optional(),
  details: z.any().optional(),
});

const ListSpacesInputSchema = z.object({}).optional();
const ListSpacesOutputSchema = z.object({
  success: z.boolean(),
  spaces: z.array(z.any()).optional(),
  totalResults: z.number().optional(),
  error: z.string().optional(),
  details: z.any().optional(),
});

const GetSpaceInputSchema = z.object({
  spaceId: z.string().describe('The UUID of the space to fetch.'),
});

const GetSpaceOutputSchema = z.object({
  success: z.boolean(),
  space: z.any().optional(),
  error: z.string().optional(),
  details: z.any().optional(),
});

const CreateSpaceInputSchema = z.object({
  name: z
    .string()
    .describe(
      'A unique name for the space. If a space with this name already exists, its ID will be returned instead of creating a duplicate.'
    ),
  embedderId: z
    .string()
    .describe(
      'The ID of the embedder model that converts text into vector representations for similarity search.'
    ),
  chunkSize: z
    .number()
    .optional()
    .default(256)
    .describe('Number of characters per chunk when splitting documents.'),
  chunkOverlap: z
    .number()
    .optional()
    .default(25)
    .describe('Number of overlapping characters between consecutive chunks.'),
  keepStrategy: z
    .enum(['KEEP_END', 'KEEP_START', 'DISCARD'])
    .optional()
    .default('KEEP_END')
    .describe('Where to attach the separator when splitting.'),
  lengthMeasurement: z
    .enum(['CHARACTER_COUNT', 'TOKEN_COUNT'])
    .optional()
    .default('CHARACTER_COUNT')
    .describe('How chunk size is measured.'),
  labels: z
    .record(z.string())
    .optional()
    .describe(
      'Optional key-value labels attached to the space at creation time.'
    ),
});

const CreateSpaceOutputSchema = z.object({
  success: z.boolean(),
  spaceId: z.string().optional(),
  name: z.string().optional(),
  embedderId: z.string().optional(),
  message: z.string().optional(),
  reused: z.boolean().optional(),
  error: z.string().optional(),
  details: z.any().optional(),
});

const UpdateSpaceInputSchema = z.object({
  spaceId: z.string().describe('The UUID of the space to update.'),
  name: z.string().optional().describe('New name for the space.'),
  publicRead: z
    .boolean()
    .optional()
    .describe('Whether the space is publicly readable.'),
  replaceLabels: z
    .record(z.string())
    .optional()
    .describe(
      'Replace ALL existing labels with this map. Mutually exclusive with mergeLabels.'
    ),
  mergeLabels: z
    .record(z.string())
    .optional()
    .describe(
      'Merge these labels into existing labels (adds/overwrites individual keys).'
    ),
  defaultChunkingConfig: z
    .any()
    .optional()
    .describe(
      'Optional new chunking config. Pass the full GoodMem chunking config object (e.g., {recursive: {...}}).'
    ),
});

const UpdateSpaceOutputSchema = z.object({
  success: z.boolean(),
  space: z.any().optional(),
  error: z.string().optional(),
  details: z.any().optional(),
});

const DeleteSpaceInputSchema = z.object({
  spaceId: z.string().describe('The UUID of the space to delete.'),
});

const DeleteSpaceOutputSchema = z.object({
  success: z.boolean(),
  spaceId: z.string().optional(),
  message: z.string().optional(),
  error: z.string().optional(),
  details: z.any().optional(),
});

// ---------------------------------------------------------------------------
// Zod schemas: memories
// ---------------------------------------------------------------------------

const CreateMemoryInputSchema = z.object({
  spaceId: z.string().describe('The ID of the space to store the memory in.'),
  filePath: z
    .string()
    .optional()
    .describe(
      'Absolute path to a file to store as memory (PDF, DOCX, image, etc.). Content type is auto-detected from the file extension.'
    ),
  textContent: z
    .string()
    .optional()
    .describe(
      'Plain text content to store as memory. If both filePath and textContent are provided, the file takes priority.'
    ),
  source: z
    .string()
    .optional()
    .describe(
      'Where this memory came from (e.g., "google-drive", "gmail"). Stored in metadata.source.'
    ),
  author: z
    .string()
    .optional()
    .describe(
      'The author or creator of the content. Stored in metadata.author.'
    ),
  tags: z
    .string()
    .optional()
    .describe(
      'Comma-separated tags for categorization (e.g., "legal,research,important"). Stored in metadata.tags as an array.'
    ),
  metadata: z
    .record(z.any())
    .optional()
    .describe(
      'Extra key-value metadata as JSON. Merged with source, author, and tags fields.'
    ),
});

const CreateMemoryOutputSchema = z.object({
  success: z.boolean(),
  memoryId: z.string().optional(),
  spaceId: z.string().optional(),
  status: z.string().optional(),
  contentType: z.string().optional(),
  fileName: z.string().optional().nullable(),
  message: z.string().optional(),
  error: z.string().optional(),
  details: z.any().optional(),
});

const ListMemoriesInputSchema = z.object({
  spaceId: z
    .string()
    .describe(
      'The UUID of the space whose memories you want to list. (GoodMem requires a space scope to list memories.)'
    ),
  statusFilter: z
    .enum(['PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'])
    .optional()
    .describe('Restrict results to memories in this processing status.'),
  includeContent: z
    .boolean()
    .optional()
    .default(false)
    .describe(
      'When true, each returned memory includes its original document content alongside the metadata.'
    ),
  sortBy: z
    .enum(['created_at', 'updated_at'])
    .optional()
    .describe('Field used to sort the returned memories.'),
  sortOrder: z
    .enum(['ASCENDING', 'DESCENDING'])
    .optional()
    .describe('Sort direction applied to sortBy.'),
});

const ListMemoriesOutputSchema = z.object({
  success: z.boolean(),
  memories: z.array(z.any()).optional(),
  totalResults: z.number().optional(),
  spaceId: z.string().optional(),
  error: z.string().optional(),
  details: z.any().optional(),
});

const RetrieveMemoriesInputSchema = z.object({
  query: z
    .string()
    .describe(
      'A natural language query used to find semantically similar memory chunks.'
    ),
  spaceIds: z
    .array(z.string())
    .describe('One or more space IDs to search across.'),
  maxResults: z
    .number()
    .optional()
    .default(5)
    .describe('Limit the number of returned memories.'),
  includeMemoryDefinition: z
    .boolean()
    .optional()
    .default(true)
    .describe('Fetch the full memory metadata alongside the matched chunks.'),
  waitForIndexing: z
    .boolean()
    .optional()
    .default(true)
    .describe(
      'Retry up to maxWaitSeconds when no results are found. Enable this when memories were just added and may still be processing.'
    ),
  maxWaitSeconds: z
    .number()
    .optional()
    .default(10)
    .describe(
      'Maximum time in seconds to keep polling for results when waitForIndexing is true.'
    ),
  pollInterval: z
    .number()
    .optional()
    .default(2)
    .describe(
      'Seconds to wait between polling attempts when waitForIndexing is true.'
    ),
  rerankerId: z
    .string()
    .optional()
    .describe('Optional reranker model UUID to improve result ordering.'),
  llmId: z
    .string()
    .optional()
    .describe(
      'Optional LLM UUID to generate contextual responses alongside retrieved chunks.'
    ),
  relevanceThreshold: z
    .number()
    .optional()
    .describe(
      'Minimum score (0-1) for including results. Only used when rerankerId or llmId is set.'
    ),
  llmTemperature: z
    .number()
    .optional()
    .describe(
      'Creativity setting for LLM generation (0-2). Only used when llmId is set.'
    ),
  chronologicalResort: z
    .boolean()
    .optional()
    .default(false)
    .describe('Reorder results by creation time instead of relevance score.'),
  metadataFilter: z
    .string()
    .optional()
    .describe(
      "Server-side filter applied to every space in spaceIds. Accepts a SQL-style JSONPath expression, for example \"CAST(val('$.category') AS TEXT) = 'feat'\" to return only memories whose metadata.category equals 'feat'."
    ),
});

const RetrieveMemoriesOutputSchema = z.object({
  success: z.boolean(),
  resultSetId: z.string().optional(),
  results: z.array(z.any()).optional(),
  memories: z.array(z.any()).optional(),
  totalResults: z.number().optional(),
  query: z.string().optional(),
  abstractReply: z.any().optional(),
  message: z.string().optional(),
  error: z.string().optional(),
  details: z.any().optional(),
});

const GetMemoryInputSchema = z.object({
  memoryId: z
    .string()
    .describe('The UUID of the memory to fetch (returned by Create Memory).'),
  includeContent: z
    .boolean()
    .optional()
    .default(true)
    .describe(
      'Fetch the original document content of the memory in addition to its metadata.'
    ),
});

const GetMemoryOutputSchema = z.object({
  success: z.boolean(),
  memory: z.any().optional(),
  content: z.any().optional(),
  contentError: z.string().optional(),
  error: z.string().optional(),
  details: z.any().optional(),
});

const DeleteMemoryInputSchema = z.object({
  memoryId: z
    .string()
    .describe('The UUID of the memory to delete (returned by Create Memory).'),
});

const DeleteMemoryOutputSchema = z.object({
  success: z.boolean(),
  memoryId: z.string().optional(),
  message: z.string().optional(),
  error: z.string().optional(),
  details: z.any().optional(),
});

// ---------------------------------------------------------------------------
// Plugin definition
// ---------------------------------------------------------------------------

/**
 * Genkit plugin for GoodMem, the retrieval-augmented generation (RAG)
 * memory backend for AI agents.
 *
 * Registers 11 GoodMem tools that can be used with any Genkit agent or flow:
 * list_embedders, list_spaces, get_space, create_space, update_space,
 * delete_space, create_memory, list_memories, retrieve_memories, get_memory,
 * delete_memory.
 *
 * @param params - GoodMem connection parameters (baseUrl, apiKey).
 * @returns A GenkitPlugin that registers the GoodMem tools.
 *
 * @example
 * ```ts
 * import { genkit } from 'genkit';
 * import { goodmem } from 'genkitx-goodmem';
 *
 * const ai = genkit({
 *   plugins: [
 *     goodmem({
 *       baseUrl: 'http://localhost:8080',
 *       apiKey: process.env.GOODMEM_API_KEY!,
 *     }),
 *   ],
 * });
 * ```
 */
export function goodmem(params: GoodMemPluginParams): GenkitPlugin {
  if (!params.baseUrl) {
    throw new Error(
      'GoodMem plugin requires a baseUrl. ' +
        'Please provide the URL of your GoodMem API server.'
    );
  }
  if (!params.apiKey) {
    throw new Error(
      'GoodMem plugin requires an apiKey. ' +
        'Please pass in the API key or set the GOODMEM_API_KEY environment variable.'
    );
  }
  return genkitPlugin('goodmem', async (ai: Genkit) => {
    const baseUrl = normalizeBaseUrl(params.baseUrl);
    const { apiKey } = params;

    // ----- List Embedders --------------------------------------------------
    ai.defineTool(
      {
        name: 'goodmem/list_embedders',
        description:
          'List all embedder models available in the GoodMem instance. Embedders convert text into vector representations and are referenced when creating spaces.',
        inputSchema: ListEmbeddersInputSchema,
        outputSchema: ListEmbeddersOutputSchema,
      },
      async () => {
        try {
          const embedders = await listEmbedders(params);
          return {
            success: true,
            embedders,
            totalResults: embedders.length,
          };
        } catch (error: any) {
          return {
            success: false,
            error: error.message || 'Failed to list embedders',
            details: error.response?.body || String(error),
          };
        }
      }
    );

    // ----- List Spaces -----------------------------------------------------
    ai.defineTool(
      {
        name: 'goodmem/list_spaces',
        description:
          'List all spaces visible to the API key. A space is a logical container for organizing related memories.',
        inputSchema: ListSpacesInputSchema,
        outputSchema: ListSpacesOutputSchema,
      },
      async () => {
        try {
          const spaces = await listSpaces(params);
          return {
            success: true,
            spaces,
            totalResults: spaces.length,
          };
        } catch (error: any) {
          return {
            success: false,
            error: error.message || 'Failed to list spaces',
            details: error.response?.body || String(error),
          };
        }
      }
    );

    // ----- Get Space -------------------------------------------------------
    ai.defineTool(
      {
        name: 'goodmem/get_space',
        description: 'Fetch a specific GoodMem space by its UUID.',
        inputSchema: GetSpaceInputSchema,
        outputSchema: GetSpaceOutputSchema,
      },
      async (input) => {
        const { spaceId } = input;
        try {
          const space = await apiJson(
            `${baseUrl}/v1/spaces/${spaceId}`,
            apiKey
          );
          return { success: true, space };
        } catch (error: any) {
          return {
            success: false,
            error: error.message || 'Failed to get space',
            details: error.response?.body || String(error),
          };
        }
      }
    );

    // ----- Create Space ----------------------------------------------------
    ai.defineTool(
      {
        name: 'goodmem/create_space',
        description:
          'Create a new GoodMem space or reuse an existing one with the same name. A space is a logical container for organizing related memories, configured with an embedder that converts text to vector embeddings.',
        inputSchema: CreateSpaceInputSchema,
        outputSchema: CreateSpaceOutputSchema,
      },
      async (input) => {
        const {
          name,
          embedderId,
          chunkSize,
          chunkOverlap,
          keepStrategy,
          lengthMeasurement,
          labels,
        } = input;

        // Reuse-on-name: check if a space with this name already exists.
        try {
          const spaces = await listSpaces(params);
          const existing = spaces.find((s: any) => s.name === name);
          if (existing) {
            return {
              success: true,
              spaceId: existing.spaceId,
              name: existing.name,
              embedderId,
              message: 'Space already exists, reusing existing space',
              reused: true,
            };
          }
        } catch {
          // If listing fails, proceed to create.
        }

        const requestBody: any = {
          name,
          spaceEmbedders: [{ embedderId, defaultRetrievalWeight: 1.0 }],
          defaultChunkingConfig: {
            recursive: {
              chunkSize: chunkSize ?? 256,
              chunkOverlap: chunkOverlap ?? 25,
              separators: ['\n\n', '\n', '. ', ' ', ''],
              keepStrategy: keepStrategy ?? 'KEEP_END',
              separatorIsRegex: false,
              lengthMeasurement: lengthMeasurement ?? 'CHARACTER_COUNT',
            },
          },
        };
        if (labels && Object.keys(labels).length > 0) {
          requestBody.labels = labels;
        }

        try {
          const response = await apiJson(`${baseUrl}/v1/spaces`, apiKey, {
            method: 'POST',
            body: JSON.stringify(requestBody),
          });
          return {
            success: true,
            spaceId: response.spaceId,
            name: response.name,
            embedderId,
            message: 'Space created successfully',
            reused: false,
          };
        } catch (error: any) {
          return {
            success: false,
            error: error.message || 'Failed to create space',
            details: error.response?.body || String(error),
          };
        }
      }
    );

    // ----- Update Space ----------------------------------------------------
    ai.defineTool(
      {
        name: 'goodmem/update_space',
        description:
          'Update a GoodMem space. Supports renaming, toggling publicRead, and modifying labels via replaceLabels (overwrite all) or mergeLabels (merge into existing).',
        inputSchema: UpdateSpaceInputSchema,
        outputSchema: UpdateSpaceOutputSchema,
      },
      async (input) => {
        const {
          spaceId,
          name,
          publicRead,
          replaceLabels,
          mergeLabels,
          defaultChunkingConfig,
        } = input;

        if (replaceLabels && mergeLabels) {
          return {
            success: false,
            error:
              'replaceLabels and mergeLabels are mutually exclusive. Pass only one.',
          };
        }

        const requestBody: any = {};
        if (name !== undefined) requestBody.name = name;
        if (publicRead !== undefined) requestBody.publicRead = publicRead;
        if (replaceLabels !== undefined)
          requestBody.replaceLabels = replaceLabels;
        if (mergeLabels !== undefined) requestBody.mergeLabels = mergeLabels;
        if (defaultChunkingConfig !== undefined)
          requestBody.defaultChunkingConfig = defaultChunkingConfig;

        if (Object.keys(requestBody).length === 0) {
          return {
            success: false,
            error:
              'No fields provided to update. Pass at least one of: name, publicRead, replaceLabels, mergeLabels, defaultChunkingConfig.',
          };
        }

        try {
          const space = await apiJson(
            `${baseUrl}/v1/spaces/${spaceId}`,
            apiKey,
            { method: 'PUT', body: JSON.stringify(requestBody) }
          );
          return { success: true, space };
        } catch (error: any) {
          return {
            success: false,
            error: error.message || 'Failed to update space',
            details: error.response?.body || String(error),
          };
        }
      }
    );

    // ----- Delete Space ----------------------------------------------------
    ai.defineTool(
      {
        name: 'goodmem/delete_space',
        description:
          'Permanently delete a GoodMem space. All memories inside the space are deleted as well.',
        inputSchema: DeleteSpaceInputSchema,
        outputSchema: DeleteSpaceOutputSchema,
      },
      async (input) => {
        const { spaceId } = input;
        try {
          const res = await apiFetch(
            `${baseUrl}/v1/spaces/${spaceId}`,
            apiKey,
            { method: 'DELETE' }
          );
          if (!res.ok) {
            let detail: string;
            try {
              const errBody = await res.json();
              detail =
                errBody.message || errBody.error || JSON.stringify(errBody);
            } catch {
              detail = await res.text().catch(() => res.statusText);
            }
            throw new Error(`GoodMem API error (${res.status}): ${detail}`);
          }
          return {
            success: true,
            spaceId,
            message: 'Space deleted successfully',
          };
        } catch (error: any) {
          return {
            success: false,
            error: error.message || 'Failed to delete space',
            details: error.response?.body || String(error),
          };
        }
      }
    );

    // ----- Create Memory ---------------------------------------------------
    ai.defineTool(
      {
        name: 'goodmem/create_memory',
        description:
          'Store a document as a new memory in a GoodMem space. The memory is processed asynchronously, chunked into searchable pieces, and embedded into vectors. Accepts a file path or plain text.',
        inputSchema: CreateMemoryInputSchema,
        outputSchema: CreateMemoryOutputSchema,
      },
      async (input) => {
        const {
          spaceId,
          filePath,
          textContent,
          source,
          author,
          tags,
          metadata,
        } = input;

        const requestBody: any = { spaceId };
        let fileName: string | null = null;

        if (filePath) {
          if (!fs.existsSync(filePath)) {
            return {
              success: false,
              error: `File not found: ${filePath}`,
            };
          }
          const fileBuffer = fs.readFileSync(filePath);
          const base64 = fileBuffer.toString('base64');
          const ext = path.extname(filePath).replace('.', '');
          const detectedMime = getMimeType(ext);
          const mimeType = detectedMime || 'application/octet-stream';
          fileName = path.basename(filePath);

          if (mimeType.startsWith('text/')) {
            requestBody.contentType = mimeType;
            requestBody.originalContent = fileBuffer.toString('utf-8');
          } else {
            requestBody.contentType = mimeType;
            requestBody.originalContentB64 = base64;
          }
        } else if (textContent) {
          requestBody.contentType = 'text/plain';
          requestBody.originalContent = textContent;
        } else {
          return {
            success: false,
            error:
              'No content provided. Please provide a filePath or textContent.',
          };
        }

        const mergedMetadata: Record<string, any> = {};
        if (metadata && typeof metadata === 'object') {
          Object.assign(mergedMetadata, metadata);
        }
        if (source) mergedMetadata.source = source;
        if (author) mergedMetadata.author = author;
        if (tags) {
          mergedMetadata.tags = tags
            .split(',')
            .map((t: string) => t.trim())
            .filter((t: string) => t.length > 0);
        }
        if (Object.keys(mergedMetadata).length > 0) {
          requestBody.metadata = mergedMetadata;
        }

        try {
          const response = await apiJson(`${baseUrl}/v1/memories`, apiKey, {
            method: 'POST',
            body: JSON.stringify(requestBody),
          });
          return {
            success: true,
            memoryId: response.memoryId,
            spaceId: response.spaceId,
            status: response.processingStatus || 'PENDING',
            contentType: requestBody.contentType,
            fileName,
            message: 'Memory created successfully',
          };
        } catch (error: any) {
          return {
            success: false,
            error: error.message || 'Failed to create memory',
            details: error.response?.body || String(error),
          };
        }
      }
    );

    // ----- List Memories ---------------------------------------------------
    ai.defineTool(
      {
        name: 'goodmem/list_memories',
        description:
          'List the memories stored in a specific GoodMem space (GoodMem scopes memory listing to a single space).',
        inputSchema: ListMemoriesInputSchema,
        outputSchema: ListMemoriesOutputSchema,
      },
      async (input) => {
        const { spaceId, statusFilter, includeContent, sortBy, sortOrder } =
          input;
        const params = new URLSearchParams();
        if (includeContent) params.set('includeContent', 'true');
        if (statusFilter) params.set('statusFilter', statusFilter);
        if (sortBy) params.set('sortBy', sortBy);
        if (sortOrder) params.set('sortOrder', sortOrder);
        const query = params.toString();
        const url = `${baseUrl}/v1/spaces/${spaceId}/memories${
          query ? `?${query}` : ''
        }`;
        try {
          const body = await apiJson<any>(url, apiKey);
          const memories = Array.isArray(body) ? body : body?.memories || [];
          return {
            success: true,
            spaceId,
            memories,
            totalResults: memories.length,
          };
        } catch (error: any) {
          return {
            success: false,
            error: error.message || 'Failed to list memories',
            details: error.response?.body || String(error),
          };
        }
      }
    );

    // ----- Retrieve Memories -----------------------------------------------
    ai.defineTool(
      {
        name: 'goodmem/retrieve_memories',
        description:
          'Perform similarity-based semantic retrieval across one or more GoodMem spaces. Returns matching chunks ranked by relevance, with optional reranking, LLM-generated abstract reply, relevance threshold, and chronological resorting.',
        inputSchema: RetrieveMemoriesInputSchema,
        outputSchema: RetrieveMemoriesOutputSchema,
      },
      async (input) => {
        const {
          query,
          spaceIds,
          maxResults,
          includeMemoryDefinition,
          waitForIndexing,
          maxWaitSeconds,
          pollInterval,
          rerankerId,
          llmId,
          relevanceThreshold,
          llmTemperature,
          chronologicalResort,
          metadataFilter,
        } = input;

        const spaceKeys = spaceIds
          .filter((id: string) => id && id.length > 0)
          .map((spaceId: string) => {
            const key: { spaceId: string; filter?: string } = { spaceId };
            if (metadataFilter && metadataFilter.length > 0) {
              key.filter = metadataFilter;
            }
            return key;
          });

        if (spaceKeys.length === 0) {
          return {
            success: false,
            error: 'At least one space must be selected.',
          };
        }

        const requestBody: any = {
          message: query,
          spaceKeys,
          requestedSize: maxResults ?? 5,
          fetchMemory: includeMemoryDefinition !== false,
        };

        if (rerankerId || llmId) {
          const config: any = {};
          if (rerankerId) config.reranker_id = rerankerId;
          if (llmId) config.llm_id = llmId;
          if (relevanceThreshold !== undefined && relevanceThreshold !== null)
            config.relevance_threshold = relevanceThreshold;
          if (llmTemperature !== undefined && llmTemperature !== null)
            config.llm_temp = llmTemperature;
          if (maxResults !== undefined && maxResults !== null)
            config.max_results = maxResults;
          if (chronologicalResort === true) config.chronological_resort = true;

          requestBody.postProcessor = {
            name: 'com.goodmem.retrieval.postprocess.ChatPostProcessorFactory',
            config,
          };
        }

        const maxWaitMs = (maxWaitSeconds ?? 10) * 1000;
        const pollIntervalMs = (pollInterval ?? 2) * 1000;
        const shouldWait = waitForIndexing !== false;
        const startTime = Date.now();
        let lastResult: any = null;

        try {
          do {
            const res = await apiFetch(
              `${baseUrl}/v1/memories:retrieve`,
              apiKey,
              {
                method: 'POST',
                headers: { Accept: 'application/x-ndjson' },
                body: JSON.stringify(requestBody),
              }
            );

            if (!res.ok) {
              let detail: string;
              try {
                const errBody = await res.json();
                detail =
                  errBody.message || errBody.error || JSON.stringify(errBody);
              } catch {
                detail = await res.text().catch(() => res.statusText);
              }
              throw new Error(`GoodMem API error (${res.status}): ${detail}`);
            }

            const responseText = await res.text();
            const results: any[] = [];
            const memories: any[] = [];
            let resultSetId = '';
            let abstractReply: any = null;

            const lines = responseText.trim().split('\n');
            for (const line of lines) {
              let jsonStr = line.trim();
              if (!jsonStr) continue;
              if (jsonStr.startsWith('data:')) {
                jsonStr = jsonStr.substring(5).trim();
              }
              if (jsonStr.startsWith('event:') || jsonStr === '') continue;

              try {
                const item = JSON.parse(jsonStr);

                if (item.resultSetBoundary) {
                  resultSetId = item.resultSetBoundary.resultSetId;
                } else if (item.memoryDefinition) {
                  memories.push(item.memoryDefinition);
                } else if (item.abstractReply) {
                  abstractReply = item.abstractReply;
                } else if (item.retrievedItem) {
                  results.push({
                    chunkId: item.retrievedItem.chunk?.chunk?.chunkId,
                    chunkText: item.retrievedItem.chunk?.chunk?.chunkText,
                    memoryId: item.retrievedItem.chunk?.chunk?.memoryId,
                    relevanceScore: item.retrievedItem.chunk?.relevanceScore,
                    memoryIndex: item.retrievedItem.chunk?.memoryIndex,
                  });
                }
              } catch {
                // skip non-JSON lines
              }
            }

            lastResult = {
              success: true,
              resultSetId,
              results,
              memories,
              totalResults: results.length,
              query,
              ...(abstractReply ? { abstractReply } : {}),
            };

            if (results.length > 0 || !shouldWait) {
              return lastResult;
            }

            const elapsed = Date.now() - startTime;
            if (elapsed >= maxWaitMs) {
              return {
                ...lastResult,
                message:
                  'No results found after waiting for indexing. Memories may still be processing.',
              };
            }

            await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
          } while (true);
        } catch (error: any) {
          return {
            success: false,
            error: error.message || 'Failed to retrieve memories',
            details: error.response?.body || String(error),
          };
        }
      }
    );

    // ----- Get Memory ------------------------------------------------------
    ai.defineTool(
      {
        name: 'goodmem/get_memory',
        description:
          'Fetch a specific GoodMem memory record by its ID, including metadata, processing status, and optionally the original content.',
        inputSchema: GetMemoryInputSchema,
        outputSchema: GetMemoryOutputSchema,
      },
      async (input) => {
        const { memoryId, includeContent } = input;
        try {
          const memory = await apiJson(
            `${baseUrl}/v1/memories/${memoryId}`,
            apiKey
          );

          const result: any = { success: true, memory };

          if (includeContent) {
            try {
              const content = await apiJson(
                `${baseUrl}/v1/memories/${memoryId}/content`,
                apiKey
              );
              result.content = content;
            } catch (contentError: any) {
              result.contentError =
                'Failed to fetch content: ' +
                (contentError.message || 'Unknown error');
            }
          }

          return result;
        } catch (error: any) {
          return {
            success: false,
            error: error.message || 'Failed to get memory',
            details: error.response?.body || String(error),
          };
        }
      }
    );

    // ----- Delete Memory ---------------------------------------------------
    ai.defineTool(
      {
        name: 'goodmem/delete_memory',
        description:
          'Permanently delete a GoodMem memory and its associated chunks and vector embeddings.',
        inputSchema: DeleteMemoryInputSchema,
        outputSchema: DeleteMemoryOutputSchema,
      },
      async (input) => {
        const { memoryId } = input;
        try {
          const res = await apiFetch(
            `${baseUrl}/v1/memories/${memoryId}`,
            apiKey,
            { method: 'DELETE' }
          );
          if (!res.ok) {
            let detail: string;
            try {
              const errBody = await res.json();
              detail =
                errBody.message || errBody.error || JSON.stringify(errBody);
            } catch {
              detail = await res.text().catch(() => res.statusText);
            }
            throw new Error(`GoodMem API error (${res.status}): ${detail}`);
          }

          return {
            success: true,
            memoryId,
            message: 'Memory deleted successfully',
          };
        } catch (error: any) {
          return {
            success: false,
            error: error.message || 'Failed to delete memory',
            details: error.response?.body || String(error),
          };
        }
      }
    );
  });
}

export default goodmem;
