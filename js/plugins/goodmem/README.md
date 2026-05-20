# GoodMem plugin for Genkit

GoodMem gives AI agents retrieval-augmented generation (RAG) memory. Store documents in a space and GoodMem chunks, embeds, and indexes them so your agent can pull back the most relevant passages on any question. This plugin exposes GoodMem operations as Genkit tools that work with any Genkit agent or flow.

## Prerequisites

### 1. Install GoodMem

You need a running GoodMem instance. Install it on your VM or local machine.

**Visit:** [https://goodmem.ai/](https://goodmem.ai/)

Follow the installation instructions for your platform (Docker, local installation, or cloud deployment).

### 2. Create an embedder

Spaces and memories require an embedder model on your GoodMem instance. You can list embedders programmatically with the `goodmem/list_embedders` tool or the `listEmbedders()` helper.

### 3. Get your API key

Obtain an API key from your GoodMem instance. Keys start with `gm_`.

## Installing the plugin

```bash
npm i --save genkitx-goodmem
```

## Using the plugin

```ts
import { genkit } from 'genkit';
import { goodmem } from 'genkitx-goodmem';

const ai = genkit({
  plugins: [
    goodmem({
      baseUrl: process.env.GOODMEM_BASE_URL || 'http://localhost:8080',
      apiKey: process.env.GOODMEM_API_KEY!,
    }),
  ],
});
```

Once the plugin is loaded, 11 tools are automatically registered and available to any Genkit agent or flow.

## Tool naming

All tools sit under the `goodmem/` namespace and use snake_case action names. For example: `goodmem/create_space`, `goodmem/list_embedders`. The `<plugin>/<action>` shape follows Genkit's convention; snake_case keeps action names consistent across the surface.

## Available tools

### `goodmem/list_embedders`

List all embedder models available in the GoodMem instance.

**Input:** none.
**Output:** `{ success, embedders, totalResults }`.

### `goodmem/list_spaces`

List all spaces visible to the API key.

**Input:** none.
**Output:** `{ success, spaces, totalResults }`.

### `goodmem/get_space`

Fetch a single space by its UUID.

**Input:**

- **spaceId** (required): UUID of the space.

**Output:** `{ success, space }`.

### `goodmem/create_space`

Create a new space (a container for memories) with configurable settings. If a space with the same name already exists, it is reused instead of creating a duplicate.

**Input:**

- **name** (required): unique name for the space.
- **embedderId** (required): ID of the embedder model that converts text to vector embeddings.
- **chunkSize** (default: 256): number of characters per chunk when splitting documents.
- **chunkOverlap** (default: 25): overlapping characters between consecutive chunks.
- **keepStrategy** (default: `"KEEP_END"`): where to attach the separator when splitting (`"KEEP_END"`, `"KEEP_START"`, or `"DISCARD"`).
- **lengthMeasurement** (default: `"CHARACTER_COUNT"`): how chunk size is measured (`"CHARACTER_COUNT"` or `"TOKEN_COUNT"`).
- **labels** (optional): map of string labels to attach at creation time.

**Output:** `{ success, spaceId, name, embedderId, reused, message }`.

### `goodmem/update_space`

Update a GoodMem space. Supports renaming, toggling `publicRead`, modifying labels, and changing the default chunking config.

**Input:**

- **spaceId** (required): UUID of the space to update.
- **name** (optional): new name.
- **publicRead** (optional): whether the space is publicly readable.
- **replaceLabels** (optional): replace all existing labels with this map.
- **mergeLabels** (optional): merge these labels into the existing labels, adding or overwriting individual keys. Mutually exclusive with `replaceLabels`.
- **defaultChunkingConfig** (optional): new chunking config, for example `{ recursive: { chunkSize: 512, ... } }`.

**Output:** `{ success, space }`.

### `goodmem/delete_space`

Permanently delete a GoodMem space. All memories inside the space are deleted as well.

**Input:**

- **spaceId** (required): UUID of the space.

**Output:** `{ success, spaceId, message }`.

### `goodmem/create_memory`

Store a document or plain text as a memory in a space. The content is automatically chunked and embedded for semantic search.

**Input:**

- **spaceId** (required): ID of the space to store the memory in.
- **filePath** (optional): absolute path to a file (PDF, DOCX, TXT, images, etc.). Content type is auto-detected.
- **textContent** (optional): plain text content. If both `filePath` and `textContent` are provided, the file takes priority.
- **source** (optional): where this memory came from, for example `"google-drive"` or `"gmail"`.
- **author** (optional): the author or creator of the content.
- **tags** (optional): comma-separated tags for categorization, for example `"legal,research,important"`.
- **metadata** (optional): extra key-value metadata as JSON.

**Output:** `{ success, memoryId, spaceId, status, contentType, fileName, message }`.

### `goodmem/list_memories`

List the memories stored in a specific GoodMem space (GoodMem scopes memory listing to a single space).

**Input:**

- **spaceId** (required): UUID of the space.
- **statusFilter** (optional): restrict results to memories in this processing status (`"PENDING"`, `"PROCESSING"`, `"COMPLETED"`, or `"FAILED"`).
- **includeContent** (default: false): when true, each returned memory includes its original document content alongside the metadata.
- **sortBy** (optional): field used to sort the returned memories (`"created_at"` or `"updated_at"`).
- **sortOrder** (optional): sort direction (`"ASCENDING"` or `"DESCENDING"`).

**Output:** `{ success, spaceId, memories, totalResults }`.

### `goodmem/retrieve_memories`

Perform semantic search across one or more spaces to find relevant memory chunks. Supports advanced post-processing with reranking, LLM-generated contextual responses, score thresholds, and chronological resorting.

**Input:**

- **query** (required): natural language search query.
- **spaceIds** (required): array of space IDs to search across.
- **maxResults** (default: 5): limit the number of returned results.
- **includeMemoryDefinition** (default: true): fetch full memory metadata alongside matched chunks.
- **waitForIndexing** (default: true): retry up to `maxWaitSeconds` when no results are found. Useful when memories were just added and may still be processing.
- **maxWaitSeconds** (default: 10): maximum time in seconds to keep polling for results when `waitForIndexing` is true.
- **pollInterval** (default: 2): seconds to wait between polling attempts when `waitForIndexing` is true.
- **rerankerId** (optional): UUID of a reranker model to improve result ordering.
- **llmId** (optional): UUID of an LLM that generates a contextual `abstractReply` alongside the retrieved chunks.
- **relevanceThreshold** (optional, 0-1): minimum score for including results. Only used when `rerankerId` or `llmId` is set.
- **llmTemperature** (optional, 0-2): creativity setting for LLM generation. Only used when `llmId` is set.
- **chronologicalResort** (default: false): reorder results by creation time instead of relevance score.
- **metadataFilter** (optional): server-side SQL-style JSONPath expression applied to every space in `spaceIds`. For example, `CAST(val('$.category') AS TEXT) = 'feat'` returns only memories whose `metadata.category` equals `feat`.

The advanced options (`rerankerId`, `llmId`, `relevanceThreshold`, `llmTemperature`, `maxResults`, `chronologicalResort`) are forwarded to the GoodMem retrieve API as a `postProcessor` block:

```json
{
  "postProcessor": {
    "name": "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory",
    "config": {
      "reranker_id": "...",
      "llm_id": "...",
      "relevance_threshold": 0.5,
      "llm_temp": 0.2,
      "max_results": 5,
      "chronological_resort": false
    }
  }
}
```

**Output:** `{ success, resultSetId, results, memories, totalResults, query, abstractReply?, message? }`.

### `goodmem/get_memory`

Retrieve a specific memory by its ID, including metadata, processing status, and optionally the original content.

**Input:**

- **memoryId** (required): the UUID of the memory to fetch.
- **includeContent** (default: true): fetch the original document content in addition to metadata.

**Output:** `{ success, memory, content?, contentError? }`.

### `goodmem/delete_memory`

Permanently delete a memory and all its associated chunks and vector embeddings.

**Input:**

- **memoryId** (required): the UUID of the memory to delete.

**Output:** `{ success, memoryId, message }`.

## Helper functions

The plugin also exports two helper functions for programmatic use. The `goodmem/list_spaces` and `goodmem/list_embedders` tools call into these:

```ts
import { listSpaces, listEmbedders } from 'genkitx-goodmem';

// List all spaces
const spaces = await listSpaces({ baseUrl: '...', apiKey: '...' });

// List all embedders
const embedders = await listEmbedders({ baseUrl: '...', apiKey: '...' });
```

## Example: full workflow

```ts
import { genkit, z } from 'genkit';
import { goodmem } from 'genkitx-goodmem';

const ai = genkit({
  plugins: [
    goodmem({
      baseUrl: 'http://localhost:8080',
      apiKey: process.env.GOODMEM_API_KEY!,
    }),
  ],
});

const memoryFlow = ai.defineFlow(
  { name: 'memoryFlow', inputSchema: z.string(), outputSchema: z.any() },
  async (query) => {
    // Look up an embedder
    const listEmbedders = await ai.registry.lookupAction(
      '/tool/goodmem/list_embedders'
    );
    const embeddersResp = await listEmbedders({});
    const embedderId = embeddersResp.embedders[0].embedderId;

    // Create (or reuse) a space
    const createSpace = await ai.registry.lookupAction(
      '/tool/goodmem/create_space'
    );
    const space = await createSpace({
      name: 'my-knowledge-base',
      embedderId,
    });

    // Store a memory
    const createMemory = await ai.registry.lookupAction(
      '/tool/goodmem/create_memory'
    );
    await createMemory({
      spaceId: space.spaceId,
      textContent: 'The capital of France is Paris.',
      source: 'manual',
    });

    // Retrieve relevant memories with reranking and an LLM abstract reply
    const retrieve = await ai.registry.lookupAction(
      '/tool/goodmem/retrieve_memories'
    );
    return retrieve({
      query,
      spaceIds: [space.spaceId],
      rerankerId: process.env.GOODMEM_RERANKER_ID,
      llmId: process.env.GOODMEM_LLM_ID,
      relevanceThreshold: 0.3,
      llmTemperature: 0.2,
      maxResults: 5,
    });
  }
);
```

A runnable demo with three scenarios (persistent project knowledge, a scribe and analyst pipeline, and metadata-driven retrieval) lives at `js/testapps/goodmem/`.

## Testing

Run the unit test suite (mocked fetch, exercises every tool):

```bash
npm run test
```

The sources for this package are in the main [Genkit](https://github.com/firebase/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [official Genkit documentation](https://genkit.dev/docs/get-started/).

License: Apache 2.0
