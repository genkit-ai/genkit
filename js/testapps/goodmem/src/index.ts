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

/**
 * End-to-end demo for the genkitx-goodmem plugin. GoodMem gives AI agents
 * retrieval-augmented generation (RAG) memory; this demo wires it into
 * Genkit.
 *
 * Runs three scenarios against a live GoodMem server and an OpenAI-compatible
 * model:
 *
 *   1. Persistent project context across sessions. Store project facts in
 *      a space, then retrieve them from a separate call so the model can
 *      answer a follow-up question grounded in the stored context.
 *   2. Two-role team knowledge pipeline. A scribe step stores team notes;
 *      an analyst step retrieves and summarizes them. Each step runs
 *      inside its own Genkit span so the role shows up in the trace.
 *   3. Structured team activity log. Store release entries tagged by
 *      category, then use the metadataFilter parameter on
 *      retrieve_memories to fetch only the feature entries.
 *
 * Required environment variables:
 *   GOODMEM_API_KEY, GOODMEM_BASE_URL, OPENAI_API_KEY.
 * Optional for self-signed local TLS:
 *   NODE_TLS_REJECT_UNAUTHORIZED=0.
 */

import { openAI } from '@genkit-ai/compat-oai/openai';
import { genkit } from 'genkit';
import { runInNewSpan } from 'genkit/tracing';
import { goodmem } from 'genkitx-goodmem';

const BASE_URL = process.env.GOODMEM_BASE_URL ?? 'https://localhost:8080';
const API_KEY = process.env.GOODMEM_API_KEY;
const OPENAI_KEY = process.env.OPENAI_API_KEY;

if (!API_KEY) {
  console.error('Set GOODMEM_API_KEY before running this demo.');
  process.exit(1);
}
if (!OPENAI_KEY) {
  console.error('Set OPENAI_API_KEY before running this demo.');
  process.exit(1);
}

const ai = genkit({
  plugins: [openAI(), goodmem({ baseUrl: BASE_URL, apiKey: API_KEY })],
});

const MODEL = openAI.model('gpt-4o-mini');

const PROJECT_FACTS = [
  "I'm building a customer support assistant for our SaaS product.",
  'The team uses Python 3.12 with FastAPI and Postgres.',
  'For tests we use pytest with at least 80% coverage required.',
];

const TEAM_NOTES = [
  'Q2 goal: reduce customer support response time to under 2 hours.',
  'Our main services are auth-service, billing-service, and notifications-service.',
  'Known issue: notifications-service drops messages during high load.',
  'Team retro: the CI pipeline is too slow; we should parallelize tests.',
];

interface ReleaseEntry {
  content: string;
  category: 'feat' | 'fix' | 'chore' | 'docs';
}

const RELEASE_LOG: ReleaseEntry[] = [
  { content: 'Added user profile editing to the dashboard.', category: 'feat' },
  { content: 'Built the CSV export feature.', category: 'feat' },
  { content: 'Resolved slow login on the mobile app.', category: 'fix' },
  { content: 'Fixed crash when opening large attachments.', category: 'fix' },
  { content: 'Upgraded Python version across services.', category: 'chore' },
  {
    content: 'Updated the API reference for billing endpoints.',
    category: 'docs',
  },
];

const INDEX_WAIT_MS = 6_000;

interface ToolResult {
  success: boolean;
  error?: string;
  [key: string]: unknown;
}

function section(title: string): void {
  console.log('\n' + '='.repeat(70));
  console.log(`  ${title}`);
  console.log('='.repeat(70));
}

async function callTool<T extends ToolResult>(
  name: string,
  input: Record<string, unknown>
): Promise<T> {
  const action = await ai.registry.lookupAction(`/tool/${name}`);
  if (!action) {
    throw new Error(`Tool not found: ${name}`);
  }
  const result = (await (action as (input: unknown) => Promise<unknown>)(
    input
  )) as T;
  if (!result.success) {
    throw new Error(`${name} failed: ${result.error ?? 'unknown error'}`);
  }
  return result;
}

async function pickEmbedderId(): Promise<string> {
  const result = await callTool<
    ToolResult & { embedders?: Array<{ embedderId?: string; id?: string }> }
  >('goodmem/list_embedders', {});
  const first = result.embedders?.[0];
  if (!first) {
    throw new Error(
      'No embedders found on the GoodMem server. Register one and retry.'
    );
  }
  const embedderId = first.embedderId ?? first.id;
  if (!embedderId) {
    throw new Error('First embedder is missing an id field.');
  }
  return embedderId;
}

async function createSpace(name: string, embedderId: string): Promise<string> {
  const result = await callTool<ToolResult & { spaceId?: string }>(
    'goodmem/create_space',
    { name, embedderId }
  );
  if (!result.spaceId) {
    throw new Error(`create_space for ${name} returned no spaceId.`);
  }
  return result.spaceId;
}

async function storeMemory(
  spaceId: string,
  textContent: string,
  metadata?: Record<string, unknown>,
  source?: string
): Promise<void> {
  const input: Record<string, unknown> = { spaceId, textContent };
  if (source) input.source = source;
  if (metadata) input.metadata = metadata;
  await callTool('goodmem/create_memory', input);
}

interface RetrievedChunk {
  chunkId?: string;
  chunkText?: string;
  memoryId?: string;
  relevanceScore?: number;
  memoryIndex?: number;
}

async function retrieveChunks(
  spaceId: string,
  query: string,
  options: { maxResults?: number; metadataFilter?: string } = {}
): Promise<RetrievedChunk[]> {
  const result = await callTool<ToolResult & { results?: RetrievedChunk[] }>(
    'goodmem/retrieve_memories',
    {
      query,
      spaceIds: [spaceId],
      maxResults: options.maxResults ?? 5,
      waitForIndexing: true,
      maxWaitSeconds: 15,
      pollInterval: 2,
      ...(options.metadataFilter
        ? { metadataFilter: options.metadataFilter }
        : {}),
    }
  );
  return result.results ?? [];
}

async function answerGrounded(
  question: string,
  chunks: RetrievedChunk[],
  task: string
): Promise<string> {
  const context = chunks
    .map((c) => `- ${(c.chunkText ?? '').trim()}`)
    .join('\n');
  const { text } = await ai.generate({
    model: MODEL,
    prompt:
      `${task}\n\n` +
      `Use ONLY the context below. If the context does not contain the ` +
      `answer, say you do not know.\n\n` +
      `Context:\n${context}\n\nQuestion: ${question}`,
  });
  return text.trim();
}

async function deleteSpaceSilently(spaceId: string): Promise<void> {
  try {
    await callTool('goodmem/delete_space', { spaceId });
  } catch (error) {
    console.warn(
      `  Cleanup warning for ${spaceId}: ${(error as Error).message}`
    );
  }
}

async function scenario1PersistentMemory(embedderId: string): Promise<string> {
  section('Scenario 1: persistent project context across sessions');
  const spaceId = await createSpace('genkit-goodmem-persistent', embedderId);
  console.log(`  Space: ${spaceId}`);

  console.log(`  Storing ${PROJECT_FACTS.length} project facts.`);
  for (const fact of PROJECT_FACTS) {
    await storeMemory(spaceId, fact, undefined, 'project-facts');
  }

  console.log(`  Waiting ${INDEX_WAIT_MS / 1000}s for indexing.`);
  await new Promise((r) => setTimeout(r, INDEX_WAIT_MS));

  const question = 'Remind me what our coverage requirement is.';
  console.log(`\n  Question: ${question}`);

  const chunks = await retrieveChunks(spaceId, question, { maxResults: 3 });
  for (const [i, chunk] of chunks.entries()) {
    const score = chunk.relevanceScore?.toFixed(3) ?? 'n/a';
    const preview = (chunk.chunkText ?? '').slice(0, 120);
    console.log(`    [${i + 1}] score=${score} :: ${preview}`);
  }

  const answer = await answerGrounded(
    question,
    chunks,
    'Answer the user question in one short sentence.'
  );
  console.log(`\n  Answer: ${answer}`);
  return spaceId;
}

async function scenario2ScribeAnalyst(embedderId: string): Promise<string> {
  section('Scenario 2: two-role team knowledge pipeline');
  const spaceId = await createSpace('genkit-goodmem-team', embedderId);
  console.log(`  Space: ${spaceId}`);

  await runInNewSpan(
    {
      metadata: { name: 'scribe.store-notes' },
      labels: { role: 'scribe', 'goodmem.space_id': spaceId },
    },
    async () => {
      console.log(`  Scribe storing ${TEAM_NOTES.length} team notes.`);
      for (const note of TEAM_NOTES) {
        await storeMemory(spaceId, note, undefined, 'team-notes');
      }
    }
  );

  console.log(`  Waiting ${INDEX_WAIT_MS / 1000}s for indexing.`);
  await new Promise((r) => setTimeout(r, INDEX_WAIT_MS));

  const question = 'What do we know about our services and current priorities?';

  const summary = await runInNewSpan<string>(
    {
      metadata: { name: 'analyst.summarize' },
      labels: { role: 'analyst', 'goodmem.space_id': spaceId },
    },
    async () => {
      console.log(`\n  Analyst question: ${question}`);
      const chunks = await retrieveChunks(spaceId, question, {
        maxResults: 5,
      });
      for (const [i, chunk] of chunks.entries()) {
        const preview = (chunk.chunkText ?? '').slice(0, 120);
        console.log(`    [${i + 1}] ${preview}`);
      }
      return answerGrounded(
        question,
        chunks,
        'Write a concise team summary in two or three short bullets.'
      );
    }
  );

  console.log(`\n  Summary:\n${indent(summary, '    ')}`);
  return spaceId;
}

async function scenario3MetadataFiltering(embedderId: string): Promise<string> {
  section('Scenario 3: structured team activity log');
  const spaceId = await createSpace('genkit-goodmem-release-log', embedderId);
  console.log(`  Space: ${spaceId}`);

  console.log(`  Storing ${RELEASE_LOG.length} tagged release entries.`);
  for (const entry of RELEASE_LOG) {
    await storeMemory(
      spaceId,
      entry.content,
      { category: entry.category },
      'release-log'
    );
  }

  console.log(`  Waiting ${INDEX_WAIT_MS / 1000}s for indexing.`);
  await new Promise((r) => setTimeout(r, INDEX_WAIT_MS));

  const filter = "CAST(val('$.category') AS TEXT) = 'feat'";
  const question = "Show me the new features we've shipped.";
  console.log(`\n  Question: ${question}`);
  console.log(`  Filter:   ${filter}`);

  const chunks = await retrieveChunks(spaceId, question, {
    maxResults: 10,
    metadataFilter: filter,
  });
  for (const [i, chunk] of chunks.entries()) {
    const preview = (chunk.chunkText ?? '').slice(0, 120);
    console.log(`    [${i + 1}] ${preview}`);
  }

  const releaseNotes = await answerGrounded(
    question,
    chunks,
    'Write a release-notes bullet list of the new features.'
  );
  console.log(`\n  Release notes:\n${indent(releaseNotes, '    ')}`);
  return spaceId;
}

function indent(text: string, prefix: string): string {
  return text
    .split('\n')
    .map((line) => prefix + line)
    .join('\n');
}

async function main(): Promise<void> {
  console.log('\n  GoodMem + Genkit demo');
  console.log(`  Server: ${BASE_URL}`);

  const embedderId = await pickEmbedderId();
  console.log(`  Embedder: ${embedderId}`);

  const spaceIds: string[] = [];
  try {
    spaceIds.push(await scenario1PersistentMemory(embedderId));
    spaceIds.push(await scenario2ScribeAnalyst(embedderId));
    spaceIds.push(await scenario3MetadataFiltering(embedderId));
  } finally {
    section('Cleanup');
    for (const spaceId of spaceIds) {
      console.log(`  Deleting space ${spaceId}`);
      await deleteSpaceSilently(spaceId);
    }
  }

  if (process.env.GENKIT_ENV === 'dev') {
    console.log(
      '\n  Done. The Genkit Developer UI is running at ' +
        'http://localhost:4000 for trace inspection. Press Ctrl+C to ' +
        'shut it down.\n'
    );
  } else {
    console.log(
      '\n  Done. To inspect traces in the Genkit Developer UI, rerun ' +
        'with `pnpm genkit:start`. The UI will be at ' +
        'http://localhost:4000 while the process is alive.\n'
    );
  }
}

main().catch((err) => {
  console.error('\nDemo failed:', err);
  process.exitCode = 1;
});
