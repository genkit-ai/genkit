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
 * Manual smoke test against the live OpenAI Responses API.
 *
 * Run from `js/plugins/compat-oai/`:
 *
 *   OPENAI_API_KEY=sk-... pnpm tsx scripts/smoke_responses.ts
 *
 * Optional model override:
 *
 *   OPENAI_API_KEY=sk-... SMOKE_MODEL=gpt-5-nano pnpm tsx scripts/smoke_responses.ts
 *
 * Exits 1 on any failure so this can also be wired into a CI gate later.
 */

import { genkit } from 'genkit';
import openAI, { openAIResponses } from '../src/openai';

const MODEL = process.env.SMOKE_MODEL ?? 'gpt-5-mini';

async function assert(name: string, cond: unknown, detail?: unknown) {
  if (!cond) {
    console.error('✗ %s', name, detail ?? '');
    process.exitCode = 1;
  } else {
    console.log('✓ %s', name);
  }
}

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    console.error('OPENAI_API_KEY is not set; refusing to call live API.');
    process.exit(2);
  }
  const ai = genkit({ plugins: [openAI(), openAIResponses()] });

  // ---- Smoke 1: plain non-streaming text generation ----
  console.log(`\n=== Smoke 1: plain text via ${MODEL} ===`);
  const r1 = await ai.generate({
    model: openAI.responsesModel(MODEL),
    prompt: 'Reply with exactly the single word: PONG',
  });
  console.log('text:', JSON.stringify(r1.text));
  console.log('finishReason:', r1.finishReason);
  console.log(
    'responseId:',
    (r1.custom as { responseId?: string } | undefined)?.responseId
  );
  await assert(
    'plain text returned',
    typeof r1.text === 'string' && r1.text.trim().length > 0,
    r1
  );
  await assert(
    'finishReason is stop',
    r1.finishReason === 'stop',
    r1.finishReason
  );

  // ---- Smoke 2: web_search_preview returns citations on metadata ----
  console.log(`\n=== Smoke 2: web_search_preview citations ===`);
  const r2 = await ai.generate({
    model: openAI.responsesModel(MODEL),
    prompt:
      'In one sentence, what is the latest news about OpenAI ' +
      '(today or this week)? Cite at least one source.',
    config: {
      builtInTools: [{ type: 'web_search_preview' }],
    },
  });
  console.log('text:', r2.text);
  let foundCitation = false;
  for (const part of r2.message?.content ?? []) {
    const meta = part.metadata as
      | { citations?: Array<{ type: string; url?: string }> }
      | undefined;
    if (meta?.citations && meta.citations.length > 0) {
      foundCitation = true;
      console.log('citations:', JSON.stringify(meta.citations, null, 2));
      break;
    }
  }
  await assert(
    'at least one url_citation surfaced via metadata.citations',
    foundCitation,
    r2.message?.content
  );

  // ---- Smoke 3: streaming chunks arrive in order ----
  console.log(`\n=== Smoke 3: streaming ===`);
  const chunks: string[] = [];
  const r3 = await ai.generate({
    model: openAI.responsesModel(MODEL),
    prompt: 'Count from 1 to 5 inclusive, separated by commas. No prose.',
    onChunk: (chunk) => {
      const text = (chunk.content?.[0] as { text?: string } | undefined)?.text;
      if (text) {
        chunks.push(text);
      }
    },
  });
  console.log(`received ${chunks.length} streaming chunks`);
  console.log('final text:', r3.text);
  await assert('received >=2 streaming chunks', chunks.length >= 2, chunks);
  await assert(
    'streamed chunks reconstruct close to final text',
    r3.text?.includes('1') && r3.text?.includes('5'),
    r3.text
  );

  // ---- Smoke 4: previousResponseId chaining ----
  console.log(`\n=== Smoke 4: previousResponseId chaining ===`);
  const seed = await ai.generate({
    model: openAI.responsesModel(MODEL),
    prompt: 'My favorite color is teal. Acknowledge in one short sentence.',
    config: { store: true },
  });
  const seedId = (seed.custom as { responseId?: string } | undefined)
    ?.responseId;
  console.log('seed responseId:', seedId);
  await assert('seed turn carries responseId', !!seedId, seed);

  if (seedId) {
    const followup = await ai.generate({
      model: openAI.responsesModel(MODEL),
      prompt: 'What is my favorite color? Answer with one word only.',
      config: { previousResponseId: seedId, store: true },
    });
    console.log('followup text:', followup.text);
    await assert(
      'followup recalls "teal" via server-side state',
      /teal/i.test(followup.text ?? ''),
      followup.text
    );
  }

  if (process.exitCode === 1) {
    console.error('\nSMOKE FAILED — see ✗ markers above.');
  } else {
    console.log('\nSMOKE OK — all assertions passed.');
  }
}

main().catch((e) => {
  console.error('SMOKE THREW:', e);
  process.exit(1);
});
