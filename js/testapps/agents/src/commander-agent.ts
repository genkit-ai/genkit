/**
 * Copyright 2026 Google LLC
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
 * Incident Commander — demonstrates background delegation (`async: true`)
 *
 * The orchestrator sample delegates and waits: its turn is blocked for as long
 * as the sub-agent runs. This agent cannot afford that. It is an incident
 * commander, and an incident has two clocks that do not agree. The
 * investigation takes as long as it takes. The status update is owed now.
 *
 * So every investigation starts with `background: true`. The delegation tool
 * returns a task ID at once and the sub-agent keeps running after the tool
 * call returns, so the commander posts its first update while the
 * investigators are still reading logs. It collects the results afterwards
 * with `wait_for_background_tasks`: first with a short timeout, so a slow
 * investigation turns into an interim update instead of silence, then without
 * one.
 *
 * The parts worth reading:
 *
 *   - `async: true` on the middleware. It adds the `background` flag to every
 *     delegation tool and the three task tools that go with it: check, wait
 *     and abort.
 *   - Both sub-agents have a session store. A background delegation records a
 *     pending snapshot as the durable task, so a store is what makes an agent
 *     able to run in the background at all; a store-less agent is refused at
 *     launch and the commander is told to delegate to it synchronously.
 *   - The task ID (`<agent>:<snapshotId>`) comes back in the delegation tool
 *     result, so it lands in the conversation. Nothing else tracks the task,
 *     which is why a commander rebuilt from its history alone can still
 *     collect the answer.
 *   - Neither investigator is told anything about this conversation. History
 *     is never forwarded to a sub-agent that has a store, so the task text has
 *     to stand alone and each investigator pulls its own input with tools.
 *   - `read_status_board` is work the commander can do with the time
 *     background delegation gives it back. A blocking orchestrator has no such
 *     time: its turn is inside the sub-agent call.
 *   - `maxTurns`. Launching, posting, reading the board, waiting, posting and
 *     waiting again are all tool rounds, so an orchestrator that collects in
 *     the background needs a higher ceiling than one that blocks on each
 *     delegation.
 *
 * Try it with:
 *
 *     checkout-api is throwing 500s and customers can't pay
 *
 * Watch the order of the tool calls: two delegations, then post_status, then
 * the waits, with read_status_board filling the time in between. The task IDs
 * are visible in the wait tool's input.
 */

import { agents, retry } from '@genkit-ai/middleware';
import { z } from 'genkit';
import { InMemorySessionStore } from 'genkit/beta';
import { ai, defaultModel } from './genkit.js';

// ---------------------------------------------------------------------------
// The incident: checkout-api started failing at 14:03. The logs and the
// deploy history each hold one half of the answer, and neither investigator
// can reach the other's half.
// ---------------------------------------------------------------------------

const searchLogs = ai.defineTool(
  {
    name: 'searchLogs',
    description:
      'Searches the recent logs of a service and returns matching lines with timestamps.',
    inputSchema: z.object({
      service: z.string().describe('Service name, e.g. "checkout-api".'),
      query: z
        .string()
        .optional()
        .describe('Optional substring to filter log lines by.'),
    }),
    outputSchema: z.object({ lines: z.array(z.string()) }),
  },
  async ({ service, query }) => {
    // The investigation "takes as long as it takes": long enough for the
    // commander's first wait to time out and turn into an interim update.
    await new Promise((resolve) => setTimeout(resolve, 8_000));
    const lines = [
      `14:03:12 ${service} ERROR connection refused to payments-db-v2:5432 (x412 in 60s)`,
      `14:03:12 ${service} ERROR POST /checkout 500 upstream=payments-db-v2 timeout`,
      `14:02:58 ${service} INFO  config reloaded PAYMENTS_DB_HOST=payments-db-v2`,
      `13:59:40 ${service} INFO  POST /checkout 200 (p95 210ms)`,
    ];
    return {
      lines: query ? lines.filter((l) => l.includes(query)) : lines,
    };
  }
);

const listDeploys = ai.defineTool(
  {
    name: 'listDeploys',
    description:
      'Lists the recent deploys and configuration changes of a service, newest first.',
    inputSchema: z.object({
      service: z.string().describe('Service name, e.g. "checkout-api".'),
    }),
    outputSchema: z.object({ deploys: z.array(z.string()) }),
  },
  async ({ service }) => ({
    deploys: [
      `14:02 ${service} config change: PAYMENTS_DB_HOST payments-db -> payments-db-v2 (ticket INFRA-881, migration not yet cut over)`,
      `13:10 ${service} v2.41.0 deployed (cart rounding fix)`,
      `09:45 ${service} v2.40.3 deployed`,
    ],
  })
);

// ---------------------------------------------------------------------------
// The investigators. Each has its own session store: that is what lets the
// commander delegate to it in the background.
// ---------------------------------------------------------------------------

export const logAnalyst = ai.defineAgent({
  name: 'logAnalyst',
  description:
    "Reads a service's recent logs and reports what they show: the error signature, when it started, and the blast radius.",
  model: defaultModel,
  system: `You are an SRE log analyst. Use searchLogs to read the logs of the service you are asked about, then report in a few sentences: the error signature, when it started, what it points at, and how widespread it is. Report only what the logs show.`,
  tools: [searchLogs],
  maxTurns: 6,
  store: new InMemorySessionStore(),
  use: [retry()],
});

export const deployHistorian = ai.defineAgent({
  name: 'deployHistorian',
  description:
    'Checks what changed: recent deploys and configuration changes for a service, with timestamps.',
  model: defaultModel,
  system: `You are a release engineer. Use listDeploys to look up the recent changes of the service you are asked about, then report in a few sentences which changes landed shortly before the reported time and what each one touched.`,
  tools: [listDeploys],
  maxTurns: 6,
  store: new InMemorySessionStore(),
  use: [retry()],
});

// ---------------------------------------------------------------------------
// The commander's own tools: the status feed it owes updates to, and a board
// it can read while the investigators work.
// ---------------------------------------------------------------------------

/** Updates posted during the current incident, newest last. */
const statusLog: { at: string; update: string }[] = [];

const postStatus = ai.defineTool(
  {
    name: 'post_status',
    description:
      'Posts an update to the incident status feed. Post as soon as you know anything, and again whenever the picture changes.',
    inputSchema: z.object({
      update: z.string().describe('The status update, two sentences at most.'),
    }),
    outputSchema: z.object({ posted: z.boolean(), at: z.string() }),
  },
  async ({ update }) => {
    const at = new Date().toISOString();
    statusLog.push({ at, update });
    return { posted: true, at };
  }
);

const readStatusBoard = ai.defineTool(
  {
    name: 'read_status_board',
    description:
      'Reads the live status board: the latest signals from monitoring and support. Cheap to call; the board changes often.',
    inputSchema: z.object({}),
    outputSchema: z.object({ signals: z.array(z.string()) }),
  },
  async () => {
    const useful = [
      'Support: 37 tickets in the last 10 minutes, all "payment failed at checkout".',
      'Monitoring: checkout-api error rate 94%, other services nominal.',
      'Monitoring: payments-db (old host) healthy, 0 connections in the last minute.',
    ];
    const noise = [
      'Facilities: the 4th floor coffee machine is being descaled.',
      'Marketing: newsletter open rate up 2% week over week.',
      'IT: printer queue on floor 2 cleared.',
    ];
    const pick = (list: string[]) =>
      list[Math.floor(Math.random() * list.length)];
    return { signals: [pick(useful), pick(noise)] };
  }
);

// ---------------------------------------------------------------------------
// The commander. Delegates in the background, posts while it waits.
// ---------------------------------------------------------------------------

export const commanderAgent = ai.defineAgent({
  name: 'commanderAgent',
  model: defaultModel,
  system: `You are the incident commander for a production incident. You owe the status feed an update now, and a root cause as soon as one is known.

Work like this:
1. Immediately delegate two investigations IN THE BACKGROUND (set "background": true on each delegation): ask the log analyst what the logs of the affected service show, and ask the deploy historian what changed in that service recently. Each task description must stand alone: the investigators know nothing about this conversation.
2. Post a first status update with post_status right away, while they work.
3. Read the status board with read_status_board once or twice; use what is relevant, ignore what is not.
4. Collect the investigations with wait_for_background_tasks. Use a short timeoutSeconds (about 5) the first time; if some are still pending, post an interim update, then wait again without a timeout.
5. Post a final update with the root cause and the fix, then answer the user with the same.`,
  tools: [postStatus, readStatusBoard],
  maxTurns: 20,
  use: [
    agents({
      agents: ['logAnalyst', 'deployHistorian'],
      async: true,
    }),
    retry(),
  ],
});

// ---------------------------------------------------------------------------
// Test flow — runs one incident and returns the status feed with the answer.
// ---------------------------------------------------------------------------

export const testCommanderAgent = ai.defineFlow(
  {
    name: 'testCommanderAgent',
    inputSchema: z
      .string()
      .default("checkout-api is throwing 500s and customers can't pay"),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    statusLog.length = 0;
    const chat = commanderAgent.chat();
    const turn = chat.sendStream(text);
    for await (const chunk of turn.stream) {
      sendChunk(chunk.raw);
    }
    const res = await turn.response;
    return { answer: res.text, statusUpdates: [...statusLog] };
  }
);
