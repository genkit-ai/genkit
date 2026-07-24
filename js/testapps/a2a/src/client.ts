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
 * A2A client demo — drives the concierge agent using the *real* `@a2a-js/sdk`
 * client, proving end-to-end interop with the `@genkit-ai/a2a` server.
 *
 * It walks through the full feature matrix:
 *   1. Discover the agent card.
 *   2. Stream a planning turn (tool calls -> artifact updates -> completed).
 *   3. Trigger and resolve a human-in-the-loop interrupt
 *      (terminal `input-required` -> resume the same task -> completed).
 *   4. Fetch the resumed task with getTask (rebuilt from the agent snapshot).
 *   5. A follow-up turn on the same contextId to show session memory.
 *
 * Start the server first (`pnpm server`), then run `pnpm client`.
 */

import type {
  DataPart,
  Message,
  MessageSendParams,
  Part,
  Task,
  TaskArtifactUpdateEvent,
  TaskStatusUpdateEvent,
} from '@a2a-js/sdk';
import { A2AClient } from '@a2a-js/sdk/client';

const SERVER_URL = process.env.A2A_URL ?? 'http://localhost:4000';

/** Generates a client-side message id. */
function newId(): string {
  return crypto.randomUUID();
}

/** Renders an array of A2A parts as a short human-readable string. */
function renderParts(parts: Part[]): string {
  return parts
    .map((p) => {
      if (p.kind === 'text') return p.text;
      if (p.kind === 'data') {
        const type = (p.metadata as Record<string, unknown> | undefined)?.[
          'genkit:type'
        ];
        return `[${type ?? 'data'} ${JSON.stringify(p.data)}]`;
      }
      if (p.kind === 'file') return '[file]';
      return '';
    })
    .join('');
}

/**
 * Consumes a streaming send, printing events as they arrive. Returns the
 * terminal task (the one carried by the final status-update event), if any.
 */
async function streamTurn(
  client: A2AClient,
  params: MessageSendParams,
  label: string
): Promise<Task | undefined> {
  console.log(`\n=== ${label} ===`);
  console.log(`> ${renderParts(params.message.parts)}`);

  let lastTask: Task | undefined;
  let streamedText = '';

  for await (const event of client.sendMessageStream(params)) {
    switch (event.kind) {
      case 'task': {
        lastTask = event as Task;
        console.log(
          `  · task ${lastTask.id.slice(0, 8)} [${lastTask.status.state}]`
        );
        break;
      }
      case 'status-update': {
        const e = event as TaskStatusUpdateEvent;
        console.log(
          `  · status -> ${e.status.state}${e.final ? ' (final)' : ''}`
        );
        if (e.status.message) {
          const text = renderParts(e.status.message.parts);
          if (text) console.log(`    message: ${text}`);
        }
        if (lastTask) lastTask = { ...lastTask, status: e.status };
        break;
      }
      case 'artifact-update': {
        const e = event as TaskArtifactUpdateEvent;
        const chunk = renderParts(e.artifact.parts);
        streamedText += chunk;
        process.stdout.write(chunk);
        break;
      }
      default:
        break;
    }
  }
  if (streamedText) console.log();
  return lastTask;
}

/**
 * Finds the interrupt tool-request data part carried in an `input-required`
 * status message (tagged `genkit:interrupt`).
 */
function findInterrupt(task: Task | undefined): DataPart | undefined {
  const parts = task?.status.message?.parts ?? [];
  return parts.find(
    (p): p is DataPart =>
      p.kind === 'data' &&
      (p.metadata as Record<string, unknown> | undefined)?.[
        'genkit:interrupt'
      ] !== undefined
  );
}

async function main() {
  // 1. Discover the agent.
  const client = await A2AClient.fromCardUrl(
    `${SERVER_URL}/.well-known/agent-card.json`
  );
  const card = await client.getAgentCard();
  console.log(`Connected to "${card.name}" v${card.version}`);
  console.log(`  ${card.description}`);

  // A shared contextId ties all turns into one server-side session.
  const contextId = newId();

  // 2. Plan a trip — exercises tool calls + streamed text.
  await streamTurn(
    client,
    {
      message: {
        kind: 'message',
        messageId: newId(),
        role: 'user',
        contextId,
        parts: [
          {
            kind: 'text',
            text: 'I want to plan a trip to Tokyo. What is the weather and what flights are available?',
          },
        ],
      },
    },
    'Turn 1: plan the trip'
  );

  // 3. Ask to book — the agent pauses on the confirmBooking interrupt.
  const bookingTask = await streamTurn(
    client,
    {
      message: {
        kind: 'message',
        messageId: newId(),
        role: 'user',
        contextId,
        parts: [
          {
            kind: 'text',
            text: 'Great, book me the cheapest flight on Genkit Air.',
          },
        ],
      },
    },
    'Turn 2: request booking (expect input-required)'
  );

  const interrupt = findInterrupt(bookingTask);
  if (!interrupt || !bookingTask) {
    console.log(
      '\n⚠️  Expected an interrupt but none arrived; the model may have skipped confirmation.'
    );
    return;
  }

  // The interrupt data part is the paused toolRequest: { name, ref, input }.
  const toolRequest = interrupt.data as {
    name: string;
    ref?: string;
    input?: unknown;
  };
  console.log(
    `\n🙋 Agent is asking to confirm: ${JSON.stringify(toolRequest.input)}`
  );

  // 4. Resume the SAME task with a tool-response data part (the resolved
  //    interrupt). Tagging it `genkit:type=toolResponse` tells the server to
  //    feed it back into the agent's `resume.respond`.
  const resumeMessage: Message = {
    kind: 'message',
    messageId: newId(),
    role: 'user',
    contextId,
    taskId: bookingTask.id, // <-- target the paused task
    parts: [
      {
        kind: 'data',
        data: {
          name: toolRequest.name,
          ref: toolRequest.ref,
          output: { confirmed: true, note: 'Looks great, book it!' },
        },
        metadata: { 'genkit:type': 'toolResponse' },
      },
    ],
  };
  await streamTurn(
    client,
    { message: resumeMessage },
    'Turn 3: resume with approval (expect completed)'
  );

  // 5. Inspect the task via getTask. For a server-managed agent the task id is
  //    the originating turn's Genkit snapshot id, and the resume advanced its
  //    pointer to a new snapshot. The handler resolves that and rebuilds the
  //    Task straight from the agent's SessionStore — no separate task copy.
  console.log('\n=== getTask: inspect the resumed task ===');
  const response = await client.getTask({ id: bookingTask.id });
  if ('error' in response) {
    console.log(`  ⚠️  getTask failed: ${response.error.message}`);
  } else {
    const fetched = response.result;
    console.log(
      `  · task ${fetched.id.slice(0, 8)} [${fetched.status.state}], ` +
        `${fetched.history?.length ?? 0} history message(s)`
    );
    const lastMsg = fetched.history?.at(-1);
    if (lastMsg) {
      console.log(`    last: ${lastMsg.role}: ${renderParts(lastMsg.parts)}`);
    }
  }

  // 6. Follow-up turn on the same context — shows session memory.
  await streamTurn(
    client,
    {
      message: {
        kind: 'message',
        messageId: newId(),
        role: 'user',
        contextId,
        parts: [
          {
            kind: 'text',
            text: 'Remind me where I am flying and on which airline.',
          },
        ],
      },
    },
    'Turn 4: follow-up (session memory)'
  );

  console.log('\n✅ Demo complete.');
}

main().catch((err) => {
  console.error('\n❌ Client error:', err);
  process.exitCode = 1;
});
