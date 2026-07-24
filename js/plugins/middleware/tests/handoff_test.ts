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

import * as assert from 'assert';
import { z } from 'genkit';
import { genkit } from 'genkit/beta';
import { describe, it } from 'node:test';
import { handoff } from '../src/handoff.js';

const triagePersona = {
  name: 'triage',
  description: 'Figures out the issue and routes the user.',
  system: 'You triage support requests.',
};
const refundPersona = {
  name: 'refund',
  description: 'Handles refund requests.',
  system: 'You handle refunds.',
  tools: ['lookupOrder', 'issueRefund'],
};
const billingPersona = {
  name: 'billing',
  description: 'Handles billing questions.',
  system: 'You handle billing.',
  tools: ['getInvoice'],
};

/** Registers no-op tools used by personas so their names resolve. */
function defineSupportTools(ai: ReturnType<typeof genkit>) {
  ai.defineTool(
    {
      name: 'lookupOrder',
      description: 'Look up an order.',
      inputSchema: z.object({ id: z.string() }),
      outputSchema: z.string(),
    },
    async () => 'order-ok'
  );
  ai.defineTool(
    {
      name: 'issueRefund',
      description: 'Issue a refund.',
      inputSchema: z.object({ id: z.string() }),
      outputSchema: z.string(),
    },
    async () => 'refunded'
  );
  ai.defineTool(
    {
      name: 'getInvoice',
      description: 'Get an invoice.',
      inputSchema: z.object({ id: z.string() }),
      outputSchema: z.string(),
    },
    async () => 'invoice'
  );
}

function systemText(req: { messages?: any[] }): string {
  const sys = req.messages?.find((m) => m.role === 'system');
  return (sys?.content ?? []).map((p: any) => p.text ?? '').join('\n');
}

describe('handoff middleware', () => {
  it('injects transfer tools and the default persona system prompt', async () => {
    const ai = genkit({});
    defineSupportTools(ai);

    let capturedSystem = '';
    let capturedTools: string[] | undefined;
    const model = ai.defineModel(
      { name: 'handoff-default-' + Math.random() },
      async (req) => {
        capturedSystem = systemText(req);
        capturedTools = req.tools?.map((t: any) => t.name);
        return {
          message: { role: 'model' as const, content: [{ text: 'hello' }] },
        };
      }
    );

    await ai.generate({
      model,
      prompt: 'hi',
      use: [
        handoff({
          personas: [triagePersona, refundPersona, billingPersona],
          defaultPersona: 'triage',
        }),
      ],
    });

    // Default persona is triage.
    assert.ok(
      capturedSystem.includes('"triage"'),
      'System should activate the triage persona'
    );
    assert.ok(
      capturedSystem.includes('You triage support requests.'),
      'System should contain triage instructions'
    );
    // Transfer tools to other personas are listed.
    assert.ok(capturedSystem.includes('transfer_to_refund'));
    assert.ok(capturedSystem.includes('transfer_to_billing'));

    // All transfer tools are available; no persona business tools exposed yet
    // (triage has none).
    assert.ok(capturedTools?.includes('transfer_to_refund'));
    assert.ok(capturedTools?.includes('transfer_to_triage'));
    assert.ok(
      !capturedTools?.includes('issueRefund'),
      'Refund tools should not be exposed while triage is active'
    );
  });

  it('swaps system prompt and tools after a transfer', async () => {
    const ai = genkit({});
    defineSupportTools(ai);

    const turns: { system: string; tools: string[] }[] = [];
    let turn = 0;
    const model = ai.defineModel(
      { name: 'handoff-swap-' + Math.random() },
      async (req) => {
        turn++;
        turns.push({
          system: systemText(req),
          tools: (req.tools ?? []).map((t: any) => t.name),
        });
        if (turn === 1) {
          // Triage decides to transfer to refund.
          return {
            message: {
              role: 'model' as const,
              content: [
                {
                  toolRequest: {
                    name: 'transfer_to_refund',
                    input: { reason: 'user wants a refund' },
                  },
                },
              ],
            },
          };
        }
        // After the transfer, the refund persona responds.
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'I can help with your refund.' }],
          },
        };
      }
    );

    const res = await ai.generate({
      model,
      prompt: 'I want my money back',
      use: [
        handoff({
          personas: [triagePersona, refundPersona, billingPersona],
          defaultPersona: 'triage',
        }),
      ],
    });

    assert.strictEqual(turns.length, 2, 'Should have two model turns');

    // Turn 1: triage active, no refund tools.
    assert.ok(turns[0].system.includes('"triage"'));
    assert.ok(!turns[0].tools.includes('issueRefund'));

    // Turn 2: refund active, refund tools exposed, transfer tools still there.
    assert.ok(
      turns[1].system.includes('"refund"'),
      'After transfer the refund persona should be active'
    );
    assert.ok(turns[1].system.includes('You handle refunds.'));
    assert.ok(
      turns[1].tools.includes('lookupOrder'),
      'Refund persona tools should now be exposed'
    );
    assert.ok(turns[1].tools.includes('issueRefund'));
    assert.ok(
      turns[1].tools.includes('transfer_to_triage'),
      'Transfer tools should remain available so the persona can hand back'
    );
    // Billing's tool should still be hidden.
    assert.ok(!turns[1].tools.includes('getInvoice'));

    assert.ok(res.text.includes('refund'));
  });

  it('supports handing back to the default persona', async () => {
    const ai = genkit({});
    defineSupportTools(ai);

    const systems: string[] = [];
    let turn = 0;
    const model = ai.defineModel(
      { name: 'handoff-back-' + Math.random() },
      async (req) => {
        turn++;
        systems.push(systemText(req));
        if (turn === 1) {
          return {
            message: {
              role: 'model' as const,
              content: [
                { toolRequest: { name: 'transfer_to_refund', input: {} } },
              ],
            },
          };
        }
        if (turn === 2) {
          // Refund hands back to triage.
          return {
            message: {
              role: 'model' as const,
              content: [
                { toolRequest: { name: 'transfer_to_triage', input: {} } },
              ],
            },
          };
        }
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'Anything else?' }],
          },
        };
      }
    );

    await ai.generate({
      model,
      prompt: 'refund then something else',
      use: [
        handoff({
          personas: [triagePersona, refundPersona, billingPersona],
        }),
      ],
    });

    assert.strictEqual(systems.length, 3);
    assert.ok(systems[0].includes('"triage"'), 'turn 1 triage');
    assert.ok(systems[1].includes('"refund"'), 'turn 2 refund');
    assert.ok(
      systems[2].includes('"triage"'),
      'turn 3 should be back to triage'
    );
  });

  it('persists the active persona across separate generate calls (multi-turn)', async () => {
    const ai = genkit({});
    defineSupportTools(ai);

    // Simulate history where a transfer to billing already happened.
    const priorMessages = [
      { role: 'user' as const, content: [{ text: 'I have a billing issue' }] },
      {
        role: 'model' as const,
        content: [
          {
            toolRequest: {
              name: 'transfer_to_billing',
              ref: '1',
              input: {},
            },
          },
        ],
      },
      {
        role: 'tool' as const,
        content: [
          {
            toolResponse: {
              name: 'transfer_to_billing',
              ref: '1',
              output: {
                transferred: true,
                to: 'billing',
                message: 'Transferred to the "billing" agent.',
              },
            },
          },
        ],
      },
      {
        role: 'model' as const,
        content: [{ text: 'I can help with billing.' }],
      },
    ];

    let capturedSystem = '';
    let capturedTools: string[] = [];
    const model = ai.defineModel(
      { name: 'handoff-persist-' + Math.random() },
      async (req) => {
        capturedSystem = systemText(req);
        capturedTools = (req.tools ?? []).map((t: any) => t.name);
        return {
          message: {
            role: 'model' as const,
            content: [{ text: 'here is your invoice' }],
          },
        };
      }
    );

    await ai.generate({
      model,
      messages: [
        ...priorMessages,
        { role: 'user', content: [{ text: 'what is my latest invoice?' }] },
      ],
      use: [
        handoff({
          personas: [triagePersona, refundPersona, billingPersona],
          defaultPersona: 'triage',
        }),
      ],
    });

    assert.ok(
      capturedSystem.includes('"billing"'),
      'Billing persona should remain active based on history'
    );
    assert.ok(capturedTools.includes('getInvoice'));
    assert.ok(!capturedTools.includes('issueRefund'));
  });

  it('enforces maxTransfers', async () => {
    const ai = genkit({});
    defineSupportTools(ai);

    let turn = 0;
    const model = ai.defineModel(
      { name: 'handoff-max-' + Math.random() },
      async () => {
        turn++;
        if (turn <= 3) {
          // Keep bouncing between personas.
          const target = turn % 2 === 1 ? 'refund' : 'triage';
          return {
            message: {
              role: 'model' as const,
              content: [
                { toolRequest: { name: `transfer_to_${target}`, input: {} } },
              ],
            },
          };
        }
        return {
          message: { role: 'model' as const, content: [{ text: 'done' }] },
        };
      }
    );

    const res = await ai.generate({
      model,
      prompt: 'bounce around',
      use: [
        handoff({
          personas: [triagePersona, refundPersona],
          maxTransfers: 2,
        }),
      ],
    });

    const toolMsgs = res.messages.filter((m) => m.role === 'tool');
    const limitHit = toolMsgs.some((m) =>
      m.content.some((p) => {
        const out = p.toolResponse?.output as { message?: string };
        return out?.message?.includes('Transfer limit reached');
      })
    );
    assert.ok(limitHit, 'Should hit the transfer limit');
  });

  it('supports a custom tool prefix', async () => {
    const ai = genkit({});
    defineSupportTools(ai);

    let capturedSystem = '';
    let capturedTools: string[] = [];
    const model = ai.defineModel(
      { name: 'handoff-prefix-' + Math.random() },
      async (req) => {
        capturedSystem = systemText(req);
        capturedTools = (req.tools ?? []).map((t: any) => t.name);
        return {
          message: { role: 'model' as const, content: [{ text: 'ok' }] },
        };
      }
    );

    await ai.generate({
      model,
      prompt: 'hi',
      use: [
        handoff({
          personas: [triagePersona, refundPersona],
          toolPrefix: 'route_to',
        }),
      ],
    });

    assert.ok(capturedSystem.includes('route_to_refund'));
    assert.ok(capturedTools.includes('route_to_refund'));
  });

  it('throws if no personas are provided', () => {
    const ai = genkit({});
    assert.throws(() => {
      handoff.instantiate({
        config: { personas: [] },
        ai,
        pluginConfig: undefined,
      });
    }, /at least one persona/);
  });

  it('throws if defaultPersona is not a configured persona', () => {
    const ai = genkit({});
    assert.throws(() => {
      handoff.instantiate({
        config: {
          personas: [triagePersona],
          defaultPersona: 'nope',
        },
        ai,
        pluginConfig: undefined,
      });
    }, /defaultPersona/);
  });
});
