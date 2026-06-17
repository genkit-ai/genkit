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
import { genkit, z } from 'genkit';
import { describe, it } from 'node:test';
import { toolApproval } from '../src/tool-approval.js';

describe('toolApproval middleware', () => {
  function createToolModel(ai: any, toolName: string) {
    let turn = 0;
    return ai.defineModel(
      { name: `pm-${toolName}-${Math.random()}` },
      async () => {
        turn++;
        if (turn === 1) {
          return {
            message: {
              role: 'model',
              content: [{ toolRequest: { name: toolName, input: {} } }],
            },
          };
        }
        return { message: { role: 'model', content: [{ text: 'done' }] } };
      }
    );
  }

  it('allows approved tools to execute', async () => {
    const ai = genkit({});

    // Register a tool
    const mockTool = ai.defineTool(
      {
        name: 'good_tool',
        description: 'a tool',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => 'good'
    );

    const pm = createToolModel(ai, 'good_tool');

    const result = (await ai.generate({
      model: pm,
      prompt: 'use tool',
      tools: [mockTool],
      use: [toolApproval({ approved: ['good_tool'] })],
    })) as any;

    const toolMsg = result.messages.find((m: any) => m.role === 'tool');
    assert.ok(toolMsg);
    assert.match(toolMsg.content[0].toolResponse.output, /good/);
    assert.notStrictEqual(result.finishReason, 'interrupted');
  });

  it('interrupts unapproved tools', async () => {
    const ai = genkit({});

    const mockTool = ai.defineTool(
      {
        name: 'bad_tool',
        description: 'a tool',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => 'bad'
    );

    const pm = createToolModel(ai, 'bad_tool');

    const result = (await ai.generate({
      model: pm,
      prompt: 'use tool',
      tools: [mockTool],
      use: [toolApproval({ approved: ['good_tool'] })],
    })) as any;

    // It should be interrupted
    assert.strictEqual(result.finishReason, 'interrupted');
    const interruptPart = result.message?.content.find(
      (p: any) => p.metadata?.interrupt
    );
    assert.ok(interruptPart);
    assert.match(
      interruptPart.metadata.interrupt.message,
      /Tool not in approved list/
    );
  });

  it('allows unapproved tools if running on resumed execution', async () => {
    const ai = genkit({});

    const mockTool = ai.defineTool(
      {
        name: 'bad_tool_approved',
        description: 'a tool',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => 'bad_but_approved'
    );

    let turn = 0;
    const pm = ai.defineModel({ name: 'pm-' + Math.random() }, async () => {
      turn++;
      if (turn === 1) {
        return {
          message: {
            role: 'model',
            content: [
              {
                toolRequest: { name: 'bad_tool_approved', input: {} },
              },
            ],
          },
        };
      }
      return { message: { role: 'model', content: [{ text: 'done' }] } };
    });

    // First call, should be interrupted because it's not approved
    const response1 = (await ai.generate({
      model: pm,
      prompt: 'use tool',
      tools: [mockTool],
      use: [toolApproval({ approved: ['good_tool'] })],
    })) as any;

    assert.strictEqual(response1.finishReason, 'interrupted');

    // Get the tool request part that was interrupted
    const interruptPart = response1.message?.content.find(
      (p: any) => p.toolRequest && p.toolRequest.name === 'bad_tool_approved'
    );
    assert.ok(interruptPart);

    // Call ToolAction.restart to prepare for resume
    const restartedPart = mockTool.restart(interruptPart, {
      toolApproved: true,
    });

    // Second call, should execute because it's resumed
    const response2 = (await ai.generate({
      model: pm,
      messages: response1.messages, // pass history
      resume: { restart: [restartedPart] },
      tools: [mockTool],
      use: [toolApproval({ approved: ['good_tool'] })],
    })) as any;

    assert.notStrictEqual(response2.finishReason, 'interrupted');
    const toolMsg = response2.messages.find((m: any) => m.role === 'tool');
    assert.ok(toolMsg);
    assert.match(toolMsg.content[0].toolResponse.output, /bad_but_approved/);
  });
});
