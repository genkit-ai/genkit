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

/**
 * Agent conformance test runner.
 *
 * Reads the shared spec from tests/specs/agent.yaml and executes each test
 * case against harness-provided agent implementations. See
 * docs/agents-conformance-testing.md for the full spec format reference and
 * harness requirements.
 */

import { stripUndefinedProps, z } from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import * as assert from 'assert';
import { readFileSync } from 'fs';
import { beforeEach, describe, it } from 'node:test';
import { parse } from 'yaml';

import {
  defineAgent,
  defineCustomAgent,
  type Agent,
  type AgentStreamChunk,
} from '../src/session-flow.js';
import { InMemorySessionStore } from '../src/session.js';
import { defineTool, interrupt } from '../src/tool.js';
import { defineProgrammableModel, type ProgrammableModel } from './helpers.js';

initNodeFeatures();

// ---------------------------------------------------------------------------
// Spec parsing & validation
// ---------------------------------------------------------------------------

// These Zod schemas mirror the canonical definitions in
// genkit-tools/common/src/types/agents-conformance.ts. We define them
// locally here because js/ai does not depend on genkit-tools/common.

const OutputAssertionsSchema = z.object({
  message: z.any().optional(),
  hasSnapshotId: z.boolean().optional(),
  stateContains: z.any().optional(),
  artifactsContain: z.array(z.any()).optional(),
});

const SnapshotAssertionsSchema = z.object({
  parentId: z.string().optional(),
  status: z.string().optional(),
  stateContains: z.any().optional(),
  errorContains: z.any().optional(),
});

const SendInvocationSchema = z.object({
  type: z.literal('send'),
  init: z.any().optional(),
  inputs: z.array(z.any()).optional(),
  modelResponses: z.array(z.any()).optional(),
  streamChunks: z.array(z.array(z.any())).optional(),
  expectChunks: z.array(z.any()).optional(),
  expectOutput: OutputAssertionsSchema.optional(),
  captureSnapshotId: z.string().optional(),
  captureState: z.string().optional(),
});

const GetSnapshotDataInvocationSchema = z.object({
  type: z.literal('getSnapshotData'),
  snapshotId: z.string(),
  expectSnapshot: SnapshotAssertionsSchema.optional(),
});

const AbortInvocationSchema = z.object({
  type: z.literal('abort'),
  snapshotId: z.string(),
  // YAML `~` (null) means "expect undefined/absent". A string value means
  // "expect exactly this status".
  expectPreviousStatus: z.string().nullable().optional(),
});

const WaitUntilCompletedInvocationSchema = z.object({
  type: z.literal('waitUntilCompleted'),
  snapshotId: z.string(),
  timeoutMs: z.number().optional(),
  expectSnapshot: SnapshotAssertionsSchema.optional(),
});

const SpecStepSchema = z.discriminatedUnion('type', [
  SendInvocationSchema,
  GetSnapshotDataInvocationSchema,
  AbortInvocationSchema,
  WaitUntilCompletedInvocationSchema,
]);

const SpecTestSchema = z.object({
  name: z.string(),
  description: z.string().optional(),
  agent: z.string(),
  steps: z.array(SpecStepSchema),
});

const SpecSuiteSchema = z.object({
  tests: z.array(SpecTestSchema),
});

type SendInvocation = z.infer<typeof SendInvocationSchema>;
type SpecStep = z.infer<typeof SpecStepSchema>;

// ---------------------------------------------------------------------------
// Template resolution
// ---------------------------------------------------------------------------

/**
 * Recursively resolves `{{name}}` template references in any value using the
 * provided captures map. Only simple `{{name}}` syntax is supported.
 */
function resolveTemplates(value: any, captures: Map<string, any>): any {
  if (typeof value === 'string') {
    const match = value.match(/^\{\{(\w+)\}\}$/);
    if (match) {
      const name = match[1];
      if (!captures.has(name)) {
        throw new Error(
          `Template reference '{{${name}}}' not found in captures`
        );
      }
      return captures.get(name);
    }
    // Also handle inline template replacements (partial string)
    return value.replace(/\{\{(\w+)\}\}/g, (_, name) => {
      if (!captures.has(name)) {
        throw new Error(
          `Template reference '{{${name}}}' not found in captures`
        );
      }
      const v = captures.get(name);
      return typeof v === 'string' ? v : JSON.stringify(v);
    });
  }
  if (Array.isArray(value)) {
    return value.map((item) => resolveTemplates(item, captures));
  }
  if (value && typeof value === 'object') {
    const resolved: Record<string, any> = {};
    for (const [k, v] of Object.entries(value)) {
      resolved[k] = resolveTemplates(v, captures);
    }
    return resolved;
  }
  return value;
}

// ---------------------------------------------------------------------------
// "Contains" assertion helpers
// ---------------------------------------------------------------------------

/**
 * Asserts that `actual` contains all fields specified in `expected`.
 * For arrays (like messages), asserts that all expected items appear in order
 * as a contiguous subsequence within the actual array.
 */
function assertContains(actual: any, expected: any, path: string = ''): void {
  if (expected === undefined || expected === null) return;

  if (Array.isArray(expected)) {
    assert.ok(
      Array.isArray(actual),
      `Expected array at ${path}, got ${typeof actual}`
    );
    // Find the expected items as a contiguous subsequence in actual
    assertContainsSubsequence(actual, expected, path);
    return;
  }

  if (typeof expected === 'object') {
    assert.ok(
      actual && typeof actual === 'object',
      `Expected object at ${path}, got ${typeof actual}`
    );
    for (const [key, val] of Object.entries(expected)) {
      assertContains(actual[key], val, `${path}.${key}`);
    }
    return;
  }

  assert.deepStrictEqual(
    actual,
    expected,
    `Mismatch at ${path}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`
  );
}

/**
 * Asserts that all items in `expected` appear in `actual` in order.
 * Each expected item is matched against actual items using deep-contains logic.
 * Items must appear in the same relative order but need not be contiguous.
 */
function assertContainsSubsequence(
  actual: any[],
  expected: any[],
  path: string
): void {
  let actualIdx = 0;
  for (let i = 0; i < expected.length; i++) {
    let found = false;
    while (actualIdx < actual.length) {
      try {
        assertContains(actual[actualIdx], expected[i], `${path}[${actualIdx}]`);
        found = true;
        actualIdx++;
        break;
      } catch {
        actualIdx++;
      }
    }
    if (!found) {
      assert.fail(
        `Expected item at ${path}[${i}] not found in actual array.\n` +
          `  Expected: ${JSON.stringify(expected[i])}\n` +
          `  Actual array: ${JSON.stringify(actual)}`
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Strip helpers
// ---------------------------------------------------------------------------

/**
 * Recursively strips undefined values from an object/array for clean
 * comparison. Note: `null` is preserved — stripping it could mask real
 * differences (e.g. a field explicitly set to `null` vs absent).
 */
function deepStrip(value: any): any {
  if (value === undefined) return undefined;
  if (value === null) return null;
  if (Array.isArray(value)) {
    return value.map(deepStrip);
  }
  if (typeof value === 'object') {
    const out: Record<string, any> = {};
    for (const [k, v] of Object.entries(value)) {
      const stripped = deepStrip(v);
      if (stripped !== undefined) {
        out[k] = stripped;
      }
    }
    return out;
  }
  return value;
}

// ---------------------------------------------------------------------------
// Harness setup
// ---------------------------------------------------------------------------

interface HarnessAgents {
  [name: string]: Agent;
}

function setupHarness(
  registry: Registry,
  pm: ProgrammableModel
): HarnessAgents {
  // --- Tools ---
  defineTool(
    registry,
    { name: 'testTool', description: 'A simple test tool' },
    async () => 'tool called'
  );

  // interruptTool is registered via the `interrupt()` helper rather than
  // `defineTool()` because it uses the interrupt mechanism (pausing execution
  // and returning the tool request to the client for external resolution).
  // The helper returns an action definition that must be manually registered.
  const interruptToolDef = interrupt({
    name: 'interruptTool',
    description: 'An interrupt tool',
    inputSchema: z.object({ query: z.string() }),
    outputSchema: z.object({ answer: z.string() }),
  });
  registry.registerAction('tool', interruptToolDef);

  // --- Agents ---

  // promptAgent: client-managed, no tools
  const promptAgent = defineAgent(registry, {
    name: 'promptAgent',
    model: 'programmableModel',
    config: { temperature: 1 },
  });

  // promptAgentWithStore: server-managed
  const promptAgentWithStore = defineAgent(registry, {
    name: 'promptAgentWithStore',
    model: 'programmableModel',
    config: { temperature: 1 },
    store: new InMemorySessionStore(),
  });

  // promptAgentWithTools: client-managed, with testTool
  const promptAgentWithTools = defineAgent(registry, {
    name: 'promptAgentWithTools',
    model: 'programmableModel',
    config: { temperature: 1 },
    tools: ['testTool'],
  });

  // promptAgentWithInterrupt: server-managed, with interruptTool
  const promptAgentWithInterrupt = defineAgent(registry, {
    name: 'promptAgentWithInterrupt',
    model: 'programmableModel',
    config: { temperature: 1 },
    tools: ['interruptTool'],
    store: new InMemorySessionStore(),
  });

  // --- Phase 2: Custom agents for detach, abort, artifacts, state ---

  // customAgentBlocking: server-managed, blocks until abort signal fires.
  // Used for abort-while-pending tests.
  const customAgentBlocking = defineCustomAgent(
    registry,
    {
      name: 'customAgentBlocking',
      store: new InMemorySessionStore(),
    },
    async (sess, { abortSignal }) => {
      await sess.run(async () => {
        await new Promise<void>((resolve) => {
          if (abortSignal?.aborted) {
            resolve();
            return;
          }
          abortSignal?.addEventListener('abort', () => resolve(), {
            once: true,
          });
        });
      });
      return {
        message: { role: 'model', content: [{ text: 'unblocked' }] },
      };
    }
  );

  // customAgentFailing: server-managed, throws during processing.
  // Used for detach + background failure tests.
  const customAgentFailing = defineCustomAgent(
    registry,
    {
      name: 'customAgentFailing',
      store: new InMemorySessionStore(),
    },
    async (sess) => {
      await sess.run(async () => {
        throw new Error('intentional failure');
      });
      return {
        message: { role: 'model', content: [{ text: 'unreachable' }] },
      };
    }
  );

  // customAgentWithArtifacts: client-managed, adds and updates artifacts.
  const customAgentWithArtifacts = defineCustomAgent(
    registry,
    {
      name: 'customAgentWithArtifacts',
    },
    async (sess) => {
      await sess.run(async () => {
        sess.session.addArtifacts([{ name: 'doc1', parts: [{ text: 'v1' }] }]);
        sess.session.addArtifacts([{ name: 'doc1', parts: [{ text: 'v2' }] }]);
        sess.session.addArtifacts([
          { name: 'doc2', parts: [{ text: 'other' }] },
        ]);
      });
      return {
        artifacts: sess.session.getArtifacts(),
        message: { role: 'model', content: [{ text: 'done' }] },
      };
    }
  );

  // customAgentWithCustomState: client-managed, increments a counter on
  // each turn. Useful for verifying custom state persistence across
  // invocations.
  const customAgentWithCustomState = defineCustomAgent(
    registry,
    {
      name: 'customAgentWithCustomState',
    },
    async (sess) => {
      await sess.run(async () => {
        const prev = (sess.session.getCustom() as any) || {};
        const counter = (prev.counter || 0) + 1;
        sess.session.updateCustom(() => ({ counter }));
      });
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
      };
    }
  );

  return {
    promptAgent,
    promptAgentWithStore,
    promptAgentWithTools,
    promptAgentWithInterrupt,
    customAgentBlocking,
    customAgentFailing,
    customAgentWithArtifacts,
    customAgentWithCustomState,
  };
}

// ---------------------------------------------------------------------------
// Invocation executors
// ---------------------------------------------------------------------------

async function executeSendInvocation(
  agent: Agent,
  pm: ProgrammableModel,
  invocation: SendInvocation,
  captures: Map<string, any>
): Promise<void> {
  const resolvedInvocation = resolveTemplates(invocation, captures);

  // Program the model
  if (resolvedInvocation.modelResponses || resolvedInvocation.streamChunks) {
    let reqCounter = 0;
    pm.handleResponse = async (req, sc) => {
      if (resolvedInvocation.streamChunks?.[reqCounter] && sc) {
        for (const chunk of resolvedInvocation.streamChunks[reqCounter]) {
          sc(chunk);
        }
      }
      return resolvedInvocation.modelResponses?.[reqCounter++]!;
    };
  }

  // Start the bidi session
  const init = resolvedInvocation.init || {};
  const session = agent.streamBidi(init);

  // Send all inputs
  const inputs = resolvedInvocation.inputs || [];
  for (const input of inputs) {
    session.send(input);
  }
  session.close();

  // Collect stream chunks
  const chunks: AgentStreamChunk[] = [];
  for await (const chunk of session.stream) {
    chunks.push(chunk);
  }

  // Get the output
  const output = await session.output;

  // --- Assertions ---

  // expectChunks: strict ordered comparison
  if (resolvedInvocation.expectChunks) {
    const strippedChunks = chunks.map((c) => deepStrip(stripUndefinedProps(c)));
    const expectedChunks = resolvedInvocation.expectChunks.map((c: any) =>
      deepStrip(c)
    );

    // For strict comparison, we compare each expected chunk against actual.
    // turnEnd chunks may contain dynamic snapshotIds, so we do field-level
    // matching: if the expected chunk has a field, it must match; extra fields
    // in actual are allowed for turnEnd only.
    assert.strictEqual(
      strippedChunks.length,
      expectedChunks.length,
      `Expected ${expectedChunks.length} chunks, got ${strippedChunks.length}.\n` +
        `  Actual: ${JSON.stringify(strippedChunks)}\n` +
        `  Expected: ${JSON.stringify(expectedChunks)}`
    );

    for (let i = 0; i < expectedChunks.length; i++) {
      const expected = expectedChunks[i];
      const actual = strippedChunks[i];

      if (expected.turnEnd !== undefined) {
        // turnEnd: verify turnEnd key exists, but snapshotId is dynamic
        assert.ok(
          actual.turnEnd !== undefined,
          `Chunk ${i}: expected turnEnd, got ${JSON.stringify(actual)}`
        );
      } else if (expected.modelChunk !== undefined) {
        assertContains(
          actual.modelChunk,
          expected.modelChunk,
          `chunk[${i}].modelChunk`
        );
      } else if (expected.artifact !== undefined) {
        assertContains(
          actual.artifact,
          expected.artifact,
          `chunk[${i}].artifact`
        );
      } else {
        // Generic deep comparison for other chunk types
        assertContains(actual, expected, `chunk[${i}]`);
      }
    }
  }

  // expectOutput
  if (resolvedInvocation.expectOutput) {
    const expect = resolvedInvocation.expectOutput;

    // message: strict comparison
    if (expect.message) {
      assertContains(
        deepStrip(stripUndefinedProps(output.message)),
        deepStrip(expect.message),
        'output.message'
      );
    }

    // hasSnapshotId
    if (expect.hasSnapshotId) {
      assert.ok(
        output.snapshotId && typeof output.snapshotId === 'string',
        `Expected output to have a snapshotId, got: ${output.snapshotId}`
      );
    }

    // stateContains: partial matching
    if (expect.stateContains) {
      assert.ok(output.state, 'Expected output to have state');
      assertContains(output.state, expect.stateContains, 'output.state');
    }

    // artifactsContain: partial matching
    if (expect.artifactsContain) {
      assert.ok(output.artifacts, 'Expected output to have artifacts');
      for (const expectedArt of expect.artifactsContain) {
        const found = output.artifacts!.find(
          (a: any) => a.name === expectedArt.name
        );
        assert.ok(
          found,
          `Expected artifact "${expectedArt.name}" not found in output`
        );
        assertContains(found, expectedArt, `artifact(${expectedArt.name})`);
      }
    }
  }

  // Capture values for subsequent invocations
  if (invocation.captureSnapshotId) {
    assert.ok(
      output.snapshotId,
      `captureSnapshotId '${invocation.captureSnapshotId}' requested but output has no snapshotId`
    );
    captures.set(invocation.captureSnapshotId, output.snapshotId);
  }

  if (invocation.captureState) {
    assert.ok(
      output.state,
      `captureState '${invocation.captureState}' requested but output has no state`
    );
    captures.set(invocation.captureState, output.state);
  }
}

async function executeGetSnapshotDataInvocation(
  agent: Agent,
  invocation: SpecStep,
  captures: Map<string, any>
): Promise<void> {
  const resolved = resolveTemplates(invocation, captures);
  const snapshotId = resolved.snapshotId;

  assert.ok(snapshotId, 'getSnapshotData invocation requires snapshotId');

  const snapshot = await agent.getSnapshotData(snapshotId);
  assert.ok(snapshot, `Snapshot ${snapshotId} not found`);

  if (resolved.expectSnapshot) {
    const expect = resolved.expectSnapshot;

    if (expect.parentId !== undefined) {
      assert.strictEqual(
        snapshot.parentId,
        expect.parentId,
        `Expected parentId '${expect.parentId}', got '${snapshot.parentId}'`
      );
    }

    if (expect.status !== undefined) {
      assert.strictEqual(
        snapshot.status,
        expect.status,
        `Expected status '${expect.status}', got '${snapshot.status}'`
      );
    }

    if (expect.stateContains) {
      assertContains(snapshot.state, expect.stateContains, 'snapshot.state');
    }

    if (expect.errorContains) {
      assert.ok(snapshot.error, 'Expected snapshot to have error');
      assertContains(snapshot.error, expect.errorContains, 'snapshot.error');
    }
  }
}

async function executeAbortInvocation(
  agent: Agent,
  invocation: SpecStep,
  captures: Map<string, any>
): Promise<void> {
  const resolved = resolveTemplates(invocation, captures);
  const snapshotId = resolved.snapshotId;

  assert.ok(snapshotId, 'abort invocation requires snapshotId');

  const previousStatus = await agent.abort(snapshotId);

  // The `expectPreviousStatus` key being present (even as null/~) means we
  // should assert.  YAML `~` maps to JS `null`; agent.abort() returns
  // `undefined` for non-existent snapshots — treat both as "absent".
  if ('expectPreviousStatus' in resolved) {
    const expected = resolved.expectPreviousStatus ?? undefined;
    assert.strictEqual(
      previousStatus,
      expected,
      `Expected previous status '${expected}', got '${previousStatus}'`
    );
  }
}

async function executeWaitUntilCompletedInvocation(
  agent: Agent,
  invocation: SpecStep,
  captures: Map<string, any>
): Promise<void> {
  const resolved = resolveTemplates(invocation, captures);
  const snapshotId = resolved.snapshotId;
  const timeoutMs = resolved.timeoutMs || 5000;

  assert.ok(snapshotId, 'waitUntilCompleted invocation requires snapshotId');

  const terminalStatuses = new Set(['done', 'failed', 'aborted']);
  const startTime = Date.now();

  let snapshot: any;
  while (Date.now() - startTime < timeoutMs) {
    snapshot = await agent.getSnapshotData(snapshotId);
    if (snapshot && terminalStatuses.has(snapshot.status)) {
      break;
    }
    await new Promise((r) => setTimeout(r, 100));
  }

  assert.ok(snapshot, `Snapshot ${snapshotId} not found after waiting`);
  assert.ok(
    terminalStatuses.has(snapshot.status),
    `Snapshot ${snapshotId} did not reach terminal status within ${timeoutMs}ms. Status: ${snapshot.status}`
  );

  if (resolved.expectSnapshot) {
    const expect = resolved.expectSnapshot;

    if (expect.status !== undefined) {
      assert.strictEqual(snapshot.status, expect.status);
    }

    if (expect.stateContains) {
      assertContains(snapshot.state, expect.stateContains, 'snapshot.state');
    }

    if (expect.errorContains) {
      assert.ok(snapshot.error, 'Expected snapshot to have error');
      assertContains(snapshot.error, expect.errorContains, 'snapshot.error');
    }
  }
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------

const specPath = '../../tests/specs/agent.yaml';
const spec = SpecSuiteSchema.parse(parse(readFileSync(specPath, 'utf-8')));

describe('Agent conformance spec', () => {
  let registry: Registry;
  let pm: ProgrammableModel;
  let agents: HarnessAgents;

  beforeEach(() => {
    registry = new Registry();
    registry.apiStability = 'beta';
    pm = defineProgrammableModel(registry);
    agents = setupHarness(registry, pm);
  });

  for (const test of spec.tests) {
    it(test.name, async () => {
      const agent = agents[test.agent];
      assert.ok(agent, `Unknown agent '${test.agent}' in test '${test.name}'`);

      const captures = new Map<string, any>();

      for (let i = 0; i < test.steps.length; i++) {
        const step = test.steps[i];
        const label = `step[${i}] (${step.type})`;

        try {
          switch (step.type) {
            case 'send':
              await executeSendInvocation(agent, pm, step, captures);
              break;
            case 'getSnapshotData':
              await executeGetSnapshotDataInvocation(agent, step, captures);
              break;
            case 'abort':
              await executeAbortInvocation(agent, step, captures);
              break;
            case 'waitUntilCompleted':
              await executeWaitUntilCompletedInvocation(agent, step, captures);
              break;
            default:
              assert.fail(`Unknown step type: ${(step as any).type}`);
          }
        } catch (e: any) {
          // Wrap error with step context for better diagnostics
          const wrapped = new Error(
            `${label} in test '${test.name}' failed: ${e.message}`
          );
          wrapped.stack = e.stack;
          throw wrapped;
        }
      }
    });
  }
});
