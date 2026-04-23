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

import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import { z, type Genkit } from 'genkit';
import { genkit, type GenkitBeta } from 'genkit/beta';
import { appendSpan } from 'genkit/tracing';
import {
  __forceFlushSpansForTesting,
  __getSpanExporterForTesting,
} from '../src/gcpOpenTelemetry.js';
import { enableGoogleCloudTelemetry } from '../src/index.js';

jest.mock('../src/auth.js', () => {
  const original = jest.requireActual('../src/auth.js');
  return {
    ...(original || {}),
    resolveCurrentPrincipal: jest.fn().mockImplementation(() => {
      return Promise.resolve({
        projectId: 'test',
        serviceAccountEmail: 'test@test.com',
      });
    }),
    credentialsFromEnvironment: jest.fn().mockImplementation(() => {
      return Promise.resolve({
        projectId: 'test',
        credentials: {
          client_email: 'test@genkit.com',
          private_key: '-----BEGIN PRIVATE KEY-----',
        },
      });
    }),
  };
});

describe('GoogleCloudTracing', () => {
  let ai: GenkitBeta;

  beforeAll(async () => {
    process.env.GCLOUD_PROJECT = 'test';
    process.env.GENKIT_ENV = 'dev';
    await enableGoogleCloudTelemetry({
      projectId: 'test',
      forceDevExport: false,
    });
    ai = genkit({});
  });
  beforeEach(async () => {
    __getSpanExporterForTesting().reset();
  });
  afterAll(async () => {
    await ai.stopServers();
  });

  it('writes traces', async () => {
    const testFlow = createFlow(ai, 'testFlow');

    await testFlow();

    const spans = await getExportedSpans();
    assert.equal(spans.length, 1);
    assert.equal(spans[0].name, 'testFlow');
  });

  it('Adjusts attributes to support GCP trace filtering', async () => {
    const testFlow = createFlow(ai, 'testFlow');

    await testFlow();

    const spans = await getExportedSpans();
    // Check some common attributes
    assert.equal(spans[0].attributes['genkit/name'], 'testFlow');
    assert.equal(spans[0].attributes['genkit/type'], 'action');
    assert.equal(spans[0].attributes['genkit/metadata/subtype'], 'flow');
    // Ensure we have no attributes with ':' because these are awkward to use in
    // Cloud Trace.
    const spanAttrKeys = Object.entries(spans[0].attributes).map(([k, v]) => k);
    for (const key in spanAttrKeys) {
      assert.equal(key.indexOf(':'), -1);
    }
  });

  it('sub actions are contained within flows', async () => {
    const testFlow = createFlow(ai, 'testFlow', async () => {
      return await ai.run('subAction', async () => {
        return await ai.run('subAction2', async () => {
          return 'done';
        });
      });
    });

    await testFlow();

    const spans = await getExportedSpans();
    assert.equal(spans.length, 3);
    assert.equal(spans[2].name, 'testFlow');
    assert.equal(spans[2].parentSpanId, undefined);
    assert.equal(spans[1].name, 'subAction');
    assert.equal(spans[1].parentSpanId, spans[2].spanContext().spanId);
    assert.equal(spans[0].name, 'subAction2');
    assert.equal(spans[0].parentSpanId, spans[1].spanContext().spanId);
  });

  it('different flows run independently', async () => {
    const testFlow1 = createFlow(ai, 'testFlow1');
    const testFlow2 = createFlow(ai, 'testFlow2');

    await testFlow1();
    await testFlow2();

    const spans = await getExportedSpans();
    assert.equal(spans.length, 2);
    assert.equal(spans[0].parentSpanId, undefined);
    assert.equal(spans[1].parentSpanId, undefined);
  });

  it('labels failed spans', async () => {
    const testFlow = createFlow(ai, 'badFlow', async () => {
      return await ai.run('badStep', async () => {
        throw new Error('oh no!');
      });
    });
    await assert.rejects(async () => {
      await testFlow();
    });

    const spans = await getExportedSpans();

    expect(spans).toHaveLength(2);

    expect(spans[0].name).toEqual('badStep');
    expect(spans[0].attributes['genkit/failedSpan']).toEqual('badStep');
    expect(spans[0].attributes['genkit/failedPath']).toEqual(
      '/{badFlow,t:flow}/{badStep,t:flowStep}'
    );

    expect(spans[1].attributes['genkit/isRoot']).toEqual(true);
    expect(spans[1].attributes['genkit/rootState']).toEqual('error');
    expect(spans[1].attributes['genkit/state']).toEqual('error');
    expect(spans[1].attributes['genkit/failedSpan']).toBeUndefined();
    expect(spans[1].attributes['genkit/failedPath']).toBeUndefined();
  });

  it('labels multiple failed spans', async () => {
    const testFlow = createFlow(ai, 'badFlow', async () => {
      await Promise.all([
        ai.run('sub1', async () => {
          throw new Error('oh no!');
        }),
        ai.run('sub2', async () => {
          throw new Error('oh no!');
        }),
      ]);
      return 'root is ok';
    });
    await assert.rejects(async () => {
      await testFlow();
    });

    const spans = await getExportedSpans();

    expect(spans).toHaveLength(3);

    const rootSpan = spans.find((s) => s.name === 'badFlow')!;
    const sub1Span = spans.find((s) => s.name === 'sub1')!;
    const sub2Span = spans.find((s) => s.name === 'sub2')!;

    expect(rootSpan.attributes['genkit/failedSpan']).toBeUndefined();
    expect(rootSpan.attributes['genkit/failedPath']).toBeUndefined();

    expect(sub1Span.attributes['genkit/failedSpan']).toEqual('sub1');
    expect(sub1Span.attributes['genkit/failedPath']).toEqual(
      '/{badFlow,t:flow}/{sub1,t:flowStep}'
    );

    expect(sub2Span.attributes['genkit/failedSpan']).toEqual('sub2');
    expect(sub1Span.attributes['genkit/failedPath']).toEqual(
      '/{badFlow,t:flow}/{sub1,t:flowStep}'
    );
  });

  it('labels the root feature', async () => {
    const testFlow = createFlow(ai, 'niceFlow', async () => {
      return ai.run('niceStep', async () => {});
    });
    await testFlow();

    const spans = await getExportedSpans();
    assert.equal(spans[0].name, 'niceStep');
    assert.equal(spans[0].attributes['genkit/feature'], undefined);
    assert.equal(spans[1].name, 'niceFlow');
    assert.equal(spans[1].attributes['genkit/feature'], 'niceFlow');
    assert.equal(spans[1].attributes['genkit/rootState'], 'success');
  });

  it('marks the root feature failed when it is the failure', async () => {
    const testFlow = createFlow(ai, 'failingFlow', async () => {
      await ai.run('good step', async () => {
        return 'nothing going on here';
      });
      throw new Error('oops!');
    });
    await assert.rejects(async () => {
      await testFlow();
    });

    const spans = await getExportedSpans();

    assert.equal(spans.length, 2);
    assert.equal(spans[0].attributes['genkit/state'], 'success');
    assert.equal(spans[1].attributes['genkit/name'], 'failingFlow');
    assert.equal(spans[1].attributes['genkit/failedSpan'], 'failingFlow');
    assert.equal(
      spans[1].attributes['genkit/failedPath'],
      '/{failingFlow,t:flow}'
    );
    assert.equal(spans[1].attributes['genkit/isRoot'], true);
    assert.equal(spans[1].attributes['genkit/rootState'], 'error');
  });

  it('adds the genkit/model label for model actions', async () => {
    const echoModel = ai.defineModel(
      {
        name: 'echoModel',
      },
      async (request) => {
        return {
          message: {
            role: 'model',
            content: [
              {
                text:
                  'Echo: ' +
                  request.messages
                    .map((m) => m.content.map((c) => c.text).join())
                    .join(),
              },
            ],
          },
          finishReason: 'stop',
        };
      }
    );
    const testFlow = createFlow(ai, 'modelFlow', async () => {
      return ai.run('runFlow', async () => {
        await ai.generate({
          model: echoModel,
          prompt: 'Testing model telemetry',
        });
      });
    });

    await testFlow();

    const spans = await getExportedSpans();

    assert.equal(spans[0].name, 'echoModel');
    assert.equal(spans[0].attributes['genkit/model'], 'echoModel');
    assert.equal(spans[1].name, 'generate');
    assert.equal(spans[2].name, 'runFlow');
    assert.equal(spans[3].name, 'modelFlow');
  });

  it('attaches additional span', async () => {
    await appendSpan(
      'trace1',
      'parent1',
      { name: 'span-name', metadata: { metadata_key: 'metadata_value' } },
      { ['label_key']: 'label_value' }
    );

    const spans = await getExportedSpans();
    const span = spans.find((it) => it.name === 'span-name');
    assert.equal(Object.keys(span?.attributes || {}).length, 3);
    assert.equal(span?.attributes['genkit/name'], 'span-name');
    assert.equal(span?.attributes['label_key'], 'label_value');
    assert.equal(
      span?.attributes['genkit/metadata/metadata_key'],
      'metadata_value'
    );
  });

  /** Helper to create a flow with no inputs or outputs */
  function createFlow(
    ai: Genkit,
    name: string,
    fn: () => Promise<any> = async () => {}
  ) {
    return ai.defineFlow(
      {
        name,
        inputSchema: z.void(),
        outputSchema: z.void(),
      },
      fn
    );
  }

  /** Polls the in memory metric exporter until the genkit scope is found. */
  async function getExportedSpans(maxAttempts = 200): Promise<ReadableSpan[]> {
    __forceFlushSpansForTesting();
    var attempts = 0;
    while (attempts++ < maxAttempts) {
      await new Promise((resolve) => setTimeout(resolve, 50));
      const found = __getSpanExporterForTesting().getFinishedSpans();
      if (found.length > 0) {
        return found;
      }
    }
    assert.fail(`Timed out while waiting for spans to be exported.`);
  }
});
