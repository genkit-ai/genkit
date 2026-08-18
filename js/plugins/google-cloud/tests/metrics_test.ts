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
  it,
  jest,
} from '@jest/globals';
import type {
  DataPoint,
  Histogram,
  HistogramMetricData,
  ScopeMetrics,
  SumMetricData,
} from '@opentelemetry/sdk-metrics';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import { genkit, z, type GenerateResponseData, type Genkit } from 'genkit';
import { SPAN_TYPE_ATTR, appendSpan } from 'genkit/tracing';
import {
  GcpOpenTelemetry,
  __forceFlushSpansForTesting,
  __getMetricExporterForTesting,
  __getSpanExporterForTesting,
  enableGoogleCloudTelemetry,
} from '../src/index.js';

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

describe('GoogleCloudMetrics', () => {
  let ai: Genkit;

  beforeAll(async () => {
    process.env.GCLOUD_PROJECT = 'test';
    process.env.GENKIT_ENV = 'dev';
    await enableGoogleCloudTelemetry({
      projectId: 'test',
      forceDevExport: false,
      metricExportIntervalMillis: 100,
      metricExportTimeoutMillis: 100,
    });
    ai = genkit({});
  });
  beforeEach(async () => {
    __getMetricExporterForTesting().reset();
    __getSpanExporterForTesting().reset();
  });
  afterAll(async () => {
    await ai.stopServers();
  });

  it('writes feature metrics for a successful flow', async () => {
    const testFlow = createFlow(ai, 'testFlow');

    await testFlow();
    await testFlow();

    await getExportedSpans();

    const requestCounter = await getCounterMetric('genkit/feature/requests');
    const latencyHistogram = await getHistogramMetric('genkit/feature/latency');
    assert.equal(requestCounter.value, 2);
    assert.equal(requestCounter.attributes.name, 'testFlow');
    assert.equal(requestCounter.attributes.source, 'ts');
    assert.equal(requestCounter.attributes.status, 'success');
    assert.ok(requestCounter.attributes.sourceVersion);
    assert.equal(latencyHistogram.value.count, 2);
    assert.equal(latencyHistogram.attributes.name, 'testFlow');
    assert.equal(latencyHistogram.attributes.source, 'ts');
    assert.equal(latencyHistogram.attributes.status, 'success');
    assert.ok(latencyHistogram.attributes.sourceVersion);
  });

  it('writes feature metrics for a failing flow', async () => {
    const testFlow = createFlow(ai, 'testFlow', async () => {
      return explode();
    });

    await assert.rejects(async () => {
      await testFlow();
    });

    await getExportedSpans();

    const requestCounter = await getCounterMetric('genkit/feature/requests');
    assert.equal(requestCounter.value, 1);
    assert.equal(requestCounter.attributes.name, 'testFlow');
    assert.equal(requestCounter.attributes.source, 'ts');
    assert.equal(requestCounter.attributes.error, 'TypeError');
    assert.equal(requestCounter.attributes.status, 'failure');
  }, 10000); //timeout

  it('writes feature metrics for an action', async () => {
    const testAction = createAction(ai, 'featureAction');

    await testAction(null);
    await testAction(null);

    await getExportedSpans();

    const requestCounter = await getCounterMetric('genkit/feature/requests');
    const latencyHistogram = await getHistogramMetric('genkit/feature/latency');
    assert.equal(requestCounter.value, 2);
    assert.equal(requestCounter.attributes.name, 'featureAction');
    assert.equal(requestCounter.attributes.source, 'ts');
    assert.equal(requestCounter.attributes.status, 'success');
    assert.ok(requestCounter.attributes.sourceVersion);
    assert.equal(latencyHistogram.value.count, 2);
    assert.equal(latencyHistogram.attributes.name, 'featureAction');
    assert.equal(latencyHistogram.attributes.source, 'ts');
    assert.equal(latencyHistogram.attributes.status, 'success');
    assert.ok(latencyHistogram.attributes.sourceVersion);
  });

  it('writes feature metrics for generate', async () => {
    const testModel = createTestModel(ai, 'helloModel');
    await ai.generate({ model: testModel, prompt: 'Hi' });
    await ai.generate({ model: testModel, prompt: 'Yo' });

    await getExportedSpans();

    const requestCounter = await getCounterMetric('genkit/feature/requests');
    const latencyHistogram = await getHistogramMetric('genkit/feature/latency');
    assert.equal(requestCounter.value, 2);
    assert.equal(requestCounter.attributes.name, 'generate');
    assert.equal(requestCounter.attributes.source, 'ts');
    assert.equal(requestCounter.attributes.status, 'success');
    assert.ok(requestCounter.attributes.sourceVersion);
    assert.equal(latencyHistogram.value.count, 2);
    assert.equal(latencyHistogram.attributes.name, 'generate');
    assert.equal(latencyHistogram.attributes.source, 'ts');
    assert.equal(latencyHistogram.attributes.status, 'success');
    assert.ok(latencyHistogram.attributes.sourceVersion);
  });

  it('writes generate metrics for a successful model action', async () => {
    const testModel = createTestModel(ai, 'testModel');

    await ai.generate({
      model: testModel,
      prompt: 'test prompt',
      config: {
        maxOutputTokens: 7,
      },
    });

    await getExportedSpans();

    const requestCounter = await getCounterMetric(
      'genkit/ai/generate/requests'
    );
    const inputTokenCounter = await getCounterMetric(
      'genkit/ai/generate/input/tokens'
    );
    const outputTokenCounter = await getCounterMetric(
      'genkit/ai/generate/output/tokens'
    );
    const thoughtTokenCounter = await getCounterMetric(
      'genkit/ai/generate/thinking/tokens'
    );
    const inputCharacterCounter = await getCounterMetric(
      'genkit/ai/generate/input/characters'
    );
    const outputCharacterCounter = await getCounterMetric(
      'genkit/ai/generate/output/characters'
    );
    const inputImageCounter = await getCounterMetric(
      'genkit/ai/generate/input/images'
    );
    const outputImageCounter = await getCounterMetric(
      'genkit/ai/generate/output/images'
    );
    const latencyHistogram = await getHistogramMetric(
      'genkit/ai/generate/latency'
    );
    assert.equal(requestCounter.value, 1);
    assert.equal(inputTokenCounter.value, 10);
    assert.equal(outputTokenCounter.value, 14);
    assert.equal(thoughtTokenCounter.value, 5);
    assert.equal(inputCharacterCounter.value, 8);
    assert.equal(outputCharacterCounter.value, 16);
    assert.equal(inputImageCounter.value, 1);
    assert.equal(outputImageCounter.value, 3);
    assert.equal(latencyHistogram.value.count, 1);
    for (const metric of [
      requestCounter,
      inputTokenCounter,
      outputTokenCounter,
      thoughtTokenCounter,
      inputCharacterCounter,
      outputCharacterCounter,
      inputImageCounter,
      outputImageCounter,
      latencyHistogram,
    ]) {
      assert.equal(metric.attributes.modelName, 'testModel');
      assert.equal(metric.attributes.source, 'ts');
      assert.equal(metric.attributes.status, 'success');
      assert.equal(metric.attributes.featureName, 'generate');
      assert.ok(metric.attributes.sourceVersion);
    }
  });

  it('writes generate metrics for a failing model action', async () => {
    const testModel = createModel(ai, 'failingTestModel', async () => {
      return explode();
    });

    await assert.rejects(async () => {
      return ai.generate({
        model: testModel,
        prompt: 'test prompt',
      });
    });

    await getExportedSpans();

    const requestCounter = await getCounterMetric(
      'genkit/ai/generate/requests'
    );
    assert.equal(requestCounter.value, 1);
    assert.equal(requestCounter.attributes.modelName, 'failingTestModel');
    assert.equal(requestCounter.attributes.source, 'ts');
    assert.equal(requestCounter.attributes.status, 'failure');
    assert.equal(requestCounter.attributes.error, 'TypeError');
    assert.ok(requestCounter.attributes.sourceVersion);
  }, 10000); //timeout

  it('writes feature label to generate metrics when running inside a flow', async () => {
    const testModel = createModel(ai, 'testModel', async () => {
      return {
        message: {
          role: 'user',
          content: [
            {
              text: 'response',
            },
          ],
        },
        finishReason: 'stop',
        usage: {
          inputTokens: 10,
          outputTokens: 14,
          thoughtsTokens: 5,
          inputCharacters: 8,
          outputCharacters: 16,
          inputImages: 1,
          outputImages: 3,
        },
      };
    });
    const flow = createFlow(ai, 'testFlow', async () => {
      return await ai.generate({
        model: testModel,
        prompt: 'test prompt',
      });
    });

    await flow();

    await getExportedSpans();

    const metrics = [
      await getCounterMetric('genkit/ai/generate/requests'),
      await getCounterMetric('genkit/ai/generate/input/tokens'),
      await getCounterMetric('genkit/ai/generate/output/tokens'),
      await getCounterMetric('genkit/ai/generate/thinking/tokens'),
      await getCounterMetric('genkit/ai/generate/input/characters'),
      await getCounterMetric('genkit/ai/generate/output/characters'),
      await getCounterMetric('genkit/ai/generate/input/images'),
      await getCounterMetric('genkit/ai/generate/output/images'),
      await getHistogramMetric('genkit/ai/generate/latency'),
    ];
    for (const metric of metrics) {
      assert.equal(metric.attributes.featureName, 'testFlow');
    }
  });

  describe('path metrics', () => {
    it('writes no path metrics for a successful flow', async () => {
      const flow = createFlow(ai, 'pathTestFlow', async () => {
        await ai.run('step1', async () => {
          return await ai.run('substep_a', async () => {
            return await ai.run('substep_b', async () => 'res1');
          });
        });
        await ai.run('step2', async () => 'res2');
        return;
      });

      await flow();

      await getExportedSpans();

      await assert.rejects(async () => {
        await getCounterDataPoints('genkit/feature/path/requests');
      });
      await assert.rejects(async () => {
        await getHistogramDataPoints('genkit/feature/path/latency');
      });
    });

    it('writes path metrics for a failing flow with exception in root', async () => {
      const flow = createFlow(ai, 'testFlow', async () => {
        await ai.run('sub-action', async () => {
          return 'done';
        });
        return Promise.reject(new Error('failed'));
      });

      await assert.rejects(async () => {
        await flow();
      });

      await getExportedSpans();

      const reqPoints = await getCounterDataPoints(
        'genkit/feature/path/requests'
      );
      const reqStatuses = reqPoints.map((p) => [
        p.attributes.path,
        p.attributes.status,
      ]);
      assert.deepEqual(reqStatuses, [['/{testFlow,t:flow}', 'failure']]);
      const latencyPoints = await getHistogramDataPoints(
        'genkit/feature/path/latency'
      );
      const latencyStatuses = latencyPoints.map((p) => [
        p.attributes.path,
        p.attributes.status,
      ]);
      assert.deepEqual(latencyStatuses, [['/{testFlow,t:flow}', 'failure']]);
    }, 10000); //timeout

    it('writes path metrics for a failing flow with exception in subaction', async () => {
      const flow = createFlow(ai, 'testFlow', async () => {
        await ai.run('sub-action-1', async () => {
          await ai.run('sub-action-2', async () => {
            return Promise.reject(new Error('failed'));
          });
          return 'done';
        });
        return 'done';
      });

      await assert.rejects(async () => {
        await flow();
      });

      await getExportedSpans();

      const reqPoints = await getCounterDataPoints(
        'genkit/feature/path/requests'
      );
      const reqStatuses = reqPoints.map((p) => [
        p.attributes.path,
        p.attributes.status,
      ]);
      assert.deepEqual(reqStatuses, [
        [
          '/{testFlow,t:flow}/{sub-action-1,t:flowStep}/{sub-action-2,t:flowStep}',
          'failure',
        ],
      ]);
      const latencyPoints = await getHistogramDataPoints(
        'genkit/feature/path/latency'
      );
      const latencyStatuses = latencyPoints.map((p) => [
        p.attributes.path,
        p.attributes.status,
      ]);
      assert.deepEqual(latencyStatuses, [
        [
          '/{testFlow,t:flow}/{sub-action-1,t:flowStep}/{sub-action-2,t:flowStep}',
          'failure',
        ],
      ]);
    }, 10000); //timeout

    it('writes path metrics for a flow with exception in action', async () => {
      const flow = createFlow(ai, 'testFlow', async () => {
        await ai.run('sub-action-1', async () => {
          await ai.run('sub-action-2', async () => {
            return 'done';
          });
          return Promise.reject(new Error('failed'));
        });
        return 'done';
      });

      await assert.rejects(async () => {
        await flow();
      });

      await getExportedSpans();

      const reqPoints = await getCounterDataPoints(
        'genkit/feature/path/requests'
      );
      const reqStatuses = reqPoints.map((p) => [
        p.attributes.path,
        p.attributes.status,
      ]);
      assert.deepEqual(reqStatuses, [
        ['/{testFlow,t:flow}/{sub-action-1,t:flowStep}', 'failure'],
      ]);
      const latencyPoints = await getHistogramDataPoints(
        'genkit/feature/path/latency'
      );
      const latencyStatuses = latencyPoints.map((p) => [
        p.attributes.path,
        p.attributes.status,
      ]);
      assert.deepEqual(latencyStatuses, [
        ['/{testFlow,t:flow}/{sub-action-1,t:flowStep}', 'failure'],
      ]);
    }, 10000); //timeout

    it('writes path metrics for a flow with an exception in a serial action', async () => {
      const flow = createFlow(ai, 'testFlow', async () => {
        await ai.run('sub-action-1', async () => {
          return 'done';
        });
        await ai.run('sub-action-2', async () => {
          return Promise.reject(new Error('failed'));
        });
        return 'done';
      });

      await assert.rejects(async () => {
        await flow();
      });

      await getExportedSpans();

      const reqPoints = await getCounterDataPoints(
        'genkit/feature/path/requests'
      );
      const reqStatuses = reqPoints.map((p) => [
        p.attributes.path,
        p.attributes.status,
      ]);
      assert.deepEqual(reqStatuses, [
        ['/{testFlow,t:flow}/{sub-action-2,t:flowStep}', 'failure'],
      ]);
      const latencyPoints = await getHistogramDataPoints(
        'genkit/feature/path/latency'
      );
      const latencyStatuses = latencyPoints.map((p) => [
        p.attributes.path,
        p.attributes.status,
      ]);
      assert.deepEqual(latencyStatuses, [
        ['/{testFlow,t:flow}/{sub-action-2,t:flowStep}', 'failure'],
      ]);
    }, 10000); //timeout

    it('writes path metrics for a flow multiple failing actions', async () => {
      const flow = createFlow(ai, 'testFlow', async () => {
        await Promise.all([
          ai.run('sub1', async () => {
            return explode();
          }),
          ai.run('sub2', async () => {
            return explode();
          }),
        ]);
        return 'not failing';
      });

      await assert.rejects(async () => {
        await flow();
      });

      await getExportedSpans();

      const reqPoints = await getCounterDataPoints(
        'genkit/feature/path/requests'
      );
      const reqStatuses = reqPoints.map((p) => [
        p.attributes.path,
        p.attributes.status,
      ]);
      assert.deepEqual(reqStatuses, [
        ['/{testFlow,t:flow}/{sub1,t:flowStep}', 'failure'],
        ['/{testFlow,t:flow}/{sub2,t:flowStep}', 'failure'],
      ]);
      const latencyPoints = await getHistogramDataPoints(
        'genkit/feature/path/latency'
      );
      const latencyStatuses = latencyPoints.map((p) => [
        p.attributes.path,
        p.attributes.status,
      ]);
      assert.deepEqual(latencyStatuses, [
        ['/{testFlow,t:flow}/{sub1,t:flowStep}', 'failure'],
        ['/{testFlow,t:flow}/{sub2,t:flowStep}', 'failure'],
      ]);
    }, 10000); //timeout
  });

  it('writes user feedback metrics', async () => {
    await appendSpan(
      'trace1',
      'parent1',
      {
        name: 'user-feedback',
        path: '/{flowName}',
        metadata: {
          subtype: 'userFeedback',
          feedbackValue: 'negative',
          textFeedback: 'terrible',
        },
      },
      { [SPAN_TYPE_ATTR]: 'userEngagement' }
    );

    await getExportedSpans();
    const dataPoints = await getCounterDataPoints('genkit/engagement/feedback');

    const points = dataPoints.map((p) => [
      p.attributes.name,
      p.attributes.value,
      p.attributes.hasText,
      p.attributes.source,
    ]);
    assert.deepEqual(points, [['flowName', 'negative', true, 'ts']]);
    assert.ok(dataPoints[0].attributes.sourceVersion);
  });

  it('writes user acceptance metrics', async () => {
    await appendSpan(
      'trace1',
      'parent1',
      {
        name: 'user-acceptance',
        path: '/{flowName}',
        metadata: { subtype: 'userAcceptance', acceptanceValue: 'rejected' },
      },
      { [SPAN_TYPE_ATTR]: 'userEngagement' }
    );

    await getExportedSpans();
    const dataPoints = await getCounterDataPoints(
      'genkit/engagement/acceptance'
    );

    const points = dataPoints.map((p) => [
      p.attributes.name,
      p.attributes.value,
      p.attributes.source,
    ]);
    assert.deepEqual(points, [['flowName', 'rejected', 'ts']]);
    assert.ok(dataPoints[0].attributes.sourceVersion);
  });

  describe('Configuration', () => {
    it('should export only traces', async () => {
      const telemetry = new GcpOpenTelemetry({
        export: true,
        disableMetrics: true,
        disableTraces: false,
      } as any);
      assert.equal(telemetry['shouldExportTraces'](), true);
      assert.equal(telemetry['shouldExportMetrics'](), false);
    });

    it('should export only metrics', async () => {
      const telemetry = new GcpOpenTelemetry({
        export: true,
        disableTraces: true,
        disableMetrics: false,
      } as any);
      assert.equal(telemetry['shouldExportTraces'](), false);
      assert.equal(telemetry['shouldExportMetrics'](), true);
    });
  });

  /** Polls the in memory metric exporter until the genkit scope is found. */
  async function getGenkitMetrics(
    name = 'genkit',
    maxAttempts = 100
  ): Promise<ScopeMetrics | undefined> {
    var attempts = 0;
    while (attempts++ < maxAttempts) {
      await new Promise((resolve) => setTimeout(resolve, 50));
      const found = __getMetricExporterForTesting()
        .getMetrics()
        .find((e) => e.scopeMetrics.map((sm) => sm.scope.name).includes(name));
      if (found) {
        return found.scopeMetrics.find((e) => e.scope.name === name);
      }
    }
    assert.fail(`Waiting for metric ${name} but it has not been written.`);
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

  /** Finds all datapoints for a counter metric with the given name in the in memory exporter */
  async function getCounterDataPoints(
    metricName: string
  ): Promise<Array<DataPoint<number>>> {
    const genkitMetrics = await getGenkitMetrics();
    if (genkitMetrics) {
      const counterMetric: SumMetricData = genkitMetrics.metrics.find(
        (e) =>
          e.descriptor.name === metricName && e.descriptor.type === 'COUNTER'
      ) as SumMetricData;
      if (counterMetric) {
        return counterMetric.dataPoints;
      }
      assert.fail(
        `No counter metric named ${metricName} was found. Only found: ${genkitMetrics.metrics.map((e) => e.descriptor.name)}`
      );
    } else {
      assert.fail(`No genkit metrics found.`);
    }
  }

  /** Finds a counter metric with the given name in the in memory exporter */
  async function getCounterMetric(
    metricName: string
  ): Promise<DataPoint<number>> {
    const counter = await getCounterDataPoints(metricName).then((points) =>
      points.at(-1)
    );
    if (counter === undefined) {
      assert.fail(`Counter not found`);
    } else {
      return counter;
    }
  }

  /**
   * Finds all datapoints for a histogram metric with the given name in the in
   * memory exporter.
   */
  async function getHistogramDataPoints(
    metricName: string
  ): Promise<Array<DataPoint<Histogram>>> {
    const genkitMetrics = await getGenkitMetrics();
    if (genkitMetrics === undefined) {
      assert.fail('Genkit metrics not found');
    } else {
      const histogramMetric = genkitMetrics.metrics.find(
        (e) =>
          e.descriptor.name === metricName && e.descriptor.type === 'HISTOGRAM'
      ) as HistogramMetricData;
      if (histogramMetric === undefined) {
        assert.fail(
          `No histogram metric named ${metricName} was found. Only found: ${genkitMetrics.metrics.map((e) => e.descriptor.name)}`
        );
      } else {
        return histogramMetric.dataPoints;
      }
    }
  }

  /** Finds a histogram metric with the given name in the in memory exporter */
  async function getHistogramMetric(
    metricName: string
  ): Promise<DataPoint<Histogram>> {
    const metric = await getHistogramDataPoints(metricName).then((points) =>
      points.at(-1)
    );
    if (metric === undefined) {
      assert.fail('Metric not found');
    } else {
      return metric;
    }
  }

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
        outputSchema: z.any(),
      },
      fn
    );
  }

  /** Helper to create an action with no inputs or outputs */
  function createAction(
    ai: Genkit,
    name: string,
    fn: () => Promise<void> = async () => {}
  ) {
    return ai.defineTool(
      {
        name,
        description: "I don't do much...",
      },
      fn
    );
  }

  /** Helper to create a model that returns the value produced by the given
   * response function. */
  function createModel(
    ai: Genkit,
    name: string,
    respFn: () => Promise<GenerateResponseData>
  ) {
    return ai.defineModel({ name }, (req) => respFn());
  }

  function createTestModel(ai: Genkit, name: string) {
    return createModel(ai, name, async () => {
      return {
        message: {
          role: 'model',
          content: [
            {
              text: 'Oh hello',
            },
          ],
        },
        finishReason: 'stop',
        usage: {
          inputTokens: 10,
          outputTokens: 14,
          inputCharacters: 8,
          outputCharacters: 16,
          inputImages: 1,
          outputImages: 3,
          thoughtsTokens: 5,
        },
      };
    });
  }
});

function explode() {
  const nothing: { missing?: any } = { missing: 1 };
  delete nothing.missing;
  return nothing.missing.explode;
}
