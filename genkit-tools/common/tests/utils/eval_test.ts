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

import { describe, expect, it, vi } from 'vitest';
import * as configModule from '../../src/plugin/config';
import type { TraceData } from '../../src/types/trace';
import { getEvalExtractors } from '../../src/utils/eval';
import { MockTrace } from './trace';

const CONTEXT_TEXTS = [
  'are about 10 times larger, making them particularly difficult for humans to ignore.',
  'they are very big animals',
];

describe('eval utils', () => {
  describe('models', () => {
    it('works with default extractors', async () => {
      const spy = vi.spyOn(configModule, 'findToolsConfig');
      spy.mockReturnValue(Promise.resolve(null));
      // Mock trace mocks flows, but the logic of extractors should be unaffected.
      const trace = new MockTrace('My input', 'My output').getTrace();

      const extractors = await getEvalExtractors('/model/googleai/gemini-pro');

      expect(Object.keys(extractors).sort()).toEqual(
        ['input', 'output', 'context'].sort()
      );
      expect(extractors.input(trace)).toEqual('My input');
      expect(extractors.output(trace)).toEqual('My output');
      expect(extractors.context(trace)).toEqual([]);
    });
  });

  it('returns default extractors when no config provided', async () => {
    const spy = vi.spyOn(configModule, 'findToolsConfig');
    spy.mockReturnValue(Promise.resolve(null));
    const trace = new MockTrace('My input', 'My output')
      .addSpan({
        stepName: 'retrieverStep',
        spanType: 'action',
        retrieverConfig: {
          query: 'What are cats?',
          text: CONTEXT_TEXTS,
        },
      })
      .getTrace();

    const extractors = await getEvalExtractors('/flow/multiSteps');

    expect(Object.keys(extractors).sort()).toEqual(
      ['input', 'output', 'context'].sort()
    );
    expect(extractors.input(trace)).toEqual('My input');
    expect(extractors.output(trace)).toEqual('My output');
    expect(extractors.context(trace)).toEqual(CONTEXT_TEXTS);
  });

  it('returns custom extractors by stepName', async () => {
    const config: configModule.ToolsConfig = {
      evaluators: [
        {
          actionRef: '/flow/multiSteps',
          extractors: {
            output: 'step1',
          },
        },
      ],
    };
    const spy = vi.spyOn(configModule, 'findToolsConfig');
    spy.mockReturnValue(Promise.resolve(config));
    const trace = new MockTrace('My input', 42)
      .addSpan({
        stepName: 'retrieverStep',
        spanType: 'action',
        retrieverConfig: {
          query: 'What are cats?',
          text: CONTEXT_TEXTS,
        },
      })
      .addSpan({
        stepName: 'step1',
        spanType: 'flowStep',
        input: 'step-input',
        output: { out: 'my-object-output' },
      })
      .getTrace();

    const extractors = await getEvalExtractors('/flow/multiSteps');

    expect(extractors.input(trace)).toEqual('My input');
    expect(extractors.output(trace)).toEqual({ out: 'my-object-output' });
    expect(extractors.context(trace)).toEqual(CONTEXT_TEXTS);
  });

  it('returns custom extractors by stepSelector', async () => {
    const config: configModule.ToolsConfig = {
      evaluators: [
        {
          actionRef: '/flow/multiSteps',
          extractors: {
            output: { inputOf: 'step2' },
            context: { outputOf: 'step3-array' },
          },
        },
      ],
    };
    const spy = vi.spyOn(configModule, 'findToolsConfig');
    spy.mockReturnValue(Promise.resolve(config));
    const trace = new MockTrace('My input', 42)
      .addSpan({
        stepName: 'retrieverStep',
        spanType: 'action',
        retrieverConfig: {
          query: 'What are cats?',
          text: CONTEXT_TEXTS,
        },
      })
      .addSpan({
        stepName: 'step2',
        spanType: 'flowStep',
        input: 'step2-input',
        output: 'step2-output',
      })
      .addSpan({
        stepName: 'step3-array',
        spanType: 'flowStep',
        input: 'step3-input',
        output: ['Hello', 'World'],
      })
      .getTrace();

    const extractors = await getEvalExtractors('/flow/multiSteps');

    expect(extractors.input(trace)).toEqual('My input');
    expect(extractors.output(trace)).toEqual('step2-input');
    expect(extractors.context(trace)).toEqual(['Hello', 'World']);
  });

  it('returns custom extractors by trace function', async () => {
    const config: configModule.ToolsConfig = {
      evaluators: [
        {
          actionRef: '/flow/multiSteps',
          extractors: {
            input: (trace: TraceData) => {
              return Object.values(trace.spans)
                .filter(
                  (s) =>
                    s.attributes['genkit:type'] === 'action' &&
                    s.attributes['genkit:metadata:subtype'] !== 'retriever'
                )
                .map((s) => {
                  const inputValue = JSON.parse(
                    s.attributes['genkit:input'] as string
                  ).start.input;
                  if (!inputValue) {
                    return '';
                  }
                  return inputValue + ' TEST TEST TEST';
                });
            },
            output: { inputOf: 'step2' },
            context: { outputOf: 'step3-array' },
          },
        },
      ],
    };
    const spy = vi.spyOn(configModule, 'findToolsConfig');
    spy.mockReturnValue(Promise.resolve(config));
    const trace = new MockTrace('My input', 42)
      .addSpan({
        stepName: 'retrieverStep',
        spanType: 'action',
        retrieverConfig: {
          query: 'What are cats?',
          text: CONTEXT_TEXTS,
        },
      })
      .addSpan({
        stepName: 'step2',
        spanType: 'flowStep',
        input: 'step2-input',
        output: 'step2-output',
      })
      .addSpan({
        stepName: 'step3-array',
        spanType: 'flowStep',
        input: 'step3-input',
        output: ['Hello', 'World'],
      })
      .getTrace();

    const extractors = await getEvalExtractors('/flow/multiSteps');

    expect(extractors.input(trace)).toEqual(['My input TEST TEST TEST']);
    expect(extractors.output(trace)).toEqual('step2-input');
    expect(extractors.context(trace)).toEqual(['Hello', 'World']);
  });

  it('returns runs default extractors when trace fails', async () => {
    const spy = vi.spyOn(configModule, 'findToolsConfig');
    spy.mockReturnValue(Promise.resolve(null));
    const trace = new MockTrace('My input', '', 'error')
      .addSpan({
        stepName: 'retrieverStep',
        spanType: 'action',
        retrieverConfig: {
          query: 'What are cats?',
          text: CONTEXT_TEXTS,
        },
      })
      .getTrace();

    const extractors = await getEvalExtractors('/flow/multiSteps');

    expect(Object.keys(extractors).sort()).toEqual(
      ['input', 'output', 'context'].sort()
    );
    expect(extractors.input(trace)).toEqual('My input');
    expect(extractors.output(trace)).toEqual('');
    expect(extractors.context(trace)).toEqual(CONTEXT_TEXTS);
  });
});
