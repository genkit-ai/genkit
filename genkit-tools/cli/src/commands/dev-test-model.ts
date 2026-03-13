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
  GenerateRequestData,
  GenerateResponseChunkData,
  GenerateResponseData,
  GenerateResponseSchema,
  Part,
} from '@genkit-ai/tools-common';
import {
  GenkitToolsError,
  RuntimeManager,
} from '@genkit-ai/tools-common/manager';
import { findProjectRoot, logger } from '@genkit-ai/tools-common/utils';
import { Command } from 'commander';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import { parse } from 'yaml';
import { startDevProcessManager, startManager } from '../utils/manager-utils';

interface TestOptions {
  supports: string;
  fromFile?: string;
}

type TestCase = {
  name: string;
  input: GenerateRequestData;
  validators: string[];
  stream?: boolean;
};

type TestSuite = {
  model: string;
  supports?: string[];
  tests?: TestCase[];
};

const getMessageText = (response: GenerateResponseData): string | undefined => {
  const message = response.message || response.candidates?.[0]?.message;
  return message?.content?.[0]?.text;
};

const getMessageContent = (response: GenerateResponseData) => {
  const message = response.message || response.candidates?.[0]?.message;
  return message?.content;
};

const getMediaPart = (response: GenerateResponseData) => {
  const content = getMessageContent(response);
  return content?.find((p: Part) => p.media);
};

const imageBase64 =
  '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';

const VALIDATORS: Record<
  string,
  (
    response: GenerateResponseData,
    arg?: string,
    chunks?: GenerateResponseChunkData[]
  ) => void
> = {
  'has-tool-request': (response, toolName) => {
    const content = getMessageContent(response);
    if (!content || !Array.isArray(content)) {
      throw new Error(
        `Response missing message content. Full response: ${JSON.stringify(
          response,
          null,
          2
        )}`
      );
    }
    const toolRequest = content.find((c: Part) => c.toolRequest);
    if (!toolRequest) {
      throw new Error(
        `Model did not return a tool request. Content: ${JSON.stringify(
          content,
          null,
          2
        )}`
      );
    }
    if (toolName && toolRequest.toolRequest?.name !== toolName) {
      throw new Error(
        `Expected tool request '${toolName}', got '${toolRequest.toolRequest?.name}'`
      );
    }
  },
  'valid-json': (response) => {
    const content = getMessageContent(response);
    if (!content || !Array.isArray(content)) {
      throw new Error(
        `Response missing message content. Full response: ${JSON.stringify(
          response,
          null,
          2
        )}`
      );
    }
    const textPart = content.find((c: Part) => c.text);
    if (!textPart) {
      throw new Error(
        `Model did not return text content for JSON. Content: ${JSON.stringify(
          content,
          null,
          2
        )}`
      );
    }
    try {
      JSON.parse(textPart.text!);
    } catch (e) {
      throw new Error(
        `Response text is not valid JSON. Text: ${textPart.text}`
      );
    }
  },
  'text-includes': (response, expected) => {
    const text = getMessageText(response);
    if (
      !text ||
      (expected && !text.toLowerCase().includes(expected.toLowerCase()))
    ) {
      throw new Error(
        `Response text does not include '${expected}'. Text: ${text}`
      );
    }
  },
  'stream-text-includes': (response, expected, chunks) => {
    if (!chunks || chunks.length === 0) {
      throw new Error('Streaming expected but no chunks were received');
    }

    const streamedText = chunks
      .map((c) => c.content?.find((p: any) => p.text)?.text || '')
      .join('');

    if (expected && !streamedText.includes(expected)) {
      throw new Error(`Streaming response did not include ${expected}'`);
    }
  },
  'stream-has-tool-request': (response, toolName, chunks) => {
    if (!chunks || chunks.length === 0) {
      throw new Error('Streaming expected but no chunks were received');
    }

    const hasTool = chunks.some((c) =>
      c.content?.some((p: any) => !!p.toolRequest)
    );
    if (!hasTool) {
      throw new Error('No tool request found in the streamed chunks');
    }

    if (toolName) {
      VALIDATORS['has-tool-request'](response, toolName);
    }
  },
  'stream-valid-json': (response, arg, chunks) => {
    if (!chunks || chunks.length === 0) {
      throw new Error('Streaming expected but no chunks were received');
    }

    const streamedText = chunks
      .map((c) => c.content?.find((p: any) => p.text)?.text || '')
      .join('');

    if (!streamedText.trim()) {
      throw new Error('Streamed response contained no text');
    }
    try {
      JSON.parse(streamedText);
    } catch (e) {
      throw new Error(`Streamed text is not valid JSON: ${streamedText}`);
    }
  },
  'text-starts-with': (response, expected) => {
    const text = getMessageText(response);
    if (!text || (expected && !text.trim().startsWith(expected))) {
      throw new Error(
        `Response text does not start with '${expected}'. Text: ${text}`
      );
    }
  },
  'text-not-empty': (response) => {
    const text = getMessageText(response);
    if (!text || text.trim().length === 0) {
      throw new Error('Response text is empty');
    }
  },
  'valid-media': (response, type) => {
    const mediaPart = getMediaPart(response);
    if (!mediaPart) {
      throw new Error(`Model did not return ${type || 'media'} part.`);
    }
    if (type) {
      if (
        mediaPart.media?.contentType &&
        !mediaPart.media.contentType.startsWith(`${type}/`)
      ) {
        throw new Error(
          `Expected ${type} content type, got ${mediaPart.media.contentType}`
        );
      }
    }
    if (type === 'image') {
      const url = mediaPart.media?.url;
      if (!url) throw new Error('Media part missing URL');
      if (url.startsWith('data:')) {
        if (!url.startsWith('data:image/')) {
          throw new Error('Invalid data URL content type for image');
        }
      } else if (url.startsWith('http')) {
        try {
          new URL(url);
        } catch (e) {
          throw new Error(`Invalid URL: ${url}`);
        }
      } else {
        throw new Error(`Unknown URL format: ${url}`);
      }
    }
  },
  reasoning: (response) => {
    const content = getMessageContent(response);

    if (!content || !Array.isArray(content)) {
      throw new Error(`Response is missing message content`);
    }

    const hasReasoning = content.some((p: any) => !!p.reasoning);
    if (!hasReasoning) {
      throw new Error(`reasoning content not found`);
    }
  },
};

const TEST_CASES: Record<string, TestCase> = {
  'tool-request': {
    name: 'Tool Request Conformance',
    input: {
      messages: [
        {
          role: 'user',
          content: [{ text: 'What is the weather in New York? Use the tool.' }],
        },
      ],
      tools: [
        {
          name: 'weather',
          description: 'Get the weather for a city',
          inputSchema: {
            type: 'object',
            properties: {
              city: { type: 'string' },
            },
            required: ['city'],
          },
        },
      ],
    },
    validators: ['has-tool-request:weather'],
  },
  'structured-output': {
    name: 'Structured Output Conformance',
    input: {
      messages: [
        {
          role: 'user',
          content: [{ text: 'Generate a profile for John Doe.' }],
        },
      ],
      output: {
        format: 'json',
        schema: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            age: { type: 'number' },
          },
          required: ['name', 'age'],
        },
        constrained: true,
      },
    },
    validators: ['valid-json'],
  },
  multiturn: {
    name: 'Multiturn Conformance',
    input: {
      messages: [
        { role: 'user', content: [{ text: 'My name is Genkit.' }] },
        { role: 'model', content: [{ text: 'Hello Genkit.' }] },
        { role: 'user', content: [{ text: 'What is my name?' }] },
      ],
    },
    validators: ['text-includes:Genkit'],
  },
  'streaming-multiturn': {
    name: 'Multiturn Conformance with streaming',
    stream: true,
    input: {
      messages: [
        { role: 'user', content: [{ text: 'My name is Genkit.' }] },
        { role: 'model', content: [{ text: 'Hello Genkit.' }] },
        { role: 'user', content: [{ text: 'What is my name?' }] },
      ],
    },
    validators: ['stream-text-includes:Genkit'],
  },
  'streaming-tool-request': {
    name: 'Tool Request Conformance with streaming',
    stream: true,
    input: {
      messages: [
        {
          role: 'user',
          content: [
            { text: 'What is the weather in New York? Use the weather tool' },
          ],
        },
      ],
      tools: [
        {
          name: 'weather',
          description: 'Get the weather for a city',
          inputSchema: {
            type: 'object',
            properties: {
              city: { type: 'string' },
            },
            required: ['city'],
          },
        },
      ],
    },
    validators: ['stream-has-tool-request:weather'],
  },
  'streaming-structured-output': {
    name: 'Structured Output Conformance with streaming',
    stream: true,
    input: {
      messages: [
        {
          role: 'user',
          content: [{ text: 'Generate a movie review for John Wick' }],
        },
      ],
      output: {
        format: 'json',
        schema: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            rating: { type: 'number' },
          },
          required: ['name', 'rating'],
        },
        constrained: true,
      },
    },
    validators: ['stream-valid-json'],
  },
  'system-role': {
    name: 'System Role Conformance',
    input: {
      messages: [
        {
          role: 'system',
          content: [
            {
              text: "IMPORTANT: your response are machine processed, always start/prefix your response with 'RESPONSE:', ex: 'RESPONSE: hello'",
            },
          ],
        },
        { role: 'user', content: [{ text: 'hello' }] },
      ],
    },
    validators: ['text-starts-with:RESPONSE:'],
  },
  'input-image-base64': {
    name: 'Image Input (Base64) Conformance',
    input: {
      messages: [
        {
          role: 'user',
          content: [
            { text: 'What text do you see in this image?' },
            {
              media: {
                url: `data:image/png;base64,${imageBase64}`,
                contentType: 'image/png',
              },
            },
          ],
        },
      ],
    },
    validators: ['text-includes:genkit'],
  },
  'input-image-url': {
    name: 'Image Input (URL) Conformance',
    input: {
      messages: [
        {
          role: 'user',
          content: [
            { text: 'What is this logo?' },
            {
              media: {
                url: 'https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png',
                contentType: 'image/png',
              },
            },
          ],
        },
      ],
    },
    validators: ['text-includes:google'],
  },
  'input-video-youtube': {
    name: 'Video Input (YouTube) Conformance',
    input: {
      messages: [
        {
          role: 'user',
          content: [
            { text: 'Describe this video.' },
            {
              media: {
                url: 'https://www.youtube.com/watch?v=3p1P5grjXIQ',
                contentType: 'video/mp4',
              },
            },
          ],
        },
      ],
    },
    validators: ['text-not-empty'],
  },
  'output-audio': {
    name: 'Audio Output (TTS) Conformance',
    input: {
      messages: [{ role: 'user', content: [{ text: 'Say hello.' }] }],
    },
    validators: ['valid-media:audio'],
  },
  'output-image': {
    name: 'Image Output (Generation) Conformance',
    input: {
      messages: [
        {
          role: 'user',
          content: [{ text: 'Generate an image of a cat.' }],
        },
      ],
    },
    validators: ['valid-media:image'],
  },
};

async function waitForRuntime(manager: RuntimeManager) {
  // Poll for runtimes
  for (let i = 0; i < 20; i++) {
    if (manager.listRuntimes().length > 0) return;
    await new Promise((r) => setTimeout(r, 500));
  }
  logger.warn('Runtime not detected after 10 seconds.');
}

/**
 * Waits until all model actions referenced by the test suites are registered
 * with the runtime's reflection server. This prevents 404 errors caused by
 * dispatching tests before the runtime has finished registering actions.
 */
async function waitForActions(
  manager: RuntimeManager,
  suites: TestSuite[]
): Promise<void> {
  const requiredKeys = new Set<string>();
  for (const suite of suites) {
    if (suite.model) {
      const key = suite.model.startsWith('/')
        ? suite.model
        : `/model/${suite.model}`;
      requiredKeys.add(key);
    }
  }
  if (requiredKeys.size === 0) return;

  const maxAttempts = 60; // 30 seconds total
  const delayMs = 500;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const actions = await manager.listActions();
      const registeredKeys = new Set(Object.keys(actions));
      const missing = [...requiredKeys].filter((k) => !registeredKeys.has(k));
      if (missing.length === 0) {
        logger.info(
          `All ${requiredKeys.size} model actions registered. Starting tests.`
        );
        return;
      }
      if (attempt % 10 === 0 && attempt > 0) {
        logger.info(
          `Waiting for ${missing.length} model action(s) to register: ${missing.join(', ')}...`
        );
      }
    } catch (e) {
      // Actions endpoint not ready yet, keep polling.
      logger.debug(`Polling for actions failed, will retry: ${e}`);
    }
    await new Promise((r) => setTimeout(r, delayMs));
  }
  logger.warn(
    'Not all model actions registered after 30 seconds. Proceeding anyway.'
  );
}

async function runTest(
  manager: RuntimeManager,
  model: string,
  testCase: TestCase
): Promise<boolean> {
  logger.info(`Running test: ${testCase.name}...`);
  try {
    // Adjust model name if needed (e.g. /model/ prefix)
    const modelKey = model.startsWith('/') ? model : `/model/${model}`;
    const shouldStream = !!testCase.stream;

    const collectedChunks: any[] = [];

    const actionResponse = await manager.runAction(
      {
        key: modelKey,
        input: testCase.input,
      },
      shouldStream
        ? (chunk) => {
            collectedChunks.push(chunk);
          }
        : undefined
    );

    if (shouldStream && collectedChunks.length === 0) {
      throw new Error('Streaming requested but no chunks received.');
    }
    const response = GenerateResponseSchema.parse(actionResponse.result);

    for (const v of testCase.validators) {
      const [valName, ...args] = v.split(':');
      const arg = args.join(':');
      const validator = VALIDATORS[valName];
      if (!validator) throw new Error(`Unknown validator: ${valName}`);
      validator(response, arg, collectedChunks);
    }

    logger.info(`✅ Passed: ${testCase.name}`);
    return true;
  } catch (e) {
    if (e instanceof GenkitToolsError) {
      logger.error(
        `❌ Failed: ${testCase.name} - ${
          e.data?.stack || JSON.stringify(e.data?.details) || e
        }`
      );
    } else if (e instanceof Error) {
      logger.error(`❌ Failed: ${testCase.name} - ${e.message}`);
    } else {
      logger.error(`❌ Failed: ${testCase.name} - ${JSON.stringify(e)}`);
    }
    return false;
  }
}

async function runTestSuite(
  manager: RuntimeManager,
  suite: TestSuite,
  defaultSupports: string[]
): Promise<{ passed: number; failed: number }> {
  const supports = suite.supports || (suite.tests ? [] : defaultSupports);

  logger.info(`Testing model: ${suite.model}`);

  const promises: Promise<boolean>[] = [];

  // Built-in conformance tests
  for (const support of supports) {
    const testCase = TEST_CASES[support];
    if (testCase) {
      promises.push(runTest(manager, suite.model, testCase));
    } else {
      logger.warn(`Unknown capability: ${support}`);
    }
  }

  // Custom tests
  if (suite.tests) {
    for (const test of suite.tests) {
      const customTestCase: TestCase = {
        name: test.name || 'Custom Test',
        input: test.input,
        validators: test.validators || [],
        stream: test.stream,
      };
      promises.push(runTest(manager, suite.model, customTestCase));
    }
  }

  const results = await Promise.all(promises);
  const passed = results.filter((r) => r).length;
  const failed = results.filter((r) => !r).length;

  return { passed, failed };
}

export const devTestModel = new Command('dev:test-model')
  .description('Test a model against the Genkit model specification')
  .argument('[modelOrCmd]', 'Model name or command')
  .argument('[args...]', 'Command arguments')
  .option(
    '--supports <list>',
    'Comma-separated list of supported capabilities (tool-request, structured-output, multiturn, system-role, input-image-base64, input-image-url, input-video-youtube, output-audio, output-image, streaming-multiturn, reasoning)',
    'tool-request,structured-output,multiturn,system-role,input-image-base64,input-image-url,streaming-multiturn,streaming-tool-request,streaming-structured-output'
  )
  .option('--from-file <file>', 'Path to a file containing test payloads')
  .action(
    async (
      modelOrCmd: string | undefined,
      args: string[] | undefined,
      options: TestOptions
    ) => {
      const projectRoot = await findProjectRoot();

      let cmd: string[] = [];
      let defaultModelName: string | undefined;

      if (options.fromFile) {
        if (modelOrCmd) cmd.push(modelOrCmd);
        if (args) cmd.push(...args);
      } else {
        if (!modelOrCmd) {
          logger.error('Model name is required unless --from-file is used.');
          process.exitCode = 1;
          return;
        }
        defaultModelName = modelOrCmd;
        if (args) cmd = args;
      }

      let manager: RuntimeManager;

      if (cmd.length > 0) {
        const result = await startDevProcessManager(
          projectRoot,
          cmd[0],
          cmd.slice(1)
        );
        manager = result.manager;
      } else {
        manager = await startManager({ projectRoot, manageHealth: false });
      }

      await waitForRuntime(manager);

      try {
        let totalPassed = 0;
        let totalFailed = 0;

        let suites: TestSuite[] = [];

        if (options.fromFile) {
          const filePath = resolve(projectRoot, options.fromFile);
          const fileContent = readFileSync(filePath, 'utf-8');
          let parsed;
          if (filePath.endsWith('.yaml') || filePath.endsWith('.yml')) {
            parsed = parse(fileContent);
          } else {
            parsed = JSON.parse(fileContent);
          }
          suites = Array.isArray(parsed) ? parsed : [parsed];
        } else {
          if (!defaultModelName) throw new Error('Model name required');
          suites = [{ model: defaultModelName }];
        }

        const defaultSupports = options.supports
          .split(',')
          .map((s) => s.trim());

        // Wait for all model actions to be registered before dispatching tests.
        // This prevents 404 errors when the runtime is still initializing.
        await waitForActions(manager, suites);

        for (const suite of suites) {
          if (!suite.model) {
            logger.error('Model name required in test suite.');
            totalFailed++;
            continue;
          }
          const { passed, failed } = await runTestSuite(
            manager,
            suite,
            defaultSupports
          );
          totalPassed += passed;
          totalFailed += failed;
        }

        logger.info('--------------------------------------------------');
        logger.info(
          `Tests Completed: ${totalPassed} Passed, ${totalFailed} Failed`
        );

        if (totalFailed > 0) {
          process.exitCode = 1;
        }
      } catch (e) {
        logger.error('Error running tests:', e);
        process.exitCode = 1;
      } finally {
        if (manager) {
          await manager.stop();
        }
      }
    }
  );
