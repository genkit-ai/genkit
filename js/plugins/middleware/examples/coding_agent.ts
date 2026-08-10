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

import { googleAI } from '@genkit-ai/google-genai';
import * as fs from 'fs';
import {
  genkit,
  restartTool,
  type GenerateResponse,
  type MessageData,
  type ToolRequestPart,
} from 'genkit';
import * as path from 'path';
import * as readline from 'readline';
import { filesystem, retry, skills, toolApproval } from '../src/index.js';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function askQuestion(query: string): Promise<string> {
  return new Promise((resolve) => rl.question(query, resolve));
}

async function main() {
  const ai = genkit({
    plugins: [
      googleAI(),
      filesystem.plugin(),
      skills.plugin(),
      toolApproval.plugin(),
      retry.plugin(),
    ],
  });

  const currentDir = process.cwd();
  const fsRoot = path.join(currentDir, 'workspace');
  const skillsRoot = path.join(currentDir, 'examples', 'skills');

  // Ensure workspace exists
  if (!fs.existsSync(fsRoot)) {
    fs.mkdirSync(fsRoot, { recursive: true });
  }

  console.log('--- Coding Agent ---');
  console.log('Type your request. To exit, type "exit".');

  let messages: MessageData[] = [
    {
      role: 'system',
      content: [
        {
          text:
            `You are a helpful coding agent. Very terse but thoughtful and careful.\n` +
            `Your working directory is in ${fsRoot}, you are not allowed to access anything outside it.\n` +
            `Use skills. ALWAYS start by analyzing the current state of the workspace, ` +
            `there might be something already there.`,
        },
      ],
    },
  ];

  while (true) {
    const input = await askQuestion('\n> ');
    if (input.trim().toLowerCase() === 'exit') {
      break;
    }

    try {
      let interruptRestart: ToolRequestPart[] | undefined;
      let response: GenerateResponse;

      while (true) {
        response = await ai.generate({
          model: 'googleai/gemini-flash-latest',
          prompt: interruptRestart ? undefined : input,
          messages: messages,
          resume: interruptRestart ? { restart: interruptRestart } : undefined,
          use: [
            toolApproval({
              approved: ['read_file', 'list_files', 'use_skill'],
            }),
            skills({ skillPaths: [skillsRoot] }),
            filesystem({ rootDirectory: fsRoot, allowWriteAccess: true }),
          ],
          maxTurns: 20,
        });

        if (response.finishReason !== 'interrupted') {
          break;
        }

        const interrupts = response.interrupts;
        if (!interrupts || interrupts.length === 0) {
          console.log('Interrupted but no interrupt record found.');
          break;
        }

        const approvedInterrupts: ToolRequestPart[] = [];
        for (const interrupt of interrupts) {
          console.log('\n*** Tool Approval Required ***');
          console.log(`Tool: ${interrupt.toolRequest.name}`);
          console.log(`Input: ${JSON.stringify(interrupt.toolRequest.input)}`);

          const approval = await askQuestion('Approve? (y/N): ');
          if (approval.trim().toLowerCase() === 'y') {
            approvedInterrupts.push(
              restartTool(interrupt, { toolApproved: true })
            );
          }
        }

        if (approvedInterrupts.length > 0) {
          console.log('Resuming...');
          interruptRestart = approvedInterrupts;
          messages = response.messages;
        } else {
          console.log('Tool denied.');
          break;
        }
      }

      console.log(`\nAI Response:\n${response.text}`);
      messages = response.messages;
    } catch (e) {
      console.error('Error during generation:', e);
    }
  }
  rl.close();
}

main();
