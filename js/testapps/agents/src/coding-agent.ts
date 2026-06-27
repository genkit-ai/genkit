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
 * Coding Agent — a full-featured AI coding assistant
 *
 * Demonstrates a rich combination of middleware:
 *
 *   • `filesystem` — gives the agent list_files, read_file, write_file, and
 *     search_and_replace tools scoped to a workspace directory
 *   • `skills` — loads coding conventions / language guides on demand via
 *     the use_skill tool
 *   • `toolApproval` — requires user approval (interrupt) before file writes
 *     and edits; reads are auto-approved
 *   • `retry` — automatic retry on transient model errors
 *   • `run_shell` — custom tool with AI-powered safety gate that uses a fast
 *     model to evaluate shell commands before execution
 *
 * The agent operates in a sandboxed `workspace/` directory. Users can ask it
 * to create projects, edit files, refactor code, explain code, run builds,
 * install dependencies, and more.
 *
 * The tool approval pattern means that every write_file and search_and_replace
 * call triggers an interrupt — the client sees the proposed change and can
 * approve or deny it before it's executed.
 *
 * The run_shell tool has its own AI-powered safety gate: a cheap/fast model
 * evaluates each command. Safe commands (npm install, tsc, etc.) run
 * automatically; risky commands (rm -rf, system modifications) trigger an
 * interrupt so the user can approve or deny.
 */

import { filesystem, retry, skills, toolApproval } from '@genkit-ai/middleware';
import { exec } from 'child_process';
import * as fs from 'fs';
import { z } from 'genkit';
import { FileSessionStore } from 'genkit/beta';
import * as path from 'path';
import { promisify } from 'util';
import { ai, liteModel } from './genkit.js';

const execAsync = promisify(exec);

// File-based session store — persists sessions across server restarts,
// which is useful for a coding assistant where conversations can be long.
const store = new FileSessionStore<{}>('./.snapshots-coding');

// Resolve the workspace directory relative to the project root.
const WORKSPACE_DIR = path.resolve(__dirname, '..', 'workspace');
const SKILLS_DIR = path.resolve(__dirname, '..', 'skills');

// ---------------------------------------------------------------------------
// ask_user interrupt — lets the model ask the user a question with options
// ---------------------------------------------------------------------------

const askUser = ai.defineInterrupt({
  name: 'ask_user',
  description:
    'Ask the user a question when you need clarification, a preference, ' +
    'or a decision. Provide a clear question and 2-5 suggested options. ' +
    'The user can pick one of the options or write their own answer.',
  inputSchema: z.object({
    question: z.string().describe('The question to ask the user'),
    options: z
      .array(z.string())
      .min(2)
      .max(5)
      .describe('Suggested answer options for the user to choose from'),
  }),
  outputSchema: z.object({
    answer: z.string().describe("The user's selected or written answer"),
  }),
});

const runShell = ai.defineTool(
  {
    name: 'run_shell',
    description:
      'Execute a shell command in the workspace directory. Use for running ' +
      'build commands, installing dependencies, running scripts, testing, etc. ' +
      'Commands are safety-checked automatically; risky commands will require ' +
      'user approval.',
    inputSchema: z.object({
      command: z.string().describe('The shell command to execute'),
    }),
    outputSchema: z.object({
      stdout: z.string(),
      stderr: z.string(),
      exitCode: z.number(),
    }),
  },
  async (input, ctx) => {
    // Check if this is a resumed (user-approved) invocation.
    // When a user approves a risky command, resume.restart sets
    // metadata.resumed = { toolApproved: true }.
    const isApproved = ctx.metadata?.resumed?.toolApproved === true;

    if (!isApproved) {
      // AI-powered safety gate — use a fast/cheap model to evaluate the command.
      const safetyCheck = await ai.generate({
        model: liteModel,
        prompt: `You are a shell command safety evaluator. Evaluate the following shell command for safety.

Command: "${input.command}"
Working directory: A sandboxed workspace directory.

Consider these factors:
- Does it try to access files outside the workspace (e.g. /, /etc, ~, ..)?
- Is it destructive (rm -rf, format, mkfs, dd, etc.)?
- Does it modify system configuration or env permanently?
- Does it install system-wide packages or modify global state?
- Does it access network in a dangerous way (curl | bash, wget + exec, etc.)?
- Could it expose sensitive information (env vars, keys, passwords)?

Simple development commands like npm install, npx, tsc, node, cat, ls, mkdir,
echo, grep, find, git, python, etc. within the workspace are SAFE.

Respond with JSON.`,
        output: {
          schema: z.object({
            verdict: z
              .enum(['safe', 'risky'])
              .describe('Whether the command is safe or risky'),
            reason: z
              .string()
              .describe(
                'Brief explanation of why the command is safe or risky'
              ),
          }),
        },
      });

      const { verdict, reason } = safetyCheck.output!;

      if (verdict === 'risky') {
        // Interrupt — the framework will return this tool request to the
        // client. The client shows the command + reason and asks for approval.
        // If approved, the tool is restarted with { toolApproved: true }.
        ctx.interrupt({
          command: input.command,
          reason,
          verdict: 'risky',
        });
        // ctx.interrupt() throws, so we never reach here.
      }
    }

    // Execute the command in the workspace directory.
    try {
      const { stdout, stderr } = await execAsync(input.command, {
        cwd: WORKSPACE_DIR,
        timeout: 30_000, // 30 second timeout
        maxBuffer: 1024 * 1024, // 1MB output limit
        env: {
          ...process.env,
          HOME: WORKSPACE_DIR,
        },
      });
      return { stdout: stdout || '', stderr: stderr || '', exitCode: 0 };
    } catch (err: any) {
      return {
        stdout: err.stdout || '',
        stderr: err.stderr || err.message || 'Command failed',
        exitCode: err.code ?? 1,
      };
    }
  }
);

// ---------------------------------------------------------------------------
// Agent definition
// ---------------------------------------------------------------------------

export const codingAgent = ai.defineAgent({
  name: 'codingAgent',
  description:
    'An expert AI coding assistant that can read, create, edit files, ' +
    'and run shell commands in a sandboxed workspace.',
  system: `You are an expert AI coding assistant working in a sandboxed workspace directory.

You have access to filesystem tools to interact with the workspace:
- **list_files**: List files and directories in the workspace
- **read_file**: Read the contents of a file
- **write_file**: Create a new file or overwrite an existing one
- **search_and_replace**: Make surgical edits to existing files
- **run_shell**: Execute shell commands (npm install, tsc, node, etc.)
- **ask_user**: Ask the user a question when you need clarification or a choice

You also have access to a skills library with coding conventions and best practices.
Use the **use_skill** tool to load relevant skills before starting work.

## Rules

1. **Always explore first**: Use list_files and read_file to understand the existing codebase before making changes.
2. **Load relevant skills**: If a skill matches the task (e.g. "typescript" for TS work), load it first.
3. **Prefer surgical edits**: Use search_and_replace for small changes to existing files. Only use write_file for new files or complete rewrites.
4. **Use run_shell for builds & tests**: After writing code, use run_shell to build, lint, or test it when appropriate.
5. **Explain your work**: Before each file operation, explain what you're about to do and why. After, confirm what was done.
6. **One step at a time**: Don't try to create an entire project in one turn. Break complex tasks into logical steps.
7. **Handle errors gracefully**: If a file operation or shell command fails, explain the error and suggest a fix.
8. **Ask when uncertain**: If the user's request is ambiguous, involves a choice between alternatives, or asks you to suggest options — **always use the ask_user tool** to present options and let the user choose. Never just list options in plain text; use the tool so the user gets interactive buttons.

## Response Format

Use markdown for all responses. Use code blocks with language tags for code snippets.
When showing file changes, use diff-style formatting when helpful.`,
  // Standalone tools (not from middleware).
  tools: [runShell, askUser],
  use: [
    // Tool approval MUST come before filesystem so that ToolInterruptError
    // propagates without being caught by filesystem's error handler.
    // Reads and run_shell are auto-approved; writes require user confirmation.
    toolApproval({
      approved: [
        'list_files',
        'read_file',
        'use_skill',
        'run_shell',
        'ask_user',
      ],
    }),
    // Filesystem tools scoped to the workspace directory.
    filesystem({
      rootDirectory: WORKSPACE_DIR,
      allowWriteAccess: true,
    }),
    // Skills library — coding conventions, language guides, etc.
    skills({
      skillPaths: [SKILLS_DIR],
    }),
    // Automatic retry on transient model errors.
    retry(),
  ],
  // Server-side store is required for interrupt-based tool approval.
  store,
  maxTurns: 30,
});

// ---------------------------------------------------------------------------
// Test flow — for programmatic / CLI testing
//
// Uses the ergonomic chat API. Auto-approves all tool interrupts so the
// agent can complete its task without user input.
// ---------------------------------------------------------------------------

export const testCodingAgent = ai.defineFlow(
  {
    name: 'testCodingAgent',

    inputSchema: z
      .string()
      .default(
        'Create a simple TypeScript hello world file called hello.ts in the workspace.'
      ),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    const chat = codingAgent.chat();

    let turn = chat.sendStream(text);
    for await (const chunk of turn.stream) {
      sendChunk(chunk.raw);
    }
    let res = await turn.response;

    // Auto-approve all tool interrupts for testing. `res.interrupts` surfaces
    // each paused tool request with `respond`/`restart` builders.
    let maxResumes = 10;
    while (maxResumes-- > 0) {
      const interrupt = res.interrupts[0];
      if (!interrupt) break;

      if (interrupt.name === 'ask_user') {
        // Respond pattern: provide the tool output directly via the interrupt's
        // `respond` builder. The tool never re-executes.
        const firstOption =
          (interrupt.input as { options?: string[] })?.options?.[0] || 'Yes';
        sendChunk({ status: `Auto-answering ask_user: "${firstOption}"` });

        turn = chat.resumeStream({
          respond: [
            {
              ...interrupt.respond({ answer: firstOption }),
              metadata: { interruptResponse: true },
            },
          ],
        });
      } else {
        // Restart pattern: re-issue the original tool request via the
        // interrupt's `restart` builder, tagging it as approved so the tool
        // (write_file, search_and_replace, run_shell) executes this time.
        sendChunk({ status: `Auto-approving tool: ${interrupt.name}` });

        turn = chat.resumeStream({
          restart: [
            {
              ...interrupt.restart(),
              metadata: { resumed: { toolApproved: true } },
            },
          ],
        });
      }

      for await (const chunk of turn.stream) {
        sendChunk(chunk.raw);
      }
      res = await turn.response;
    }

    return res.raw;
  }
);

// ---------------------------------------------------------------------------
// Workspace browser flows — expose the workspace directory as Genkit flows
// so the web UI can browse files using runFlow() instead of raw fetch().
// ---------------------------------------------------------------------------

/** Schema for a single file/directory entry in the workspace. */
const WorkspaceFileSchema: z.ZodType<WorkspaceFile> = z.lazy(() =>
  z.object({
    name: z.string(),
    path: z.string(),
    type: z.enum(['file', 'directory']),
    children: z.array(WorkspaceFileSchema).optional(),
  })
);

interface WorkspaceFile {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: WorkspaceFile[];
}

/** List all files and directories in the workspace, recursively. */
export const listWorkspaceFiles = ai.defineFlow(
  {
    name: 'listWorkspaceFiles',
    inputSchema: z.void(),
    outputSchema: z.object({ files: z.array(WorkspaceFileSchema) }),
  },
  async () => {
    const files = await walkDirectory(WORKSPACE_DIR, WORKSPACE_DIR);
    return { files };
  }
);

/** Read the contents of a single file in the workspace. */
export const readWorkspaceFile = ai.defineFlow(
  {
    name: 'readWorkspaceFile',
    inputSchema: z.string().describe('Relative path within the workspace'),
    outputSchema: z.object({
      path: z.string(),
      content: z.string(),
    }),
  },
  async (filePath) => {
    const fullPath = path.resolve(WORKSPACE_DIR, filePath);
    if (!fullPath.startsWith(WORKSPACE_DIR)) {
      throw new Error('Path outside workspace');
    }
    const content = fs.readFileSync(fullPath, 'utf8');
    return { path: filePath, content };
  }
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Recursively list files in a directory, sorted (directories first). */

async function walkDirectory(
  dir: string,
  rootDir: string
): Promise<WorkspaceFile[]> {
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });
  const result: WorkspaceFile[] = [];

  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const fullPath = path.join(dir, entry.name);
    const relativePath = path.relative(rootDir, fullPath);

    if (entry.isDirectory()) {
      const children = await walkDirectory(fullPath, rootDir);
      result.push({
        name: entry.name,
        path: relativePath,
        type: 'directory',
        children,
      });
    } else {
      result.push({ name: entry.name, path: relativePath, type: 'file' });
    }
  }

  return result.sort((a, b) => {
    if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}
