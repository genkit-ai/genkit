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
 * Custom State Agent — Task Tracker
 *
 * Demonstrates features not covered by other samples:
 *   • `session.updateCustom()` / `session.getCustom()` — typed custom state
 *   • Tools that mutate structured state inside the session
 *   • Reading `result.state.custom` on the client to display live state
 *   • Uses `defineAgent` (not defineCustomAgent) — custom state works
 *     seamlessly with the standard agent API
 *
 * The user chats naturally ("Add buy groceries", "Mark task 1 done",
 * "What's left?") and the model uses tools to mutate a typed task list
 * stored in `session.custom`.
 */

import { z } from 'genkit';
import { ai } from './genkit.js';

// ---------------------------------------------------------------------------
// Typed custom state
// ---------------------------------------------------------------------------

interface TaskItem {
  id: number;
  title: string;
  done: boolean;
}

interface TaskState {
  tasks: TaskItem[];
  nextId: number;
}

// ---------------------------------------------------------------------------
// Tools — the model calls these to mutate custom state
// ---------------------------------------------------------------------------

const addTask = ai.defineTool(
  {
    name: 'addTask',
    description:
      'Add a new task to the task list. Returns the newly created task.',
    inputSchema: z.object({
      title: z.string().describe('Short description of the task'),
    }),
    outputSchema: z.object({
      id: z.number(),
      title: z.string(),
      done: z.boolean(),
    }),
  },
  async (input) => {
    const session = ai.currentSession<TaskState>();
    let newTask!: TaskItem;
    session.updateCustom((state) => {
      const s = state || { tasks: [], nextId: 1 };
      newTask = { id: s.nextId, title: input.title, done: false };
      s.tasks.push(newTask);
      s.nextId++;
      return s;
    });
    return newTask;
  }
);

const toggleTask = ai.defineTool(
  {
    name: 'toggleTask',
    description:
      'Toggle a task between done and not-done by its ID. Returns the updated task or an error message.',
    inputSchema: z.object({
      id: z.number().describe('The task ID to toggle'),
    }),
    outputSchema: z.object({
      success: z.boolean(),
      task: z
        .object({
          id: z.number(),
          title: z.string(),
          done: z.boolean(),
        })
        .optional(),
      error: z.string().optional(),
    }),
  },
  async (input) => {
    const session = ai.currentSession<TaskState>();
    let result: { success: boolean; task?: TaskItem; error?: string };
    session.updateCustom((state) => {
      const s = state || { tasks: [], nextId: 1 };
      const task = s.tasks.find((t) => t.id === input.id);
      if (task) {
        task.done = !task.done;
        result = { success: true, task: { ...task } };
      } else {
        result = { success: false, error: `Task ${input.id} not found` };
      }
      return s;
    });
    return result!;
  }
);

const removeTask = ai.defineTool(
  {
    name: 'removeTask',
    description: 'Remove a task from the list by its ID.',
    inputSchema: z.object({
      id: z.number().describe('The task ID to remove'),
    }),
    outputSchema: z.object({
      success: z.boolean(),
      error: z.string().optional(),
    }),
  },
  async (input) => {
    const session = ai.currentSession<TaskState>();
    let result: { success: boolean; error?: string };
    session.updateCustom((state) => {
      const s = state || { tasks: [], nextId: 1 };
      const idx = s.tasks.findIndex((t) => t.id === input.id);
      if (idx >= 0) {
        s.tasks.splice(idx, 1);
        result = { success: true };
      } else {
        result = { success: false, error: `Task ${input.id} not found` };
      }
      return s;
    });
    return result!;
  }
);

// ---------------------------------------------------------------------------
// Agent — uses defineAgent (the standard shortcut API)
// Custom state works seamlessly — tools call ai.currentSession().updateCustom()
// ---------------------------------------------------------------------------

export const taskAgent = ai.defineAgent<{}, TaskState>({
  name: 'taskPrompt',
  model: 'googleai/gemini-flash-latest',
  system: `You are a concise task management assistant. Help the user manage their task list.

Rules:
- Use the addTask tool to add new tasks.
- Use the toggleTask tool to mark tasks done or undone.
- Use the removeTask tool to delete tasks.
- Be brief and friendly. After modifying tasks, confirm what you did.`,
  tools: [addTask, toggleTask, removeTask],
});

// ---------------------------------------------------------------------------
// Test flow
// ---------------------------------------------------------------------------

export const testTaskAgent = ai.defineFlow(
  {
    name: 'testTaskAgent',
    inputSchema: z.string().default('Add a task: buy groceries'),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    const res = await taskAgent.run(
      {
        messages: [{ role: 'user' as const, content: [{ text }] }],
      },
      {
        init: {
          state: {
            custom: { tasks: [], nextId: 1 } as TaskState,
            messages: [],
            artifacts: [],
          },
        },
        onChunk: sendChunk,
      }
    );

    return {
      message: res.result.message,
      customState: res.result.state?.custom,
    };
  }
);
