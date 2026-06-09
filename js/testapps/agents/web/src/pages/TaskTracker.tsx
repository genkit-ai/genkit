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

import { remoteAgent, type AgentChat } from 'genkit/beta/client';
import { useCallback, useMemo, useRef, useState } from 'react';
import { ChatUI, type ChatMessage } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Task Tracker — Custom State Agent
//
// Demonstrates features not covered by other samples:
//   • `session.updateCustom()` / `session.getCustom()` — typed custom state
//     maintained alongside message history inside the session
//   • Tools that mutate structured state inside the session
//   • Reading `chat.state` to display structured state alongside chat
//   • Uses `defineAgent` (not defineCustomAgent) — custom state works
//     seamlessly with the standard agent API
//   • Client-managed multi-turn — the `remoteAgent` client threads state
//     across turns automatically
//
// The user chats naturally ("Add buy groceries", "Mark task 1 done") and
// the model uses tools to manage a typed task list stored in session.custom.
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/taskAgent';

interface TaskItem {
  id: number;
  title: string;
  done: boolean;
}

interface TaskState {
  tasks: TaskItem[];
  nextId: number;
}

export default function TaskTracker() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);

  // Task state — extracted from chat.state each turn
  const [tasks, setTasks] = useState<TaskItem[]>([]);

  // Typed HTTP client. Tracks session state across turns for us.
  const agent = useMemo(() => remoteAgent<TaskState>({ url: ENDPOINT }), []);

  // The conversation. Created on first send (seeding the initial custom state)
  // and reused for follow-up turns so history is threaded automatically.
  const chatRef = useRef<AgentChat<TaskState> | null>(null);

  const handleSend = useCallback(
    async (text: string) => {
      if (loading) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');

      // Lazily create the chat, seeding the initial custom state so the tools
      // have something to work with on the very first turn.
      if (!chatRef.current) {
        chatRef.current = agent.chat({
          state: {
            custom: { tasks: [], nextId: 1 },
            messages: [],
            artifacts: [],
          },
        });
      }
      const chat = chatRef.current;

      try {
        const turn = chat.sendStream(text);

        let accumulated = '';
        for await (const chunk of turn.stream) {
          // ── Tool calls/responses — render inline from the raw chunk ──
          for (const part of chunk.raw.modelChunk?.content ?? []) {
            if (part.toolRequest) {
              const tr = part.toolRequest;
              setMessages((prev) => [
                ...prev,
                {
                  role: 'tool',
                  text: `🔧 ${tr.name}(${JSON.stringify(tr.input)})`,
                },
              ]);
            } else if (part.toolResponse) {
              const tr = part.toolResponse;
              setMessages((prev) => [
                ...prev,
                {
                  role: 'tool',
                  text: `✅ ${tr.name} → ${JSON.stringify(tr.output)}`,
                },
              ]);
            }
          }

          // ── Model text chunks ──
          if (chunk.text) {
            accumulated = chunk.accumulatedText;
            setStreamingText(accumulated);
          }
        }

        const res = await turn.response;
        setStreamingText('');

        // Reflect the authoritative final custom state in the sidebar.
        if (chat.state?.tasks) {
          setTasks([...chat.state.tasks]);
        }

        setMessages((prev) => [
          ...prev,
          { role: 'model', text: res.text || accumulated },
        ]);
      } catch (err: any) {
        setStreamingText('');
        setMessages((prev) => [
          ...prev,
          { role: 'system', text: `Error: ${err.message}` },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [agent, loading]
  );

  const doneCount = tasks.filter((t) => t.done).length;
  const totalCount = tasks.length;

  return (
    <div className="task-tracker-layout">
      <ChatUI
        title="Task Tracker"
        description="Chat-based task management powered by custom state."
        suggestions={[
          'Add buy groceries',
          'Add finish the report by Friday',
          'What tasks do I have?',
        ]}
        messages={messages}
        streamingText={streamingText}
        loading={loading}
        onSend={handleSend}
      />

      {/* Task State Panel — live view of session.custom */}
      <aside className="task-sidebar">
        <h3>📋 Task List</h3>
        <p className="task-sidebar-hint">
          Live view of <code>session.custom</code> — read from{' '}
          <code>chat.state</code> after each turn.
        </p>

        {tasks.length === 0 ? (
          <div className="task-empty">
            No tasks yet. Ask the agent to add some!
          </div>
        ) : (
          <>
            <div className="task-progress">
              {doneCount}/{totalCount} completed
              <div className="task-progress-bar">
                <div
                  className="task-progress-fill"
                  style={{
                    width:
                      totalCount > 0
                        ? `${(doneCount / totalCount) * 100}%`
                        : '0%',
                  }}
                />
              </div>
            </div>
            <ul className="task-list">
              {tasks.map((task) => (
                <li
                  key={task.id}
                  className={`task-item ${task.done ? 'task-done' : ''}`}>
                  <span className="task-checkbox">
                    {task.done ? '✅' : '⬜'}
                  </span>
                  <span className="task-title">
                    #{task.id}: {task.title}
                  </span>
                </li>
              ))}
            </ul>
          </>
        )}

        <details className="task-state-raw" open={false}>
          <summary>🔍 Raw Custom State JSON</summary>
          <pre className="task-state-json">
            {tasks.length > 0
              ? JSON.stringify(tasks, null, 2)
              : '(no state yet)'}
          </pre>
        </details>

        <hr className="task-divider" />

        <h4>📋 How It Works</h4>
        <ol className="task-howto">
          <li>
            The backend uses <code>defineAgent</code> — the standard agent API
            handles model calls and tool dispatch automatically.
          </li>
          <li>
            Three tools (<code>addTask</code>, <code>toggleTask</code>,{' '}
            <code>removeTask</code>) mutate the typed{' '}
            <code>session.custom</code> state via{' '}
            <code>ai.currentSession().updateCustom()</code>.
          </li>
          <li>
            After each turn, the client reads <code>chat.state</code> to update
            the task list panel.
          </li>
          <li>
            The <code>remoteAgent</code> client threads the full session state
            (messages + custom) across turns automatically.
          </li>
        </ol>

        <h4>Key APIs</h4>
        <pre className="task-code">{`// Backend — update custom state in a tool
const session = ai.currentSession();
session.updateCustom((state) => {
  state.tasks.push({ id: state.nextId++, title, done: false });
  return state;
});

// Client — read custom state from the chat
const agent = remoteAgent<TaskState>({ url: '/api/taskAgent' });
const chat = agent.chat({ state: { custom: { tasks: [], nextId: 1 } } });
await chat.send('Add buy groceries');
setTasks(chat.state.tasks);`}</pre>
      </aside>
    </div>
  );
}
