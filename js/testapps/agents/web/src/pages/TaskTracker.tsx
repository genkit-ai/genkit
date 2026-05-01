import { useCallback, useRef, useState } from 'react';
import { streamFlow } from 'genkit/beta/client';
import type { AgentInit, AgentInput, AgentOutput, AgentStreamChunk } from 'genkit/beta';
import { ChatUI, type Message } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Task Tracker — Custom State Agent
//
// Demonstrates features not covered by other samples:
//   • `session.updateCustom()` / `session.getCustom()` — typed custom state
//     maintained alongside message history inside the session
//   • Tools that mutate structured state inside the session
//   • Reading `result.state.custom` to display structured state alongside chat
//   • Uses `defineAgent` (not defineCustomAgent) — custom state works
//     seamlessly with the standard agent API
//   • Client-managed multi-turn via `init: { state }` round-tripping
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
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);

  // Task state — extracted from result.state.custom each turn
  const [tasks, setTasks] = useState<TaskItem[]>([]);

  // Session state — round-tripped to the server each turn
  const stateRef = useRef<any>(undefined);

  const handleSend = useCallback(
    async (text: string) => {
      if (loading) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');

      // ── Build the request ──────────────────────────────────────────────
      const input: AgentInput = {
        messages: [{ role: 'user', content: [{ text }] }],
      };

      // On first turn init is empty; subsequent turns send back state.
      // We ensure the custom state is initialised so the tools have
      // something to work with on the very first turn.
      const init: AgentInit = stateRef.current
        ? { state: stateRef.current }
        : {
            state: {
              custom: { tasks: [], nextId: 1 } as TaskState,
              messages: [],
              artifacts: [],
            },
          };

      try {
        // ── Stream the response ────────────────────────────────────────
        const response = streamFlow<AgentOutput, AgentStreamChunk, AgentInit>({
          url: ENDPOINT,
          input,
          init,
        });

        let accumulated = '';
        for await (const chunk of response.stream) {
          // ── Model text/tool chunks ──────────────────────────────────
          const mc = chunk?.modelChunk;
          if (!mc) continue;

          for (const part of mc.content || []) {
            if (part.text) {
              accumulated += part.text;
              setStreamingText(accumulated);
            } else if (part.toolRequest) {
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
        }

        // ── Read the final result ──────────────────────────────────────
        const result = await response.output;
        setStreamingText('');

        // Save session state for the next turn.
        if (result?.state) {
          stateRef.current = result.state;

          // Extract and display the custom task state.
          const custom = result.state.custom as TaskState | undefined;
          if (custom?.tasks) {
            setTasks([...custom.tasks]);
          }
        }

        const replyText = extractText(result);
        setMessages((prev) => [
          ...prev,
          { role: 'model', text: replyText || accumulated },
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
    [loading]
  );

  const doneCount = tasks.filter((t) => t.done).length;
  const totalCount = tasks.length;

  return (
    <div className="task-tracker-layout">
      <ChatUI
        title="Task Tracker"
        description='Chat-based task management powered by custom state. Try: "Add buy groceries", "Mark task 1 done", "What tasks do I have?"'
        messages={messages}
        streamingText={streamingText}
        loading={loading}
        onSend={handleSend}
      />

      {/* Task State Panel — live view of session.custom */}
      <aside className="task-sidebar">
        <h3>📋 Task List</h3>
        <p className="task-sidebar-hint">
          Live view of <code>session.custom</code> — updated each turn from{' '}
          <code>result.state.custom</code>.
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
                    width: totalCount > 0 ? `${(doneCount / totalCount) * 100}%` : '0%',
                  }}
                />
              </div>
            </div>
            <ul className="task-list">
              {tasks.map((task) => (
                <li key={task.id} className={`task-item ${task.done ? 'task-done' : ''}`}>
                  <span className="task-checkbox">{task.done ? '✅' : '⬜'}</span>
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
            {stateRef.current?.custom
              ? JSON.stringify(stateRef.current.custom, null, 2)
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
            After each turn, the client reads{' '}
            <code>result.state.custom</code> to update the task list panel.
          </li>
          <li>
            The full <code>state</code> blob (messages + custom) is sent back
            on the next turn via <code>{'init: { state }'}</code>.
          </li>
        </ol>

        <h4>Key APIs</h4>
        <pre className="task-code">{`// Backend — update custom state in a tool
const session = ai.currentSession();
session.updateCustom((state) => {
  state.tasks.push({ id: state.nextId++, title, done: false });
  return state;
});

// Client — read custom state from result
const custom = result.state.custom;
setTasks(custom.tasks);`}</pre>
      </aside>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helper: extract displayable text from a session flow result
// ---------------------------------------------------------------------------
function extractText(result: AgentOutput): string {
  if (!result) return '(no result)';
  const msg = result.message;
  if (!msg) return JSON.stringify(result, null, 2);
  const parts: string[] = [];
  for (const p of msg.content || []) {
    if (p.text) parts.push(p.text);
  }
  return parts.join('') || JSON.stringify(result, null, 2);
}
