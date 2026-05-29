/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

import { useMemo } from 'react';
import { ChatUI, type ChatMessage } from '../components/ChatUI';
import { useGenkitAgent } from '../genkit-react';

// Task Tracker — custom state demonstration. v2 hook surfaces
// `agent.customState` as a typed reactive value derived from
// `result.state.custom` on every turn.

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
  const agent = useGenkitAgent<TaskState>({ url: ENDPOINT });

  const handleSend = (text: string) => {
    if (agent.phase === 'streaming') return;
    agent.submit({ messages: [{ role: 'user', content: [{ text }] }] });
  };

  const chatMessages = useMemo<ChatMessage[]>(
    () => agent.messages.flatMap(messageToChatRows),
    [agent.messages]
  );

  const tasks = agent.customState?.tasks ?? [];
  const doneCount = tasks.filter((t) => t.done).length;
  const totalCount = tasks.length;

  return (
    <div className="task-tracker-layout">
      <ChatUI
        title="Task Tracker — v2"
        description="Chat-based task management with typed custom state."
        suggestions={[
          'Add buy groceries',
          'Add finish the report by Friday',
          'What tasks do I have?',
        ]}
        messages={chatMessages}
        streamingText={agent.streamingText}
        loading={agent.phase === 'streaming'}
        onSend={handleSend}
      />

      <aside className="task-sidebar">
        <h3>📋 Task List</h3>
        <p className="task-sidebar-hint">
          <code>agent.customState</code> is a typed reactive view of{' '}
          <code>session.custom</code>. Updates automatically each turn.
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

        <details className="task-state-raw">
          <summary>🔍 Raw customState JSON</summary>
          <pre className="task-state-json">
            {agent.customState
              ? JSON.stringify(agent.customState, null, 2)
              : '(no state yet)'}
          </pre>
        </details>

        <hr className="task-divider" />
        <h4>Page code</h4>
        <pre className="task-code">{`const agent = useGenkitAgent<TaskState>({ url });

// Typed reactive accessor — updates each turn
const tasks = agent.customState?.tasks ?? [];`}</pre>
      </aside>
    </div>
  );
}

function messageToChatRows(msg: any): ChatMessage[] {
  const rows: ChatMessage[] = [];
  const textParts: string[] = [];
  for (const p of msg.content ?? []) {
    if (p.text) textParts.push(p.text);
    if (p.toolRequest) {
      rows.push({
        role: 'tool',
        text: `🔧 ${p.toolRequest.name}(${JSON.stringify(p.toolRequest.input)})`,
      });
    }
    if (p.toolResponse) {
      rows.push({
        role: 'tool',
        text: `✅ ${p.toolResponse.name} → ${JSON.stringify(p.toolResponse.output)}`,
      });
    }
  }
  if (textParts.length > 0) {
    rows.unshift({ role: msg.role, text: textParts.join('') });
  }
  return rows;
}
