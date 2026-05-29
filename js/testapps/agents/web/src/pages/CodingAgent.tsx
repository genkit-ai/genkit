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

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { ChatUI, type ChatMessage } from '../components/ChatUI';
import { useGenkitAgent, useGenkitRunFlow } from '../genkit-react';

// ---------------------------------------------------------------------------
// Coding Agent — AI coding assistant with filesystem access
//
// This is the most advanced sample, combining multiple Genkit patterns:
//
// Backend APIs demonstrated:
//   • `defineAgent` with middleware composition (filesystem, skills,
//     toolApproval, retry)
//   • `defineInterrupt` for the ask_user tool (respond pattern)
//   • `defineTool` with AI-powered safety gate and manual interrupt
//   • `defineFlow` for workspace browser (listWorkspaceFiles, readWorkspaceFile)
//   • `FileSessionStore` for persistent sessions & interrupt resumption
//   • `getSnapshotDataAction` for restoring sessions from URL
//
// Client APIs demonstrated:
//   • `streamFlow()` for streaming agent responses
//   • `runFlow()` for non-streaming workspace file operations
//   • Two interrupt resumption patterns:
//     - **Restart pattern** (resume.restart) — for write_file, search_and_replace,
//       run_shell. Re-executes the tool with `{ toolApproved: true }` metadata.
//     - **Respond pattern** (resume.respond) — for ask_user. Sends the
//       user's answer directly without re-executing the tool.
//   • Session continuity via snapshotId
//   • Streaming reasoning/thinking content
// ---------------------------------------------------------------------------

// Use relative paths so the Vite dev server's `/api` proxy handles
// routing to the backend (matches every other page in this testapp).
const ENDPOINT = '/api/codingAgent';
const STATE_ENDPOINT = '/api/codingAgent/state';
const WORKSPACE_FILES_ENDPOINT = '/api/workspace/files';
const WORKSPACE_FILE_ENDPOINT = '/api/workspace/file';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface WorkspaceFile {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: WorkspaceFile[];
}

interface PendingApproval {
  toolName: string;
  ref?: string;
  input: any;
  snapshotId: string;
}

interface PendingQuestion {
  question: string;
  options: string[];
  ref?: string;
  snapshotId: string;
}

/** Part shape used by AgentMessage content + filesystem middleware metadata. */
interface AgentPart {
  text?: string;
  reasoning?: string;
  toolRequest?: { name: string; input?: any; ref?: string };
  toolResponse?: { name: string; output?: any; ref?: string };
  metadata?: {
    filesystemMiddlewareTool?: string;
    filePath?: string;
    [k: string]: unknown;
  };
  [k: string]: unknown;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const INTERRUPT_TOOLS = new Set([
  'ask_user',
  'write_file',
  'search_and_replace',
  'run_shell',
]);

/** Should this part be hidden from the chat transcript? */
function shouldSkipPart(p: AgentPart): boolean {
  // Hide raw read_file payloads and filesystem-middleware text injections —
  // we render them as compact "📖 Reading …" tool bubbles instead.
  if (p.metadata?.filesystemMiddlewareTool) return true;
  return false;
}

export default function CodingAgent() {
  const { snapshotId: urlSnapshotId } = useParams<{ snapshotId: string }>();
  const navigate = useNavigate();

  // Capture the URL-provided snapshotId at mount-time so the agent hook's
  // resume effect doesn't re-fire when our own navigate() pushes a new
  // snapshotId mid-session.
  const initialUrlSnapshotId = useRef(urlSnapshotId).current;

  // The headline hook — handles streaming, in-stream interrupts, snapshot
  // restoration, continuation round-trip, and aborts.
  const agent = useGenkitAgent({
    url: ENDPOINT,
    stateUrl: STATE_ENDPOINT,
    resumeFromSnapshotId: initialUrlSnapshotId,
  });

  // System-only chat entries (approvals, denials, ask_user answers,
  // restore failures, etc.) that don't exist in the agent's message
  // history. Indexed by `agent.messages.length` at insertion time so we
  // can interleave them with the agent's own messages in display order.
  const [synthEvents, setSynthEvents] = useState<
    Array<{ id: number; afterIndex: number; message: ChatMessage }>
  >([]);
  const synthEventIdRef = useRef(0);
  const pushSynth = useCallback(
    (message: ChatMessage) => {
      setSynthEvents((prev) => [
        ...prev,
        {
          id: ++synthEventIdRef.current,
          afterIndex: agent.messages.length,
          message,
        },
      ]);
    },
    [agent.messages.length]
  );

  // Local UI state — dialogs are driven from the agent's interrupt signal.
  const [approval, setApproval] = useState<PendingApproval | null>(null);
  const [question, setQuestion] = useState<PendingQuestion | null>(null);
  const [customAnswer, setCustomAnswer] = useState('');

  // Track whether we're still hydrating the URL-loaded session.
  const [restoring, setRestoring] = useState(!!urlSnapshotId);
  useEffect(() => {
    if (restoring && (agent.phase !== 'idle' || agent.error)) {
      setRestoring(false);
    }
  }, [restoring, agent.phase, agent.error]);

  // File explorer state
  const [files, setFiles] = useState<WorkspaceFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [fileLoading, setFileLoading] = useState(false);

  // Sibling endpoint hooks — no raw `runFlow` imports needed.
  const filesFlow = useGenkitRunFlow<void, { files: WorkspaceFile[] }>({
    url: WORKSPACE_FILES_ENDPOINT,
  });
  const fileFlow = useGenkitRunFlow<string, { path: string; content: string }>({
    url: WORKSPACE_FILE_ENDPOINT,
  });

  // ── Fetch workspace file tree ──────────────────────────────────────────
  // Depend on the stable `.run` reference, not the full hook result —
  // otherwise the inevitable re-render after `setFiles` would recreate
  // `refreshFiles`, re-firing the useEffect and looping forever.
  const filesRun = filesFlow.run;
  const refreshFiles = useCallback(async () => {
    try {
      const data = await filesRun();
      setFiles(data.files || []);
    } catch {
      // ignore — workspace may not exist yet
    }
  }, [filesRun]);

  // Load files on mount.
  useEffect(() => {
    refreshFiles();
  }, [refreshFiles]);

  // Refresh files whenever a turn finishes — the agent may have created or
  // edited files. Watch for the streaming → terminal phase transition.
  const prevPhaseRef = useRef(agent.phase);
  useEffect(() => {
    const wasActive =
      prevPhaseRef.current === 'streaming' ||
      prevPhaseRef.current === 'background';
    const nowTerminal =
      agent.phase === 'done' ||
      agent.phase === 'awaiting-interrupt' ||
      agent.phase === 'error';
    if (wasActive && nowTerminal) refreshFiles();
    prevPhaseRef.current = agent.phase;
  }, [agent.phase, refreshFiles]);

  // Mirror the agent's snapshotId into the URL so the session is bookmarkable.
  useEffect(() => {
    if (agent.snapshotId && agent.snapshotId !== urlSnapshotId) {
      navigate(`/coding-agent/${agent.snapshotId}`, { replace: true });
    }
  }, [agent.snapshotId, urlSnapshotId, navigate]);

  // ── Fetch a single file's content ──────────────────────────────────────
  const fileRun = fileFlow.run;
  const viewFile = useCallback(
    async (filePath: string) => {
      setSelectedFile(filePath);
      setFileLoading(true);
      try {
        const data = await fileRun(filePath);
        setFileContent(data.content || '');
      } catch {
        setFileContent('(failed to load file)');
      } finally {
        setFileLoading(false);
      }
    },
    [fileRun]
  );

  // ── Drive approval/question dialogs from agent.pendingInterrupt ───────
  // The hook fires this on in-stream interrupt events. Restored interrupts
  // (from a snapshot whose last turn was an interrupt) are handled by the
  // separate effect below since they don't pass through the event stream.
  useEffect(() => {
    const pi = agent.pendingInterrupt;
    if (!pi) return;
    const sid = agent.snapshotId ?? '';
    if (pi.toolName === 'ask_user') {
      const ask = pi.input as { question?: string; options?: string[] };
      setQuestion({
        question: ask?.question || 'What would you like to do?',
        options: ask?.options || [],
        ref: pi.toolCallId,
        snapshotId: sid,
      });
    } else {
      setApproval({
        toolName: pi.toolName,
        ref: pi.toolCallId,
        input: pi.input,
        snapshotId: sid,
      });
    }
  }, [agent.pendingInterrupt, agent.snapshotId]);

  // ── Detect restored interrupts when hydrating from URL ─────────────────
  // The hook's resume effect loads the snapshot's messages but does not
  // surface a `pendingInterrupt` for restored state. Sniff the last message
  // once after restoration.
  const restoreSniffedRef = useRef(false);
  useEffect(() => {
    if (restoring) return;
    if (restoreSniffedRef.current) return;
    if (!initialUrlSnapshotId) return;
    restoreSniffedRef.current = true;
    const last = agent.messages[agent.messages.length - 1];
    if (!last) return;
    const sid = agent.snapshotId ?? '';
    for (const p of (last.content || []) as AgentPart[]) {
      const tr = p.toolRequest;
      if (!tr) continue;
      if (tr.name === 'ask_user') {
        const ask = tr.input as { question?: string; options?: string[] };
        setQuestion({
          question: ask?.question || 'What would you like to do?',
          options: ask?.options || [],
          ref: tr.ref,
          snapshotId: sid,
        });
        return;
      }
      if (INTERRUPT_TOOLS.has(tr.name)) {
        setApproval({
          toolName: tr.name,
          ref: tr.ref,
          input: tr.input,
          snapshotId: sid,
        });
        return;
      }
    }
  }, [restoring, agent.messages, agent.snapshotId, initialUrlSnapshotId]);

  // ── Surface streaming errors as a system message in the transcript ────
  const prevErrorRef = useRef<Error | null>(null);
  useEffect(() => {
    if (agent.error && agent.error !== prevErrorRef.current) {
      pushSynth({ role: 'system', text: `Error: ${agent.error.message}` });
    }
    prevErrorRef.current = agent.error;
  }, [agent.error, pushSynth]);

  // ── Send a regular user message ──────────────────────────────────────
  const handleSend = useCallback(
    (text: string) => {
      if (agent.phase === 'streaming' || approval || question) return;
      agent.submit({
        messages: [{ role: 'user', content: [{ text }] }],
      });
    },
    [agent, approval, question]
  );

  // ── Respond to a tool approval interrupt ──────────────────────────────
  const handleApprovalResponse = useCallback(
    (approved: boolean) => {
      if (!approval) return;
      const a = approval;
      setApproval(null);
      pushSynth({
        role: 'system',
        text: approved
          ? `✅ Approved: ${a.toolName}`
          : `❌ Denied: ${a.toolName}`,
      });
      if (approved) {
        // resume.restart with the toolApproved flag — read by toolApproval
        // middleware (write_file, search_and_replace) and by run_shell's
        // own handler. We submit explicitly (rather than via
        // restartInterrupt) because the interrupt may have been restored
        // from a snapshot, in which case the hook's `pendingInterrupt`
        // wasn't populated by an in-stream event.
        agent.submit({
          resume: {
            restart: [
              {
                toolRequest: {
                  name: a.toolName,
                  ref: a.ref,
                  input: a.input,
                },
                metadata: { resumed: { toolApproved: true } },
              },
            ],
          },
        });
      } else {
        // For denial, send a fresh user message so the model knows the tool
        // was rejected. resume.restart would loop the interrupt.
        agent.submit({
          messages: [
            {
              role: 'user',
              content: [
                {
                  text:
                    `I denied the "${a.toolName}" tool call` +
                    (a.toolName === 'run_shell'
                      ? ` for command: "${a.input?.command}".`
                      : ` for file: "${a.input?.filePath}".`) +
                    ` Please continue without executing it, or suggest an alternative.`,
                },
              ],
            },
          ],
        });
      }
    },
    [agent, approval, pushSynth]
  );

  // ── Respond to an ask_user interrupt ───────────────────────────────────
  const handleQuestionResponse = useCallback(
    (answer: string) => {
      if (!question) return;
      const q = question;
      setQuestion(null);
      setCustomAnswer('');
      pushSynth({ role: 'system', text: `💬 Answer: ${answer}` });
      // Same rationale as handleApprovalResponse: restored interrupts
      // don't surface as agent.pendingInterrupt, so submit explicitly.
      agent.submit({
        resume: {
          respond: [
            {
              toolResponse: {
                name: 'ask_user',
                ref: q.ref,
                output: { answer },
              },
            },
          ],
        },
      });
    },
    [agent, question, pushSynth]
  );

  // ── Derive the displayed chat transcript from agent.messages ──────────
  // Tool calls / responses are filtered (interrupt tools rendered as
  // dialogs, read_file rendered as a compact "📖 Reading" bubble),
  // filesystem-middleware text injections are dropped, and synthetic
  // system events are interleaved at their insertion points.
  const messages = useMemo<ChatMessage[]>(() => {
    const out: ChatMessage[] = [];
    const insertSynthsAt = (idx: number) => {
      for (const e of synthEvents) {
        if (e.afterIndex === idx) out.push(e.message);
      }
    };
    insertSynthsAt(0);
    agent.messages.forEach((m, i) => {
      const role = m.role as ChatMessage['role'];
      const content = (m.content || []) as AgentPart[];

      // Text body — combine text parts, skip middleware-injected payloads.
      const textBuf: string[] = [];
      const reasoningBuf: string[] = [];
      for (const p of content) {
        if (shouldSkipPart(p)) continue;
        if (p.text) textBuf.push(p.text);
        if (p.reasoning) reasoningBuf.push(p.reasoning);
      }
      if (textBuf.length > 0 || reasoningBuf.length > 0) {
        out.push({
          role,
          text: textBuf.join(''),
          reasoning: reasoningBuf.join('') || undefined,
        });
      }

      // Tool requests — skip interrupt tools (rendered as dialogs); render
      // read_file as a compact "📖 Reading {path}" bubble.
      for (const p of content) {
        const tr = p.toolRequest;
        if (!tr) continue;
        if (INTERRUPT_TOOLS.has(tr.name)) continue;
        const tmsg = formatToolRequest(tr.name, tr.input);
        out.push({ role: 'tool', ...tmsg });
      }

      // Tool responses — skip read_file (the "📖 Reading" bubble covers it).
      for (const p of content) {
        const tr = p.toolResponse;
        if (!tr) continue;
        if (tr.name === 'read_file') continue;
        const tmsg = formatToolResponse(tr.name, tr.output);
        out.push({ role: 'tool', ...tmsg });
      }

      insertSynthsAt(i + 1);
    });
    return out;
  }, [agent.messages, synthEvents]);

  const loading = agent.phase === 'streaming';

  // ── Restoring state — show loading UI while fetching snapshot ──────────
  if (restoring) {
    return (
      <div className="coding-agent-layout">
        <div className="chat-panel">
          <div className="chat-header">
            <h2>Coding Agent</h2>
            <span className="chat-desc">Restoring session…</span>
          </div>
          <div className="chat-messages">
            <div className="message">
              <div className="message-role">system</div>
              <div className="message-text loading">
                Restoring session from snapshot {urlSnapshotId}…
              </div>
            </div>
          </div>
        </div>
        <aside className="file-explorer" />
      </div>
    );
  }

  return (
    <div className="coding-agent-layout">
      {/* Main chat panel */}
      <ChatUI
        title="Coding Agent"
        description="AI coding assistant with filesystem access, skills, tool approval, and shell execution."
        suggestions={[
          'I want to build something fun. Ask me to pick 3 quick/simple project ideas.',
          'Create a TypeScript Express hello world app',
          'List the files in the workspace',
          'Create a Python script that generates fibonacci numbers',
        ]}
        messages={messages}
        streamingText={agent.streamingText}
        streamingReasoning={agent.streamingReasoning}
        loading={loading}
        onSend={handleSend}
        inputDisabled={!!approval || !!question}
        renderMarkdown
        headerAction={
          <Link to="/coding-agent" className="btn-new-session" reloadDocument>
            ✨ New Session
          </Link>
        }>
        {/* Tool approval dialog */}
        {approval && (
          <div className="approval-dialog">
            <h3>⚠️ Tool Approval Required</h3>
            <div className="approval-tool-name">
              <span className="approval-label">Tool:</span>{' '}
              <code>{approval.toolName}</code>
            </div>

            {approval.toolName === 'write_file' && (
              <div className="approval-details">
                <div className="approval-file-path">
                  <span className="approval-label">File:</span>{' '}
                  <code>{approval.input?.filePath}</code>
                </div>
                <div className="approval-content-preview">
                  <span className="approval-label">Content:</span>
                  <pre className="approval-code">
                    {approval.input?.content || '(empty)'}
                  </pre>
                </div>
              </div>
            )}

            {approval.toolName === 'search_and_replace' && (
              <div className="approval-details">
                <div className="approval-file-path">
                  <span className="approval-label">File:</span>{' '}
                  <code>{approval.input?.filePath}</code>
                </div>
                <div className="approval-content-preview">
                  <span className="approval-label">Edits:</span>
                  {(approval.input?.edits || []).map(
                    (edit: string, i: number) => (
                      <pre key={i} className="approval-code approval-diff">
                        {edit}
                      </pre>
                    )
                  )}
                </div>
              </div>
            )}

            {approval.toolName === 'run_shell' && (
              <div className="approval-details">
                <div className="approval-file-path">
                  <span className="approval-label">Command:</span>{' '}
                  <code className="approval-command">
                    {approval.input?.command}
                  </code>
                </div>
                <div className="approval-content-preview">
                  <div className="approval-warning">
                    🛡️ This shell command was flagged as potentially dangerous
                    by the AI safety gate. Review the command carefully before
                    approving.
                  </div>
                </div>
              </div>
            )}

            <div className="approval-buttons">
              <button
                className="btn btn-approve"
                onClick={() => handleApprovalResponse(true)}>
                ✅ Approve
              </button>
              <button
                className="btn btn-deny"
                onClick={() => handleApprovalResponse(false)}>
                ❌ Deny
              </button>
            </div>
          </div>
        )}

        {/* Ask user question dialog */}
        {question && (
          <div className="ask-user-dialog">
            <h3>❓ Question from Agent</h3>
            <p className="ask-user-question">{question.question}</p>

            <div className="ask-user-options">
              {question.options.map((opt, i) => (
                <button
                  key={i}
                  className="ask-user-option"
                  onClick={() => handleQuestionResponse(opt)}>
                  {opt}
                </button>
              ))}
            </div>

            <div className="ask-user-custom-section">
              <span className="ask-user-custom-label">Or write your own:</span>
              <div className="ask-user-custom-row">
                <input
                  type="text"
                  className="ask-user-custom"
                  placeholder="Type your answer…"
                  value={customAnswer}
                  onChange={(e) => setCustomAnswer(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && customAnswer.trim()) {
                      handleQuestionResponse(customAnswer.trim());
                    }
                  }}
                />
                <button
                  className="btn btn-send"
                  disabled={!customAnswer.trim()}
                  onClick={() => {
                    if (customAnswer.trim()) {
                      handleQuestionResponse(customAnswer.trim());
                    }
                  }}>
                  Send
                </button>
              </div>
            </div>
          </div>
        )}
      </ChatUI>

      {/* File explorer sidebar */}
      <aside className="file-explorer">
        <div className="file-explorer-header">
          <h3>📁 Workspace</h3>
          <button
            className="btn-refresh-files"
            onClick={refreshFiles}
            title="Refresh file list">
            🔄
          </button>
        </div>

        {files.length === 0 ? (
          <p className="file-explorer-empty">Workspace is empty.</p>
        ) : (
          <div className="file-tree">
            <FileTree
              files={files}
              selectedFile={selectedFile}
              onSelect={viewFile}
            />
          </div>
        )}

        {/* File content viewer */}
        {selectedFile && (
          <div className="file-viewer">
            <div className="file-viewer-header">
              <span className="file-viewer-path">{selectedFile}</span>
              <button
                className="file-viewer-close"
                onClick={() => {
                  setSelectedFile(null);
                  setFileContent('');
                }}>
                ✕
              </button>
            </div>
            <pre className="file-viewer-content">
              {fileLoading ? 'Loading…' : fileContent}
            </pre>
          </div>
        )}
      </aside>
    </div>
  );
}

// ---------------------------------------------------------------------------
// File Tree component
// ---------------------------------------------------------------------------

function FileTree({
  files,
  selectedFile,
  onSelect,
  depth = 0,
}: {
  files: WorkspaceFile[];
  selectedFile: string | null;
  onSelect: (path: string) => void;
  depth?: number;
}) {
  return (
    <>
      {files.map((f) => (
        <div key={f.path}>
          {f.type === 'directory' ? (
            <DirectoryNode
              file={f}
              selectedFile={selectedFile}
              onSelect={onSelect}
              depth={depth}
            />
          ) : (
            <button
              className={`file-tree-item ${selectedFile === f.path ? 'selected' : ''}`}
              style={{ paddingLeft: `${12 + depth * 16}px` }}
              onClick={() => onSelect(f.path)}>
              <span className="file-icon">📄</span>
              <span className="file-name">{f.name}</span>
            </button>
          )}
        </div>
      ))}
    </>
  );
}

function DirectoryNode({
  file,
  selectedFile,
  onSelect,
  depth,
}: {
  file: WorkspaceFile;
  selectedFile: string | null;
  onSelect: (path: string) => void;
  depth: number;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <button
        className="file-tree-item file-tree-dir"
        style={{ paddingLeft: `${12 + depth * 16}px` }}
        onClick={() => setExpanded(!expanded)}>
        <span className="file-icon">{expanded ? '📂' : '📁'}</span>
        <span className="file-name">{file.name}</span>
      </button>
      {expanded && file.children && (
        <FileTree
          files={file.children}
          selectedFile={selectedFile}
          onSelect={onSelect}
          depth={depth + 1}
        />
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Tool display formatters
// ---------------------------------------------------------------------------

/** Truncate a string, adding ellipsis if it exceeds maxLen. */
function truncate(s: string, maxLen = 200): string {
  return s.length > maxLen ? s.substring(0, maxLen) + '…' : s;
}

/** Show first N lines of content with a "(+X more lines)" note. */
function previewLines(content: string, maxLines = 20): string {
  const lines = content.split('\n');
  if (lines.length <= maxLines) return content;
  return (
    lines.slice(0, maxLines).join('\n') +
    `\n… (+${lines.length - maxLines} more lines)`
  );
}

interface ToolMsg {
  text: string;
  detail?: string;
}

/** Pretty-format a tool request for inline display. */
function formatToolRequest(name: string, input: any): ToolMsg {
  switch (name) {
    case 'write_file':
      return {
        text: `📝 Writing ${input?.filePath || 'file'}`,
        detail: previewLines(input?.content || '', 20),
      };

    case 'read_file':
      return { text: `📖 Reading ${input?.filePath || 'file'}` };

    case 'list_files':
      return { text: `📁 Listing files in ${input?.directory || '/'}` };

    case 'search_and_replace': {
      const file = input?.filePath || 'file';
      const edits: any[] = input?.edits || [];
      if (edits.length === 0) return { text: `✏️ Editing ${file}` };
      const diffPreview = edits
        .slice(0, 3)
        .map((e: any) => {
          if (typeof e === 'string') return truncate(e, 150);
          const search = truncate(String(e.search || ''), 80);
          const replace = truncate(String(e.replace || ''), 80);
          return `"${search}" → "${replace}"`;
        })
        .join('\n');
      const moreNote =
        edits.length > 3 ? `\n… (+${edits.length - 3} more edits)` : '';
      return { text: `✏️ Editing ${file}`, detail: diffPreview + moreNote };
    }

    case 'run_shell':
      return { text: `🖥️ $ ${input?.command || '(unknown command)'}` };

    case 'use_skill':
      return { text: `📚 Loading skill: ${input?.skillName || '(unknown)'}` };

    default: {
      const inputStr =
        typeof input === 'object'
          ? truncate(JSON.stringify(input), 300)
          : truncate(String(input ?? ''), 300);
      return { text: `🔧 ${name}`, detail: inputStr };
    }
  }
}

/** Pretty-format a tool response for inline display. */
function formatToolResponse(name: string, output: any): ToolMsg {
  const outputStr =
    typeof output === 'string' ? output : JSON.stringify(output);

  switch (name) {
    case 'write_file':
      return { text: '✅ File written' };

    case 'read_file':
      return { text: '✅ File content:', detail: previewLines(outputStr, 15) };

    case 'list_files': {
      // Parse JSON array and show a nice file list
      let fileList = outputStr;
      try {
        const files = typeof output === 'string' ? JSON.parse(output) : output;
        if (Array.isArray(files)) {
          fileList = files
            .map((f: any) => `${f.isDirectory ? '📁' : '📄'} ${f.path}`)
            .join('\n');
        }
      } catch {
        /* use raw string */
      }
      return { text: '✅ Files:', detail: fileList || '(empty)' };
    }

    case 'search_and_replace':
      return { text: '✅ Edits applied' };

    case 'run_shell': {
      // Extract stdout/stderr from structured output
      let shellText = outputStr;
      if (typeof output === 'object' && output !== null) {
        const parts: string[] = [];
        if (output.stdout) parts.push(output.stdout);
        if (output.stderr) parts.push(`(stderr) ${output.stderr}`);
        if (output.exitCode !== undefined && output.exitCode !== 0)
          parts.push(`Exit code: ${output.exitCode}`);
        shellText = parts.join('\n') || '(no output)';
      }
      return { text: '✅ Shell output:', detail: truncate(shellText, 500) };
    }

    case 'use_skill':
      return { text: '✅ Skill loaded' };

    default:
      return { text: `✅ ${name}`, detail: truncate(outputStr, 400) };
  }
}
