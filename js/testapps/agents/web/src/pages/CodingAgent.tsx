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

import type { MessageData, Part } from 'genkit/beta';
import {
  AgentError,
  remoteAgent,
  runFlow,
  type AgentChat,
  type AgentResponse,
  type AgentTurn,
} from 'genkit/beta/client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { ChatUI, type ChatMessage } from '../components/ChatUI';

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
//
// Client APIs demonstrated (the ergonomic `remoteAgent` client):
//   • `agent.chat()` — a stateful conversation that tracks snapshotId/state
//   • `chat.send()` / `turn.stream` / `turn.response` for streaming turns
//   • `agent.loadChat({ snapshotId })` to restore a session from the URL
//   • `res.interrupts` to detect paused tool requests (ask_user + approvals)
//   • Two interrupt resumption patterns via `chat.resume()`:
//     - **Restart pattern** (resume.restart) — for write_file,
//       search_and_replace, run_shell. Re-executes the tool with
//       `{ toolApproved: true }` metadata.
//     - **Respond pattern** (resume.respond) — for ask_user. Sends the user's
//       answer directly without re-executing the tool.
//   • `runFlow()` for non-streaming workspace file operations (plain flows)
// ---------------------------------------------------------------------------

const API_BASE = 'http://localhost:8080';
const ENDPOINT = `${API_BASE}/api/codingAgent`;
const WORKSPACE_FILES_ENDPOINT = `${API_BASE}/api/workspace/files`;
const WORKSPACE_FILE_ENDPOINT = `${API_BASE}/api/workspace/file`;

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
}

interface PendingQuestion {
  question: string;
  options: string[];
  ref?: string;
}

/** Extended part type for parts that carry filesystem middleware metadata. */
type PartWithMetadata = Part & {
  metadata?: {
    filesystemMiddlewareTool?: string;
    filePath?: string;
  };
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CodingAgent() {
  const { snapshotId: urlSnapshotId } = useParams<{ snapshotId: string }>();
  const navigate = useNavigate();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [streamingReasoning, setStreamingReasoning] = useState('');
  const [loading, setLoading] = useState(false);
  const [restoring, setRestoring] = useState(!!urlSnapshotId);
  const [approval, setApproval] = useState<PendingApproval | null>(null);
  const [question, setQuestion] = useState<PendingQuestion | null>(null);
  const [customAnswer, setCustomAnswer] = useState('');

  // File explorer state
  const [files, setFiles] = useState<WorkspaceFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [fileLoading, setFileLoading] = useState(false);

  // The agent client and the live chat that tracks snapshotId/state/messages.
  // The coding agent uses a server-managed FileSessionStore, so the chat
  // threads the snapshotId for us across turns and resumes.
  const agent = useMemo(() => remoteAgent({ url: ENDPOINT }), []);
  const chatRef = useRef<AgentChat | null>(null);

  // ── Fetch workspace file tree via runFlow() ────────────────────────────
  const refreshFiles = useCallback(async () => {
    try {
      const data = await runFlow<{ files: WorkspaceFile[] }>({
        url: WORKSPACE_FILES_ENDPOINT,
      });
      setFiles(data.files || []);
    } catch {
      // ignore — workspace may not exist yet
    }
  }, []);

  // Load files on mount and after each agent response
  useEffect(() => {
    refreshFiles();
  }, [refreshFiles]);

  // ── Restore session from snapshotId in the URL ─────────────────────────
  useEffect(() => {
    if (!urlSnapshotId) return;

    let cancelled = false;

    async function restore() {
      try {
        // loadChat fetches the snapshot and returns a chat with the full
        // message history + state restored, ready to continue.
        const chat = await agent.loadChat({ snapshotId: urlSnapshotId! });
        if (cancelled) return;

        chatRef.current = chat;
        setMessages(messagesToChat(chat.messages));

        // If the last message has a pending interrupt (e.g. ask_user,
        // write_file approval), trigger the dialog so the user can respond.
        const allMessages = chat.messages;
        if (allMessages.length > 0) {
          const lastMsg = allMessages[allMessages.length - 1];
          for (const p of lastMsg.content || []) {
            if (!p.toolRequest) continue;
            const tr = p.toolRequest;
            if (tr.name === 'ask_user') {
              const askInput = tr.input as
                | { question?: string; options?: string[] }
                | undefined;
              setQuestion({
                question: askInput?.question || 'What would you like to do?',
                options: askInput?.options || [],
                ref: tr.ref,
              });
              break;
            } else if (
              tr.name === 'write_file' ||
              tr.name === 'search_and_replace' ||
              tr.name === 'run_shell'
            ) {
              setApproval({
                toolName: tr.name,
                ref: tr.ref,
                input: tr.input,
              });
              break;
            }
          }
        }
      } catch (err: any) {
        if (!cancelled) {
          setMessages([
            {
              role: 'system',
              text: `Failed to restore session: ${err.message}`,
            },
          ]);
        }
      } finally {
        if (!cancelled) setRestoring(false);
      }
    }

    restore();
    return () => {
      cancelled = true;
    };
  }, []); // Only run on mount

  // ── Fetch a single file's content via runFlow() ────────────────────────
  const viewFile = useCallback(async (filePath: string) => {
    setSelectedFile(filePath);
    setFileLoading(true);
    try {
      const data = await runFlow<{ path: string; content: string }>({
        url: WORKSPACE_FILE_ENDPOINT,
        input: filePath,
      });
      setFileContent(data.content || '');
    } catch {
      setFileContent('(failed to load file)');
    } finally {
      setFileLoading(false);
    }
  }, []);

  // ── Send a regular user message ──────────────────────────────────────
  const handleSend = useCallback(
    async (text: string) => {
      if (loading || approval || question) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');
      setStreamingReasoning('');

      if (!chatRef.current) {
        chatRef.current = agent.chat();
      }
      const chat = chatRef.current;

      try {
        const res = await streamTurn(chat.sendStream(text));
        processResult(res);
      } catch (err: unknown) {
        handleTurnError(err);
      } finally {
        setLoading(false);
        refreshFiles();
      }
    },
    [loading, approval, question, refreshFiles, agent]
  );

  // ── Respond to a tool approval interrupt ──────────────────────────────
  const handleApprovalResponse = useCallback(
    async (approved: boolean) => {
      if (!approval || !chatRef.current) return;
      const currentApproval = approval;
      const chat = chatRef.current;
      setApproval(null);
      setLoading(true);
      setStreamingText('');
      setStreamingReasoning('');

      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          text: approved
            ? `✅ Approved: ${currentApproval.toolName}`
            : `❌ Denied: ${currentApproval.toolName}`,
        },
      ]);

      try {
        let res: AgentResponse;
        if (approved) {
          // Restart pattern — resume the interrupted tool by re-issuing the
          // original tool request tagged with `{ toolApproved: true }`. This
          // metadata is checked by:
          //   • toolApproval middleware (write_file, search_and_replace)
          //   • run_shell tool handler (risky shell commands)
          res = await streamTurn(
            chat.resumeStream({
              restart: [
                {
                  toolRequest: {
                    name: currentApproval.toolName,
                    ref: currentApproval.ref,
                    input: currentApproval.input,
                  },
                  metadata: { resumed: { toolApproved: true } },
                },
              ],
            })
          );
        } else {
          // For denial, send a user message so the model knows the tool was
          // rejected. We don't restart for denial because that would re-trigger
          // the interrupt in a loop.
          const denialText =
            `I denied the "${currentApproval.toolName}" tool call` +
            (currentApproval.toolName === 'run_shell'
              ? ` for command: "${currentApproval.input?.command}".`
              : ` for file: "${currentApproval.input?.filePath}".`) +
            ` Please continue without executing it, or suggest an alternative.`;
          res = await streamTurn(
            chat.sendStream({
              messages: [{ role: 'user', content: [{ text: denialText }] }],
            })
          );
        }
        processResult(res);
      } catch (err: unknown) {
        handleTurnError(err, 'resuming');
      } finally {
        setLoading(false);
        refreshFiles();
      }
    },
    [approval, refreshFiles]
  );

  // ── Respond to an ask_user interrupt ───────────────────────────────────
  const handleQuestionResponse = useCallback(
    async (answer: string) => {
      if (!question || !chatRef.current) return;
      const currentQuestion = question;
      const chat = chatRef.current;
      setQuestion(null);
      setCustomAnswer('');
      setLoading(true);
      setStreamingText('');
      setStreamingReasoning('');

      setMessages((prev) => [
        ...prev,
        { role: 'system', text: `💬 Answer: ${answer}` },
      ]);

      try {
        // Respond pattern — send the tool response directly via resume.respond.
        // The tool never executes; we provide the output ourselves.
        const res = await streamTurn(
          chat.resumeStream({
            respond: [
              {
                toolResponse: {
                  name: 'ask_user',
                  ref: currentQuestion.ref,
                  output: { answer },
                },
              },
            ],
          })
        );
        processResult(res);
      } catch (err: unknown) {
        handleTurnError(err, 'resuming');
      } finally {
        setLoading(false);
        refreshFiles();
      }
    },
    [question, refreshFiles]
  );

  // ── Shared: stream a turn, render chunks, and append the model reply ──
  async function streamTurn(turn: AgentTurn): Promise<AgentResponse> {
    let accumulated = '';
    let accumulatedReasoning = '';

    for await (const chunk of turn.stream) {
      const content = chunk.raw.modelChunk?.content as
        | PartWithMetadata[]
        | undefined;
      if (!content) continue;

      for (const part of content) {
        const p = part as PartWithMetadata;
        if (p.reasoning) {
          // Accumulate reasoning/thinking content
          accumulatedReasoning += p.reasoning;
          setStreamingReasoning(accumulatedReasoning);
        } else if (part.text) {
          // Filesystem middleware injects file contents as text chunks —
          // show as a tool message but don't dump raw content into streaming text
          const fsMeta = p.metadata;
          if (fsMeta?.filesystemMiddlewareTool) {
            const toolName = fsMeta.filesystemMiddlewareTool;
            const filePath = fsMeta.filePath || '';
            if (toolName === 'read_file' && filePath) {
              setMessages((prev) => [
                ...prev,
                { role: 'tool', text: `📖 Reading ${filePath}` },
              ]);
            }
            continue;
          }
          accumulated += part.text;
          setStreamingText(accumulated);
        } else if (part.toolRequest) {
          const tr = part.toolRequest;
          // Skip interrupt tools — they're shown in approval/question dialogs
          // and would appear twice (once in original stream, once on resume)
          if (
            tr.name === 'ask_user' ||
            tr.name === 'write_file' ||
            tr.name === 'search_and_replace' ||
            tr.name === 'read_file'
          )
            continue;
          const msg = formatToolRequest(tr.name, tr.input);
          setMessages((prev) => [...prev, { role: 'tool', ...msg }]);
        } else if (part.toolResponse) {
          const tr = part.toolResponse;
          // Skip read_file response — the "📖 Reading" message already covers it
          if (tr.name === 'read_file') continue;
          const msg = formatToolResponse(tr.name, tr.output);
          setMessages((prev) => [...prev, { role: 'tool', ...msg }]);
        }
      }
    }

    const res = await turn.response;
    setStreamingText('');
    setStreamingReasoning('');

    const replyText = res.text;
    if (accumulated || replyText || accumulatedReasoning) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'model',
          text: replyText || accumulated || '',
          reasoning: accumulatedReasoning || undefined,
        },
      ]);
    }

    return res;
  }

  // ── Process a result: update the URL & detect interrupts ─────────────
  function processResult(res: AgentResponse) {
    // The chat tracks the latest snapshotId — push it into the URL so the
    // user can bookmark or reload.
    const snapshotId = chatRef.current?.snapshotId;
    if (snapshotId) {
      navigate(`/coding-agent/${snapshotId}`, { replace: true });
    }

    // res.interrupts surfaces all paused tool requests — both ask_user
    // (defineInterrupt) and the toolApproval-middleware approvals.
    const interrupt = res.interrupts[0];
    if (!interrupt) return;

    if (interrupt.name === 'ask_user') {
      const input = interrupt.input as
        | { question?: string; options?: string[] }
        | undefined;
      setQuestion({
        question: input?.question || 'What would you like to do?',
        options: input?.options || [],
        ref: interrupt.ref,
      });
    } else {
      // Tool approval interrupt (write_file, search_and_replace, run_shell)
      setApproval({
        toolName: interrupt.name,
        ref: interrupt.ref,
        input: interrupt.input,
      });
    }
  }

  // ── Shared: surface a turn-level or transport error ──────────────────
  function handleTurnError(err: unknown, verb = '') {
    setStreamingText('');
    setStreamingReasoning('');
    const prefix = verb ? `Error ${verb}` : 'Error';
    if (err instanceof AgentError) {
      // A turn (including a resume) can fail gracefully — the chat keeps the
      // last-good snapshot so the user can keep working.
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          text: `${prefix} (${err.status}): ${err.message}`,
        },
      ]);
      return;
    }
    const errMsg = err instanceof Error ? err.message : String(err);
    setMessages((prev) => [
      ...prev,
      { role: 'system', text: `${prefix}: ${errMsg}` },
    ]);
  }

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
        streamingText={streamingText}
        streamingReasoning={streamingReasoning}
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Rebuild displayable chat messages from a restored session history. */
function messagesToChat(history: MessageData[]): ChatMessage[] {
  const restored: ChatMessage[] = [];
  for (const msg of history) {
    const role = msg.role as ChatMessage['role'];

    // Filter out filesystem middleware read_file text parts
    // (same filtering we apply during streaming).
    const textParts = (msg.content || [])
      .filter((p: PartWithMetadata) => {
        if (!p.text) return false;
        // Skip read_file content injected by filesystem middleware
        if (p.metadata?.filesystemMiddlewareTool) return false;
        return true;
      })
      .map((p: Part) => p.text);

    if (textParts.length > 0) {
      restored.push({ role, text: textParts.join('') });
    }

    // Show tool calls/responses from history, but skip read_file responses
    // (the raw file content is too verbose).
    for (const p of (msg.content as Part[]) || []) {
      if (p.toolRequest) {
        const tmsg = formatToolRequest(p.toolRequest.name, p.toolRequest.input);
        restored.push({ role: 'tool', ...tmsg });
      }
      if (p.toolResponse) {
        if (p.toolResponse.name === 'read_file') continue;
        const tmsg = formatToolResponse(
          p.toolResponse.name,
          p.toolResponse.output
        );
        restored.push({ role: 'tool', ...tmsg });
      }
    }
  }
  return restored;
}
