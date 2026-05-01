import { useCallback, useRef, useState } from 'react';
import { runFlow } from 'genkit/beta/client';
import type { AgentInit, AgentInput, AgentOutput } from 'genkit/beta';

// ---------------------------------------------------------------------------
// Prompt Variables — showcases `defineAgent` with multiple
// runtime input variables (tone, format, audience).
//
// Demonstrates:
//   • runFlow() for simple non-streaming requests
//   • Passing multiple prompt input variables via `init`
//   • Live system prompt preview — see exactly what the LLM receives
//   • Multi-turn session via `init: { state }` round-tripping
//   • Changing any variable resets the session (client-side decision)
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/writerAgent';

const TONES = [
  'Professional',
  'Casual',
  'Humorous',
  'Academic',
  'Dramatic',
  'Friendly',
  'Sarcastic',
];

const FORMATS = [
  'Paragraph',
  'Bullet Points',
  'Numbered List',
  'Email',
  'Tweet Thread',
  'Poem',
  'Dialogue',
];

const AUDIENCES = [
  'General',
  'Children (ages 8-12)',
  'Teenagers',
  'Developers',
  'Executives',
  'Students',
  'Scientists',
];

/** Build the rendered system prompt for the live preview. */
function renderSystemPrompt(tone: string, format: string, audience: string) {
  return `You are a versatile writing assistant.

Tone: ${tone}
Format: ${format}
Target audience: ${audience}

Follow these rules strictly:
- Always write in the specified tone.
- Always structure your response using the specified format.
- Always tailor your language and complexity to the target audience.

Help the user with whatever writing task they request.`;
}

export default function PromptVariables() {
  const [sourceText, setSourceText] = useState('');
  const [result, setResult] = useState('');
  const [tone, setTone] = useState('Professional');
  const [format, setFormat] = useState('Paragraph');
  const [audience, setAudience] = useState('General');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Session state — returned by the server, sent back on the next turn.
  const stateRef = useRef<any>(undefined);
  // Track which variable combo was used for the current session so we can
  // reset when anything changes.
  const sessionVarsRef = useRef<string>('');

  const currentVarsKey = `${tone}|${format}|${audience}`;

  const handleSubmit = useCallback(async () => {
    if (!sourceText.trim() || loading) return;
    setLoading(true);
    setError('');
    setResult('');

    // ── Build the request ──────────────────────────────────────────────
    const input: AgentInput = {
      messages: [{ role: 'user', content: [{ text: sourceText }] }],
    };

    // If any variable changed, start a fresh session.
    // When starting a new session, we pass the current dropdown values.
    // On subsequent turns, we just pass the state to reuse them.
    let init: AgentInit = {};
    if (stateRef.current && sessionVarsRef.current === currentVarsKey) {
      init = { state: stateRef.current };
    } else {
      init = { state: { inputVariables: { tone, format, audience } } };
    }

    try {
      const res = await runFlow<AgentOutput, AgentInit>({ url: ENDPOINT, input, init });

      // Save session state for multi-turn.
      if (res?.state) {
        stateRef.current = res.state;
        sessionVarsRef.current = currentVarsKey;
      }

      // Extract text from response.
      const msg = res?.message;
      if (msg?.content) {
        const texts = msg.content
          .filter((p: any) => p.text)
          .map((p: any) => p.text);
        setResult(texts.join(''));
      } else {
        setResult(JSON.stringify(res, null, 2));
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [sourceText, currentVarsKey, loading]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const varsChanged = stateRef.current && sessionVarsRef.current !== currentVarsKey;

  return (
    <div className="page-with-sidebar">
      <div className="pv-page">
        {/* ── Header ─────────────────────────────────────────────────── */}
        <div className="chat-header">
          <h2>🎛️ Prompt Variables</h2>
          <span className="chat-desc">
            Adjust prompt input variables at runtime. Uses{' '}
            <code>defineAgent</code> with{' '}
            <code>{'{{ tone }}'}</code>, <code>{'{{ format }}'}</code>,{' '}
            <code>{'{{ audience }}'}</code>.
          </span>
        </div>

        {/* ── Variable controls ──────────────────────────────────────── */}
        <div className="pv-controls">
          <label>
            <span className="pv-control-label">Tone</span>
            <select
              value={tone}
              onChange={(e) => setTone(e.target.value)}
              disabled={loading}
            >
              {TONES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </label>

          <label>
            <span className="pv-control-label">Format</span>
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value)}
              disabled={loading}
            >
              {FORMATS.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </label>

          <label>
            <span className="pv-control-label">Audience</span>
            <select
              value={audience}
              onChange={(e) => setAudience(e.target.value)}
              disabled={loading}
            >
              {AUDIENCES.map((a) => (
                <option key={a} value={a}>{a}</option>
              ))}
            </select>
          </label>

          {varsChanged && (
            <span className="pv-vars-changed">
              ⚠ Variables changed — next request starts a new session
            </span>
          )}
        </div>

        {/* ── Live system prompt preview ─────────────────────────────── */}
        <details className="pv-prompt-preview">
          <summary>📄 Live System Prompt Preview</summary>
          <pre className="pv-prompt-code">
            {renderSystemPrompt(tone, format, audience)}
          </pre>
        </details>

        {/* ── Input + Output panels ──────────────────────────────────── */}
        <div className="pv-panels">
          <div className="pv-panel">
            <h3>Your Request</h3>
            <textarea
              className="pv-textarea"
              placeholder="e.g. Write a short intro about AI safety…"
              value={sourceText}
              onChange={(e) => setSourceText(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              rows={8}
            />
          </div>

          <div className="pv-panel">
            <h3>Result</h3>
            <div className="pv-output">
              {loading && <span className="loading-text">Generating…</span>}
              {error && <span className="error-text">Error: {error}</span>}
              {!loading && !error && (
                <span>{result || 'Output will appear here'}</span>
              )}
            </div>
          </div>
        </div>

        {/* ── Actions ────────────────────────────────────────────────── */}
        <div className="pv-actions">
          <button
            className="btn btn-send"
            onClick={handleSubmit}
            disabled={loading || !sourceText.trim()}
          >
            {loading ? 'Generating…' : 'Generate'}
          </button>
          <span className="pv-hint">⌘+Enter to generate</span>
        </div>
      </div>

      {/* ── How It Works sidebar ───────────────────────────────────── */}
      <aside className="info-sidebar">
        <h3>📋 How It Works</h3>
        <ol>
          <li>
            The backend defines a <strong>Dotprompt</strong> with three input
            variables: <code>{'{{ tone }}'}</code>,{' '}
            <code>{'{{ format }}'}</code>, and{' '}
            <code>{'{{ audience }}'}</code>.
          </li>
          <li>
            <code>defineAgent</code> wraps this prompt into an
            agent. The <code>defaultInput</code> provides fallback values
            for all three variables.
          </li>
          <li>
            The client sends the user's message via <code>runFlow()</code>. When
            starting a new session, the client passes the dropdown values in the 
            <code>init</code> object.
          </li>
          <li>
            Changing any variable <strong>resets the session</strong>: the client
            simply passes the new values in <code>init</code> without the{' '}
            <code>state</code>, causing a fresh session with the new prompt configuration.
          </li>
        </ol>

        <h4>Agent Definition</h4>
        <pre>{`ai.defineAgent({
  name: 'writerPrompt',
  model: 'googleai/gemini-flash-latest',
  input: { schema: z.object({ tone: z.string(), format: z.string(), audience: z.string() }) },
  system: \`You are a writing assistant. Tone: {{ tone }} …\`,
  defaultInput: {
    tone: 'Professional',
    format: 'Paragraph',
    audience: 'General',
  },
});`}</pre>

        <h4>Client Code</h4>
        <pre>{`const result = await runFlow({
  url: '/api/writerAgent',
  input: {
    messages: [{
      role: 'user',
      content: [{ text }],
    }],
  },
  // pass state to continue, or new variables to reset
  init: state ? { state } : { state: { inputVariables: { tone, format, audience } } },
});`}</pre>

        <h4>Session Reset Logic</h4>
        <p>
          The live preview above shows the <em>rendered</em> system prompt. When
          you change a dropdown, the preview updates instantly — but the server
          doesn't know yet. The next request simply passes the new variables
          without <code>state</code>, forcing a new session with the updated
          variables. This is a <strong>client-side decision</strong>, not a
          server feature.
        </p>
      </aside>
    </div>
  );
}
