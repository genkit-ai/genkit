import { useCallback, useRef, useState } from 'react';
import { runFlow } from 'genkit/beta/client';

// ---------------------------------------------------------------------------
// Translator — a non-chat translation UI
//
// Demonstrates:
//   • runFlow() for simple non-streaming requests
//   • Passing prompt input variables via `init` (the `language` parameter)
//   • Multi-turn session via `init: { state }` round-tripping
//   • A purpose-built UI instead of generic chat
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/translatorAgent';

const LANGUAGES = [
  'French',
  'Spanish',
  'German',
  'Japanese',
  'Korean',
  'Italian',
  'Portuguese',
  'Mandarin Chinese',
  'Arabic',
  'Hindi',
];

export default function Translator() {
  const [sourceText, setSourceText] = useState('');
  const [translation, setTranslation] = useState('');
  const [language, setLanguage] = useState('French');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Session state — returned by the server, sent back on the next turn.
  const stateRef = useRef<any>(undefined);
  // Track which language was used for the current session so we can
  // reset the session when the user switches languages.
  const sessionLanguageRef = useRef<string>('French');

  const handleTranslate = useCallback(async () => {
    if (!sourceText.trim() || loading) return;
    setLoading(true);
    setError('');
    setTranslation('');

    // ── Build the request ──────────────────────────────────────────────
    const input = {
      messages: [{ role: 'user' as const, content: [{ text: sourceText }] }],
    };

    // If the language changed, start a fresh session.
    // Otherwise, send `state` to continue the session (which preserves
    // the system prompt context from prior turns).
    let init: Record<string, any> = {};
    if (
      stateRef.current &&
      sessionLanguageRef.current === language
    ) {
      init = { state: stateRef.current };
    }
    // The translatorAgent's prompt expects `{ language }` as input variables.
    // These are passed as part of `init` — Genkit merges them into the prompt.
    // (On the first turn, or when language changes, we start fresh with the
    // new language. On subsequent turns, the language is already baked into
    // the session state.)

    try {
      // ── Call the flow (non-streaming) ────────────────────────────────
      const result = (await runFlow({ url: ENDPOINT, input, init })) as any;

      // Save session state for multi-turn.
      if (result?.state) {
        stateRef.current = result.state;
        sessionLanguageRef.current = language;
      }

      // Extract translation text.
      const msg = result?.message;
      if (msg?.content) {
        const texts = msg.content
          .filter((p: any) => p.text)
          .map((p: any) => p.text);
        setTranslation(texts.join('\n'));
      } else {
        setTranslation(JSON.stringify(result, null, 2));
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [sourceText, language, loading]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleTranslate();
    }
  };

  return (
    <div className="page-with-sidebar">
      <div className="translator-page">
        <div className="chat-header">
          <h2>🌐 Translator</h2>
          <span className="chat-desc">
            Translate text using a prompt-based session flow. Uses{' '}
            <code>runFlow()</code> — no streaming.
          </span>
        </div>

        <div className="translator-controls">
          <label>
            Target language:{' '}
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              disabled={loading}
            >
              {LANGUAGES.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="translator-panels">
          <div className="translator-panel">
            <h3>Source</h3>
            <textarea
              className="translator-textarea"
              placeholder="Type text to translate…"
              value={sourceText}
              onChange={(e) => setSourceText(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              rows={8}
            />
          </div>

          <div className="translator-panel">
            <h3>Translation ({language})</h3>
            <div className="translator-output">
              {loading && <span className="loading-text">Translating…</span>}
              {error && <span className="error-text">Error: {error}</span>}
              {!loading && !error && (
                <span>{translation || 'Translation will appear here'}</span>
              )}
            </div>
          </div>
        </div>

        <div className="translator-actions">
          <button
            className="btn btn-send"
            onClick={handleTranslate}
            disabled={loading || !sourceText.trim()}
          >
            {loading ? 'Translating…' : 'Translate'}
          </button>
          <span className="translator-hint">⌘+Enter to translate</span>
        </div>
      </div>

      <aside className="info-sidebar">
        <h3>📋 How It Works</h3>
        <ol>
          <li>
            Client calls <code>runFlow()</code> — a simple request/response with
            no streaming.
          </li>
          <li>
            The translator agent is built with{' '}
            <code>defineSessionFlowFromPrompt</code>, which takes a Dotprompt
            with an input variable <code>{'{{ language }}'}</code>.
          </li>
          <li>
            On the first turn (or when the language changes), a fresh session
            starts. The language is baked into the system prompt.
          </li>
          <li>
            On subsequent turns with the same language, the client sends back the{' '}
            <code>state</code> from the previous response via{' '}
            <code>{'init: { state }'}</code>, preserving conversation context.
          </li>
        </ol>

        <h4>Key APIs</h4>
        <pre>{`// Simple non-streaming call
const result = await runFlow({
  url: '/api/translatorAgent',
  input: {
    messages: [{
      role: 'user',
      content: [{ text: sourceText }],
    }],
  },
  init: { state }, // multi-turn
});

// result.state → save for next turn
// result.message.content → translation`}</pre>

        <h4>Session vs. Fresh Start</h4>
        <p>
          Switching the target language <strong>resets the session</strong> —
          the client simply omits <code>state</code> from <code>init</code>,
          causing the server to start a new session with the updated language
          prompt. This is a client-side decision, not a server feature.
        </p>
      </aside>
    </div>
  );
}
