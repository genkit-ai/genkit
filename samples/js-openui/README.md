# Genkit + OpenUI streaming generative UI

This sample combines [Genkit](https://genkit.dev/) with
[OpenUI](https://www.openui.com/) in one streaming chat flow. Genkit owns the
model provider and streamed model call. OpenUI's `AgentInterface` owns the chat
shell and in-memory conversation state, supplies the model's component
contract, renders the generated OpenUI Lang, and turns follow-up and form
actions into the next Genkit turn.

Removing Genkit removes the model and streaming path. Removing OpenUI removes
the generated UI contract, parser, renderer, charts, form state, and action
events. The integration uses Genkit's normal `generateStream()` API rather than
adding a second model client.

## Supported versions

The checked-in `package-lock.json` is the source of truth. This sample was built
with Genkit and `@genkit-ai/compat-oai` 1.41.0,
`@openuidev/react-ui` 0.13.6, `@openuidev/react-headless` 0.9.7,
`@openuidev/react-lang` 0.2.11, React 19.2.0, and Node.js 20 or newer.

## Set up and run

1. Install dependencies:

   ```bash
   npm install
   ```

2. Set an OpenAI API key in the server environment. Optionally select another
   OpenAI model:

   ```bash
   export OPENAI_API_KEY=your-key
   export OPENAI_MODEL=gpt-4o-mini
   ```

   The browser never reads these variables. Do not use a `VITE_` prefix for a
   provider key.

3. Start the Vite client and Genkit server:

   ```bash
   npm run dev
   ```

4. Open [http://localhost:5173](http://localhost:5173). The Genkit endpoint is
   on port 3400 and is reached through Vite's `/api` proxy.

Use `npm test`, `npm run typecheck`, and `npm run build` to run the focused
checks and production build. After a build, `OPENAI_API_KEY=... npm start`
serves the production app at [http://localhost:3400](http://localhost:3400).

## Architecture

```text
prompt or UI action -> AgentInterface -> POST /api/chat -> Genkit generateStream() -> OpenAI
OpenUI renderer     <- OpenAI SSE adapter <- completion-shaped SSE <- Genkit stream
```

`src/openui-prompt.ts` compiles the system prompt at runtime from the exact
`openuiChatLibrary` object that `src/App.tsx` passes to `AgentInterface`. There
is no hand-written schema or generated prompt artifact to keep in sync. If the
library dependency changes, reinstall, run `npm test`, and inspect the parser
test before committing the new lockfile.

`fetchLLM()` pairs `openAIMessageFormat` with `openAIAdapter`, and the Express
route converts that wire format into Genkit history before returning
completion-shaped server-sent events. The API key stays on the server.

`AgentInterface` uses its internal `ThemeProvider` with a visible blue accent;
the provider is not disabled. Its built-in `ContinueConversation` handling
creates exactly one ordinary user turn for either a follow-up or form submit.
`src/message-adapter.ts` separates AgentInterface's inline display and context
sections so form values and action context reach Genkit without leaking
persisted form state into assistant history. Storage is intentionally omitted,
so conversations are in memory for this sample.

## Acceptance prompts

The page includes buttons for these deterministic prompts:

- Chart: `Show quarterly revenue as a labeled bar chart: Q1 120, Q2 180, Q3 150, Q4 240. End with two relevant follow-up suggestions.`
- Follow-up: `Compare the strongest and weakest quarter, then add a FollowUpBlock with two next questions.`
- Form: `Create a validated project estimate form with fields for project name, team size, and notes. Add a primary Submit button that sends the completed values to you.`

For the form, try `Aurora-731`, `7`, and
`Prioritize accessibility and charts`. Required-field validation should block
an incomplete submit. A valid submit makes exactly one new Genkit request, and
the next rendered response should acknowledge the submitted project name and at
least one other value. Clicking a generated follow-up likewise creates one user
turn and one new Genkit request in the same conversation.
