# Genkit + Hono + @genkit-ai/fetch

This sample runs a [Hono](https://hono.dev) server and exposes Genkit flows over HTTP using [@genkit-ai/fetch](https://www.npmjs.com/package/@genkit-ai/fetch). It uses the Web API `Request`/`Response` so the same code can run on Node, Cloudflare Workers, Deno, or Bun.

## Setup

1. Set `GOOGLE_GENAI_API_KEY` (or `GOOGLE_API_KEY` / `GEMINI_API_KEY`).
2. Install and build:

   ```bash
   pnpm install
   pnpm run build
   ```

3. Start the server:

   ```bash
   pnpm start
   ```

   Or run with Genkit Dev UI:

   ```bash
   pnpm run genkit:dev
   ```

   By default the server listens on http://localhost:3780.

## Usage

- **GET /** – Info and list of flow names.
- **POST /api/hello** – Flow with string input. Body: `{ "data": "World" }`.
- **POST /api/greeting** – Flow with object input. Body: `{ "data": { "name": "Alice" } }`.
- **POST /api/streaming** – Streaming flow. Body: `{ "data": { "prompt": "Say hi in 3 words" } }`. Use `Accept: text/event-stream` or `?stream=true` to stream.
- **POST /api/secureGreeting** – Same as greeting but requires auth. Header: `Authorization: Bearer open-sesame`. Body: `{ "data": { "name": "Alice" } }`. Uses `withFlowOptions(flow, { contextProvider })` to attach a context provider.

Example:

```bash
curl -X POST http://localhost:3780/api/hello \
  -H "Content-Type: application/json" \
  -d '{"data": "Hono"}'

# With auth (secureGreeting):
curl -X POST http://localhost:3780/api/secureGreeting \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer open-sesame" \
  -d '{"data": {"name": "Alice"}}'
```

## How it works

- **Hono** handles routing; **@genkit-ai/fetch**’s `handleFlows(request, flows, pathPrefix)` turns a Web `Request` into a Genkit flow call and returns a Web `Response`.
- Flow URLs are `POST /api/<flowName>` with body `{ "data": <input> }`, matching the [Genkit callable protocol](https://firebase.google.com/docs/genkit/reference/js/client).

**Developing in the genkit repo:** In `package.json` set `"@genkit-ai/fetch": "file:../../js/plugins/web"`, then build the web plugin (`cd ../../js/plugins/web && pnpm run build`) and run `pnpm install` from this directory.
