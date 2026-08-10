# Express Integration

Demonstrates integrating Genkit flows with an Express.js server — including
authentication context, streaming responses, and the `expressHandler` utility.

## Features Demonstrated

| Feature | Endpoint | Description |
|---------|----------|-------------|
| Flow via `expressHandler` | `POST /jokeFlow` | Genkit flow exposed via Express with auth context |
| Flow handler (no auth) | `POST /jokeHandler` | Flow exposed without auth validation |
| Direct flow invocation | `GET /jokeWithFlow` | Call a flow directly from a route handler |
| Raw streaming | `GET /jokeStream` | Chunked transfer encoding with `ai.generate` |
| Auth context | `Authorization` header | Token-based auth with context validation |
| Context providers | `auth()` factory | Reusable auth context provider pattern |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager

### API Keys

```bash
export GEMINI_API_KEY='<your-api-key>'
```

### Build and Install

From the repo root:

```bash
pnpm install
pnpm run setup
```

## Run the Sample

```bash
pnpm build && pnpm start
```

The Express server starts on port `5000` (or `$PORT`).

## Testing This Demo

1. **Test with auth** (requires `Authorization: open sesame` header):
   ```bash
   curl http://localhost:5000/jokeFlow?stream=true \
     -d '{"data": "banana"}' \
     -H "Content-Type: application/json" \
     -H "Authorization: open sesame"
   ```

2. **Test without auth**:
   ```bash
   curl http://localhost:5000/jokeHandler?stream=true \
     -d '{"data": "banana"}' \
     -H "Content-Type: application/json"
   ```

3. **Test direct flow invocation**:
   ```bash
   curl "http://localhost:5000/jokeWithFlow?subject=banana"
   ```

4. **Test raw streaming**:
   ```bash
   curl "http://localhost:5000/jokeStream?subject=banana"
   ```

5. **Expected behavior**:
   - Authenticated requests (`open sesame`) succeed
   - Unauthenticated requests return `PERMISSION_DENIED`
   - Streaming endpoints deliver text incrementally
   - `?stream=true` enables streaming on `expressHandler` endpoints
