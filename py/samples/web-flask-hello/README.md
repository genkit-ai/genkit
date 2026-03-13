# Flask + Genkit Integration

Serve Genkit AI flows as Flask HTTP endpoints with context propagation from
request headers.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                  FLASK + GENKIT INTEGRATION                      │
│                                                                  │
│  HTTP Request                    Flask App                       │
│  ───────────                    ─────────                       │
│  POST /chat                                                      │
│  Authorization: JohnDoe         ┌──────────────────────┐        │
│  {"name": "Mittens"}    ──────► │ genkit_flask_handler  │        │
│                                 │                      │        │
│                                 │  1. Extract headers   │        │
│                                 │  2. Build context     │        │
│                                 │  3. Run Genkit flow   │        │
│                                 └──────────┬───────────┘        │
│                                            │                     │
│                                            ▼                     │
│                                 ┌──────────────────────┐        │
│                                 │  @ai.flow()          │        │
│                                 │  say_hi(input, ctx)   │        │
│                                 │                      │        │
│                                 │  ctx.context =        │        │
│                                 │  {"username":"JohnDoe"}│        │
│                                 └──────────────────────┘        │
│                                                                  │
│  ◄──── AI-generated joke about Mittens for user JohnDoe         │
└─────────────────────────────────────────────────────────────────┘
```

## Features Demonstrated

| Feature | Code | Description |
|---------|------|-------------|
| Flask Integration | `@genkit_flask_handler(ai)` | Connect Flask routes to Genkit flows |
| Context Provider | `my_context_provider()` | Extract request headers into flow context |
| Header-based Auth | `request.headers["authorization"]` | Pass username from HTTP headers |
| Flow Context Access | `ctx.context.get("username")` | Read context inside the flow |
| Streaming Support | `ctx.send_chunk` | Stream response chunks to the client |

## ELI5: Key Concepts

| Concept | ELI5 |
|---------|------|
| **Flask** | A simple Python web framework — like a waiter that takes HTTP requests and serves responses |
| **`genkit_flask_handler`** | Connects Flask routes to Genkit flows — does the plumbing so you focus on AI logic |
| **Context Provider** | A function that adds extra info to each request — like adding the username from headers |
| **Request Headers** | Metadata sent with HTTP requests — like the "From:" on an envelope |

## Quick Start

```bash
export GEMINI_API_KEY=your-api-key
./run.sh
```

## Setup

### Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API key" and create a key

```bash
export GEMINI_API_KEY='your-api-key'
```

### Run the Sample

```bash
./run.sh
```

Or manually:

```bash
genkit start -- uv run flask --app src/main.py run
```

## Testing This Demo

### Via curl

```bash
# Basic request
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"data": {"name": "Mittens"}}'

# With authorization header (username passed as context)
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: JohnDoe" \
  -d '{"data": {"name": "Mittens"}}'
```

### Via Dev UI (http://localhost:4000)

- [ ] Run the `say_hi` flow
- [ ] Verify AI-generated response

### Expected Behavior

- POST `/chat` returns an AI-generated joke
- `Authorization` header is extracted as `username` in flow context
- Flow accesses username via `ctx.context.get("username")`

## Development

The `run.sh` script uses `watchmedo` for hot reloading on file changes.
