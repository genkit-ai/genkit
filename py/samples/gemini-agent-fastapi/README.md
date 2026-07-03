# Gemini Agent on Cloud Run — a full agent as one FastAPI service

Deploy a real agent — tools, sessions, streaming, per-user auth, and token accounting — as **one FastAPI service** on Google Cloud Run.

The agent, its tools, its sessions, and your API all live in the same container: code you can read, fork, and ship. It's a self-contained, approachable way to see how a production agent fits together.

Built on [Genkit](https://genkit.dev) with Gemini. An AI Studio key runs it locally; Cloud Run runs it in production.

## The single-service pattern

Google Cloud gives you a few good ways to run an agent in production, and this sample shows one of them end to end: **the agent and its HTTP API in a single FastAPI service.**

Your agent logic, its tools, session handling, and the routes your frontend calls all live in one container. When a request comes in, FastAPI hands it to the agent in the same process and streams the result straight back — no extra hop, no second service to wire up.

That keeps a few things simple:

- **One thing to deploy.** A single Cloud Run container is your whole backend. `gcloud run deploy --source .` and you're live.
- **Your API is just FastAPI.** Add routes, middleware, and auth the way you already do in any Python web app.
- **Everything is readable.** Sessions, tools, streaming, and tracing are files in this repo you can open and change — a good way to learn how an agent actually works before you reach for more managed options.
- **Portable.** It runs on Cloud Run, in Docker locally, or anywhere else that runs a container.

As your needs grow, Google Cloud has heavier-duty options for managed scaling, evaluation, and enterprise distribution. This sample is the friendly on-ramp: get a real agent — tools, sessions, streaming, auth — running and understood, then reach for more when you need it.

## Quickstart

### 1. Install dependencies

From the Genkit Python workspace root:

```bash
cd py
uv sync
```

Or from this sample directory (standalone):

```bash
uv sync
```

### 2. Get a free Gemini API key

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Get a key at [Google AI Studio](https://aistudio.google.com/apikey).

### 3. Start a datastore

This app stores sessions, profiles, and usage in a real datastore — there's no
in-memory fallback, because a store that forgets on restart loses user data in
production. For local dev, run the Firestore emulator:

```bash
gcloud emulators firestore start --host-port=localhost:8080
```

Then, in the terminal you'll run the app from:

```bash
export FIRESTORE_EMULATOR_HOST="localhost:8080"
export GOOGLE_CLOUD_PROJECT="demo-local"
uv run python scripts/seed_demo.py   # seed the demo user's profile (optional)
```

### 4. Start server and Dev UI

```bash
cd py/samples/gemini-agent-fastapi
uv run genkit start -- uvicorn app.server:app --reload --port 8000
```

- **FastAPI docs:** http://localhost:8000/docs
- **Genkit Dev UI:** http://localhost:4000

### 5. Sign in and chat

**Option A — curl (see the raw wire format)**

```bash
# Dev login (demo user)
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"demo@example.com","password":"demo1234"}' | jq -r .access_token)

# Start a new session — NDJSON stream + final {"result": ...}
curl -N -X POST 'http://localhost:8000/api/chat' \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"message":"What orders do I have?"}'

# Resume a session (sessionId from the prior turn's result)
curl -N -X POST 'http://localhost:8000/api/chat?session_id=YOUR_SESSION_ID' \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"message":"Tell me more about the first one."}'

# List sessions
curl -s http://localhost:8000/api/sessions \
  -H "Authorization: Bearer $TOKEN" | jq
```

**Option B — the `AgentChat` client (streaming + sessions in a few lines)**

The same routes are a first-class Genkit client. `remote_agent` points at `/api/chat`, and `chat.send()` streams a turn and remembers the session — no manual NDJSON parsing or session-id juggling.

```python
import asyncio
import httpx
from genkit.agent import remote_agent
from genkit._core._http_client import get_cached_client

BASE = 'http://localhost:8000'


async def main() -> None:
    # Dev login → bearer token.
    async with httpx.AsyncClient() as http:
        resp = await http.post(
            f'{BASE}/api/auth/login',
            json={'email': 'demo@example.com', 'password': 'demo1234'},
        )
        token = resp.json()['access_token']

    # Send the token on every agent call. (The routes are auth-protected; the
    # client reuses one httpx client per event loop, so seed it with the header.)
    get_cached_client('agent_transport', headers={'Authorization': f'Bearer {token}'})

    # The server owns the session store, so state is server-managed.
    agent = remote_agent(f'{BASE}/api/chat', state_management='server')
    chat = agent.chat()

    # Turn 1 — stream chunks as they arrive, then read the settled reply.
    turn = chat.send('What plan am I on and what are my open orders?')
    async for chunk in turn:
        if chunk.text:
            print(chunk.accumulated_text, end='\r', flush=True)
    print((await turn).text)
    print(f'[session_id={chat.session_id}]')

    # Turn 2 — same session, so the agent still remembers turn 1.
    res = await chat.send('Tell me more about the first one.')
    print(res.text)


if __name__ == '__main__':
    asyncio.run(main())
```

Run it against the local server (from the sample root, in a second terminal):

```bash
uv run python scripts/try_chat.py
```

## The full agent surface, as code

Everything a production agent needs, in files you can open — not behind a managed API.

| Capability | Location |
|---------|----------|
| Chat routes (run, snapshot, abort) + streaming | `app/api/chat.py` (via `serve_agent`) |
| Per-user auth (dev + Firebase) | `app/auth/` (swappable providers) |
| Multi-turn sessions (list + history) | `app/api/sessions.py` |
| Session list, profiles, token logs | `app/data/` (one `Storage`, swappable backend) |
| Token accounting middleware | `app/ai/middleware/token_tracker.py` |
| User-scoped tools | `app/ai/tools/user_orders.py`, `app/ai/tools/user_profile.py` |
| The agent | `app/ai/agents/copilot.py` |
| Prompt (Dotprompt) | `app/ai/prompts/copilot.prompt` |

## Scope — what's here on purpose

This sample aims to be the smallest thing that's still a *real* agent backend: enough to run in production, small enough to read in one sitting. Some production concerns are intentionally left out so the core stays legible. Each is a normal FastAPI or Google Cloud add-on you can bring in when you actually need it:

| Concern | Status | When to add it |
|---------|--------|----------------|
| Rate limiting | Not included | Before you expose the API to untrusted callers — a few lines of middleware (see below) |
| Evals | Starter set (`evals/`) | Grow it as you tune the agent's behavior |
| Long-term / semantic memory | Not included | When users expect the agent to recall facts across sessions — add a vector store |
| Metrics dashboards | Not included | When you want charts; Cloud Run + Cloud Monitoring cover the basics for free |
| Relational DB + migrations | Not included | If you outgrow Firestore and want SQL |

The goal isn't to ship every feature — it's to give you a clean, Google-native core you understand, then let you reach for managed services as you scale.

### Rate limiting (recipe)

Not wired in by default so the core stays minimal. When you need it, [slowapi](https://github.com/laurentS/slowapi) is a few lines:

```python
# app/server.py
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

```python
# app/api/chat.py — cap the expensive route
@router.post("/chat")
@limiter.limit("20/minute")
async def run_turn(request: Request, ...):
    ...
```

For per-user limits, swap `get_remote_address` for a key function that reads the uid from the request's auth. `pip install slowapi` (or add it to `pyproject.toml`) to enable.

## Directory layout

```
app/
  server.py         FastAPI app assembly (uvicorn app.server:app)
  core/             cross-cutting plumbing
    config.py       settings + one-time env bootstrap
    identity.py     stable uid helpers
    tenant.py       per-request tenant scoping
  ai/               ← your agent lives here
    __init__.py     shared Genkit runtime
    context.py      current_uid(ctx) — the signed-in user inside a tool
    agents/
    middleware/
    prompts/
    tools/          per-user tools (getMyProfile, listMyOrders)
  api/              routes: login, deps, chat, sessions
  auth/             identity providers behind one interface
    base.py         the AuthProvider interface
    dev.py          signed dev JWT + demo login (default)
    firebase.py     Firebase backend (delete to drop Firebase)
  data/             every store the app has, at a glance
    __init__.py     the catalog: user_doc / user_collection declarations
    store.py        the storage engine (Storage interface + typed helpers)
    backends/
      firestore.py  the only file that talks to a database
    session_index.py  the UI's session list (plain code over user_collection)
    agent_sessions.py the agent's snapshot store selection
  models.py         your domain types, in one place
evals/              starter eval dataset + runner
scripts/
  seed_demo.py      seed the demo user's profile into the datastore
  try_chat.py       AgentChat client example (Option B)
```

### Store your own data

Two shapes, both typed and scoped to the signed-in user. Add a model to
`app/models.py`, declare the store in `app/data/__init__.py`, use it in a route:

```python
# app/models.py
class Note(BaseModel):
    title: str
    body: str


# app/data/__init__.py
notes = user_collection('notes', Note)  # many per user  (user_doc = one per user)


# app/api/notes.py
@router.get('/notes', response_model=list[Note])
async def list_notes(user: Annotated[AuthUser, Depends(get_current_user)]) -> list[Note]:
    return await notes.list(user.uid)


@router.post('/notes')
async def add_note(body: Note, user: Annotated[AuthUser, Depends(get_current_user)]) -> dict:
    return {'id': await notes.add(user.uid, body)}
```

No backend code — `notes` works on whatever `Storage` is configured. Switch
databases by adding one module under `data/backends/` and pointing `get_storage`
at it; nothing else changes.

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/auth/login` | Dev login → bearer token |
| POST | `/api/chat` | Run one agent turn (NDJSON stream) |
| POST | `/api/chat/getSnapshot` | Load snapshot for resume/rewind |
| POST | `/api/chat/abort` | Abort a running turn |
| GET | `/api/sessions` | List sessions for the signed-in user |
| GET | `/api/sessions/{session_id}` | Session metadata + message history |

**Quickstart body:** `{"message": "..."}` with optional `?session_id=`

**Full agent wire format:** `{"input": {...}, "init": {"sessionId": "..."}}`

## Configuration

Copy `.env.example` to `.env`:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Gemini API key (required) |
| `JWT_SECRET` | Dev JWT signing secret |
| `FIREBASE_PROJECT_ID` | Datastore project + Firebase ID token verification (deployed) |
| `FIRESTORE_EMULATOR_HOST` | Point at the Firestore emulator (local dev) |
| `GOOGLE_CLOUD_PROJECT` | Project id the emulator client uses locally |
| `AUTH_DEV_MODE` | Keep `/api/auth/login` when Firebase is configured |
| `GENKIT_ENV=dev` | Enable Genkit Dev UI reflection |

A datastore is required — set `FIREBASE_PROJECT_ID` when deployed or `FIRESTORE_EMULATOR_HOST` for local dev. There's no in-memory fallback: sessions, profiles, usage, and agent snapshots all persist, so nothing is silently lost on restart. If neither is set the app fails fast with a message telling you which variable to set.

### Per-user tools (getMyProfile, listMyOrders)

Both tools read user-scoped data from the data layer using the signed-in uid
(`current_uid(ctx)` from `app/ai/context.py`) — the same pattern for any tool that
touches a user's data. `getMyProfile` loads the **public** profile via the
`profiles` store (`user_doc('profiles', UserProfile)`), which lives at:

```
{firestore_collection}/profiles/{uid}/_
```

Example document:

```json
{
  "uid": "demo_at_example_com",
  "displayName": "Demo User",
  "email": "demo@example.com",
  "plan": "Pro",
  "memberSince": "2025-11-01",
  "company": "Acme Analytics",
  "timezone": "America/Los_Angeles",
  "supportTier": "standard"
}
```

`listMyOrders` works the same way against the `orders` collection
(`user_collection('orders', Order)`). Run `uv run python scripts/seed_demo.py` to
create the demo user's profile and orders in the datastore.

## Docker / Cloud Run

Local stack with Firestore emulator:

```bash
docker compose up --build
```

Deploy to Cloud Run:

```bash
echo -n "$GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-

gcloud run deploy gemini-agent-fastapi \
  --source . \
  --region us-central1 \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --set-env-vars FIREBASE_PROJECT_ID=your-project \
  --allow-unauthenticated
```

## Production auth

1. Sign users in with the Firebase client SDK.
2. Send the Firebase ID token as `Authorization: Bearer <idToken>` on every request.
3. Set `FIREBASE_PROJECT_ID` and disable dev login (unset `AUTH_DEV_MODE`).

To move to Clerk, Supabase, or another provider, add a module in `app/auth/` that implements `AuthProvider` (from `app/auth/base.py`) and point the factory in `app/auth/__init__.py` at it. The rest of the stack only ever sees an `AuthUser` — it never knows the vendor.

## AI assistant rules

- `.cursorrules` — product intent, scope bar, and where code belongs (for Cursor/Claude Code)
- `llms-full.txt` — decision guide + copy-paste patterns (machine-readable reference)
