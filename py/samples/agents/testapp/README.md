# Agent backend

A FastAPI process with the agents in this folder mounted at `/api/<name>`.
Sessions stay in memory or on disk so it runs on `GEMINI_API_KEY` tonight.
Swap in `FirestoreSessionStore()` from `genkit-google-cloud` when you
deploy.

```bash
cd py/samples/agents
uv sync
genkit start -- uv run testapp/server.py
```

Dev UI is http://localhost:4000. HTTP is http://localhost:8080. Look at
`server.py` for the mount list.
