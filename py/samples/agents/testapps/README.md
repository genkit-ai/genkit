# Agents testapp (Python)

These agents integrate with the testapp in `js/testapps/agents`. Each
`*_agent.py` is a self-contained agent plus a `test_*` flow you can Run in the
Dev UI, and `server.py` mounts every agent behind `/api/<name>`. Swap the Node
backend out for this Python one and the existing web frontend keeps working
unchanged.

Requires `GEMINI_API_KEY`.

## Run

```bash
cd py/samples/agents
genkit start -- uv run testapps/server.py
```

- Dev UI: http://localhost:4000 — pick any `test_*` flow and Run it.
- HTTP API: http://localhost:8080 — one endpoint per agent.

Point the web frontend at this backend instead of the Node one:

```bash
cd js/testapps/agents/web && pnpm install && pnpm dev   # http://localhost:5173
```

Or run a single agent on its own:

```bash
genkit start -- uv run testapps/weather_agent.py
```
