# Agents sample

Backend examples for Genkit agents: `stream_bidi`, `send_text`, `send_resume`,
`detach`, `close`, `receive`, `output`, plus FastAPI serving for HTTP/useChat.

Requires `GEMINI_API_KEY`.

## Backend examples (primary)

From the sample directory:

```bash
cd py/samples/agents
uv sync
uv run python src/examples/basic_samples/01_define_agent_with_store.py
```

See [`src/examples/README.md`](src/examples/README.md) for the full matrix (`basic_samples/`, `branching_samples/`). Each example defines its agent inline.

## FastAPI / HTTP

```bash
cd py/samples/agents
uv run uvicorn src.main:app --port 8080
```

See [`src/examples/fastapi/README.md`](src/examples/fastapi/README.md).

Optional HTTP smoke test: `uv run python src/verify.py`

Wire format: [`WIRE_PROTOCOL.md`](WIRE_PROTOCOL.md)

## useChat UI

The Next.js frontend lives in [`../usechat-serve`](../usechat-serve). Start this
sample's backend on port 8080, then run the web app.
