# Multi-Server Sample

Run multiple ASGI servers (Litestar, Starlette) in parallel using Genkit's
`ServerManager`, with shared lifecycle management and graceful shutdown.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       ServerManager                              │
│         (coordinates all servers, handles SIGTERM)               │
│                                                                  │
│    ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│    │  Litestar     │  │  Starlette       │  │  Reflection    │  │
│    │  (Flows API)  │  │  (Reflection)    │  │  (DevUI)       │  │
│    │  :3400        │  │  :3100           │  │  :4000         │  │
│    │               │  │                  │  │                │  │
│    │  /flow/run    │  │  Genkit internal │  │  Genkit flows  │  │
│    │  /greet       │  │  API             │  │  & debugging   │  │
│    │  /__healthz   │  │                  │  │                │  │
│    │  /__serverz   │  │                  │  │                │  │
│    └──────────────┘  └──────────────────┘  └────────────────┘  │
│                                                                  │
│    Each server: own port, own lifecycle, own middleware          │
└─────────────────────────────────────────────────────────────────┘
```

## Features Demonstrated

| Feature | Code | Description |
|---------|------|-------------|
| Server Manager | `ServerManager()` | Coordinate multiple servers in one process |
| Litestar Server | `FlowsServerLifecycle` | Full-featured API server with controllers |
| Starlette Server | `ReflectionServerStarletteLifecycle` | Lightweight ASGI reflection server |
| Server Lifecycle | `AbstractBaseServer` | Custom startup/shutdown hooks per server |
| Logging Middleware | `LitestarLoggingMiddleware` | Request/response logging with timing |
| Health Checks | `/__healthz` | Server health endpoint |
| Server Info | `/__serverz` | Process info and server metadata |
| Graceful Shutdown | `terminate_all_servers()` | SIGTERM handling across all servers |
| Port Range | `ports=range(3400, 3410)` | Auto-select from a port range |
| Delayed Server Start | `add_server_after()` | Add servers after a delay |

## ELI5: Key Concepts

| Concept | ELI5 |
|---------|------|
| **ASGI** | A standard for Python web servers — like USB but for connecting web frameworks |
| **Litestar** | A modern Python web framework — fast and type-safe for building APIs |
| **Starlette** | A lightweight ASGI toolkit — the building block for frameworks like FastAPI |
| **ServerManager** | Runs multiple servers in parallel — each gets its own port and lifecycle |
| **Reflection Server** | Genkit's internal server — provides DevUI and flow execution endpoints |

## Quick Start

```bash
./run.sh
```

No API keys needed — this sample demonstrates server infrastructure, not AI generation.

## Testing This Demo

1. **Run the demo**:
   ```bash
   ./run.sh
   ```

2. **Test the Litestar server** (port 3400):
   - [ ] `GET http://localhost:3400/greet` — Greeting endpoint
   - [ ] `GET http://localhost:3400/flow/run` — Flow endpoint
   - [ ] `GET http://localhost:3400/__healthz` — Health check
   - [ ] `GET http://localhost:3400/__serverz` — Server info (includes PID)

3. **Test the Reflection server** (port 3100, dev mode only):
   - [ ] Starts automatically in dev mode (`GENKIT_ENV=dev`)

4. **Test graceful shutdown**:
   - [ ] `POST http://localhost:3400/__quitquitquitz` — Shuts down all servers
   - [ ] Or send `kill -15 <PID>` (get PID from `/__serverz`)

5. **Expected behavior**:
   - Multiple servers start concurrently on different ports
   - Each server has its own lifecycle hooks (startup/shutdown logged)
   - Logging middleware captures request/response timing
   - Graceful shutdown stops all servers cleanly

## Development

The `run.sh` script uses `watchmedo` for hot reloading on file changes.
