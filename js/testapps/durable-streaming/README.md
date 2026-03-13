# Durable Streaming

Demonstrates durable streaming with multiple stream manager backends —
in-memory, Firestore, and Realtime Database. This allows clients to
reconnect and resume consuming stream chunks even after disconnection.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Basic Streaming | `streamy` | Counts to N, streaming each step |
| Error Handling | `streamyThrowy` | Throws an error after 3 chunks |
| In-Memory Backend | `/streamy` | Stream manager using in-memory storage |
| Firestore Backend | `/streamyFirestore` | Stream manager backed by Cloud Firestore |
| RTDB Backend | `/streamyRtdb` | Stream manager backed by Realtime Database |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager
- **Firebase project** with Firestore and/or Realtime Database enabled

### Build and Install

From the repo root:

```bash
pnpm install
pnpm run setup
```

### Firebase Configuration

Set up Application Default Credentials for Firebase:

```bash
export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account-key.json'
```

Or use `gcloud`:

```bash
gcloud auth application-default login
```

## Run the Sample

```bash
pnpm build && pnpm start
```

The Express server starts on port `3500`.

## Testing This Demo

1. **Test in-memory streaming**:
   ```bash
   curl -X POST http://localhost:3500/streamy \
     -H "Content-Type: application/json" \
     -d '{"data": 5}'
   ```

2. **Test Firestore-backed streaming**:
   ```bash
   curl -X POST http://localhost:3500/streamyFirestore \
     -H "Content-Type: application/json" \
     -d '{"data": 5}'
   ```

3. **Test RTDB-backed streaming**:
   ```bash
   curl -X POST http://localhost:3500/streamyRtdb \
     -H "Content-Type: application/json" \
     -d '{"data": 5}'
   ```

4. **Test error handling**:
   ```bash
   curl -X POST http://localhost:3500/streamyThrowy \
     -H "Content-Type: application/json" \
     -d '{"data": 5}'
   ```

5. **Expected behavior**:
   - `streamy` streams count objects `{count: 0}` through `{count: N-1}`
   - `streamyThrowy` streams 3 chunks then throws an error
   - All three backends (in-memory, Firestore, RTDB) behave identically
