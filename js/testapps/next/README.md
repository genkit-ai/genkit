# Next.js Integration

Sample app to test the `@genkit-ai/next` plugin — demonstrates using Genkit
flows as Next.js API routes with streaming support.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Joke Generation | `tellJoke` | Streaming joke generation via Next.js API route |
| Next.js App Router | `appRoute()` | Expose Genkit flows as Next.js route handlers |
| Streaming | `generateStream` | Token-by-token streaming in Next.js |

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
pnpm run dev
```

The Next.js dev server starts on http://localhost:3000.

## Testing This Demo

1. **Test the joke flow** via the API route (path depends on your route setup).

2. **Expected behavior**:
   - Genkit flows work as Next.js API route handlers
   - Streaming responses deliver tokens incrementally
   - OTel root span detection is disabled (required for Next.js edge runtime)
