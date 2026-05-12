# Genkit GCP Monitoring Dashboard

A locally-running dashboard for visualizing Genkit telemetry data from Google
Cloud Monitoring and Cloud Trace.

## What it does

When you use the `@genkit-ai/google-cloud` plugin to send telemetry to GCP,
this dashboard lets you browse that data locally with a Genkit-aware UI:

- **Overview**: See all your Genkit features, request volumes, success rates,
  latencies, and token usage at a glance
- **Feature Detail**: Drill into a specific feature to see detailed charts
  and individual traces
- **Trace Viewer**: Inspect individual traces with a span tree showing the
  full execution flow, including input/output data

## Prerequisites

1. **Node.js 18+** and **pnpm**
2. **Google Cloud SDK** (`gcloud`) installed
3. A GCP project with Genkit telemetry data (traces and/or metrics)

## Setup

### 1. Authenticate with GCP

The dashboard uses Application Default Credentials (ADC). Set them up:

```bash
gcloud auth application-default login
```

This opens a browser for OAuth login and saves credentials locally.

### 2. Install dependencies

```bash
cd js/testapps/gcp-monitoring-dashboard
pnpm install
```

### 3. Run the dashboard

```bash
pnpm dev
```

This starts:
- **Frontend** at http://localhost:5173 (Vite dev server)
- **Backend** at http://localhost:3000 (Express API proxy)

Open http://localhost:5173 in your browser.

## Usage

1. **Select a project**: Use the dropdown in the top-left to pick your GCP
   project
2. **Choose a time range**: Use the selector in the top-right (1h, 6h, 24h, 7d)
3. **Browse features**: The overview page shows all Genkit features with
   summary metrics
4. **Click a feature**: Opens the detail page with charts and traces
5. **Click a trace**: Opens the trace viewer with the full span tree

## Architecture

```
Browser (localhost:5173)  →  Express Backend (localhost:3000)  →  GCP APIs
   React + Recharts              ADC Auth + Cache                Monitoring v3
   Tailwind CSS                                                  Trace v2
```

The backend acts as an authenticated proxy to GCP APIs. It caches responses
to minimize API calls:

- Metrics: 60s cache
- Trace lists: 30s cache
- Individual traces: 5min cache (traces are immutable)

## Scripts

| Command | Description |
|---------|-------------|
| `pnpm dev` | Start frontend + backend (development) |
| `pnpm dev:frontend` | Start frontend only |
| `pnpm dev:server` | Start backend only |
| `pnpm build` | Build for production |

## Troubleshooting

### "ADC not configured" error

Run `gcloud auth application-default login` and try again.

### No data showing

- Ensure your Genkit app has the `@genkit-ai/google-cloud` plugin enabled
- Verify telemetry data exists in GCP Console → Cloud Trace
- Check that you selected the correct project and time range
- Metrics may take a few minutes to appear after first being emitted

### API quota errors

The dashboard caches API responses to minimize calls. If you still hit quota
limits, try using a longer time range (fewer data points) or wait a moment
before refreshing.

## Development

See [DESIGN.md](./DESIGN.md) for the detailed design document and
[PROJECT_PLAN.md](./PROJECT_PLAN.md) for the implementation plan.

For AI-assisted development, see [AGENTS.md](./AGENTS.md) for context and
instructions.
