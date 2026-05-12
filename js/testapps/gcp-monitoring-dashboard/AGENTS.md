# GCP Monitoring Dashboard — Agent Instructions

## Project Overview

This is a locally-running Vite + React web application that visualizes Genkit
telemetry data from Google Cloud Monitoring (GCM) and Cloud Trace. It
authenticates using Application Default Credentials (ADC), fetches trace and
metric data from GCP APIs, and renders a monitoring dashboard.

**Location**: `js/testapps/gcp-monitoring-dashboard/`

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  React Frontend     │────▶│  Express Backend      │────▶│  GCP APIs       │
│  (Vite, port 5173)  │     │  (proxy, port 3000)   │     │  - Monitoring v3│
│  Recharts, Tailwind │     │  ADC auth + cache     │     │  - Trace v1     │
└─────────────────────┘     └──────────────────────┘     └─────────────────┘
```

- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS + Recharts
- **Backend**: Express + `google-auth-library` (ADC) + `googleapis`
- **Caching**: Server-side in-memory LRU cache with TTL
- **Dev mode**: Vite proxies `/api/*` to the Express backend

### Why a backend proxy?

Browser apps cannot call GCP APIs directly due to CORS restrictions and
credential security. The Express backend uses ADC (from
`gcloud auth application-default login`) to authenticate and proxies requests.

## Key Files & Directories

```
js/testapps/gcp-monitoring-dashboard/
├── AGENTS.md                 # This file — project instructions
├── DESIGN.md                 # Detailed design document
├── PROJECT_PLAN.md           # Implementation plan with checklists
├── README.md                 # User-facing readme (how to run)
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tsconfig.node.json
├── tailwind.config.js
├── postcss.config.js
├── index.html
├── server/                   # Express backend
│   ├── index.ts              # Server entry point
│   ├── auth.ts               # ADC authentication helper
│   ├── cache.ts              # In-memory LRU cache
│   ├── gcp/
│   │   ├── monitoring.ts     # Cloud Monitoring API client
│   │   └── tracing.ts        # Cloud Trace API client
│   └── routes/
│       ├── metrics.ts        # /api/metrics/* routes
│       └── traces.ts         # /api/traces/* routes
└── src/                      # React frontend
    ├── main.tsx
    ├── App.tsx
    ├── index.css             # Tailwind + global styles
    ├── api/
    │   └── client.ts         # Fetch wrapper for backend API
    ├── hooks/
    │   ├── useMetrics.ts     # React Query hooks for metrics
    │   └── useTraces.ts      # React Query hooks for traces
    ├── types/
    │   └── index.ts          # Shared TypeScript types
    ├── pages/
    │   ├── OverviewPage.tsx   # Main dashboard (Screenshot 1)
    │   ├── FeaturePage.tsx    # Feature detail (Screenshots 2, 5)
    │   └── TraceViewerPage.tsx # Trace viewer (Screenshots 3, 4)
    └── components/
        ├── Layout.tsx         # App shell with nav
        ├── TimeRangeSelector.tsx
        ├── ProjectSelector.tsx
        ├── charts/
        │   ├── RequestsChart.tsx
        │   ├── SuccessRateChart.tsx
        │   ├── LatencyChart.tsx
        │   └── TokenChart.tsx
        ├── FeatureTable.tsx
        ├── TracesList.tsx
        ├── SpanTree.tsx
        └── SpanDetail.tsx
```

## GCP APIs Used

### Cloud Monitoring API v3

- **Discovery**: `https://monitoring.googleapis.com/$discovery/rest?version=v3`
- **Time Series**: `projects.timeSeries.list` — fetches metric data over time
- **Metric Descriptors**: `projects.metricDescriptors.list` — discover available metrics

**Genkit metric names** (prefixed with `workload.googleapis.com/genkit/`):
- `genkit/feature/requests` — feature-level request counts
- `genkit/feature/latency` — feature-level latency histogram
- `genkit/feature/path/requests` — path-level request counts (errors)
- `genkit/feature/path/latency` — path-level latency
- `genkit/ai/generate/requests` — model generate request counts
- `genkit/ai/generate/latency` — model generate latency
- `genkit/ai/generate/input/tokens` — input token counts
- `genkit/ai/generate/output/tokens` — output token counts
- `genkit/ai/generate/thinking/tokens` — thinking token counts
- `genkit/ai/generate/input/characters` — input character counts
- `genkit/ai/generate/output/characters` — output character counts
- `genkit/ai/generate/input/images` — input image counts
- `genkit/ai/generate/output/images` — output image counts

**Common metric dimensions/labels**:
- `featureName`, `modelName`, `path`, `status`, `error`, `source`, `sourceVersion`

### Cloud Trace API v1

- **List traces**: `projects.traces.list` — paginated trace listing with filters (use `view=ROOTSPAN`)
- **Get trace**: `projects.traces.get` — get all spans for a trace

> **Note**: Cloud Trace v2 is write-only (for ingesting spans). Reading traces
> requires the v1 API at `https://cloudtrace.googleapis.com/v1`.

**Genkit span attributes** (normalized with `/` instead of `:`):
- `genkit/type` — action, flow, flowStep, util, userEngagement
- `genkit/name` — action name
- `genkit/path` — hierarchical path (e.g., `/{flowName}/{stepName}`)
- `genkit/state` — success, error
- `genkit/isRoot` — true for root spans
- `genkit/input` — JSON input (redacted in traces, available in logs)
- `genkit/output` — JSON output (redacted in traces, available in logs)
- `genkit/metadata/subtype` — model, tool, etc.
- `genkit/feature` — feature name (set on root spans)
- `genkit/model` — model name (set on model spans)
- `genkit/rootState` — root span state

## UI Design Reference

Screenshots in `ref/monitoring-ui-ideas/` (1.png through 5.png):

1. **Overview Page**: Summary charts (Requests, Success Rate, Latency p95) + Feature table
2. **Feature Detail**: Stability metrics bar + Traces list with filter
3. **Trace Viewer**: Span tree (left) + Span detail with flow input (right)
4. **Trace Viewer (span)**: Generate action detail showing model request JSON
5. **Feature Detail (expanded)**: All charts including token usage stacked bar

## Design Principles

- **Dark theme** matching the GCP console aesthetic shown in mockups
- **Responsive** charts that resize with the window
- **Local caching** to avoid excessive API calls (metrics: 60s TTL, traces: 30s, individual trace: 5min)
- **Progressive loading** — show data as it arrives, don't block on all APIs
- **Error handling** — graceful degradation when APIs are unavailable

## Development Workflow

```bash
# Prerequisites
gcloud auth application-default login

# Install
cd js/testapps/gcp-monitoring-dashboard
pnpm install

# Development (runs both frontend + backend)
pnpm dev

# Frontend only
pnpm dev:frontend

# Backend only
pnpm dev:server
```

## Implementation Stages

See `PROJECT_PLAN.md` for the detailed implementation plan with checklists.

The project is built in stages:
1. **Stage 1**: Project scaffolding + backend foundation
2. **Stage 2**: GCP API integration + caching
3. **Stage 3**: Overview page (charts + feature table)
4. **Stage 4**: Feature detail page
5. **Stage 5**: Trace viewer page
6. **Stage 6**: Polish, error handling, README

## Important Notes

- Input/output data is **redacted** in Cloud Trace spans (replaced with `<redacted>`).
  To show actual I/O, we need to fetch from **Cloud Logging** where the full
  content is logged as structured log entries. This is a potential future enhancement.
- The Cloud Monitoring API has rate limits. The caching layer is critical.
- Metrics use `DELTA` aggregation temporality but are converted to `CUMULATIVE`
  on the GCP side. When querying, use appropriate alignment periods.
- Span attribute keys use `/` separators in GCP (normalized from `:` by the plugin).
