# GCP Monitoring Dashboard — Project Plan

## Implementation Stages

This project is built incrementally across 6 stages. Each stage produces a
working (if incomplete) application. Stages can be implemented across multiple
conversations.

---

## Stage 1: Project Scaffolding & Backend Foundation

Set up the project structure, build tooling, and a working Express server
that can authenticate with GCP.

### Checklist

- [ ] **1.1** Create `package.json` with all dependencies
  - Frontend: react, react-dom, react-router-dom, recharts, @tanstack/react-query, lucide-react
  - Backend: express, googleapis, google-auth-library, cors
  - Dev: vite, @vitejs/plugin-react, typescript, tsx, tailwindcss, postcss, autoprefixer, concurrently
  - Scripts: `dev`, `dev:frontend`, `dev:server`, `build`
- [ ] **1.2** Create TypeScript configs (`tsconfig.json`, `tsconfig.node.json`)
- [ ] **1.3** Create Vite config with proxy to backend (`vite.config.ts`)
- [ ] **1.4** Create Tailwind config with dark theme defaults (`tailwind.config.js`, `postcss.config.js`)
- [ ] **1.5** Create `index.html` entry point
- [ ] **1.6** Create `src/main.tsx` and `src/App.tsx` with React Router skeleton
- [ ] **1.7** Create `src/index.css` with Tailwind directives and dark theme base styles
- [ ] **1.8** Create `server/index.ts` — Express server with health check endpoint
- [ ] **1.9** Create `server/auth.ts` — ADC authentication helper using `google-auth-library`
- [ ] **1.10** Create `server/cache.ts` — In-memory LRU cache with TTL
- [ ] **1.11** Verify: `pnpm install` succeeds, `pnpm dev` starts both servers
- [ ] **1.12** Create `README.md` with setup and usage instructions

### Deliverable
Running app at localhost:5173 showing a placeholder page. Backend at
localhost:3000 responding to `/api/health`. ADC auth working.

---

## Stage 2: GCP API Integration

Implement the backend GCP API clients and REST routes.

### Checklist

- [ ] **2.1** Create `server/gcp/monitoring.ts` — Cloud Monitoring API client
  - `listMetricDescriptors()` — discover available genkit metrics
  - `queryTimeSeries()` — fetch time series data with filters and alignment
  - `queryFeatureOverview()` — aggregate metrics across features
- [ ] **2.2** Create `server/gcp/tracing.ts` — Cloud Trace API client
  - `listTraces()` — list traces with filters and pagination
  - `getTrace()` — get all spans for a specific trace
  - Helper: `buildSpanTree()` — organize flat spans into tree structure
- [ ] **2.3** Create `server/routes/metrics.ts` — Metrics API routes
  - `GET /api/metrics/overview` — aggregated feature overview
  - `GET /api/metrics/timeseries` — time series for charts
- [ ] **2.4** Create `server/routes/traces.ts` — Traces API routes
  - `GET /api/traces` — list traces with filters
  - `GET /api/traces/:traceId` — get full trace
- [ ] **2.5** Create `GET /api/projects` route — list accessible projects
- [ ] **2.6** Create `POST /api/cache/clear` route
- [ ] **2.7** Wire all routes into `server/index.ts`
- [ ] **2.8** Add caching to all GCP API calls
- [ ] **2.9** Add error handling middleware (auth errors, API errors, validation)
- [ ] **2.10** Test with real GCP project: verify metrics and traces are returned
- [ ] **2.11** Create `src/api/client.ts` — Frontend API client (typed fetch wrapper)
- [ ] **2.12** Create `src/types/index.ts` — Shared TypeScript types for API responses

### Deliverable
Backend API fully functional. Can `curl localhost:3000/api/metrics/overview?projectId=...`
and get real data. Frontend has typed API client ready to use.

---

## Stage 3: Overview Page

Build the main dashboard page with charts and feature table.

### Checklist

- [ ] **3.1** Create `src/components/Layout.tsx` — App shell with header, nav, content area
- [ ] **3.2** Create `src/components/ProjectSelector.tsx` — Dropdown to select GCP project
  - Fetches project list from `/api/projects`
  - Stores selection in React context / URL param
- [ ] **3.3** Create `src/components/TimeRangeSelector.tsx` — Time range picker
  - Presets: Last 1h, 6h, 24h, 7d
  - Custom range with date pickers
  - Stores in URL params
- [ ] **3.4** Create React context for project + time range state
- [ ] **3.5** Create `src/hooks/useMetrics.ts` — TanStack Query hooks for metrics API
- [ ] **3.6** Create `src/components/charts/RequestsChart.tsx` — Area chart (Recharts)
- [ ] **3.7** Create `src/components/charts/SuccessRateChart.tsx` — Line chart, 0-100%
- [ ] **3.8** Create `src/components/charts/LatencyChart.tsx` — Line chart for latency
- [ ] **3.9** Create `src/components/FeatureTable.tsx` — Data table with columns:
  - Feature name (link), Success rate, Requests (with sparkline), Latency p95, I/O/Thinking tokens
  - Sortable columns, pagination
- [ ] **3.10** Create `src/pages/OverviewPage.tsx` — Compose all overview components
- [ ] **3.11** Style everything with dark theme matching mockups
- [ ] **3.12** Add loading skeletons and empty states

### Deliverable
Fully functional overview page showing real metrics data with interactive
charts and clickable feature table.

---

## Stage 4: Feature Detail Page

Build the per-feature detail view with expanded charts and traces list.

### Checklist

- [ ] **4.1** Create `src/components/StabilityMetrics.tsx` — Compact metrics bar
  - Total requests, success rate, latency p95, token totals
  - Trend indicators (▲▼ with percentage change)
- [ ] **4.2** Create `src/components/charts/TokenChart.tsx` — Stacked bar chart
  - Input (blue), Output (orange), Thinking (purple) tokens
  - Tab switch between "Tokens" and "Images" views
- [ ] **4.3** Create `src/hooks/useTraces.ts` — TanStack Query hooks for traces API
- [ ] **4.4** Create `src/components/TracesList.tsx` — Traces table
  - Status icon, created timestamp, truncated input, output, latency, model names
  - Filter bar (text search + status filter)
  - Sortable columns
  - Pagination (server-side via pageToken)
  - Download button (CSV export)
  - Click row → navigate to Trace Viewer
- [ ] **4.5** Create `src/components/FailedPathBanner.tsx` — Warning banner for failed paths
- [ ] **4.6** Create `src/pages/FeaturePage.tsx` — Compose all feature detail components
  - Breadcrumb: Genkit > featureName
  - Stability metrics bar
  - 2x2 chart grid (Requests, Tokens, Success Rate, Latency with p50+p95)
  - Traces section
- [ ] **4.7** Wire up React Router link from FeatureTable → FeaturePage
- [ ] **4.8** Style and test with real data

### Deliverable
Feature detail page with metrics, charts, and paginated traces list.
Full navigation flow from Overview → Feature.

---

## Stage 5: Trace Viewer Page

Build the trace viewer with span tree and detail panel.

### Checklist

- [ ] **5.1** Create `src/components/SpanTree.tsx` — Collapsible span tree
  - Tree structure built from parent/child relationships
  - Each node: status icon, span name, duration, type badge
  - Type badges: flow, step, dotprompt, util, model, tool (color-coded)
  - Expand/collapse controls
  - Click to select span
  - Highlight selected span
- [ ] **5.2** Create `src/components/SpanDetail.tsx` — Span detail panel
  - Header: span name, trace ID (copyable), status badge, type badge, duration, timestamp
  - Input section: syntax-highlighted JSON with copy button
  - Output section: syntax-highlighted JSON or `<redacted>` notice
  - For model spans: model name, token usage if available
  - Scrollable content
- [ ] **5.3** Create `src/pages/TraceViewerPage.tsx` — Compose span tree + detail
  - Two-panel layout (resizable divider?)
  - Breadcrumb: Genkit > featureName > Trace viewer
  - Auto-select root span on load
- [ ] **5.4** Add JSON syntax highlighting (simple CSS-based or use a lightweight lib)
- [ ] **5.5** Wire up React Router link from TracesList → TraceViewerPage
- [ ] **5.6** Style and test with real trace data

### Deliverable
Full trace viewer with interactive span tree. Complete navigation flow:
Overview → Feature → Trace.

---

## Stage 6: Polish & Error Handling

Final quality pass, error handling, documentation.

### Checklist

- [ ] **6.1** Add error boundaries around each page
- [ ] **6.2** Add proper error states for:
  - ADC not configured (show `gcloud auth` instructions)
  - Invalid project (show project selector)
  - No data in time range (empty state)
  - API errors (retry button)
- [ ] **6.3** Add loading skeletons for all data-dependent components
- [ ] **6.4** Responsive design check (min-width ~1024px for desktop)
- [ ] **6.5** Keyboard navigation in span tree (↑↓←→)
- [ ] **6.6** URL-driven state: project, time range, feature, trace ID all in URL
  - Shareable URLs, browser back/forward works
- [ ] **6.7** Performance: verify cache is working, no unnecessary re-renders
- [ ] **6.8** Finalize `README.md` with screenshots, troubleshooting
- [ ] **6.9** Clean up console warnings, unused imports, TODOs
- [ ] **6.10** Final review and test of complete flow

### Deliverable
Production-quality local development tool. Clean code, good error handling,
complete documentation.

---

## Dependencies Summary

### npm packages (all latest stable versions)

**Frontend:**
```
react, react-dom
react-router-dom
@tanstack/react-query
recharts
lucide-react
```

**Backend:**
```
express
googleapis
google-auth-library
cors
```

**Dev / Build:**
```
vite
@vitejs/plugin-react
typescript
tsx
tailwindcss
postcss
autoprefixer
concurrently
@types/react
@types/react-dom
@types/express
@types/cors
```

---

## Testing Strategy

For v1, testing is manual:
1. Run a Genkit app with GCP telemetry enabled
2. Generate some traces (success + failure cases)
3. Run the dashboard and verify all views show correct data
4. Check caching behavior (repeat requests should be fast)
5. Test error states (wrong project, no ADC, etc.)

Automated tests may be added in a future iteration.

---

## Current Status

**Last updated**: Stage 0 (Planning complete, ready to begin Stage 1)

| Stage | Status | Notes |
|-------|--------|-------|
| Stage 1: Scaffolding | 🔲 Not started | |
| Stage 2: API Integration | 🔲 Not started | |
| Stage 3: Overview Page | 🔲 Not started | |
| Stage 4: Feature Detail | 🔲 Not started | |
| Stage 5: Trace Viewer | 🔲 Not started | |
| Stage 6: Polish | 🔲 Not started | |
