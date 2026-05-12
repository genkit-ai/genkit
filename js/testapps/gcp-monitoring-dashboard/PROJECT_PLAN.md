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

- [x] **1.1** Create `package.json` with all dependencies
  - Frontend: react, react-dom, react-router-dom, recharts, @tanstack/react-query, lucide-react
  - Backend: express, googleapis, google-auth-library, cors
  - Dev: vite, @vitejs/plugin-react, typescript, tsx, tailwindcss, postcss, autoprefixer, concurrently
  - Scripts: `dev`, `dev:frontend`, `dev:server`, `build`
- [x] **1.2** Create TypeScript configs (`tsconfig.json`, `tsconfig.node.json`)
- [x] **1.3** Create Vite config with proxy to backend (`vite.config.ts`)
- [x] **1.4** Create Tailwind config with dark theme defaults (`tailwind.config.js`, `postcss.config.js`)
- [x] **1.5** Create `index.html` entry point
- [x] **1.6** Create `src/main.tsx` and `src/App.tsx` with React Router skeleton
- [x] **1.7** Create `src/index.css` with Tailwind directives and dark theme base styles
- [x] **1.8** Create `server/index.ts` — Express server with health check endpoint
- [x] **1.9** Create `server/auth.ts` — ADC authentication helper using `google-auth-library`
- [x] **1.10** Create `server/cache.ts` — In-memory LRU cache with TTL
- [x] **1.11** Verify: `pnpm install` succeeds, `pnpm dev` starts both servers
- [x] **1.12** Create `README.md` with setup and usage instructions

### Deliverable
Running app at localhost:5173 showing a placeholder page. Backend at
localhost:3000 responding to `/api/health`. ADC auth working.

---

## Stage 2: GCP API Integration

Implement the backend GCP API clients and REST routes.

### Checklist

- [x] **2.1** Create `server/gcp/monitoring.ts` — Cloud Monitoring API client
  - `queryTimeSeries()` — fetch time series data with filters and alignment
  - `queryFeatureOverview()` — aggregate metrics across features
  - `normalizeTimeSeries()` — convert GCP format to simplified format
  - `computeAlignmentPeriod()` — auto-compute based on time range
  - **Note**: Metric prefix is `workload.googleapis.com/genkit/` (not `custom.googleapis.com/opencensus/`)
- [x] **2.2** Create `server/gcp/tracing.ts` — Cloud Trace v1 API client
  - `listTraces()` — list traces with filters and pagination (v1 API, ROOTSPAN view)
  - `getTrace()` — get all spans for a specific trace
  - `buildSpanTree()` — organize flat spans into tree structure
  - `normalizeTraceForList()` — summary format for list view
  - `flattenSpanTree()` — flatten tree back to list
  - **Note**: Cloud Trace v2 is write-only; reading requires v1 API
- [x] **2.3** Create `server/routes/metrics.ts` — Metrics API routes
  - `GET /api/metrics/overview` — aggregated feature overview (cached)
  - `GET /api/metrics/timeseries` — time series for charts (cached)
- [x] **2.4** Create `server/routes/traces.ts` — Traces API routes
  - `GET /api/traces` — list traces with filters, pagination, shorthand params (cached)
  - `GET /api/traces/:traceId` — get full trace with span tree (cached)
- [x] **2.5** Create `GET /api/projects` route — list accessible projects (in server/index.ts)
- [x] **2.6** Create `POST /api/cache/clear` route (in server/index.ts)
- [x] **2.7** Wire all routes into `server/index.ts`
- [x] **2.8** Add caching to all GCP API calls (metrics: 60s, trace list: 30s, trace detail: 5min)
- [x] **2.9** Add error handling middleware (auth errors, API errors, validation)
- [x] **2.10** Test with real GCP project: verify metrics and traces are returned
  - Tested with `weather-gen-test-next` project
  - Fixed: ALIGN_SUM → ALIGN_DELTA for CUMULATIVE metrics
  - Verified: metrics overview, timeseries, trace list, trace detail all working
- [x] **2.11** Create `src/api/client.ts` — Frontend API client (typed fetch wrapper with ApiError class)
- [x] **2.12** Create `src/types/index.ts` — Shared TypeScript types for API responses

### Deliverable
Backend API fully functional. Can `curl localhost:3000/api/metrics/overview?projectId=...`
and get real data. Frontend has typed API client ready to use.

---

## Stage 3: Overview Page

Build the main dashboard page with charts and feature table.

### Checklist

- [x] **3.1** Update `src/components/Layout.tsx` — Sticky header with ProjectSelector + TimeRangeSelector
- [x] **3.2** Create `src/components/ProjectSelector.tsx` — Dropdown with type-in + auto-fill from ADC
- [x] **3.3** Create `src/components/TimeRangeSelector.tsx` — Preset buttons (1h/6h/24h/7d/30d) + refresh
- [x] **3.4** Create `src/contexts/DashboardContext.tsx` — React context for project + time range + URL params
- [x] **3.5** Create `src/hooks/useMetrics.ts` — TanStack Query hooks for overview + timeseries
- [x] **3.6** Create `src/components/charts/RequestsChart.tsx` — Stacked area chart (success/failure)
- [x] **3.7** Create `src/components/charts/SuccessRateChart.tsx` — Line chart 0-100% with color coding
- [x] **3.8** Create `src/components/charts/LatencyChart.tsx` — Line chart (p99) with time formatting
- [x] **3.9** Create `src/components/FeatureTable.tsx` — Sortable table with success rate badges, token counts, nav links
- [x] **3.10** Update `src/pages/OverviewPage.tsx` — 3-column chart grid + feature table
- [x] **3.11** Update `src/App.tsx` — Wrap with DashboardProvider
- [x] **3.12** Tested with real data from weather-gen-test-next (7d range)

### Deliverable
Fully functional overview page showing real metrics data with interactive
charts and clickable feature table.

---

## Stage 4: Feature Detail Page

Build the per-feature detail view with expanded charts and traces list.

### Checklist

- [x] **4.1** Create `src/components/StabilityMetrics.tsx` — Compact metrics bar
  - Total requests, success rate (color-coded), latency p95, input/output token totals
  - Failure count indicator when > 0
- [x] **4.2** Create `src/components/charts/TokenChart.tsx` — Stacked bar chart
  - Input (blue) and Output (orange) tokens over time
  - Supports featureName filter prop
- [x] **4.3** Create `src/hooks/useTraces.ts` — TanStack Query hooks for traces API
  - `useTraceList()` with feature/status filtering and pagination
  - `useTraceDetail()` with 5-min cache for immutable traces
- [x] **4.4** Create `src/components/TracesList.tsx` — Traces table
  - Status icon, timestamp, name (linked), type badge, duration
  - Status filter dropdown (All/Success/Error)
  - Server-side pagination with page tokens
  - Click row → navigate to Trace Viewer
- [x] **4.5** Add featureName filter prop to RequestsChart, SuccessRateChart, LatencyChart
  - `feature/requests` and `feature/latency` use `metric.label.name` for filtering
  - `ai/generate/input/tokens` uses `metric.label.featureName` for filtering
- [x] **4.6** Create `src/pages/FeaturePage.tsx` — Compose all feature detail components
  - Breadcrumb: Genkit > featureName with back arrow
  - Stability metrics bar (from overview data, matched by `name` field)
  - 2x2 chart grid (Requests, Tokens, Success Rate, Latency)
  - Traces section with filtering and pagination
- [x] **4.7** Wire up React Router link from FeatureTable → FeaturePage (done in Stage 3)
- [x] **4.8** Tested with real data from weather-gen-test-next (7d range)
  - Fixed: metric label `name` vs `featureName` mismatch for feature filtering
  - Fixed: overview `name` field lookup in FeaturePage
  - All charts, metrics bar, and traces list rendering correctly

### Deliverable
Feature detail page with metrics, charts, and paginated traces list.
Full navigation flow from Overview → Feature.

---

## Stage 5: Trace Viewer Page

Build the trace viewer with span tree and detail panel.

### Checklist

- [x] **5.1** Create `src/components/SpanTree.tsx` — Collapsible span tree
  - Recursive tree with expand/collapse (auto-expand first 3 levels)
  - Each node: status icon, span name, type badge (color-coded), timing bar, duration
  - Timing bars show relative position & width vs root span duration
  - Color-coded bars: red (error), orange (model), blue (flow), green (tool), purple (other)
  - Click to select span (blue left border highlight)
- [x] **5.2** Create `src/components/SpanDetail.tsx` — Span detail panel
  - Header: span name, status badge, type badge, model name badge (for model spans)
  - Metadata grid: trace ID (copyable), span ID (copyable), duration, start time, path
  - Input/Output sections with JSON syntax highlighting (CSS-based) or `<redacted>` notice
  - Copy buttons on IDs and input/output values
  - Labels section: filtered genkit-specific labels with clean display names
  - Scrollable content
- [x] **5.3** Create `src/pages/TraceViewerPage.tsx` — Compose span tree + detail
  - Two-panel 50/50 layout with full viewport height
  - Breadcrumb: Genkit > featureName > Trace viewer (links back to feature page)
  - Auto-select root span on load, shows span count
  - Loading/error/empty states
- [x] **5.4** Add JSON syntax highlighting (CSS-based: strings blue, numbers cyan, keywords red)
  - Added type badges for `executable-prompt` and `unknown` subtypes
- [x] **5.5** Wire up React Router link from TracesList → TraceViewerPage (done in Stage 1)
- [x] **5.6** Tested with real trace data from pavelj-genkit-test1
  - Verified: generate → googleai/gemini-3-flash-preview → POST span tree
  - Span selection, type badges, timing bars, model name display all working
  - Full navigation flow: Overview → Feature → Trace → back all working
  - Note: Cloud Trace v1 API may return fewer results with 7d+ time ranges + label filters; 24h range works reliably

### Deliverable
Full trace viewer with interactive span tree. Complete navigation flow:
Overview → Feature → Trace.

---

## Stage 6: Polish & Error Handling

Final quality pass, error handling, documentation.

### Checklist

- [x] **6.1** Add error boundaries around each page
  - Created `ErrorBoundary` class component with retry button
  - Wraps entire app + each page route individually
- [x] **6.2** Add proper error states for:
  - Created `AuthGate` component: checks backend health + ADC on startup
  - Shows `gcloud auth application-default login` instructions if ADC fails
  - Shows retry button for connection errors
  - Empty states already handled in all charts, tables, and traces list
- [x] **6.3** Clean up unused imports (removed `useCallback` from SpanTree)
- [ ] **6.4** _(Deferred)_ Loading skeletons, responsive design, keyboard nav
- [x] **6.5** URL-driven state already works: project, time range in URL params
  - Shareable URLs, browser back/forward works via React Router
- [x] **6.6** Performance: backend caching verified (metrics: 60s, trace list: 30s, trace detail: 5min)
  - TanStack Query staleTime prevents unnecessary refetches
- [x] **6.7** Final test of complete flow with pavelj-genkit-test1
  - Overview → Feature → Trace → back navigation all working
  - Error boundary + auth gate verified working

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

**Last updated**: Stage 6 complete — all stages done! 🎉

| Stage | Status | Notes |
|-------|--------|-------|
| Stage 1: Scaffolding | ✅ Complete | All 12 items done. Frontend + backend running, ADC auth working. |
| Stage 2: API Integration | ✅ Complete | All 12 items done. Tested with weather-gen-test-next. Fixed ALIGN_DELTA for CUMULATIVE metrics. |
| Stage 3: Overview Page | ✅ Complete | All 12 items done. Charts, feature table, project/time selectors, dark theme all working. |
| Stage 4: Feature Detail | ✅ Complete | All 8 items done. StabilityMetrics, charts with feature filtering, TracesList with pagination. |
| Stage 5: Trace Viewer | ✅ Complete | All 6 items done. SpanTree with timing bars, SpanDetail with JSON highlighting, two-panel layout. |
| Stage 6: Polish | ✅ Complete | ErrorBoundary, AuthGate with ADC instructions, cleanup. Loading skeletons & keyboard nav deferred to future iteration. |
