# GCP Monitoring Dashboard — Design Document

## 1. Problem Statement

Genkit's GCP plugin exports telemetry (traces, metrics, logs) to Google Cloud
Monitoring and Cloud Trace. Currently, users must navigate the GCP Console to
view this data, which requires familiarity with Cloud Monitoring's query
language and Cloud Trace's UI. We want a purpose-built, locally-running
dashboard that presents Genkit-specific telemetry in an intuitive way.

## 2. Goals

- **Local development tool**: Run on localhost, authenticate via ADC
- **Genkit-aware**: Understands Genkit's telemetry schema (features, actions,
  models, paths) rather than showing raw metrics
- **Three-level drill-down**: Overview → Feature → Trace
- **Fast**: Local caching to minimize API calls and provide responsive UX
- **Self-contained**: Single `pnpm dev` to start both frontend and backend

## 3. Non-Goals (for v1)

- Deployment to cloud (this is a local dev tool)
- Real-time streaming / WebSocket updates
- Cloud Logging integration for full input/output content (future enhancement)
- Multi-project comparison views
- Alerting or threshold configuration

---

## 4. Architecture

### 4.1 System Diagram

```
                    ┌──────────────────────────────────┐
                    │  Browser (localhost:5173)         │
                    │                                  │
                    │  React + Recharts + Tailwind     │
                    │  React Router (client-side)      │
                    │                                  │
                    │  Routes:                         │
                    │    /                → Overview    │
                    │    /feature/:name  → Feature     │
                    │    /trace/:id      → Trace       │
                    └──────────┬───────────────────────┘
                               │ fetch /api/*
                               ▼
                    ┌──────────────────────────────────┐
                    │  Express Server (localhost:3000)  │
                    │                                  │
                    │  ┌─────────┐  ┌──────────────┐  │
                    │  │  Auth   │  │  LRU Cache   │  │
                    │  │  (ADC)  │  │  (in-memory)  │ │
                    │  └────┬────┘  └──────┬───────┘  │
                    │       │              │           │
                    │  ┌────┴──────────────┴────────┐  │
                    │  │   GCP API Clients          │  │
                    │  │   - monitoring.v3           │  │
                    │  │   - cloudtrace.v2           │  │
                    │  └────────────┬───────────────┘  │
                    └──────────────┼───────────────────┘
                                   │ HTTPS
                                   ▼
                    ┌──────────────────────────────────┐
                    │  Google Cloud Platform            │
                    │  - Cloud Monitoring API v3        │
                    │  - Cloud Trace API v2             │
                    └──────────────────────────────────┘
```

### 4.2 Frontend Architecture

**React 18** with TypeScript, using:

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Build tool | Vite | Fast HMR, standard for modern React |
| Routing | React Router v6 | Client-side routing for SPA |
| Data fetching | TanStack Query (React Query) | Caching, refetching, loading states |
| Charts | Recharts | React-native charts, good TypeScript support |
| Styling | Tailwind CSS | Rapid dark-theme styling, utility-first |
| Icons | Lucide React | Lightweight, tree-shakeable icons |

**State Management**: No global state library needed. TanStack Query handles
server state; component state handles UI state (selected time range, expanded
spans, etc.). The project ID and time range are stored in URL params / React
context.

### 4.3 Backend Architecture

**Express** server with TypeScript, using:

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Auth | `google-auth-library` | ADC support, token refresh |
| GCP APIs | `googleapis` | Official Google API client |
| Caching | Custom LRU Map | Simple, no extra dependency |
| Build | tsx (runtime) | No compile step needed for dev |

The backend is intentionally thin — it's a proxy with caching. No database,
no sessions, no complex business logic.

### 4.4 Caching Strategy

```
┌─────────────────────────────────────────────────┐
│  Cache Layer (server/cache.ts)                  │
│                                                 │
│  Key: hash(apiName + params)                    │
│  Value: { data, timestamp, ttl }                │
│                                                 │
│  TTLs:                                          │
│  - Metric time series:  60 seconds              │
│  - Trace list:          30 seconds              │
│  - Individual trace:    5 minutes (immutable)   │
│  - Project list:        10 minutes              │
│                                                 │
│  Max entries: 1000 (LRU eviction)               │
│  Memory limit: ~50MB                            │
└─────────────────────────────────────────────────┘
```

Cache is invalidated on:
- TTL expiry (lazy — checked on read)
- Manual clear via `POST /api/cache/clear`
- Time range change (different params = different cache key)

---

## 5. API Design

### 5.1 Backend REST API

#### `GET /api/projects`

Returns list of accessible GCP projects.

Response:
```json
{
  "projects": [
    { "projectId": "my-project", "name": "My Project" }
  ]
}
```

#### `GET /api/metrics/overview`

Aggregated overview metrics for the feature table.

Query params:
- `projectId` (required): GCP project ID
- `startTime` (required): ISO 8601 timestamp
- `endTime` (required): ISO 8601 timestamp

Response:
```json
{
  "features": [
    {
      "name": "flowTouristActivities",
      "totalRequests": 342,
      "successRate": 0.0,
      "latencyP95Ms": 10200,
      "inputTokens": 114000,
      "outputTokens": 83000,
      "thinkingTokens": 0,
      "inputImages": 0,
      "outputImages": 0
    }
  ]
}
```

#### `GET /api/metrics/timeseries`

Raw time series data for charts.

Query params:
- `projectId` (required)
- `metricType` (required): e.g., `feature/requests`, `ai/generate/input/tokens`
- `startTime`, `endTime` (required)
- `featureName` (optional): filter to specific feature
- `alignmentPeriod` (optional): e.g., `3600s` (defaults based on time range)

Response:
```json
{
  "timeSeries": [
    {
      "labels": { "name": "flowTouristActivities", "status": "success" },
      "points": [
        { "time": "2026-04-23T10:00:00Z", "value": 15 },
        { "time": "2026-04-23T11:00:00Z", "value": 12 }
      ]
    }
  ]
}
```

#### `GET /api/traces`

List traces with optional filtering.

Query params:
- `projectId` (required)
- `startTime`, `endTime` (required)
- `featureName` (optional): filter by genkit feature
- `status` (optional): `success` or `error`
- `pageSize` (optional, default 20)
- `pageToken` (optional): for pagination

Response:
```json
{
  "traces": [
    {
      "traceId": "abc123...",
      "rootSpan": {
        "spanId": "def456...",
        "name": "flowTouristActivities",
        "startTime": "2026-04-23T10:18:30Z",
        "endTime": "2026-04-23T10:18:37Z",
        "status": "error",
        "attributes": { ... }
      },
      "spanCount": 12
    }
  ],
  "nextPageToken": "..."
}
```

#### `GET /api/traces/:traceId`

Full trace with all spans.

Query params:
- `projectId` (required)

Response:
```json
{
  "traceId": "abc123...",
  "spans": [
    {
      "spanId": "def456...",
      "parentSpanId": "",
      "name": "flowTouristActivities",
      "startTime": "2026-04-23T10:18:30Z",
      "endTime": "2026-04-23T10:18:37Z",
      "status": { "code": "ERROR" },
      "attributes": {
        "genkit/type": "flow",
        "genkit/name": "flowTouristActivities",
        "genkit/path": "/{flowTouristActivities}",
        "genkit/state": "error",
        "genkit/isRoot": "true",
        "genkit/input": "<redacted>",
        "genkit/output": "<redacted>"
      },
      "childSpanCount": 5
    }
  ]
}
```

#### `POST /api/cache/clear`

Clears the server-side cache. No body required.

---

## 6. UI Design

### 6.1 Page Layout

All pages share a common layout:
- **Header bar**: "Genkit" branding (left), Project selector dropdown (top-left),
  Time range selector (top-right)
- **Content area**: Page-specific content
- **Dark theme**: Dark background (#0d1117), card backgrounds (#161b22),
  borders (#30363d), text (#e6edf3)

### 6.2 Overview Page (`/`)

*Reference: Screenshot 1*

```
┌─────────────────────────────────────────────────────────┐
│  [Project ▼]              Genkit            [24h ▼]     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Requests │  │ Success Rate │  │ Latency p95  │      │
│  │ [area]   │  │ [line]       │  │ [line]       │      │
│  └──────────┘  └──────────────┘  └──────────────┘      │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Feature  │ Success │ Requests │ Latency │ I/O/T │   │
│  ├──────────┼─────────┼──────────┼─────────┼───────┤   │
│  │ flow...  │ 0%      │ ▓▓▓ 342 │ ── 10.2s│ ...   │   │
│  └──────────┴─────────┴──────────┴─────────┴───────┘   │
└─────────────────────────────────────────────────────────┘
```

**Charts**:
- **Requests**: Area chart, shows total request volume over time
- **Success Rate**: Line chart, 0-100% scale
- **Latency (p95)**: Line chart, auto-scaled

**Feature Table**:
- Clickable feature names → navigate to Feature page
- Inline sparkline for requests column
- Token counts with icons (text tokens, image counts)
- Sortable columns
- Pagination (10 items per page)

### 6.3 Feature Detail Page (`/feature/:name`)

*Reference: Screenshots 2, 5*

```
┌─────────────────────────────────────────────────────────┐
│  Genkit > featureName                       [24h ▼]     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Stability metrics                                      │
│  Total: 342  Success: 0%  Latency: 10.3s  Tokens: ...  │
│                                                         │
│  ┌──────────┐  ┌─────────────────────────────────┐      │
│  │ Requests │  │ Tokens [Tokens|Images]           │     │
│  │ [area]   │  │ [stacked bar: in/out/thinking]  │      │
│  └──────────┘  └─────────────────────────────────┘      │
│  ┌──────────────┐  ┌──────────────────────────┐         │
│  │ Success Rate │  │ Latency (p50/p95)        │         │
│  │ [line]       │  │ [multi-line]             │         │
│  └──────────────┘  └──────────────────────────┘         │
│                                                         │
│  Traces                                                 │
│  ⚠ You have one failed path                            │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Filter: [____________]              [Download]  │    │
│  │ Status │ Created │ Input │ Output │ Latency │ M │   │
│  │ ● err  │ Apr 23  │ {...} │ N/A    │ 7.65s   │ …│   │
│  └────────┴─────────┴───────┴────────┴─────────┴──┘    │
└─────────────────────────────────────────────────────────┘
```

**Stability Metrics Bar**: Compact row of key stats with trend indicators (▲▼).

**Charts** (2x2 grid):
- **Requests**: Area chart filtered to this feature
- **Tokens**: Stacked bar chart (input=blue, output=orange, thinking=purple)
  with tab switch between "Tokens" and "Images"
- **Success Rate**: Line chart for this feature
- **Latency**: Dual line chart showing p50 and p95

**Traces List**:
- Filter bar for searching by input content, status, etc.
- Sortable table with status icon, timestamp, truncated input/output, latency, model names
- Click a row → navigate to Trace Viewer
- Download button (CSV export of visible traces)
- Pagination

### 6.4 Trace Viewer Page (`/trace/:traceId`)

*Reference: Screenshots 3, 4*

```
┌─────────────────────────────────────────────────────────┐
│  Genkit > featureName > Trace viewer                    │
├─────────────┬───────────────────────────────────────────┤
│ Trace spans │  spanName                    ID: abc...   │
│             │  ● Failed  🔀 flow  ⏱ 7.6s  📅 Apr 23   │
│ ▼ ● flow   │                                           │
│   ▼ ✓ step │  Flow input                               │
│     ✓ dot  │  ┌─────────────────────────────────┐      │
│     ✓ gen  │  │ {                               │      │
│       ✓ m  │  │   "startDate": "2025-01-16",   │      │
│       ✓ P  │  │   "cities": ["New York, NY"]   │      │
│     ✓ tool │  │ }                               │      │
│   ▼ ✓ gen  │  └─────────────────────────────────┘      │
│     ✓ mod  │                                           │
│     ✓ POST │  Flow output                              │
│   ✓ render │  ┌─────────────────────────────────┐      │
│   ▼ ● step │  │ <redacted>                      │      │
│     ● dot  │  └─────────────────────────────────┘      │
│             │                                           │
├─────────────┤                                           │
│  250px      │              remaining width              │
└─────────────┴───────────────────────────────────────────┘
```

**Span Tree** (left panel, ~250px):
- Collapsible tree structure
- Each node shows: status icon (✓ green / ● red), span name, duration
- Type badge: `flow`, `step`, `dotprompt`, `util`, `model`, `tool`
- Clicking a span updates the detail panel

**Span Detail** (right panel):
- Header: span name, trace ID, status badge, type badge, duration, timestamp
- **Input section**: Formatted JSON with syntax highlighting and copy button
- **Output section**: Formatted JSON (or `<redacted>` notice)
- For model spans: show model configuration, token usage
- Scrollable content area

---

## 7. GCP API Query Patterns

### 7.1 Fetching Feature Overview Metrics

To populate the feature table, we need to query multiple metric types and
aggregate by the `name` (for features) label:

```
# Request counts per feature
metric.type = "custom.googleapis.com/opencensus/genkit/feature/requests"
group_by: metric.label.name, metric.label.status

# Latency per feature  
metric.type = "custom.googleapis.com/opencensus/genkit/feature/latency"
group_by: metric.label.name
aligner: ALIGN_PERCENTILE_95

# Token counts per feature
metric.type = "custom.googleapis.com/opencensus/genkit/ai/generate/input/tokens"
group_by: metric.label.featureName
```

### 7.2 Fetching Time Series for Charts

For each chart, we query the appropriate metric with time alignment:

| Time Range | Alignment Period |
|-----------|-----------------|
| 1 hour | 60s |
| 6 hours | 300s |
| 24 hours | 600s |
| 7 days | 3600s |

### 7.3 Fetching Traces

Cloud Trace v2 API filter syntax:
```
# All root spans for a feature
+genkit/isRoot:true +genkit/feature:flowTouristActivities

# Failed traces only
+genkit/isRoot:true +genkit/state:error

# Time-bounded (built into API params, not filter)
```

### 7.4 Building the Span Tree

Spans returned by the Trace API include `parentSpanId`. To build the tree:
1. Index all spans by `spanId`
2. Find root span (no `parentSpanId` or `parentSpanId` not in the set)
3. Recursively build children lists
4. Sort children by `startTime`

---

## 8. Error Handling

| Scenario | Behavior |
|----------|----------|
| ADC not configured | Show setup instructions with `gcloud auth` command |
| Invalid project ID | Show error banner, allow changing project |
| API quota exceeded | Show warning, serve from cache if available |
| No data in time range | Show "No data" empty state with suggestion to adjust range |
| Network error | Show error toast, retry button |
| Partial data (some APIs fail) | Show available data with warning indicators |

---

## 9. Future Enhancements (post-v1)

- **Cloud Logging integration**: Fetch actual (non-redacted) input/output from
  structured logs and correlate with trace spans
- **Live / auto-refresh mode**: Periodic refresh with configurable interval
- **Comparison view**: Compare metrics across time ranges
- **Export**: Export traces/metrics as JSON or CSV
- **Flame chart**: Alternative visualization for trace timings
- **Model cost estimation**: Calculate estimated costs based on token usage
- **Session view**: Group traces by `sessionId` / `threadName`
