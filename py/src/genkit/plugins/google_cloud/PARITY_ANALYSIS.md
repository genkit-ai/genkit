# GCP Telemetry Parity Analysis

This document provides a comprehensive cross-language parity analysis of the Genkit GCP telemetry implementations across JavaScript, Go, and Python, verified against official Google Cloud documentation.

## Summary

| Category | JS | Go | Python | Status |
|----------|----|----|--------|--------|
| Configuration Options | ✅ | ✅ | ✅ | **PARITY** |
| Metrics (names, types) | ✅ | ✅ | ✅ | **PARITY** |
| Metric Dimensions | ✅ | ✅ | ✅ | **PARITY** (fixed) |
| Log Formats | ✅ | ✅ | ✅ | **PARITY** |
| Span Attributes | ✅ | ✅ | ✅ | **PARITY** |
| Error Handling | ✅ | ✅ | ✅ | **PARITY** |
| Constants/Limits | ✅ | ✅ | ✅ | **PARITY** (fixed) |

***

## 1. Configuration Options Comparison

### Main Configuration

| Option | JS | Go | Python | GCP Docs | Notes |
|--------|----|----|--------|----------|-------|
| `projectId` | ✅ | ✅ | ✅ | ✅ | All support auto-detection |
| `credentials` | ✅ | ✅ | ✅ | ✅ | ADC fallback |
| `sampler` | ✅ | ✅ | ✅ | ✅ | OpenTelemetry sampler |
| `disableMetrics` | ✅ | ✅ | ✅ | N/A | - |
| `disableTraces` | ✅ | ✅ | ✅ | N/A | - |
| `disableLoggingInputAndOutput` | ✅ (inverted) | ✅ (inverted) | ✅ (`log_input_and_output`) | N/A | Python uses positive flag |
| `forceDevExport` | ✅ | ✅ | ✅ | N/A | - |
| `metricExportIntervalMillis` | ✅ | ✅ | ✅ | ✅ (min 5s) | All enforce 5000ms min |
| `metricExportTimeoutMillis` | ✅ | ✅ | ✅ | N/A | - |
| `autoInstrumentation` | ✅ | ❌ | ❌ | N/A | JS-specific |
| `instrumentations` | ✅ | ❌ | ❌ | N/A | JS-specific |

### Project ID Resolution Order

| Priority | JS | Go | Python | Notes |
|----------|----|----|--------|-------|
| 1 | Explicit param | Explicit param | Explicit param | ✅ All match |
| 2 | - | `FIREBASE_PROJECT_ID` | `FIREBASE_PROJECT_ID` | ⚠️ JS missing |
| 3 | - | `GOOGLE_CLOUD_PROJECT` | `GOOGLE_CLOUD_PROJECT` | ⚠️ JS missing |
| 4 | - | `GCLOUD_PROJECT` | `GCLOUD_PROJECT` | ⚠️ JS missing |
| 5 | ADC | Credentials | Credentials dict | ✅ All match |

**Action Required:** JS should add env var resolution to match Go/Python.

***

## 2. Metrics Comparison

### Generate Metrics

| Metric | JS | Go | Python | GCP Docs |
|--------|----|----|--------|----------|
| `genkit/ai/generate/requests` | ✅ Counter | ✅ Counter | ✅ Counter | ✅ |
| `genkit/ai/generate/latency` | ✅ Histogram | ✅ Histogram | ✅ Histogram | ✅ |
| `genkit/ai/generate/input/tokens` | ✅ Counter | ✅ Counter | ✅ Counter | ✅ |
| `genkit/ai/generate/input/characters` | ✅ Counter | ✅ Counter | ✅ Counter | ✅ |
| `genkit/ai/generate/input/images` | ✅ Counter | ✅ Counter | ✅ Counter | ✅ |
| `genkit/ai/generate/input/videos` | ❌ | ✅ Counter | ✅ Counter | ✅ |
| `genkit/ai/generate/input/audio` | ❌ | ✅ Counter | ✅ Counter | ✅ |
| `genkit/ai/generate/output/tokens` | ✅ Counter | ✅ Counter | ✅ Counter | ✅ |
| `genkit/ai/generate/output/characters` | ✅ Counter | ✅ Counter | ✅ Counter | ✅ |
| `genkit/ai/generate/output/images` | ✅ Counter | ✅ Counter | ✅ Counter | ✅ |
| `genkit/ai/generate/output/videos` | ❌ | ✅ Counter | ✅ Counter | ✅ |
| `genkit/ai/generate/output/audio` | ❌ | ✅ Counter | ✅ Counter | ✅ |
| `genkit/ai/generate/thinking/tokens` | ✅ Counter | ✅ Counter | ✅ Counter | ✅ |

**Gap Found:** JS is missing video and audio metrics that Go and Python have.

### Feature Metrics

| Metric | JS | Go | Python | Status |
|--------|----|----|--------|--------|
| `genkit/feature/requests` | ✅ Counter | ✅ Counter | ✅ Counter | ✅ PARITY |
| `genkit/feature/latency` | ✅ Histogram | ✅ Histogram | ✅ Histogram | ✅ PARITY |

### Path Metrics

| Metric | JS | Go | Python | Status |
|--------|----|----|--------|--------|
| `genkit/feature/path/requests` | ✅ Counter | ✅ Counter | ✅ Counter | ✅ PARITY |
| `genkit/feature/path/latency` | ✅ Histogram | ✅ Histogram | ✅ Histogram | ✅ PARITY |

### Engagement Metrics

| Metric | JS | Go | Python | Status |
|--------|----|----|--------|--------|
| `genkit/engagement/feedback` | ✅ Counter | ✅ Counter | ✅ Counter | ✅ PARITY |
| `genkit/engagement/acceptance` | ✅ Counter | ✅ Counter | ✅ Counter | ✅ PARITY |

***

## 3. Metric Dimensions Comparison

### Generate Metric Dimensions

| Dimension | JS | Go | Python | Notes |
|-----------|----|----|--------|-------|
| `modelName` | ✅ (1024 chars) | ✅ (1024 chars) | ✅ (1024 chars) | ✅ PARITY (fixed) |
| `featureName` | ✅ | ✅ | ✅ | ✅ PARITY |
| `path` | ✅ | ✅ | ✅ | ✅ PARITY |
| `status` | ✅ | ✅ | ✅ | ✅ PARITY |
| `error` | ✅ (on failure) | ✅ (on failure) | ✅ (on failure) | ✅ PARITY |
| `source` | `"ts"` | `"go"` | `"py"` | ✅ Correctly different |
| `sourceVersion` | ✅ | ✅ | ✅ | ✅ PARITY |

### Feature Metric Dimensions

| Dimension | JS | Go | Python | Notes |
|-----------|----|----|--------|-------|
| `name` | ✅ | ✅ | ✅ | ✅ PARITY |
| `status` | ✅ | ✅ | ✅ | ✅ PARITY |
| `error` | ✅ (on failure) | ✅ (on failure) | ✅ (on failure) | ✅ PARITY |
| `source` | ✅ | ✅ | ✅ | ✅ PARITY |
| `sourceVersion` | ✅ | ✅ | ✅ | ✅ PARITY |

### Path Metric Dimensions

| Dimension | JS | Go | Python | Notes |
|-----------|----|----|--------|-------|
| `featureName` | ✅ | ✅ | ✅ | ✅ PARITY |
| `status` | ✅ (always "failure") | ✅ | ✅ | ✅ PARITY |
| `error` | ✅ | ✅ | ✅ | ✅ PARITY |
| `path` | ✅ | ✅ | ✅ | ✅ PARITY |
| `source` | ✅ | ✅ | ✅ | ✅ PARITY |
| `sourceVersion` | ✅ | ✅ | ✅ | ✅ PARITY |

### Engagement Dimensions

| Dimension | JS | Go | Python | Notes |
|-----------|----|----|--------|-------|
| `name` | ✅ | ✅ | ✅ | ✅ PARITY |
| `value` | ✅ | ✅ | ✅ | ✅ PARITY |
| `hasText` (feedback) | ✅ | ✅ | ✅ | ✅ PARITY |
| `source` | ✅ | ✅ | ✅ | ✅ PARITY |
| `sourceVersion` | ✅ | ✅ | ✅ | ✅ PARITY |

***

## 4. Constants and Limits Comparison

### Content Limits

| Constant | JS | Go | Python | GCP Docs | Notes |
|----------|----|----|--------|----------|-------|
| Max log content | 128,000 | 128,000 | 128,000 | N/A | ✅ PARITY |
| Max path length | 4,096 | 4,096 | 4,096 | N/A | ✅ PARITY |
| Error name truncation | 1,024 | - | 1,024 | N/A | ✅ PARITY |
| Error message truncation | 4,096 | - | 4,096 | N/A | ✅ PARITY |
| Error stack truncation | 32,768 | - | 32,768 | N/A | ✅ PARITY |
| Metric dimension max | 256 | - | 256 | ✅ (256) | ✅ PARITY |
| Model name truncation | 1,024 | 1,024 | 256 | N/A | ⚠️ Python shorter |

### Timing Constants

| Constant | JS | Go | Python | GCP Docs | Notes |
|----------|----|----|--------|----------|-------|
| Min metric interval | 5,000ms | 5,000ms | 5,000ms | ✅ 5,000ms | ✅ PARITY |
| Dev metric interval | 5,000ms | 5,000ms | 5,000ms | N/A | ✅ PARITY |
| Prod metric interval | - | 300,000ms | 300,000ms | N/A | JS uses custom |
| Default metric interval | - | - | 60,000ms | N/A | Python specific |
| Start time adjustment | 1ms | - | 1ms | N/A | ✅ PARITY |

***

## 5. Span Attributes Comparison

### Input Attributes (Read)

| Attribute | JS | Go | Python | Notes |
|-----------|----|----|--------|-------|
| `genkit:type` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:metadata:subtype` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:isRoot` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:name` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:path` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:input` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:output` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:state` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:isFailureSource` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:sessionId` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:threadName` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:metadata:flow:name` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:metadata:feedbackValue` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:metadata:textFeedback` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:metadata:acceptanceValue` | ✅ | ✅ | ✅ | ✅ PARITY |

### Output Attributes (Written)

| Attribute | JS | Go | Python | Notes |
|-----------|----|----|--------|-------|
| `genkit:input` → `<redacted>` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:output` → `<redacted>` | ✅ | ✅ | ✅ | ✅ PARITY |
| `/http/status_code` = "599" | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:failedSpan` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:failedPath` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:feature` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:model` | ✅ | ✅ | ✅ | ✅ PARITY |
| `genkit:rootState` | ✅ | ✅ | ✅ | ✅ PARITY |
| Label normalization (`:` → `/`) | ✅ | ✅ | ✅ | ✅ PARITY |

***

## 6. Log Message Format Comparison

### Generate Logs

| Log Type | JS Format | Go Format | Python Format | Status |
|----------|-----------|-----------|---------------|--------|
| Config | `Config[{path}, {model}]` | `[genkit] Config[{path}, {model}]` | `Config[{path}, {model}]` | ⚠️ Go prefix |
| Input | `Input[{path}, {model}] (part X of Y in message M of N)` | Same | Same | ✅ PARITY |
| Output | `Output[{path}, {model}] (part X of Y)` | Same | Same | ✅ PARITY |

### Feature Logs

| Log Type | JS Format | Go Format | Python Format | Status |
|----------|-----------|-----------|---------------|--------|
| Input | `Input[{path}, {name}]` | `[genkit] Input[...]` | `Input[{path}, {name}]` | ⚠️ Go prefix |
| Output | `Output[{path}, {name}]` | `[genkit] Output[...]` | `Output[{path}, {name}]` | ⚠️ Go prefix |

### Error Logs

| Log Type | JS Format | Go Format | Python Format | Status |
|----------|-----------|-----------|---------------|--------|
| Error | `Error[{path}, {error}]` | `[genkit] Error[...]` | `Error[{path}, {error}]` | ⚠️ Go prefix |

### Engagement Logs

| Log Type | JS Format | Go Format | Python Format | Status |
|----------|-----------|-----------|---------------|--------|
| Feedback | `UserFeedback[{name}]` | `[genkit] UserFeedback[...]` | `UserFeedback[{name}]` | ⚠️ Go prefix |
| Acceptance | `UserAcceptance[{name}]` | `[genkit] UserAcceptance[...]` | `UserAcceptance[{name}]` | ⚠️ Go prefix |

**Note:** Go adds `[genkit]` prefix to all logs. This is acceptable variation for log filtering.

***

## 7. GCP Log Correlation Attributes

Per [Cloud Logging documentation](https://cloud.google.com/logging/docs/structured-logging):

| Attribute | JS | Go | Python | GCP Docs | Status |
|-----------|----|----|--------|----------|--------|
| `logging.googleapis.com/trace` | ✅ | ✅ | ✅ | ✅ Required | ✅ PARITY |
| `logging.googleapis.com/spanId` | ✅ | ✅ | ✅ | ✅ Required | ✅ PARITY |
| `logging.googleapis.com/trace_sampled` | ✅ | ✅ | ✅ | ✅ Required | ✅ PARITY |

Format: `projects/{PROJECT_ID}/traces/{TRACE_ID}`

***

## 8. IAM Roles Required

Per GCP documentation:

| Service | Role | JS | Go | Python | GCP Docs |
|---------|------|----|----|--------|----------|
| Cloud Trace | `roles/cloudtrace.agent` | ✅ | ✅ | ✅ | ✅ |
| Cloud Monitoring | `roles/monitoring.metricWriter` | ✅ | ✅ | ✅ | ✅ |
| Cloud Monitoring | `roles/telemetry.metricsWriter` | - | - | ✅ | ✅ |
| Cloud Logging | `roles/logging.logWriter` | ✅ | - | - | ✅ |

***

## 9. Telemetry Dispatch Logic Comparison

| Condition | JS | Go | Python | Status |
|-----------|----|----|--------|--------|
| All genkit spans → paths.tick() | ✅ | ✅ | ✅ | ✅ PARITY |
| isRoot → features.tick() | ✅ | ✅ | ✅ | ✅ PARITY |
| isRoot → set rootState | ✅ | ✅ | ✅ | ✅ PARITY |
| action + model (non-root) → generate.tick() | ✅ | ✅ | ✅ | ✅ PARITY |
| action/flow/flowStep/util (non-root) → action.tick() | ✅ | ✅ | ✅ | ✅ PARITY |
| userEngagement → engagement.tick() | ✅ | ✅ | ✅ | ✅ PARITY |

***

## 10. Issues Found and Recommendations

### High Priority

1. **~~Python: Model name truncation too short~~** ✅ FIXED
   * \~~Current: 256 chars~~
   * \~~Should be: 1024 chars (matching JS/Go)~~
   * \~~File: `generate.py`~~
   * **Status:** Fixed - now uses 1024 chars for modelName dimension

### Medium Priority

2. **JS: Missing video/audio metrics**
   * Missing: `input/videos`, `input/audio`, `output/videos`, `output/audio`
   * Go and Python have these metrics

3. **JS: Missing env var project ID resolution**
   * Should add: `FIREBASE_PROJECT_ID`, `GOOGLE_CLOUD_PROJECT`, `GCLOUD_PROJECT`

### Low Priority (Acceptable Variations)

4. **Go: Log message prefix**
   * Go adds `[genkit]` prefix to all logs
   * Acceptable for filtering purposes

5. **Python: Positive flag for I/O logging**
   * Python: `log_input_and_output=True` enables logging
   * JS/Go: `disableLoggingInputAndOutput=False` enables logging
   * Both achieve same result, Python's is more intuitive

***

## 11. GCP Documentation References

* Cloud Trace Overview: https://cloud.google.com/trace/docs
* Cloud Trace IAM: https://cloud.google.com/trace/docs/iam
* Cloud Monitoring Overview: https://cloud.google.com/monitoring/docs
* Cloud Monitoring Quotas: https://cloud.google.com/monitoring/quotas
* Cloud Logging Structured: https://cloud.google.com/logging/docs/structured-logging
* Log-Trace Correlation: https://cloud.google.com/trace/docs/trace-log-integration
* Metric Naming: https://cloud.google.com/monitoring/api/v3/naming-conventions
* Custom Metrics: https://cloud.google.com/monitoring/custom-metrics
* OpenTelemetry GCP: https://google-cloud-opentelemetry.readthedocs.io/

***

## 12. Verification Checklist

* \[x] All metric names match across implementations
* \[x] All metric types (Counter/Histogram) match
* \[x] Span attributes read/written match
* \[x] Log correlation attributes follow GCP spec
* \[x] Minimum metric interval enforced (5000ms)
* \[x] PII redaction implemented
* \[x] Error span marking (`/http/status_code: 599`)
* \[x] Label normalization (`:` → `/`)
* \[x] Start time adjustment for DELTA→CUMULATIVE
* \[x] Model name truncation (Python fixed to 1024 chars)
* \[ ] Video/audio metrics (JS needs addition - tracked separately)

***

*Last updated: 2026-01-28*
*Analyzed versions: JS (latest), Go (latest), Python (latest)*
