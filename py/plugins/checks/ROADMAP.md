# Checks Plugin — JS/Python Parity Roadmap

> Cross-reference between the JS canonical implementation
> (`js/plugins/checks/src/`) and the Python port
> (`py/plugins/checks/src/genkit/plugins/checks/`).

## Module-by-Module Parity

### 1. `metrics.ts` ↔ `metrics.py` — ✅ DONE

| Feature | JS | Python | Status |
|---------|:--:|:------:|:------:|
| `ChecksEvaluationMetricType` enum | ✅ | ✅ | ✅ Parity |
| 8 metric types (DANGEROUS_CONTENT, PII_SOLICITING_RECITING, HARASSMENT, SEXUALLY_EXPLICIT, HATE_SPEECH, MEDICAL_INFO, VIOLENCE_AND_GORE, OBSCENITY_AND_PROFANITY) | ✅ | ✅ | ✅ Parity |
| `ChecksEvaluationMetricConfig` (type + threshold) | ✅ | ✅ | ✅ Parity |
| `ChecksEvaluationMetric` union type | ✅ | ✅ | ✅ Parity |
| `isConfig()` type guard helper | ✅ | N/A | ✅ Not needed — Python uses `isinstance()` |

### 2. `guardrails.ts` ↔ `guardrails.py` — ✅ DONE

| Feature | JS | Python | Status |
|---------|:--:|:------:|:------:|
| `classifyContent` API call | ✅ | ✅ | ✅ Parity |
| API URL (`v1alpha/aisafety:classifyContent`) | ✅ | ✅ | ✅ Parity |
| Auth via Google ADC | ✅ `GoogleAuth` | ✅ `google.auth.default` | ✅ Parity |
| OAuth scopes (cloud-platform + checks) | ✅ | ✅ | ✅ Parity |
| `GCLOUD_SERVICE_ACCOUNT_CREDS` env var | ✅ | ✅ | ✅ Parity |
| Custom credentials parameter | ✅ `GoogleAuthOptions` | ✅ `credentials: Credentials` | ✅ Parity |
| Quota project warning | ✅ | ✅ | ✅ Parity |
| `x-goog-user-project` header | ✅ | ✅ | ✅ Parity |
| Request body shape (`input.text_input.content`, `policies[].policy_type`) | ✅ | ✅ | ✅ Parity |
| Response parsing (`policyResults[].policyType`, `score`, `violationResult`) | ✅ Zod | ✅ Pydantic | ✅ Parity |
| Threshold in policy request | ✅ | ✅ | ✅ Parity |
| Per-request token refresh | ✅ | ✅ `asyncio.to_thread` | ✅ Parity |
| Error handling with status codes | ✅ (implicit via GoogleAuth) | ✅ (explicit status code + message parsing) | ✅ Python is more detailed |

### 3. `middleware.ts` ↔ `middleware.py` — ✅ DONE

| Feature | JS | Python | Status |
|---------|:--:|:------:|:------:|
| Input guard — classify each message's text parts | ✅ | ✅ | ✅ Parity |
| Output guard — classify each candidate's text parts | ✅ | ✅ | ✅ Parity |
| Blocked response with `finishReason: 'blocked'` | ✅ | ✅ `FinishReason.BLOCKED` | ✅ Parity |
| `finishMessage` with policy names | ✅ | ✅ (`finish_message`) | ✅ Parity |
| Input violation message format | `"Model input violated..."` | `"Model input violated..."` | ✅ Parity |
| Output violation message format | `"Model output violated..."` | `"Model output violated..."` | ✅ Parity |
| Factory function signature | `checksMiddleware({auth, metrics, projectId})` | `checks_middleware(project_id, metrics, credentials)` | ✅ Idiomatic |
| Custom credentials support | ✅ | ✅ | ✅ Parity |
| Top-level `message` fallback (when `candidates` is absent) | ❌ | ✅ | ✅ Python is more robust |

### 4. `evaluation.ts` ↔ `evaluation.py` — ✅ DONE

| Feature | JS | Python | Status |
|---------|:--:|:------:|:------:|
| Evaluator registration | ✅ `ai.defineEvaluator` | ✅ `registry.define_evaluator` | ✅ Parity |
| Evaluator name: `checks/guardrails` (single) | ✅ | ✅ | ✅ Parity |
| Single API call for all policies | ✅ | ✅ | ✅ Parity |
| Returns `evaluation: [Score]` (list of per-policy results) | ✅ | ✅ | ✅ Parity |
| Per-policy result shape `{id, score, details: {reasoning}}` | ✅ | ✅ | ✅ Parity |
| `testCaseId` in response | ✅ | ✅ | ✅ Parity |
| Span tracing (`runInNewSpan`) | ✅ explicit | ✅ framework handles it | ✅ Parity |
| Null output handling | ❌ (not guarded) | ✅ (returns error) | ✅ Python is more robust |

### 5. `index.ts` ↔ `plugin.py` — ✅ DONE

| Feature | JS | Python | Status |
|---------|:--:|:------:|:------:|
| Plugin registration (`genkitPlugin`) | ✅ | ✅ `Plugin` subclass | ✅ Parity |
| Plugin name (`checks`) | ✅ | ✅ | ✅ Parity |
| `projectId` from config or env | ✅ `googleAuth.getProjectId()` | ✅ `os.environ.get('GCLOUD_PROJECT')` | ✅ Parity |
| Error on missing `projectId` | ✅ | ✅ | ✅ Parity |
| `evaluation.metrics` config | ✅ | ✅ `ChecksEvaluationConfig` | ✅ Parity |
| Custom credentials (`GoogleAuthOptions`) | ✅ | ✅ `credentials: Credentials` | ✅ Parity |
| `GCLOUD_SERVICE_ACCOUNT_CREDS` env var | ✅ | ✅ | ✅ Parity |
| Quota project warning | ✅ | ✅ | ✅ Parity |
| Standalone middleware export | ✅ | ✅ (`checks_middleware()`) | ✅ Parity |
| `define_checks_evaluators()` standalone function | N/A | ✅ | ✅ Pythonic API |

## Sample App (`provider-checks-hello`)

| Feature | Status |
|---------|:------:|
| Env var setup (`GCLOUD_PROJECT`, `GEMINI_API_KEY`) | ✅ |
| ADC scope setup (auto-prompt with `--scopes`) | ✅ |
| Checks API enablement check | ✅ |
| Middleware usage in `ai.generate(use=[...])` | ✅ |
| README with setup instructions | ✅ |

## Authentication Summary

| Method | JS | Python | Notes |
|--------|:--:|:------:|-------|
| Application Default Credentials | ✅ | ✅ | Primary method |
| Service account JSON from env | ✅ | ✅ | `GCLOUD_SERVICE_ACCOUNT_CREDS` |
| Custom credentials | ✅ | ✅ | `GoogleAuthOptions` / `Credentials` |
| OAuth scopes | ✅ | ✅ | `cloud-platform` + `checks` |
| `x-goog-user-project` header | ✅ | ✅ | For billing |
| Quota project mismatch warning | ✅ | ✅ | Logs warning |

## Remaining Items

All P0–P2 parity gaps have been resolved. The Python implementation is at
**full parity** with the JS canonical implementation.

Two optional API fields (`context` and `classifierVersion`) exist in the
Checks API discovery doc but are not used by the JS plugin and therefore
not implemented here either. They can be added as optional parameters to
`classify_content()` if a use case arises.

## API Reference

- **Endpoint**: `POST https://checks.googleapis.com/v1alpha/aisafety:classifyContent`
- **Auth**: OAuth 2.0 with scopes `cloud-platform` and `checks`
- **Docs**: https://developers.google.com/checks
- **API quickstart**: https://developers.google.com/checks/guide/api/quickstart
- **Authorization**: https://developers.google.com/checks/guide/api/auth
