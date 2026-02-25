# checks-safety-hello

Demonstrates the Google Checks AI Safety plugin for Genkit. This sample
shows how to use both **evaluators** (for scoring content safety) and
**middleware** (for blocking unsafe input/output in real-time).

## Prerequisites

1. A Google Cloud project with the **Checks API** enabled:
   ```bash
   gcloud services enable checks.googleapis.com --project=your-gcp-project-id
   ```
2. **Quota** for the Checks AI Safety ClassifyContent API. The Checks API
   is a preview/restricted API — your project must have a non-zero daily
   quota for `AiSafety.ClassifyContent requests`. If you see a `429` error
   with `quota_limit_value: '0'`, you need to
   [request a quota increase](https://cloud.google.com/docs/quotas/help/request_increase)
   for the `checks.googleapis.com` service on your project.
3. Google Cloud Application Default Credentials **with the Checks scope**:
   ```bash
   gcloud auth application-default login \
     --scopes="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/checks"
   ```
   > **Note:** Standard `gcloud auth application-default login` (without
   > `--scopes`) does **not** include the `checks` scope and will result
   > in a `403 ACCESS_TOKEN_SCOPE_INSUFFICIENT` error. The `run.sh` script
   > handles this automatically.
4. A Google GenAI API key (for the Gemini model)

## Setup

```bash
export GCLOUD_PROJECT=your-gcp-project-id
export GEMINI_API_KEY=your-api-key
```

## Running

```bash
./run.sh
```

The script will:
1. Prompt for any required environment variables not yet set
2. Check that the Checks API is enabled on your project
3. Authenticate with the required OAuth scopes (opens a browser)
4. Start the app and open the Dev UI

## What this sample demonstrates

### 1. Safety Evaluators

Registers Checks evaluators that can be used in the Genkit Dev UI to
evaluate model outputs against safety policies like `DANGEROUS_CONTENT`,
`HARASSMENT`, and `HATE_SPEECH`.

### 2. Safety Middleware

Wraps a Gemini model call with `checks_middleware` so that both input
and output are automatically classified. If either violates a policy,
the middleware returns a `blocked` response instead of the model output.

## Testing

1. Start the sample with `./run.sh`
2. Open the Genkit Dev UI (printed in the terminal)
3. Test the `safe_generate` flow with safe prompts — you should get normal responses
4. Test with unsafe prompts — the middleware should block and return a `blocked` finish reason
5. Check the evaluators tab to see registered Checks evaluators

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `403 ACCESS_TOKEN_SCOPE_INSUFFICIENT` | ADC token missing the `checks` scope | Re-run `gcloud auth application-default login --scopes=...` (see Prerequisites) |
| `429 RATE_LIMIT_EXCEEDED` with `quota_limit_value: '0'` | Project has zero quota for the Checks AI Safety API | [Request a quota increase](https://cloud.google.com/docs/quotas/help/request_increase) for `checks.googleapis.com` |
| `403 PERMISSION_DENIED` (not scope-related) | Checks API not enabled on the project | Run `gcloud services enable checks.googleapis.com --project=your-gcp-project-id` |

## See Also

- [Google Checks AI Safety](https://checks.google.com/ai-safety)
- [Genkit Documentation](https://genkit.dev/)
