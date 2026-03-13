# Third-Party Observability Hello Sample

This sample demonstrates how to export Genkit telemetry to third-party
observability platforms like Sentry, Honeycomb, Datadog, Grafana Cloud, and Axiom.

## Supported Backends

| Backend | Endpoint | Auth |
|---------|----------|------|
| **Sentry** | `SENTRY_DSN` | DSN |
| **Honeycomb** | `HONEYCOMB_API_KEY` | API Key |
| **Datadog** | `DD_API_KEY` | API Key |
| **Grafana Cloud** | `GRAFANA_OTLP_ENDPOINT` + `GRAFANA_API_KEY` | API Key |
| **Axiom** | `AXIOM_TOKEN` | API Token |
| **Custom** | Any OTLP endpoint | Headers |

## Quick Start

### Honeycomb Example

```bash
export HONEYCOMB_API_KEY="your-honeycomb-key"
export GEMINI_API_KEY="your-google-ai-key"
./run.sh
```

### Sentry Example

```bash
export SENTRY_DSN="https://key@org.ingest.sentry.io/project"
export GEMINI_API_KEY="your-google-ai-key"

# Edit src/main.py to use backend="sentry"
./run.sh
```

### Datadog Example

```bash
export DD_API_KEY="your-datadog-key"
export GEMINI_API_KEY="your-google-ai-key"

# Edit src/main.py to use backend="datadog"
./run.sh
```

## Running the Sample

```bash
./run.sh
```

## Testing with the DevUI

1. Open http://localhost:4000 in your browser
2. Navigate to the "say_hello" flow
3. Enter a name and run the flow
4. View traces in your chosen backend's dashboard

## What This Sample Demonstrates

- Multi-backend observability presets
- One-line setup for popular platforms
- PII redaction for model inputs/outputs
- Platform-agnostic telemetry export
