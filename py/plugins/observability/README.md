# Genkit Observability Plugin

> **Community Plugin** — This plugin is community-maintained and is not an
> official Google product. It is provided on an "as-is" basis.
>
> **Preview** — This plugin is in preview and may have API changes in future releases.

This plugin provides a unified way to export Genkit telemetry to any
OTLP-compatible backend with simple presets for popular services.

## Features

- **Backend Presets**: One-line setup for Sentry, Honeycomb, Datadog, Grafana, Axiom
- **Custom OTLP**: Connect to any OTLP-compatible endpoint
- **PII Redaction**: Optional redaction of model inputs/outputs for privacy
- **Platform Agnostic**: Works on any cloud (AWS, GCP, Azure) or on-prem

## When to Use This Plugin

```
"I'm on AWS but want Honeycomb"       → Use this plugin
"I'm on GCP but want Sentry"          → Use this plugin
"I'm multi-cloud, want Datadog"       → Use this plugin
"I don't care, just give me traces"   → Use this plugin
```

If you want **native platform integration** (AWS X-Ray, GCP Cloud Trace, Azure
App Insights), use the platform-specific plugins instead (aws, google-cloud, azure).

## Installation

```bash
pip install genkit-plugin-observability

# For Sentry integration
pip install genkit-plugin-observability[sentry]
```

## Quick Start

```python
from genkit.plugins.observability import configure_telemetry

# Sentry
configure_telemetry(backend="sentry", sentry_dsn="https://...")

# Honeycomb
configure_telemetry(backend="honeycomb", honeycomb_api_key="...")

# Datadog
configure_telemetry(backend="datadog", datadog_api_key="...")

# Grafana Cloud
configure_telemetry(backend="grafana", grafana_endpoint="...", grafana_api_key="...")

# Custom OTLP endpoint
configure_telemetry(
    backend="custom",
    endpoint="https://my-collector/v1/traces",
    headers={"Authorization": "Bearer ..."},
)
```

## Supported Backends

| Backend | Endpoint | Auth | Features |
|---------|----------|------|----------|
| **Sentry** | `https://{org}.ingest.sentry.io/...` | DSN | Error tracking, performance |
| **Honeycomb** | `https://api.honeycomb.io/v1/traces` | API Key | Query-based debugging |
| **Datadog** | `https://trace.agent.datadoghq.com/...` | API Key | Full-stack APM |
| **Grafana Cloud** | `https://{stack}.grafana.net/otlp` | API Key | Tempo traces |
| **Axiom** | `https://api.axiom.co/v1/traces` | API Token | Log + trace ingestion |
| **Custom** | Any OTLP endpoint | Headers | Bring your own |

## Environment Variables

| Variable | Backend | Description |
|----------|---------|-------------|
| `SENTRY_DSN` | Sentry | Your Sentry DSN |
| `HONEYCOMB_API_KEY` | Honeycomb | Honeycomb API key |
| `DD_API_KEY` | Datadog | Datadog API key |
| `GRAFANA_OTLP_ENDPOINT` | Grafana | Grafana Cloud OTLP endpoint |
| `GRAFANA_API_KEY` | Grafana | Grafana Cloud API key |
| `AXIOM_TOKEN` | Axiom | Axiom API token |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `backend` | str | required | Which backend to use |
| `service_name` | str | "genkit-app" | Name of your service |
| `service_version` | str | "1.0.0" | Version of your service |
| `environment` | str | None | Environment name (production, staging) |
| `log_input_and_output` | bool | False | Disable PII redaction |
| `force_dev_export` | bool | True | Export in dev environment |

## Trademarks

All trademarks (Sentry, Honeycomb, Datadog, Grafana, Axiom) are property of
their respective owners. This is a community plugin.
