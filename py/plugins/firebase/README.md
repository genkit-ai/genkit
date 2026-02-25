# Genkit Firebase plugin

This Genkit plugin provides a set of tools and utilities for working with
Firebase.

## Telemetry

The Firebase plugin provides easy integration with Google Cloud Observability (Cloud Trace and Cloud Monitoring).

To enable telemetry:

```python
from genkit.plugins.firebase import add_firebase_telemetry

# Enable telemetry (defaults to production-only export)
add_firebase_telemetry()
```

### Configuration

`add_firebase_telemetry` supports the following options:

- `project_id`: Firebase project ID (optional, auto-detected).
- `force_dev_export`: Set to `True` to export telemetry in dev environment (defaults to `False`).
- `log_input_and_output`: Set to `True` to log model inputs and outputs (defaults to `False` / redacted).
- `disable_metrics`: Set to `True` to disable metrics export.
- `disable_traces`: Set to `True` to disable trace export.
