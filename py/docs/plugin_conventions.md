# Python Plugin Conventions

Naming rules for plugin constructors and setup functions. The goal: a developer who already uses Google's Python libraries can pass the same arguments to a Genkit plugin without looking anything up.

## Google Cloud project: `project=`

Plugins and setup functions that take a Google Cloud project name the parameter `project`, never `project_id`. When the plugin also takes a region, it's `location`.

```python
# 1. Model plugins
ai = Genkit(plugins=[
    VertexAI(project='acme-prod', location='us-central1'),
    ModelGarden(project='acme-prod', location='us-east5'),
])

# 2. Telemetry
enable_google_cloud_telemetry(project='acme-prod')
```

### Why `project`

It's what Google's own Python client libraries use. Checked against current releases:

- `google.genai.Client(vertexai=True, project=..., location=...)`
- `vertexai.init(project=..., location=...)` and `google.cloud.aiplatform.init(project=..., location=...)`
- `google.cloud.bigquery.Client(project=..., location=...)`
- `google.cloud.storage.Client(project=...)`
- `google.cloud.firestore.Client(project=...)`
- `google.cloud.logging.Client(project=...)`
- Resource path helpers such as `PublisherClient.topic_path(project, topic)` and `SecretManagerServiceClient.secret_path(project, secret)`

We treat that as idiomatic Python on Google Cloud. A developer moving from `genai.Client(project=..., location=...)` to `VertexAI(project=..., location=...)` changes the class name and nothing else.

### Where `project_id` still appears

Some libraries we call into use `project_id`. Plugins translate at that boundary and don't pass the name through to their own signature:

- **OpenTelemetry GCP exporters**: `CloudTraceSpanExporter(project_id=...)`, `CloudMonitoringMetricsExporter(project_id=...)`, and the Cloud Logging exporter. `enable_google_cloud_telemetry(project=...)` forwards the value as their `project_id`.
- **Anthropic on Vertex**: `AsyncAnthropicVertex(project_id=..., region=...)`. Model Garden forwards `project` as `project_id`.
- **google-auth**: `google.auth.default()` returns `(credentials, project_id)`, service-account credentials expose `.project_id`, and service-account JSON has a `"project_id"` key. Those are data, not our parameters.

### Other SDKs

Go uses `ProjectID` (`googlegenai.VertexAI{ProjectID: ...}`, `modelgarden`, `googlecloud`), which follows Go's own cloud clients (`bigquery.NewClient(ctx, projectID)`) and its rule of capitalizing initialisms. Each SDK follows its own language's Google Cloud libraries, so the Python and Go names differ on purpose.
