# Genkit Google Cloud Plugin

Export Genkit telemetry to Google Cloud Trace, Cloud Monitoring, and Cloud Logging.

## Installation

```bash
uv add genkit-google-cloud genkit-google-genai
```

## Usage

```python
from genkit import Genkit
from genkit_google_cloud import enable_google_cloud_telemetry
from genkit_google_genai import GoogleAI

enable_google_cloud_telemetry(project_id='my-project')

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))
```

Requires Google Cloud Application Default Credentials (ADC) or explicit credentials.


## Sessions on Firestore

`FirestoreSessionStore` persists agent sessions durably: every turn is
stored as a snapshot, sessions survive restarts and resume across
processes, and status subscriptions enable features like a cross-process
stop button.

```python
from genkit_google_cloud import FirestoreSessionStore

store = FirestoreSessionStore()  # uses Application Default Credentials
# pass `store` where your agent setup takes a session store
```

Notes:

- Data lives under three collection roots derived from `collection`
  (default `genkit-sessions`, plus `-shards` and `-pointers`). Delete all
  three together when cleaning up.
- Multi-tenant apps pass `snapshot_path_prefix` to derive a per-tenant
  path segment from call context.
- After a regenerate/branch, resume a specific branch by `snapshot_id`;
  `session_id` lookup follows the most recently written branch.
- See the class docstring for costs, limits, and lifecycle details.
