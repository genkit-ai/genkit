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

ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')
```

Requires Google Cloud Application Default Credentials (ADC) or explicit credentials.
