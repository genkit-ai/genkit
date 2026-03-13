# genkit-plugin-checks

> **Preview** — This plugin is in preview and may have API changes in future releases.

Google Checks AI Safety plugin for [Genkit](https://genkit.dev/).

This plugin integrates [Google Checks](https://checks.google.com/ai-safety)
AI Safety guardrails into Genkit, providing both **evaluators** and **model
middleware** for content safety classification.

## Features

- **Safety Evaluators**: Register evaluators that classify content against
  Checks safety policies (dangerous content, harassment, hate speech, etc.)
- **Model Middleware**: Intercept model input/output and block content that
  violates safety policies before it reaches the user.

## Installation

```bash
pip install genkit-plugin-checks
```

## Prerequisites

1. A Google Cloud project with the Checks API enabled
2. Google Cloud Application Default Credentials (ADC) configured
3. Set the `GCLOUD_PROJECT` environment variable or pass `project_id` explicitly

## Usage

### Evaluators

Register safety evaluators to evaluate model outputs:

```python
from genkit.ai import Genkit
from genkit.plugins.checks import (
    ChecksEvaluationMetricType,
    define_checks_evaluators,
)

ai = Genkit(...)

# Register Checks evaluators
define_checks_evaluators(
    ai,
    project_id='your-gcp-project-id',
    metrics=[
        ChecksEvaluationMetricType.DANGEROUS_CONTENT,
        ChecksEvaluationMetricType.HARASSMENT,
        ChecksEvaluationMetricType.HATE_SPEECH,
    ],
)
```

### Middleware

Use as model middleware to block unsafe input/output in real-time:

```python
from genkit.plugins.checks import (
    ChecksEvaluationMetricType,
    checks_middleware,
)

response = await ai.generate(
    model='googleai/gemini-1.5-flash-latest',
    prompt='Tell me a story',
    use=[
        checks_middleware(
            project_id='your-gcp-project-id',
            metrics=[
                ChecksEvaluationMetricType.DANGEROUS_CONTENT,
                ChecksEvaluationMetricType.HARASSMENT,
            ],
        ),
    ],
)

# If content was blocked:
from genkit.core.typing import FinishReason
if response.finish_reason == FinishReason.BLOCKED:
    print(f'Blocked: {response.finish_message}')
```

### With Metric Thresholds

Configure custom sensitivity thresholds per policy:

```python
from genkit.plugins.checks import (
    ChecksEvaluationMetricConfig,
    ChecksEvaluationMetricType,
    define_checks_evaluators,
)

define_checks_evaluators(
    ai,
    project_id='your-gcp-project-id',
    metrics=[
        # Use default threshold
        ChecksEvaluationMetricType.DANGEROUS_CONTENT,
        # Stricter threshold (lower = stricter)
        ChecksEvaluationMetricConfig(
            type=ChecksEvaluationMetricType.HATE_SPEECH,
            threshold=0.3,
        ),
    ],
)
```

## Supported Policies

| Policy | Description |
|--------|-------------|
| `DANGEROUS_CONTENT` | Harmful goods, services, and activities |
| `PII_SOLICITING_RECITING` | Personal information disclosure |
| `HARASSMENT` | Bullying or abusive content |
| `SEXUALLY_EXPLICIT` | Sexually explicit content |
| `HATE_SPEECH` | Violence, hatred, or discrimination |
| `MEDICAL_INFO` | Potentially harmful health advice |
| `VIOLENCE_AND_GORE` | Gratuitous violence descriptions |
| `OBSCENITY_AND_PROFANITY` | Vulgar or profane language |

## API Reference

- **`define_checks_evaluators(ai, project_id, metrics)`** — Register evaluators
- **`checks_middleware(project_id, metrics)`** — Create model middleware
- **`Checks(project_id, evaluation)`** — Plugin class (for `Genkit(plugins=[...])`)
- **`ChecksEvaluationMetricType`** — Enum of safety policy types
- **`ChecksEvaluationMetricConfig`** — Policy config with optional threshold

## See Also

- [Google Checks AI Safety](https://checks.google.com/ai-safety)
- [Genkit Documentation](https://genkit.dev/)
- [JS Checks Plugin](https://github.com/firebase/genkit/tree/main/js/plugins/checks)
