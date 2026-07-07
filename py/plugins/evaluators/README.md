# Genkit Evaluators Plugin

Provides three rule-based evaluators matching the Go and JS implementations:

- **regex** – Tests output against a regex pattern (reference = regex string)
- **deep_equal** – Tests equality of output against reference
- **jsonata** – Evaluates a JSONata expression (reference) against output; pass if result is truthy

No LLM or API keys required.

## Installation

```bash
pip install genkit-plugin-evaluators
```

## Usage

```python
from genkit import Genkit
from genkit.plugins.evaluators import GenkitEval

ai = Genkit(plugins=[GenkitEval()])

# Run evaluation with genkit eval-flow or programmatically
evaluator = await ai.registry.resolve_evaluator('genkitEval/regex')
result = await evaluator.run(input={
    'dataset': [
        {'input': 'sample', 'output': 'banana', 'reference': 'ba?a?a'},
        {'input': 'sample', 'output': 'apple', 'reference': 'ba?a?a'},
    ],
    'evalRunId': 'test',
})
```

## Evaluators

- **genkitEval/regex** – Reference is a regex string. Output (stringified if needed) must match.
- **genkitEval/deep_equal** – Reference is the expected value. Output must equal reference.
- **genkitEval/jsonata** – Reference is a JSONata expression. Evaluated against output; pass if truthy.
