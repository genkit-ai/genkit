# Genkit Evaluators Plugin

Provides three built-in rule-based evaluators:

- **regex** – Tests output against a regex pattern (reference = regex string)
- **deep_equal** – Tests equality of output against reference
- **jsonata** – Evaluates a JSONata expression (reference) against output; pass if result is truthy

No LLM or API keys required.

## Installation

```bash
uv add genkit-evaluators
```

## Usage

```python
from genkit import Genkit
from genkit.evaluator import BaseDataPoint
from genkit_evaluators import register_genkit_evaluators

ai = Genkit()
register_genkit_evaluators(ai)

results = await ai.evaluate(
    evaluator='genkitEval/regex',
    dataset=[
        BaseDataPoint(input='sample', output='banana', reference='ba?a?a'),
        BaseDataPoint(input='sample', output='apple', reference='ba?a?a'),
    ],
)
```

Or from the CLI:

```bash
genkit eval:run datasets/example.json --evaluators=genkitEval/regex
```

## Evaluators

- **genkitEval/regex** – Reference is a regex string. Output (stringified if needed) must match.
- **genkitEval/deep_equal** – Reference is the expected value. Output must equal reference.
- **genkitEval/jsonata** – Reference is a JSONata expression. Evaluated against output; pass if truthy.
