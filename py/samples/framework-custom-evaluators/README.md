# Writing your own evaluators

This sample demonstrates how to write your own suite of custom evaluators. The evaluators in this package demonstrate how to write evaluators that leverage LLMs as well as a simple regex matcher. There are also simple test datasets to demonstrate how to use them.

## Evaluators

### Non-LLM Evaluators

#### Regex Matchers

- **Location**: `src/regex_evaluator.py`
- **Names**: `byo/regex_match_url`, `byo/regex_match_us_phone`
- **Output**: boolean

The regex evaluator is an example that does not use an LLM. It also demonstrates how to create a factory method that can be parameterized to create multiple evaluators from the same pattern.

### LLM-Based Evaluators

#### PII Detection

- **Location**: `src/pii_evaluator.py`
- **Name**: `byo/pii_detection`
- **Output**: boolean

An evaluator that attempts to detect PII in your output using an LLM judge.

#### Funniness

- **Location**: `src/funniness_evaluator.py`
- **Name**: `byo/funniness`
- **Output**: enum/categorization (`FUNNY_JOKE`, `NOT_FUNNY_JOKE`, `OFFENSIVE_JOKE`, `NOT_A_JOKE`)

An evaluator that attempts to judge if a passed statement is a joke and if it is funny.

#### Deliciousness

- **Location**: `src/deliciousness_evaluator.py`
- **Name**: `byo/deliciousness`
- **Output**: string (`yes`, `no`, `maybe`)

An evaluator that attempts to judge if a passed statement is delicious literally or metaphorically.

## Setup and Run

1. **Set environment variable**:
   ```bash
   export GEMINI_API_KEY=<your-api-key>
   ```

2. **Start the app**:
   ```bash
   ./run.sh
   ```

## Test your evaluators

**Note**: Run these commands in a separate terminal while the app is running.

### Regex evaluators:

```bash
genkit eval:run datasets/regex_dataset.json --evaluators=byo/regex_match_url,byo/regex_match_us_phone
```

### PII Detection:

```bash
genkit eval:run datasets/pii_detection_dataset.json --evaluators=byo/pii_detection
```

### Funniness:

```bash
genkit eval:run datasets/funniness_dataset.json --evaluators=byo/funniness
```

### Deliciousness:

```bash
genkit eval:run datasets/deliciousness_dataset.json --evaluators=byo/deliciousness
```

## See your results

Navigate to the `Evaluations` section in the Dev UI at http://localhost:4000.

## Note

The evaluators implemented in this sample do not consider the `input` provided to the model as part of the evaluation. Therefore, many of the test datasets provided have `input` set to `"input"`. If you are implementing an evaluator that utilizes the input provided to the model, you have to provide the actual input in this field.
