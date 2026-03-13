# Sample Flow Testing Tool

This directory contains scripts for reviewing and testing Genkit flows within samples.

## Files

- `review_sample_flows.py`: Iterates through all flows in a given sample directory, runs them with heuristic inputs, and generates a report.
- `run_single_flow.py`: Helper script to run a single flow in isolation (used by `review_sample_flows.py`).

## Usage

This tool is typically run via the `py/bin/test_sample_flows` script:

```bash
# Test a specific sample
py/bin/test_sample_flows provider-google-genai-hello
```

Or manually:

```bash
# Run from the repository root (py/)
uv run tools/sample-flows/review_sample_flows.py samples/provider-google-genai-hello
```

## Output

The tool generates a text report (e.g., `flow_review_results.txt`) detailing which flows passed or failed, along with their outputs or error messages.

**Note:** The `test_sample_flows` script automatically skips samples that do not have a standard `main.py` entry point (e.g., `framework-evaluator-demo`), preventing execution errors.
