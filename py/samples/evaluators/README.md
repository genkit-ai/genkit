# Evaluators

A regex check needs no key. A custom judge is `define_evaluator` plus
`generate()`.

```bash
uv sync
genkit eval:run datasets/genkit_eval_dataset.json --evaluators=genkitEval/regex -- uv run src/main.py
```

`match` passes. `no_match` fails.

The other datasets need `GEMINI_API_KEY`:

```bash
genkit eval:run datasets/maliciousness_dataset.json --evaluators=byo/maliciousness -- uv run src/main.py
```
