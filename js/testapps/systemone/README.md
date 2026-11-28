# System One testapp

Decisions: asking a System One model typed questions about a state and getting
calibrated answers back, instead of asking a chat model for prose and parsing
it. The model is TypeSafe's jev, reached through OpenRouter's Decisions API.
The flows follow the Genkit Java System One sample.

The output schema is the question set: each field is one question, built with
`choice()`, `score()`, or `noul()` from `@genkit-ai/systemone`, its
`.describe()` is the instructions, and the answer lands in the field: a choice
or a score with its probabilities, a yes/no question as the probability that
the answer is yes.

## Flows

| Flow | What it shows |
| ---- | ------------- |
| `triageFlow` | Three questions about one ticket in a single call, routed in code behind a confidence gate |
| `triagePromptFlow` | The same decision from a prompt file (`prompts/triage.prompt`) that renders the state as JSON |
| `screenFlow` | A yes/no battery over a message, reduced to pass, review, or block |
| `rankFlow` | Scores every passage of a shortlist against a query, one call per passage, and orders them |
| `teamFlow` | The built-in `enum` output format: the system message is the question, the values are the options |
| `toolFlow` | A choice whose options (the tools on hand) are only known at run time |

## Running

```bash
export OPENROUTER_API_KEY=...
pnpm genkit:dev
```

The testapp reads the key from `OPENROUTER_API_KEY`. Without it, the plugin is
created with `apiKey: false`, so the Dev UI still starts and lists the models,
and each flow fails with FAILED_PRECONDITION: "the plugin was created with
apiKey: false, so pass the API key as config.apiKey". Set the variable and
restart, or pass `config: { apiKey }` in a flow.

The Dev UI shows the question set on each model request and the full
distributions on each response.
