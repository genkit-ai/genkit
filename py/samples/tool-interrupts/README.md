# Tool interrupts

Usually the generate loop calls a tool, your code runs, returns a value, and then restarts the generate loop again until it terminates on its own or hits a stopping condition.

With an **interrupt**, the tool **doesn’t** finish that way: a tool can `raise Interrupt(...)` and **hand control back to your application**. Think of it as the tool saying “you handle this step”—collect input, call another service, enforce policy—**instead of** returning a final tool result in one shot.

The Genkit SDK **stops that generation turn**, surfaces the pending tool call (with your payload on `metadata["interrupt"]`), and you `generate` again later with the **same `messages`** plus `resume_respond` or `resume_restart`. Either you **inject the tool outcome** (respond) or you **ask the SDK to run the tool again** with new input and metadata.

## Samples

`respond_example.py` — Trivia: the “tool” hands off to the CLI; your answer **is** the tool result (`respond_to_interrupt` + `resume_respond`). Prompt: `prompts/trivia_host_cli.prompt`.

`approval_example.py` — Bank demo: `y` restarts the tool (`resume_restart`); `n` declines with respond (`resume_respond`). `USER_MESSAGE` is hardcoded; you only type y/n. Prompt: `prompts/bank_transfer_host_cli.prompt`.

## Run

`GEMINI_API_KEY` (Google AI plugin):

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/respond_example.py
uv run src/approval_example.py
```

From repo root:

```bash
uv run --directory py/samples/tool-interrupts python src/respond_example.py
uv run --directory py/samples/tool-interrupts python src/approval_example.py
```

Wire detail: `MESSAGE_SHAPES.md`.
