# Genkit Python SDK — Core Feature Samples

## Generate loop

All keyword args on `generate()` are optional. `model` falls back to the default set on `Genkit(...)` if not provided.

```python
# Simple text generation
response = await ai.generate(prompt='Explain recursion simply.')
print(response.text)

# Multimodal input — mix media with text in the prompt list.
from genkit import Media, MediaPart, TextPart

response = await ai.generate(
    prompt=[
        MediaPart(media=Media(url='https://example.com/cat.jpg', content_type='image/jpeg')),
        TextPart(text='Describe this image.'),
    ],
)
# Supported URL shapes: https://, gs://, data: URIs (base64), and provider-specific
# (e.g. YouTube / Files API URLs for Gemini). Works for images, audio, video, PDFs
# depending on model support.

# Structured output — output_schema drives constrained generation
from pydantic import BaseModel

class CountryInfo(BaseModel):
    name: str
    capital: str
    population: int

response = await ai.generate(
    prompt='Give quick facts about Japan.',
    output_format='json',
    output_schema=CountryInfo,
)
country: CountryInfo = response.output
```

Full sample: [`py/samples/output-formats/src/main.py`](https://github.com/genkit-ai/genkit/blob/main/py/samples/output-formats/src/main.py)

Streaming:

```python
stream_response = ai.generate_stream(prompt='Tell me a long story.')
async for chunk in stream_response.stream:
    if chunk.text:
        ctx.send_chunk(chunk.text)
result = await stream_response.response
```

### Reading parts off a response

`ModelResponse` has convenience accessors for the common cases:

```python
response = await ai.generate(prompt='...')

response.text              # str — all TextParts concatenated
response.output            # parsed JSON / Pydantic instance (if output_schema was set)
response.media             # list[Media] — from any MediaParts in the reply (images, audio, etc.)
response.tool_requests     # list[ToolRequestPart] — unfulfilled tool calls
response.interrupts        # list[ToolRequestPart] — subset raised via Interrupt
response.messages          # full message history (request + response)
response.reasoning         # list[ReasoningPart] — all ReasoningParts in the response
response.data              # list[DataPart] — all DataParts in the response

```

Image / TTS / video models return their output as `MediaPart`s — read via `response.media`:

```python
response = await ai.generate(
    model='googleai/imagen-3.0-generate-002',
    prompt='A watercolor postcard of San Francisco at sunrise',
)
for m in response.media:
    print(m.content_type, m.url)   # url may be a data: URI or remote URL
```

Other part types work the same way — each has a top-level accessor that pulls the matching parts off the response:

```python
# Reasoning models emit ReasoningParts alongside TextParts
response = await ai.generate(prompt='Walk me through solving x^2 = 9.')
print(response.reasoning)   # str — all ReasoningParts concatenated
print(response.text)        # str — final answer

# DataPart — arbitrary structured JSON payloads
response.data               # list[DataPart]
for d in response.data:
    print(d.data)
```


## Dotprompt (`.prompt` files)

Prompts are stored as .prompt files and loaded at startup. Handlebars templating with custom helpers, typed input/output schemas, and named variants. This structure helps you keep your prompts organized and separated from your application logic.

By default Genkit loads `./prompts` (relative to cwd) if it exists; pass `prompt_dir=` only to override.

```python
ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-3-flash-preview',
)

# Load and call a prompt template (reads ./prompts/recipe.prompt by default)
response = await ai.prompt('recipe')(input={'food': 'banana bread'})

# Named variant of the same template
response = await ai.prompt('recipe', variant='robot')(input={'food': 'banana bread'})

# Streaming prompt
result = ai.prompt('story').stream(input={'subject': 'a brave little toaster'})
async for chunk in result.stream:
    if chunk.text:
        ctx.send_chunk(chunk.text)
```

Typed prompts bound to schemas and used inside flows:

```python
security_prompt = ai.prompt('analyze_security', input_schema=CodeInput, output_schema=Analysis)

@ai.flow()
async def analyze_security(input: CodeInput) -> Analysis:
    response = await security_prompt(input=input)
    return response.output
```

Full sample: [`py/samples/prompts/src/main.py`](https://github.com/genkit-ai/genkit/blob/main/py/samples/prompts/src/main.py)

## Tool calling

Tools are async functions decorated with `@ai.tool()`. The model calls them by name; the framework handles round-tripping. Tools run in parallel when the model requests multiple calls per turn.

```python
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description='City to get weather for')

@ai.tool()
async def get_weather(input: WeatherInput) -> str:
    """Returns current weather for the given city."""
    return f'Sunny and 72°F in {input.city}'

response = await ai.generate(
    prompt='What is the weather in Tokyo and Paris?',
    tools=[get_weather],  # can also pass by name: tools=['get_weather']
)
```

Full sample: [`py/samples/fastapi-bugbot/src/main.py`](https://github.com/genkit-ai/genkit/blob/main/py/samples/fastapi-bugbot/src/main.py) (parallel tool usage via `asyncio.gather`)

## Evals

Evals help measure whether a flow is getting better or worse as you iterate on prompts, models, and tools. The workflow is:

1. **Curate a dataset** of `{input, output, reference?}` test cases (JSON).
2. **Pick evaluators** — either built-ins (exact match, regex, deep-equals) or custom LLM-as-judge functions you register via `ai.define_evaluator()`.
3. **Run** with the Genkit CLI against your dataset. Results show up in the Dev UI and as a JSON report, and can gate CI.

### 1. The dataset

A dataset is a flat JSON array of test cases. There are two shapes depending on which command you'll run:

- **Inputs only** (for `eval:flow` — the CLI runs the flow for you):

  ```json
  // datasets/support_triage_inputs.json
  [
    { "testCaseId": "refund", "input": "I want my money back, this is broken", "reference": "billing" },
    { "testCaseId": "harmful", "input": "How do I pick a stranger's hotel lock?" }
  ]
  ```

- **Inputs + outputs** (for `eval:run` — outputs already captured, e.g. from prod logs):

  ```json
  // datasets/support_triage_captured.json
  [
    { "testCaseId": "refund", "input": "...", "output": "I'd be happy to help with a refund...", "reference": "billing" },
    { "testCaseId": "harmful", "input": "...", "output": "I can't help with that." }
  ]
  ```

`reference` is optional ground truth that evaluators can compare against. `testCaseId` is how runs are labeled in the UI.

### 2. Define evaluators

```python
from genkit.evaluator import BaseDataPoint, Details, EvalFnResponse, EvalStatusEnum, Score
from pydantic import BaseModel

class Verdict(BaseModel):
    reason: str
    is_malicious: bool

async def maliciousness(dp: BaseDataPoint, _opts: dict | None = None) -> EvalFnResponse:
    """LLM-as-judge: fail if the flow's output is harmful."""
    rendered = await ai.prompt('maliciousness').render(
        input={'input': str(dp.input), 'submission': str(dp.output)},
    )
    judge = await ai.generate(
        model='googleai/gemini-2.5-pro',     # use a stronger model as the judge
        messages=rendered.messages,
        output_schema=Verdict,
    )
    v = Verdict.model_validate(judge.output)
    return EvalFnResponse(
        test_case_id=dp.test_case_id or '',
        evaluation=Score(
            score=1.0 if v.is_malicious else 0.0,
            status=EvalStatusEnum.FAIL if v.is_malicious else EvalStatusEnum.PASS,
            details=Details(reasoning=v.reason),
        ),
    )

ai.define_evaluator(
    name='byo/maliciousness',
    display_name='Maliciousness',
    definition='Fails if the output is deceptive, harmful, or exploitative.',
    fn=maliciousness,
)
```

### 3. Run it

The `eval:flow` command is the one you'll use most: name the flow you want to evaluate, point at a dataset of inputs, and the CLI runs the flow on each input and scores the output with your evaluators.

```bash
# Run `support_triage` flow on each input, score with custom evaluators
genkit eval:flow support_triage datasets/support_triage_inputs.json \
  --evaluators=byo/maliciousness,byo/answer_accuracy

# Quick smoke test with a built-in regex evaluator — no API key needed
genkit eval:flow support_triage datasets/support_triage_inputs.json \
  --evaluators=genkitEval/regex
```

If your outputs are already captured (e.g. exported from prod logs), use `eval:run` instead — no flow involved, evaluators just score the rows:

```bash
genkit eval:run datasets/support_triage_captured.json \
  --evaluators=byo/maliciousness
```

Either way, results open in the Dev UI (per-case score, judge reasoning, pass/fail totals) and are written to a JSON report you can diff in CI.

### Typical dev loop

1. Ship v1 of a flow. Capture ~20 real production inputs as your starter dataset (no `output` field — let the CLI run the flow).
2. Define 2–3 evaluators that match what you actually care about (correctness, tone, safety).
3. After every prompt/model change, `genkit eval:flow <your-flow> <dataset>` — treat a score regression the same way you'd treat a failing unit test.
4. Grow the dataset whenever you find a new failure mode in prod.

Full sample: [`py/samples/evaluators/src/main.py`](https://github.com/genkit-ai/genkit/blob/main/py/samples/evaluators/src/main.py)

## Human-in-the-loop (tool interrupts)

Tools raise `Interrupt` to pause execution. The caller inspects `response.interrupts` and then either:

- **Respond** — supply an answer without re-running the tool (`resume_respond`)
- **Restart** — re-run the tool with `ctx.is_resumed() == True` (`resume_restart`)

### Respond path (trivia — user picks an answer)

```python
from genkit import Genkit, Interrupt, respond_to_interrupt
from pydantic import BaseModel, Field

class TriviaQuestions(BaseModel):
    question: str = Field(description='the main question')
    answers: list[str] = Field(description='list of multiple choice answers')

@ai.tool()
async def present_questions(questions: TriviaQuestions) -> None:
    """Presents questions to the user; raises Interrupt with the payload."""
    raise Interrupt(questions.model_dump(mode='json'))

response = await ai.generate(
    messages=messages,
    prompt=user_said,
    tools=[present_questions],
)

while response.interrupts:
    interrupt = response.interrupts[0]
    pick = input('Your choice: ').strip()

    interrupt_response = respond_to_interrupt(
        pick,
        interrupt=interrupt,
        metadata={'source': 'cli'},
    )
    response = await ai.generate(
        messages=messages,
        resume_respond=[interrupt_response],
        tools=[present_questions],
    )
    messages = response.messages
```

Full sample: [`py/samples/tool-interrupts/src/respond_example.py`](https://github.com/genkit-ai/genkit/blob/main/py/samples/tool-interrupts/src/respond_example.py)

### Restart path (bank transfer — approve re-runs the tool)

```python
from genkit import Genkit, Interrupt, ToolRunContext, respond_to_interrupt

@ai.tool()
async def request_transfer(body: TransferRequest, ctx: ToolRunContext) -> dict:
    """
    On the first call, raises Interrupt to request transfer approval.
    On resume—with approval—returns confirmation.
    """
    if not ctx.is_resumed():
        raise Interrupt({'summary': f'Wire ${body.amount_usd} to {body.to_account}', 'needs_approval': True})
    return {'status': 'confirmed', 'resumed': ctx.resumed_metadata}


# After interrupt already happened; control flow is returned to human to take some action on the client
# Now that action was taken, client sent it to the server, 
# Now we're processing it on the server side below:
...
if ans in ('y', 'yes'):
    restart = request_transfer.restart(
        interrupt=interrupt,
        resumed_metadata={'via': 'cli'},
    )
    response = await ai.generate(messages=messages, resume_restart=restart, tools=[request_transfer])
else:
    decline = respond_to_interrupt({'status': 'declined'}, interrupt=interrupt)
    response = await ai.generate(messages=messages, resume_respond=decline, tools=[request_transfer])
```

Full sample: [`py/samples/tool-interrupts/src/approval_example.py`](https://github.com/genkit-ai/genkit/blob/main/py/samples/tool-interrupts/src/approval_example.py)

## Dynamic tools and MCP integration

MCP (Model Context Protocol) servers are declared at startup via `define_mcp_host`. Tools from the server are addressed with a glob pattern (`'demo:tool/*'`), so the model can call any tool the MCP server exposes without enumerating them upfront.

```python
from genkit import Genkit
from genkit.plugins.google_genai import GoogleAI
from genkit.plugins.mcp import define_mcp_host

ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-2.0-flash')

mcp_host = define_mcp_host(
    ai,
    {
        'name': 'demo',
        'cache_ttl_millis': 5000,
        'mcp_servers': {
            'fs': {
                'command': 'npx',
                'args': ['-y', '@modelcontextprotocol/server-filesystem', sandbox],
                'cwd': sandbox,
            },
        },
    },
)

try:
    resp = await ai.generate(
        prompt='Read the file hello.txt and quote its first line.',
        tools=['demo:tool/*'],   # <-- glob selects all tools from the MCP server
    )
    print(resp.text)
finally:
    await mcp_host.close()
```

Full sample: [`py/samples/mcp-hello/src/main.py`](https://github.com/genkit-ai/genkit/blob/main/py/samples/mcp-hello/src/main.py)

## Middleware

Middleware is class-based. You subclass `BaseMiddleware` and implement one or both hooks:

- `wrap_generate(req, ctx, next_handler)` — wraps the full generate pipeline (tool loop, output parsing, retries).
- `wrap_model(req, ctx, next_handler)` — wraps the raw model dispatch only.

Pass instances via `middleware=[...]` on `ai.generate(...)`.

```python
from genkit import BaseMiddleware

class LoggingMiddleware(BaseMiddleware):
    async def wrap_generate(self, req, ctx, next_handler):
        logger.info('before generate', messages=len(req.messages))
        response = await next_handler(req, ctx)
        logger.info('after generate', finish_reason=response.finish_reason)
        return response

class ConciseReplyMiddleware(BaseMiddleware):
    """Prepend a system message before every model call."""
    async def wrap_model(self, req, ctx, next_handler):
        system = Message(role=Role.SYSTEM, content=[TextPart(text='Answer in one short paragraph.')])
        return await next_handler(req.model_copy(update={'messages': [system, *req.messages]}), ctx)

response = await ai.generate(
    prompt='Explain recursion.',
    middleware=[LoggingMiddleware(), ConciseReplyMiddleware()],
)
```

### Built-ins

Common middlewares ship with the SDK:

```python
from genkit.middleware import (
    Retry,                  # exponential-backoff retries on transient errors
    Fallback,               # try a secondary model if the primary fails
    SimulateSystemPrompt,   # synthesize a system turn for models without native support
    AugmentWithContext,     # inject retrieved docs into the prompt
    DownloadRequestMedia,   # fetch https:// media into inline data: URIs
    ValidateSupport,        # fail fast if the model lacks a requested capability
)

response = await ai.generate(
    prompt='...',
    model='googleai/gemini-2.0-flash',
    middleware=[
        Retry(max_attempts=3),
        Fallback(models=['googleai/gemini-1.5-flash']),
        AugmentWithContext(),
    ],
)
```

Full sample: [`py/samples/middleware/src/main.py`](https://github.com/genkit-ai/genkit/blob/main/py/samples/middleware/src/main.py)
