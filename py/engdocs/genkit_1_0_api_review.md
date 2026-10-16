# Genkit 1.0 API Review (Python)

The Python surface we intend to ship as 1.0. It describes the state after the open API PRs land (#6681, #6697, #6698, #6699, #6708, #6709, #6757, #6763–#6770, #6775, #6777, #6314, #6655, #6668, #6712, #6714).

---

## 1. Public API Surface

```python
from genkit import (
    # The app object; every ai.* call goes through it.
    Genkit,
    # What you send a model.
    Message, Role, Part, Media, Document, ToolChoice,
    # What comes back; Failure is the error on a failed turn.
    ModelResponse, ModelResponseChunk, ModelStreamResponse, StreamResponse, FinishReason, Failure,
    # ai.embed returns Embeddings; background models return Operation[ModelResponse].
    Embedding, Operation,
    # Flows, tools, prompts, and human-in-the-loop.
    Flow, FlowContext, Tool, ToolContext, Interrupt, MultipartToolResponse, Prompt,
    # Errors: GenkitError is internal, PublicError is safe to show clients.
    GenkitError, PublicError, ErrorReason,
)

from genkit.model import (
    # Defining a model, and the ctx its fn receives.
    Model, BackgroundModel, ModelContext,
    # Typed handle to a model and its config base class.
    ModelRef, ModelConfig,
    # The request a model fn receives.
    ModelRequest, GenerateActionOptions, OutputConfig, ToolDefinition, ToolRequest, ToolResponse,
    # What a model declares it supports, and token accounting.
    ModelInfo, ModelSupports, ConstrainedSupport, ModelStage, GenerationUsage,
)

from genkit.embedder import (
    # Defining an embedder and its handle.
    Embedder, EmbedderRef, EmbedderInfo, EmbedderSupports,
    # Embed request and response.
    EmbedRequest, EmbedResponse,
)

from genkit.evaluator import (
    # Defining an evaluator: per data point or batch.
    Evaluator, BatchEvaluator, EvaluatorRef,
    # Evaluator input and scored output.
    EvaluatorRequest, BatchEvaluatorRequest, EvaluatorResponse, BaseDataPoint, Score, ScoreDetails, ScoreStatus,
)

from genkit.middleware import (
    # Subclass Middleware, override wrap_generate / wrap_model / wrap_tool, register with @ai.middleware.
    Middleware, MiddlewareContext, GenerateHookParams, ModelHookParams, ToolHookParams,
)

from genkit.formats import (
    # Custom output formats.
    Formatter, FormatterConfig, FormatHandler,
)

from genkit.web import (
    # What an adapter's context_provider receives, and its type.
    RequestData, ContextProvider,
    # Building blocks for serving flows over HTTP; error_body/error_status redact non-public errors.
    read_body, wants_stream, error_body, error_status,
)

from genkit.telemetry import (
    # OpenTelemetry wiring and custom spans.
    Instrumentation, DisposableInstrumentation, FlushableInstrumentation,
    SpanContext, SpanMetadata, SpanNext,
    configure_instrumentation, run_in_new_span, is_instrumented_by, reset_instrumentation,
)

from genkit.testing import (
    # Fake models for unit tests.
    EchoModel, ScriptedModel, StaticResponseModel,
    define_echo_model, define_scripted_model, define_static_response_model,
)

from genkit.plugin_api import (
    # What plugin authors build on.
    Plugin, Action, ActionKind, DynamicActionProvider, GenkitConfig,
    # Helpers for provider plugins.
    StatusName, StrEnum, provider_error, get_logger, loop_local_client, GENKIT_USER_AGENT, is_dev_environment,
)

from genkit.exp import (
    # Experimental: agents and sessions (genkit.exp.agent adds AgentContext for custom agent fns).
    Genkit, current_session,
)

from genkit_google_genai import (
    # Plugins; refs come from classmethods: GoogleAI.gemini('gemini-flash-latest') -> ModelRef[GeminiConfig].
    GoogleAI, VertexAI,
    # One flat config class per model family.
    GeminiConfig, GeminiImageConfig, GeminiTtsConfig, GemmaConfig, VeoConfig, LyriaConfig,
    DeepResearchConfig, AntigravityConfig,
    # Literal and nested types used inside those configs.
    ThinkingLevel, SafetySetting, HarmCategory, HarmBlockThreshold, HarmBlockMethod, ResponseModality, ...,
)

from genkit_anthropic import (
    # Plugin; Anthropic.claude('claude-sonnet-4-5') -> ModelRef[AnthropicConfig]; tool choice comes from generate(tool_choice=).
    Anthropic, AnthropicConfig, Effort, ThinkingDisplay, ThinkingMode,
)
```

### Stable and experimental `Genkit`

There are two `Genkit` classes. `genkit.Genkit` is the stable 1.0 surface. `genkit.exp.Genkit` subclasses it and adds the experimental APIs: agents (`define_agent`, `define_prompt_agent`, `define_custom_agent`, `ai.agent(name)`) and sessions (`current_session()`). The stable class never grows those methods, so switching the import is the opt-in. Everything stable works the same on the subclass.

`genkit.exp` stays experimental after 1.0 and can change in a minor release.

```python
# 1. Swap the import; the stable one is `from genkit import Genkit`
from genkit.exp import Genkit
from genkit.exp.agent import InMemorySessionStore
from genkit_google_genai import GoogleAI

# 2. Same constructor; agents and sessions are now available
ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))
host = ai.define_agent(name='host', system='Seat guests and take reservations.', store=InMemorySessionStore())

# 3. Stable calls are unchanged on the subclass
response = await ai.generate(prompt='Suggest a dish.')
print(response.text)
# => Smoked Salmon Tartine
```

---

## 2. Model Generation (`ai.generate`)

### Structured Output with Pydantic

Models enforce structured output through JSON schema constraints, parsed directly into Pydantic models.

```python
from pydantic import BaseModel, Field
from genkit import Genkit
from genkit_google_genai import GoogleAI

# 1. Initialize runtime with a default model ref
ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))

class InventoryItem(BaseModel):
    sku: str = Field(description='Stock keeping unit identifier')
    quantity: int = Field(description='Quantity available in warehouse')
    reorder_recommended: bool

# 2. Run generation constrained by schema
response = await ai.generate(
    prompt='Warehouse audit: 4 units remaining for SKU-5821.',
    output_schema=InventoryItem,
)

# 3. Access validated Pydantic model
item = response.output
print(f'{item.sku}: {item.quantity} units (Reorder: {item.reorder_recommended})')
# => SKU-5821: 4 units (Reorder: True)
```

### Reading Message Parts

A `Message` is a role plus a list of `Part`s. A `Part` is flat: exactly one of `text`, `media`, `tool_request`, `tool_response`, `data`, `reasoning`, or `resource` is set, plus optional `metadata`. Setting a second kind raises.

#### Shortcuts

```python
from genkit_google_genai import GeminiImageConfig, GoogleAI

response = await ai.generate(
    model=GoogleAI.gemini_image('gemini-2.5-flash-image'),
    prompt='Plate photo for the spring menu, then describe it in one line.',
    config=GeminiImageConfig(response_modalities=['TEXT', 'IMAGE']),
)

# 1. Visible text only; reasoning parts are excluded
print(response.text)
# => Seared scallops over pea purée, finished with mint oil.

# 2. Every media part on the reply
for media in response.media:
    print(media.content_type, media.url[:30])
    # => image/png data:image/png;base64,iVBORw0K

# 3. The raw parts: one image, one text
print(len(response.message.content))
# => 2
```

#### Every part, by kind

```python
for part in response.message.content:
    if part.text is not None:
        print('text:', part.text)
        # => text: Seared scallops over pea purée, finished with mint oil.
    elif part.media is not None:
        save_image(part.media.url, part.media.content_type)
    elif part.reasoning is not None:
        print('thinking:', part.reasoning)
    elif part.tool_request is not None:
        print('tool call:', part.tool_request.name, part.tool_request.input)
    elif part.data is not None:
        print('data:', part.data)
```

#### Walking the whole conversation

`response.messages` is the request history plus the reply. After a tool loop it includes each tool call and its result, matched by `ref`.

```python
response = await ai.generate(prompt='Is ORD-9921 delivered?', tools=[lookup_order])

for message in response.messages:
    for part in message.content:
        if part.tool_request:
            print(f'{message.role}: call {part.tool_request.name}({part.tool_request.input}) ref={part.tool_request.ref}')
        elif part.tool_response:
            print(f'{message.role}: result {part.tool_response.output} ref={part.tool_response.ref}')
        elif part.text:
            print(f'{message.role}: {part.text}')
# => user: Is ORD-9921 delivered?
# => model: call lookup_order(ORD-9921) ref=0
# => tool: result ORD-9921: delivered 14:02 ref=0
# => model: Yes, it was delivered at 14:02.
```

### Streaming Responses

Streaming yields incremental text chunks while resolving final token usage upon completion. The main form iterates `stream.stream`; iterating the `StreamResponse` itself is shorthand for the same loop:

```python
async for chunk in stream.stream:  # main form, used throughout this doc
async for chunk in stream:         # shorthand
```

```python
# 1. Initiate async streaming generator
stream = ai.generate_stream(
    prompt='Draft a release notice for maintenance on Sunday at 02:00 UTC.',
)

# 2. Consume chunks as they arrive
async for chunk in stream.stream:
    print(chunk.text, end='', flush=True)
    # => Scheduled Maintenance: Systems will be offline Sunday at 02:00 UTC.

# 3. Await final aggregated response
response = await stream.response
print(f'\nTotal tokens: {response.usage.total_tokens}')
# => Total tokens: 34
```

### Streaming Structured Output: Typed Partials (feedback wanted)

With `output_schema=Recipe`, each `chunk.output` is typed `Recipe`, but it's built from the JSON received so far without validation (`model_construct`). Genkit parses the accumulated text as partial JSON and fills the fields that have arrived. The rest are `None`, even when the field is typed `str` or `int`. Nested models and list items get the same treatment. Only `(await stream.response).output` is validated.

```python
class Ingredient(BaseModel):
    name: str
    grams: int

class Recipe(BaseModel):
    title: str
    servings: int
    ingredients: list[Ingredient]

# 1. Stream with a schema
stream = ai.generate_stream(prompt='Pancakes for 4.', output_schema=Recipe)

# 2. Each chunk.output is a Recipe with holes
async for chunk in stream.stream:
    print(chunk.output)
# => Recipe(title='Fluffy Panc', servings=None, ingredients=None)
# => Recipe(title='Fluffy Pancakes', servings=4, ingredients=[Ingredient(name='fl', grams=None)])
# => Recipe(title='Fluffy Pancakes', servings=4, ingredients=[Ingredient(name='flour', grams=200), Ingredient(name='milk', grams=None)])
# => Recipe(title='Fluffy Pancakes', servings=4, ingredients=[Ingredient(name='flour', grams=200), Ingredient(name='milk', grams=300)])

# 3. The validated value
recipe = (await stream.response).output
```

What this means for the app developer:

- Any field can be `None` mid-stream, even one typed `int`. The type checker won't catch `chunk.output.servings + 1`; it fails at runtime.
- A string can be cut short (`'Fluffy Panc'`), and nothing tells you whether a chunk is complete.
- So: read `chunk.output` for live display, guard each field, and use `(await stream.response).output` for anything that matters.

JS and Go make the same trade: JS types the partial as `T`, and Go fills missing fields with zero values.

Question for reviewers: is this the best approach, or is there a better way to handle typed partials?

### Calling a Tool Directly

`@ai.tool` returns a `Tool` you can call like the function it wraps. `await tool(input)` returns what the function returned, typed by its return annotation: a Pydantic model comes back as the instance, not a camelCase dict. Unit tests call tools this way without a model.

```python
from pydantic import BaseModel

class Quote(BaseModel):
    sku: str
    unit_price: float
    in_stock: int

# 1. A tool with a typed return
@ai.tool
async def price_lookup(sku: str) -> Quote:
    """Look up price and stock for a SKU."""
    return Quote(sku=sku, unit_price=12.5, in_stock=40)

# 2. Call it directly; the result is the Quote instance
quote = await price_lookup('SKU-5821')
print(type(quote).__name__, quote.unit_price)
# => Quote 12.5

# 3. The same tool in generate; the model sees the Quote as JSON
response = await ai.generate(prompt='How much is SKU-5821?', tools=[price_lookup])
print(response.text)
# => SKU-5821 is $12.50 and 40 are in stock.
```

A direct call fails the same way `generate` would: output that can't be sent to a model raises `GenkitError` with `INVALID_ARGUMENT`, and an interrupt tool raises `Interrupt`.

### Multipart Tool Results

A tool that returns more than a JSON value, such as structured data plus a screenshot, returns a `MultipartToolResponse`. `output` is what the model reads as the tool result; `content` is a list of `Part`s (media, text) the model can see alongside it. Annotate the return as `MultipartToolResponse[T]` so the tool's output schema is `T`.

```python
from genkit import Media, MultipartToolResponse, Part

class Shot(BaseModel):
    page: str
    width: int

# 1. Return structured output plus an image part
@ai.tool
async def capture_dashboard(page: str) -> MultipartToolResponse[Shot]:
    """Screenshot a dashboard page."""
    png_url = await browser.screenshot(page)
    return MultipartToolResponse(
        output=Shot(page=page, width=1440),
        content=[Part(media=Media(url=png_url, content_type='image/png'))],
    )

# 2. Called directly, you get the MultipartToolResponse; .output is the Shot instance
result = await capture_dashboard('/billing')
print(result.output.width, result.content[0].media.content_type)
# => 1440 image/png

# 3. In generate, the model gets the Shot as the result and looks at the image
response = await ai.generate(prompt='Is anything red on the billing dashboard?', tools=[capture_dashboard])
print(response.text)
# => The overdue invoices tile is red: 3 invoices past 30 days.
```

A plain `return value` is the same as `MultipartToolResponse(output=value)`.

How it reaches the model:

1. `generate` runs every tool the model requested and builds one `role='tool'` message with a `toolResponse` part per call. Each part carries `output` (JSON-dumped) and `content` (the parts).
2. That message is appended to the history, and `generate` calls the model again with it. The tool's `content` isn't sent as a separate user message.
3. The model plugin decides how `content` goes on the wire. Gemini attaches it to the same function result (`FunctionResponse.parts`). The OpenAI, Anthropic, Ollama and Bedrock plugins send only `output` today and drop `content`.

### Streaming Tool Chunks

A tool declares its chunk type with `ToolContext[ChunkT]`, separate from its return type. The same tool streams two ways.

#### Inside `generate_stream()`: partial tool responses

Each `ctx.send_chunk()` arrives on the generate stream as a partial tool response (`part.is_partial`). Partial parts stay out of `response.messages`, so the model never sees mid-run progress on the next turn.

```python
from pydantic import BaseModel
from genkit import ToolContext

class DeployInput(BaseModel):
    service: str

class DeployStep(BaseModel):
    step: str
    percent: int

class Rollout(BaseModel):
    url: str
    revision: str

# 1. Tool streams DeployStep chunks and returns a Rollout
@ai.tool
async def deploy_service(input: DeployInput, ctx: ToolContext[DeployStep]) -> Rollout:
    ctx.send_chunk(DeployStep(step='building image', percent=25))
    ctx.send_chunk(DeployStep(step='pushing layers', percent=75))
    return Rollout(url=f'https://{input.service}.run.app', revision='rev-0042')

# 2. Model calls the tool; partial chunks arrive live, then the final return
stream = ai.generate_stream(prompt='Deploy payments', tools=[deploy_service])
async for chunk in stream.stream:
    for part in chunk.tool_responses:
        tr = part.tool_response
        if part.is_partial:
            print(f'[{tr.name}] progress: {tr.output}')
            # => [deploy_service] progress: step='building image' percent=25
            # => [deploy_service] progress: step='pushing layers' percent=75
        else:
            print(f'[{tr.name}] done: {tr.output}')
            # => [deploy_service] done: url='https://payments.run.app' revision='rev-0042'
    if chunk.text:
        print(chunk.text, end='')

# 3. History holds the final Rollout only
response = await stream.response
```

#### Direct: `tool.stream()`

Called without a model, the tool is a `StreamResponse[DeployStep, Rollout]`. Chunks are typed `DeployStep`; the response is the `Rollout` itself, same as `await deploy_service(...)`.

```python
# 1. Stream the tool directly
stream = deploy_service.stream(DeployInput(service='payments'))

# 2. Chunks are DeployStep
async for step in stream.stream:
    print(f'{step.step} ({step.percent}%)')
    # => building image (25%)
    # => pushing layers (75%)

# 3. Response is Rollout, not a tool-response envelope
rollout = await stream.response
print(rollout.url)
# => https://payments.run.app
```

### Interrupts (human-in-the-loop)

A tool raises `Interrupt(metadata)` to pause the turn. `generate` returns with `finish_reason == INTERRUPTED`, and the paused tool calls are on `response.interrupts`. You resume by passing the history back with one of two answers. `part.restart(...)` runs the tool again, for example after a manager approves; the tool can check `ctx.is_resumed()` and read `ctx.resumed_metadata`. `part.respond(output)` answers for the tool without running it.

`resume_restart=` and `resume_respond=` take one `Part` or a list. Pass a list when several tool calls paused in the same turn.

`Interrupt` subclasses `BaseException`, so an `except Exception` inside a tool can't swallow it and skip approval.

#### Approve, then re-run

```python
from genkit import FinishReason, Interrupt, ToolContext

# 1. Refunds over $100 pause for a manager
@ai.tool
async def issue_refund(order_id: str, amount: float, ctx: ToolContext) -> str:
    if amount > 100 and not ctx.is_resumed():
        raise Interrupt({'reason': 'needs_manager', 'amount': amount})
    approver = (ctx.resumed_metadata or {}).get('approved_by', 'auto')
    return f'refunded ${amount} on {order_id} (approved by {approver})'

response = await ai.generate(prompt='Refund $240 on ORD-9921.', tools=[issue_refund])

# 2. The turn pauses; nothing was refunded
print(response.finish_reason)
# => interrupted
paused = response.interrupts[0]
print(paused.tool_request.name, paused.metadata['interrupt'])
# => issue_refund {'reason': 'needs_manager', 'amount': 240.0}

# 3. Manager approves; re-run the same call with the approval attached
response = await ai.generate(
    messages=response.messages,
    tools=[issue_refund],
    resume_restart=paused.restart(resumed_metadata={'approved_by': 'dana'}),
)
print(response.text)
# => Done: refunded $240.00 on ORD-9921, approved by dana.
```

#### Answer for the tool

```python
from pydantic import BaseModel

class Question(BaseModel):
    question: str
    choices: list[str]

# 1. An interrupt-only tool: always pauses, the user's answer is its output
ask_guest = ai.define_interrupt('ask_guest', input_schema=Question, description='Ask the guest to choose.')

response = await ai.generate(prompt='Book a table for 4 tonight.', tools=[ask_guest])
paused = response.interrupts[0]
print(paused.tool_request.input)
# => {'question': 'Which time works?', 'choices': ['18:30', '20:45']}

# 2. Reply with the guest's choice; the tool never runs
response = await ai.generate(
    messages=response.messages,
    tools=[ask_guest],
    resume_respond=paused.respond('20:45'),
)
print(response.text)
# => Booked a table for 4 at 20:45.
```

### Provider Configuration & ModelRef

Refs come from plugin classmethods and carry their config class. `GeminiConfig` is flat: `temperature` sits next to `thinking_level`.

```python
from genkit_google_genai import GeminiConfig, GoogleAI

# 1. Typed ref: config is checked as GeminiConfig at the call site
analyst = GoogleAI.gemini('gemini-flash-latest')

# 2. Flat config: common and Gemini-only fields side by side
response = await ai.generate(
    model=analyst,
    prompt='Analyze customer churn indicators in financial records.',
    config=GeminiConfig(
        temperature=0.2,
        max_output_tokens=2048,
        thinking_level='high',
        safety_settings=[
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_LOW_AND_ABOVE'},
        ],
    ),
)

print(response.text)
# => Key churn indicators identified: payment failure frequency and login recency drop.
```

### Background Models

For jobs that outlive a request (video, deep research), a background model splits into `start` and `check` (plus optional `cancel`). `start` returns an `Operation` handle and `check` polls it. All three get a `ModelContext`; on these functions it's for `ctx.context` (per-request secrets, auth) and `ctx.abort_signal`, not for streaming.

`Operation` is generic over its output, like JS and Go. Background models return `Operation[ModelResponse]`: once `done`, `op.output` is a `ModelResponse`, so you read the video the same way you read a `generate` reply. A model fn can return a plain `Operation`; Genkit checks the output on the way out, and anything that isn't a `ModelResponse` fails with `INTERNAL`.

`check_operation` and `cancel_operation` return the type they're given (`Operation[T]` in, `Operation[T]` out). A handle you stored as JSON goes back in as the dict you saved; Genkit validates it strictly and returns `Operation[ModelResponse]`.

```python
import json
from typing import Any

from genkit import FinishReason, Media, Message, ModelResponse, Operation, Part, Role
from genkit.model import ModelContext, ModelRequest

# 1. start submits the job and returns a handle
async def start_render(request: ModelRequest, ctx: ModelContext) -> Operation:
    job_id = await render_api.submit(prompt=request.messages[-1].text, key=ctx.context['secrets']['api_key'])
    return Operation(id=job_id, done=False)

# 2. check polls the job; once done, output is a ModelResponse holding the video
async def check_render(op: Operation, ctx: ModelContext) -> Operation:
    job = await render_api.status(op.id, key=ctx.context['secrets']['api_key'])
    if not job.done:
        return op
    video = Part(media=Media(url=job.url, content_type='video/mp4'))
    op.output = ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[video]),
    )
    op.done = True
    return op

ai.define_background_model('acme/promo-video', start=start_render, check=check_render)

# 3. Caller starts the job; BackgroundModel fills in the type argument
secrets = {'secrets': {'api_key': RENDER_KEY}}
op = await ai.generate_operation(model='acme/promo-video', prompt='15s promo for the spring menu', context=secrets)
# op: Operation[ModelResponse]
print(op.id, op.done)
# => job_7f3a False

# 4. Store the handle as JSON
await jobs.save(op.id, op.model_dump_json())

# 5. Later, e.g. in another request: pass the stored dict back, with the same context
stored: dict[str, Any] = json.loads(await jobs.load('job_7f3a'))
op = await ai.check_operation(stored, context=secrets)
# op: Operation[ModelResponse]

# 6. Poll until done; Operation[T] in, Operation[T] out
while not op.done:
    await asyncio.sleep(5)
    op = await ai.check_operation(op, context=secrets)
print(op.output.media[0].url)
# => https://cdn.acme.dev/job_7f3a.mp4
```

The caller never writes `Operation[ModelResponse]`; it comes from `generate_operation` and the stored-dict path, and `check_operation` keeps it. Write the type argument only to annotate your own code, e.g. `async def poll(op: Operation[ModelResponse]) -> str`. A bare `Operation` is `Operation[Any]`, which is what model fns return.

A stored dict with a typo'd or unknown key (`'Done'`) or no `id` fails with `INVALID_ARGUMENT` naming the field.

Plugins build the same thing with `BackgroundModel(name, start=..., check=...)` and return it from `list_actions()`. Gemini's Veo comes back from `GoogleAI.veo('veo-3.1-generate-001')`.

---

## 3. Mid-Turn Error Recovery

A tool that fails partway through a multi-step turn doesn't throw away the work before it. `ai.generate` returns a response with `finish_reason == FAILED` and the error on `response.error`. `response.messages` holds the history up to the last completed step; the unanswered tool call is dropped. Pass that history back to pick up where it stopped.

`response.error` is a `Failure`: a value, not an exception, with `status`, `message`, `details`, and `reason` (an `ErrorReason` such as `TOOL_FAILED`). A tool that raises a plain exception shows up as `INTERNAL` / `'internal error'`; raise `GenkitError` or `PublicError` to keep your status and message. `Failure` is exported so your own handlers can take it as a typed argument.

Setup problems (bad config, unknown model) still raise `GenkitError` before the turn starts.

```python
from genkit import ErrorReason, Failure, FinishReason, GenkitError

@ai.tool
async def lookup_order(order_id: str) -> str:
    return f'{order_id}: 2x pad thai, paid $31.50'

@ai.tool
async def issue_refund(order_id: str) -> str:
    try:
        return await payments.refund(order_id)  # first call times out, retry succeeds
    except TimeoutError as e:
        raise GenkitError(status='DEADLINE_EXCEEDED', message='payments API timed out after 10s') from e

# 1. Your own handler takes the Failure value
def page_on_call(failure: Failure) -> None:
    if failure.reason == ErrorReason.TOOL_FAILED and failure.status == 'DEADLINE_EXCEEDED':
        pager.notify(team='payments', summary=failure.message)
    elif failure.status == 'INTERNAL':
        pager.notify(team='platform', summary='refund turn failed')

try:
    # 2. lookup_order succeeds, then issue_refund fails
    response = await ai.generate(
        prompt='Refund order ORD-9921.',
        tools=[lookup_order, issue_refund],
    )

    if response.finish_reason == FinishReason.FAILED:
        print(response.error.status, response.error.reason, response.error.message)
        # => DEADLINE_EXCEEDED TOOL_FAILED payments API timed out after 10s
        page_on_call(response.error)

        # 3. History up to the failure is intact: the lookup isn't lost
        print([m.role for m in response.messages])
        # => ['user', 'model', 'tool']

        # 4. Restart from that history; the model doesn't redo the lookup
        response = await ai.generate(
            messages=response.messages,
            tools=[lookup_order, issue_refund],
        )

    print(response.text)
    # => Refunded $31.50 for ORD-9921.

except GenkitError as e:
    # Only setup errors raised before the turn starts (bad config, unknown model).
    # A failed turn comes back as finish_reason == FAILED, not an exception.
    print(e.status, e.message)
```

### Handling Interrupt in Tools

`Interrupt` subclasses `BaseException`, so a plain `except Exception` won't catch it.

When a tool raises `Interrupt`, `generate` stops the turn and hands it back as a value: `finish_reason == INTERRUPTED`, with the paused call on `response.interrupts`. Nothing is raised to your code.

In your tool, that means you can raise `Interrupt` inside a `try` with a broad `except Exception` and not worry about it. The handler skips it, so it still reaches `generate` and the turn pauses.

```python
@ai.tool
async def charge_card(amount: int) -> str:
    try:
        if amount > 500:
            raise Interrupt({'needs': 'manager approval', 'amount': amount})
        return await payments.charge(amount)
    except Exception as e:
        # Payment errors land here; Interrupt passes through to generate
        raise GenkitError(status='UNAVAILABLE', message='payments down') from e
```

---

## 4. Flows (`ai.flow`)

### Structured Input & Output

Flows encapsulate typed boundaries using Pydantic models for inputs and outputs.

```python
from pydantic import BaseModel
from genkit import Genkit, FlowContext

ai = Genkit()

class ClaimRequest(BaseModel):
    claim_id: str
    amount: float
    description: str

class ClaimDecision(BaseModel):
    approved: bool
    review_required: bool
    notes: str

# 1. Define flow with typed boundaries
@ai.flow
async def evaluate_claim(request: ClaimRequest) -> ClaimDecision:
    # 2. Invoke generation within flow span
    res = await ai.generate(
        prompt=f'Evaluate claim {request.claim_id} for ${request.amount}: {request.description}',
        output_schema=ClaimDecision,
    )
    return res.output
```

### Streaming Flow Responses

The chunk type comes from the `FlowContext[ChunkT]` annotation, so `send_chunk` and the caller's `async for` are both typed.

```python
from typing import Literal
from pydantic import BaseModel
from genkit import FlowContext

class ReportChunk(BaseModel):
    stage: Literal['fetching', 'drafting', 'done']
    text: str

class Report(BaseModel):
    quarter: str
    summary: str

# 1. Flow typed as Flow[str, Report, ReportChunk]
@ai.flow
async def quarterly_report(quarter: str, ctx: FlowContext[ReportChunk]) -> Report:
    ctx.send_chunk(ReportChunk(stage='fetching', text=f'Pulling {quarter} ledger'))

    # 2. Forward model tokens as typed chunks
    summary = ''
    stream = ai.generate_stream(prompt=f'Draft the {quarter} executive summary.')
    async for chunk in stream.stream:
        summary += chunk.text
        ctx.send_chunk(ReportChunk(stage='drafting', text=chunk.text))

    ctx.send_chunk(ReportChunk(stage='done', text=''))
    return Report(quarter=quarter, summary=summary)

# 3. Caller: chunks are ReportChunk, response is Report
stream = quarterly_report.stream('Q3')
async for chunk in stream.stream:
    print(chunk.stage, chunk.text)
    # => fetching Pulling Q3 ledger
    # => drafting Revenue grew 12%
    # => drafting  quarter over quarter...
    # => done

report = await stream.response
print(report.quarter, report.summary)
# => Q3 Revenue grew 12% quarter over quarter...
```

---

## 5. Prompts (`.prompt` files and `ai.define_prompt`)

A prompt is a template plus its generation settings (model, config, tools, output schema) behind one callable `Prompt[In, Out]`. Write it as a `.prompt` file (Dotprompt: YAML frontmatter plus a Handlebars template) or in code with `ai.define_prompt`. Both give the same object, called the same way.

### `.prompt` files and lazy loading

Put `.prompt` files in `./prompts`, or point `Genkit(prompt_dir=...)` somewhere else. Genkit registers every file under it. The file path becomes the name: `prompts/menu/card.prompt` is `menu/card`. `card.formal.prompt` is the `formal` variant of `card`, and `_footer.prompt` is a partial you include with `{{> footer}}`.

Loading is lazy. `Genkit()` only parses the files; each prompt resolves its model, schemas and defaults the first time it runs, and caches the result. So startup doesn't pay for prompts you never call, and a schema you register with `ai.define_schema` after `Genkit()` still works.

`ai.prompt('name')` returns a handle right away. A typo doesn't fail until you call it, with `NOT_FOUND`. Pass `output_schema=` so the type checker knows what `response.output` is.

`prompts/menu_card.prompt`:

```handlebars
---
model: googleai/gemini-flash-latest
config:
  temperature: 0.4
input:
  schema:
    dish: string
    diet?: string
  default:
    diet: none
output:
  schema: MenuCard
---
Write a menu card for {{dish}}. Dietary restriction: {{diet}}.
```

```python
from pydantic import BaseModel
from genkit import Genkit
from genkit_google_genai import GoogleAI

class MenuCard(BaseModel):
    title: str
    blurb: str
    allergens: list[str]

# 1. The file names MenuCard; registering it after Genkit() works because files resolve on first use
ai = Genkit(plugins=[GoogleAI()], prompt_dir='prompts')
ai.define_schema('MenuCard', MenuCard)

# 2. Look the prompt up; output_schema types response.output
menu_card = ai.prompt('menu_card', output_schema=MenuCard)

# 3. Call it; `diet` takes the file default
response = await menu_card({'dish': 'mushroom risotto'})
print(response.output.title, response.output.allergens)
# => Wild Mushroom Risotto ['dairy']
```

### Input defaults and validation

Omitted input keys are filled before the template runs, in this order: what the caller passed, then the file's `input.default`, then the Pydantic field default. The result is validated against the input schema (the file's, and the one passed to `ai.prompt`/`define_prompt` if any), so a missing required field fails with `INVALID_ARGUMENT` before the model is called.

### Defining a prompt in code

```python
from genkit_google_genai import GeminiConfig

class DishInput(BaseModel):
    dish: str
    diet: str = 'none'  # in code, defaults are Pydantic field defaults

# 1. Same settings as the file, as keyword arguments
menu_card = ai.define_prompt(
    'menu_card_inline',
    model=GoogleAI.gemini('gemini-flash-latest'),
    config=GeminiConfig(temperature=0.4),
    input_schema=DishInput,
    output_schema=MenuCard,
    prompt='Write a menu card for {{dish}}. Dietary restriction: {{diet}}.',
)

# 2. Called the same way as a file prompt
response = await menu_card(DishInput(dish='ramen'))
print(response.output.title)
# => Tonkotsu Ramen
```

`ai.define_partial(name, source)` and `ai.define_helper(name, fn)` register Handlebars partials and helpers for both kinds.

### Per-call overrides

Call keywords use `ai.generate`'s names. An omitted keyword keeps the prompt's value; `config` merges per key; `tools=[]`, `use=[]`, and `docs=[]` clear the prompt's list for this call. The output schema is fixed by the prompt.

```python
# 1. Warmer sampling and one extra tool, for this call only
response = await menu_card({'dish': 'ramen'}, config={'temperature': 0.9}, tools=[price_lookup])

# 2. A variant from menu_card.formal.prompt
formal = ai.prompt('menu_card', variant='formal', output_schema=MenuCard)
```

### Streaming and rendering

`prompt.stream()` takes the same arguments and returns a `ModelStreamResponse`, like `ai.generate_stream()`. `prompt.render()` returns the request it would send, without calling the model.

```python
# 1. Stream the reply
stream = menu_card.stream({'dish': 'ramen'})
async for chunk in stream.stream:
    print(chunk.text, end='')
card = (await stream.response).output

# 2. Inspect the rendered request without running it
request = await menu_card.render({'dish': 'ramen'})
print(request.messages[0].content[0].text)
# => Write a menu card for ramen. Dietary restriction: none.
```

---

## 6. Web Serving (`genkit.web`) & Error Redaction

There are four levels. Each one down gives you more of the route and leaves less to Genkit, and all four speak the same wire format and redact errors the same way.

1. `serve_flow`: Genkit owns the whole route.
2. `handle_genkit_request`: you write the route and its `Depends`; Genkit parses the body, picks JSON or SSE, and formats errors.
3. `stream.to_sse()`: you write the route and decide what streams; Genkit only frames the SSE.
4. `genkit.web` helpers: you write everything, and the helpers keep you on the wire format. Use this for Flask, Django, Starlette, or your own adapter.

Wire format, shared by all four:

```text
POST {"data": <input>}
  JSON  => {"result": <output>}
  SSE   => data: {"message": <chunk>}   (repeated)
           data: {"result": <output>}
  error => {"message": ..., "status": ...}  or  data: {"error": {...}}  mid-stream
```

### Shared flow

```python
from fastapi import Depends, FastAPI, Header
from genkit import FlowContext, Genkit, PublicError
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))
app = FastAPI()

# 1. FastAPI dependency that resolves the caller
async def current_user(authorization: str = Header()) -> dict[str, object]:
    return {'user_id': verify_token(authorization)}

# 2. Flow streams status lines and returns an ETA
@ai.flow
async def track_order(order_id: str, ctx: FlowContext[str]) -> str:
    if not order_id.startswith('ORD-'):
        raise PublicError('INVALID_ARGUMENT', f'{order_id} is not an order id')
    ctx.send_chunk(f'[{ctx.context["user_id"]}] locating {order_id}')
    ctx.send_chunk('courier is 2 stops away')
    return 'ETA 14:20'
```

### Level 1: `serve_flow`

One line per flow. `context_dependency` reuses your `Depends` auth; its value becomes `ctx.context`. The client picks JSON or SSE with `Accept: text/event-stream` or `?stream=true`.

```python
from genkit_fastapi import serve_flow

# 1. Mount the flow at POST /api/track_order
app.include_router(serve_flow(track_order, context_dependency=current_user), prefix='/api')

# 2. JSON
# POST /api/track_order  {"data": "ORD-9921"}
# => {"result": "ETA 14:20"}

# 3. SSE (Accept: text/event-stream)
# => data: {"message": "[u_42] locating ORD-9921"}
# => data: {"message": "courier is 2 stops away"}
# => data: {"result": "ETA 14:20"}
```

### Level 2: `handle_genkit_request`

Your route, your `Depends`, your path. Genkit still reads `{"data": ...}`, negotiates JSON vs SSE, and formats errors.

```python
from fastapi import Request
from genkit_fastapi import handle_genkit_request

# 1. Route signature and dependencies are yours
@app.post('/orders/track', response_model=None)
async def track(request: Request, user: dict[str, object] = Depends(current_user)):
    # 2. Pass the resolved context; same JSON/SSE behavior as serve_flow
    return await handle_genkit_request(request, action=track_order, context=user)
```

### Level 3: `stream.to_sse()`

You decide the input shape and what streams. `.to_sse()` on any `StreamResponse` (a flow, `ai.generate_stream()`, or a tool) or `AgentTurn` frames chunks, the final result, and mid-stream errors as SSE events. A failure mid-stream ends with one `error` frame and no `result`.

```python
from fastapi.responses import StreamingResponse

# 1. Typed query param instead of {"data": ...}; always SSE
@app.get('/orders/{order_id}/live')
async def live(order_id: str, user: dict[str, object] = Depends(current_user)):
    stream = track_order.stream(order_id, context=user)
    return StreamingResponse(stream.to_sse(), media_type='text/event-stream')

# 2. Stream a model call directly, no flow
@app.post('/support/chat')
async def chat(prompt: str, user: dict[str, object] = Depends(current_user)):
    stream = ai.generate_stream(prompt=prompt, context=user)
    return StreamingResponse(stream.to_sse(), media_type='text/event-stream')
    # => data: {"message": {"content": [{"text": "Your order"}], ...}}
    # => data: {"result": {"message": {...}, "finishReason": "stop", ...}}
```

### Level 4: `genkit.web` primitives

No adapter. Use this for another framework or when you're writing an adapter. Four helpers cover the whole contract.

```python
from starlette.requests import Request
from starlette.responses import JSONResponse
from genkit.web import error_body, error_status, read_body, wants_stream

async def track_endpoint(request: Request):
    try:
        # 1. {"data": x} -> x; {} or {"data": null} -> no input
        order_id = read_body(await request.json())

        # 2. Build the flow's context however your framework does auth
        context = {'user_id': verify_token(request.headers.get('authorization'))}

        # 3. Accept: text/event-stream or ?stream=true
        if wants_stream(accept=request.headers.get('accept'), stream=request.query_params.get('stream')):
            stream = track_order.stream(order_id, context=context)
            return StreamingResponse(stream.to_sse(), media_type='text/event-stream')

        result = await track_order.run(order_id, context=context)
        return JSONResponse({'result': result.response})
    except Exception as e:
        # 4. Redacts everything but PublicError
        return JSONResponse(error_body(e), status_code=error_status(e))
```

### Per-tenant API keys

A tenant's provider key goes in one place: `context={'secrets': {'api_key': ...}}`. The Google GenAI models read it from there and fall back to the plugin's key when it's absent.

Put it anywhere else (in `config=`, or at the top level of `context`) and the call fails with `INVALID_ARGUMENT`, pointing you at `context.secrets`. Inside a flow, `ai.generate()` without `context=` inherits the flow's context, so tools and nested calls get the same key. The Dev UI shows `secrets` and `auth` as `<redacted>`.

The key isn't stored on a background job's `Operation`, so pass it again on every `check_operation` and `cancel_operation`.

```python
from fastapi import Depends, Header, Request
from fastapi.responses import StreamingResponse
from genkit import Operation
from genkit_fastapi import handle_genkit_request
from genkit_google_genai import GoogleAI

# 1. One dependency resolves the tenant and builds the context Genkit expects
async def tenant_context(authorization: str = Header()) -> dict[str, object]:
    tenant = await tenants.from_token(authorization)
    return {
        'auth': {'tenant_id': tenant.id},
        'secrets': {'api_key': await secret_manager.get(f'gemini-key/{tenant.id}')},
    }

# 2. Level 2: the flow's ai.generate() inherits the context, key included
@app.post('/orders/track', response_model=None)
async def track(request: Request, ctx: dict[str, object] = Depends(tenant_context)):
    return await handle_genkit_request(request, action=track_order, context=ctx)

# 3. Level 3: call ai.generate_stream() directly and pass the context yourself
@app.post('/support/chat')
async def chat(prompt: str, ctx: dict[str, object] = Depends(tenant_context)):
    stream = ai.generate_stream(prompt=prompt, context=ctx)
    return StreamingResponse(stream.to_sse(), media_type='text/event-stream')

# 4. Background job: start with the tenant's key...
veo = GoogleAI.veo('veo-3.1-generate-001')

@app.post('/promos')
async def start_promo(prompt: str, ctx: dict[str, object] = Depends(tenant_context)) -> Operation:
    return await ai.generate_operation(model=veo, prompt=prompt, context=ctx)
    # => {"id": "operations/abc123", "done": false}

# 5. ...and poll with it again; the Operation doesn't carry it
@app.post('/promos/poll')
async def poll_promo(op: Operation, ctx: dict[str, object] = Depends(tenant_context)) -> Operation:
    return await ai.check_operation(op, context=ctx)
    # => {"id": "operations/abc123", "done": true, "output": {...}}

# Wrong slot: rejected before the provider is called
await ai.generate(prompt='hi', context={'api_key': key})
# => INVALID_ARGUMENT: API key belongs in context.secrets, not config or the top-level context.
#    Pass the key as context={'secrets': {'api_key': ...}}.
```

### Error redaction

In process, you get the real exception with its type, `__cause__`, and traceback. Over HTTP, `error_body` / `error_status` decide what leaves the server, at every level above:

- A `PublicError` goes out as written: its message, status and details, with the matching HTTP code.
- Input that fails the flow's schema is a 400 that names the failing fields but not the values, e.g. `Invalid input for flow 'place': order_id: Field required`. A nested action called with bad input is a server bug, so that's a 500.
- Everything else (`GenkitError`, provider errors, `RuntimeError`) is `500 {"message": "Internal Error", "status": "INTERNAL"}`. The real message stays in your logs and trace.

```python
@ai.flow
async def refund_order(order_id: str) -> str:
    if order_id == 'ORD-0000':
        # 1. Author says this text is safe for the caller
        raise PublicError('NOT_FOUND', f'{order_id} does not exist')
    # 2. Internal detail; never leaves the server
    raise RuntimeError('ledger-primary: connection pool exhausted')

# In process: the real exception
try:
    await refund_order('ORD-9921')
except RuntimeError as e:
    print(e)
    # => ledger-primary: connection pool exhausted

# Over HTTP, JSON
# POST {"data": "ORD-0000"}
# => 404 {"message": "ORD-0000 does not exist", "status": "NOT_FOUND"}
# POST {"data": "ORD-9921"}
# => 500 {"message": "Internal Error", "status": "INTERNAL"}

# Over HTTP, SSE: chunks already sent stay sent; the stream ends with one error frame
# => data: {"message": "..."}
# => data: {"error": {"message": "Internal Error", "status": "INTERNAL"}}
```

---

## 7. Telemetry & OpenTelemetry

`genkit_otel` emits `gen_ai.*` spans and metrics per the OTel GenAI semantic conventions. Your app owns the OTel SDK and exporter. Configure the instrumentation before `Genkit()` so every action registered after it is covered. It runs alongside the Dev UI trace pipeline; the two export separately.

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from genkit import Genkit
from genkit.telemetry import configure_instrumentation
from genkit_google_genai import GoogleAI
from genkit_otel import GenAiInstrumentation

# 1. App owns the OTel SDK
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint='http://localhost:4318/v1/traces')))
trace.set_tracer_provider(provider)

# 2. Configure instrumentation before creating Genkit; prompts and outputs stay off spans by default
configure_instrumentation(GenAiInstrumentation(content_capturing_mode='NO_CONTENT'))

# 3. Then create the app
ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))

# 4. Flows and model calls produce nested spans (tool spans: emit_tool_spans=True)
@ai.flow
async def audit_vendor(vendor_id: str) -> str:
    response = await ai.generate(prompt=f'Run a compliance scan for {vendor_id}.')
    return response.text

await audit_vendor('VEN-118')
# => span audit_vendor
#      span chat gemini-flash-latest  (kind=CLIENT)
#        gen_ai.operation.name=chat
#        gen_ai.request.model=gemini-flash-latest
#        gen_ai.usage.input_tokens=14
#        gen_ai.usage.output_tokens=52
```

Set `content_capturing_mode='SPAN_ONLY'` (or `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`) to record prompts and outputs on spans. An explicit `'NO_CONTENT'` wins over the env var, so it works as a PII opt-out.

---

## 8. Embedders

Embedders vectorize documents for retrieval. Register one with `ai.define_embedder(name, fn)` (or return an `Embedder` from a plugin's `list_actions()`), and call it with `ai.embed` or `ai.embed_many`.

Like every action fn, the embedder fn takes the action's input, an `EmbedRequest`. `EmbedRequest[Cfg]` types `req.options`: `ai.embed(config=...)` arrives validated, and the Dev UI renders `Cfg` as a form. A bare `EmbedRequest` gives you a dict.

```python
from pydantic import BaseModel
from genkit import Embedding, Genkit
from genkit.embedder import EmbedRequest, EmbedResponse

ai = Genkit()

# 1. Options class; defaults apply when the caller passes no config
class CrmEmbedConfig(BaseModel):
    dimensions: int = 3

# 2. EmbedRequest[CrmEmbedConfig] types req.options
async def crm_vectors(req: EmbedRequest[CrmEmbedConfig]) -> EmbedResponse:
    vector = [0.012, 0.451, -0.231, 0.118][: req.options.dimensions]
    return EmbedResponse(embeddings=[Embedding(embedding=vector) for _ in req.input])

ai.define_embedder('crm/vectors', crm_vectors)

# 3. Embed with config; a dict is validated into CrmEmbedConfig
result = await ai.embed(
    embedder='crm/vectors',
    content='Search indexed warranty policies',
    config={'dimensions': 2},
)
print(result[0].embedding)
# => [0.012, 0.451]

# 4. Bad config fails before the fn runs
await ai.embed(embedder='crm/vectors', content='...', config={'dimensions': 'two'})
# => INVALID_ARGUMENT: Invalid input for embedder 'crm/vectors': options.dimensions: Input should be a valid integer
```

---

## 9. Evaluators

Evaluators score dataset rows against reference outputs and contexts. Register a per-row evaluator with `ai.define_evaluator(name, fn, ...)` or a whole-dataset evaluator with `ai.define_batch_evaluator(name, fn, ...)`, and run it with `ai.evaluate`.

Evaluator fns follow the same rule as every other action: they take a `Request[Cfg]`. A per-row fn takes `EvaluatorRequest[Cfg]` (`req.row`, `req.options`) and returns one `EvaluatorResponse`. A batch fn takes `BatchEvaluatorRequest[Cfg]` (`req.dataset`, `req.options`) and returns one per row.

Genkit reads `Cfg` from the annotation (or `config_schema=`; given both, they must match), validates `ai.evaluate(config=...)` into it, and renders it as a Dev UI form. A bare `EvaluatorRequest` gives you `options` as a dict. `BatchEvaluatorRequest` is the wire type; the JSON is unchanged.

### Per-row evaluator

```python
from pydantic import BaseModel
from genkit import Genkit
from genkit.evaluator import BaseDataPoint, EvaluatorRequest, EvaluatorResponse, Score, ScoreDetails, ScoreStatus

ai = Genkit()

# 1. Options class; defaults apply when the caller passes no config
class RelevanceConfig(BaseModel):
    keyword: str = 'approved'

# 2. Score one row; req.options is a validated RelevanceConfig
async def relevance_evaluator(req: EvaluatorRequest[RelevanceConfig]) -> EvaluatorResponse:
    row, options = req.row, req.options
    hit = options.keyword in str(row.output).lower()
    return EvaluatorResponse(
        test_case_id=row.test_case_id or '',
        evaluation=[
            Score(
                score=1.0 if hit else 0.0,
                status=ScoreStatus.PASS if hit else ScoreStatus.FAIL,
                details=ScoreDetails(reasoning=f'Checked for {options.keyword!r}'),
            )
        ],
    )

# 3. Register; RelevanceConfig comes from the annotation
ai.define_evaluator(
    'custom/relevance',
    relevance_evaluator,
    display_name='Relevance',
    definition='Whether the output contains the expected keyword.',
)

# 4. Run with typed config; config={'keyword': 'refunded'} works too
rows = await ai.evaluate(
    evaluator='custom/relevance',
    dataset=[
        BaseDataPoint(
            test_case_id='claim-1',
            input='What is the claim status?',
            output='Claim was refunded in full',
        )
    ],
    config=RelevanceConfig(keyword='refunded'),
)
print(rows[0].evaluation[0].score, rows[0].evaluation[0].details.reasoning)
# => 1.0 Checked for 'refunded'
```

### Batch evaluator

A batch fn gets the whole dataset in one `BatchEvaluatorRequest[Cfg]`, so it can batch calls to a judge model. `req.options` is the same validated config.

```python
from genkit.evaluator import BatchEvaluatorRequest

# 1. One call per dataset; req.options is a RelevanceConfig
async def relevance_batch(req: BatchEvaluatorRequest[RelevanceConfig]) -> list[EvaluatorResponse]:
    return [
        EvaluatorResponse(
            test_case_id=row.test_case_id or '',
            evaluation=[Score(score=1.0 if req.options.keyword in str(row.output).lower() else 0.0)],
        )
        for row in req.dataset
    ]

ai.define_batch_evaluator(
    'custom/relevance_batch',
    relevance_batch,
    display_name='Relevance (batch)',
    definition='Whether each output contains the expected keyword.',
)

# 2. A dict config is validated into RelevanceConfig
rows = await ai.evaluate(
    evaluator='custom/relevance_batch',
    dataset=[
        BaseDataPoint(test_case_id='claim-1', output='Claim was refunded in full'),
        BaseDataPoint(test_case_id='claim-2', output='Claim is under review'),
    ],
    config={'keyword': 'refunded'},
)
print([row.evaluation[0].score for row in rows])
# => [1.0, 0.0]
```

---

## 10. Middleware

Middleware runs code around a generate call. A hook gets the params, can change them, calls `next_fn` to run the rest of the chain, and can change the result on the way back.

- `wrap_generate`: each iteration of the tool loop.
- `wrap_model`: each model call. `params.request` is the `ModelRequest`.
- `wrap_tool`: each tool call.
- `tools(ctx)`: extra tools to give the model for this call.

```python
from pydantic import BaseModel
from genkit import Message, Part
from genkit.middleware import Middleware, MiddlewareContext, ModelHookParams

# 1. Config is a Pydantic model; the Dev UI renders it as a form
class AllergyConfig(BaseModel):
    allergen: str

# 2. Middleware[Config] gives you self.config, typed
@ai.middleware(name='allergy_guard')
class AllergyGuard(Middleware[AllergyConfig]):
    async def wrap_model(self, params: ModelHookParams, ctx: MiddlewareContext, next_fn):
        # 3. Before: add a system message to the request
        note = Message(role='system', content=[Part.from_text(f'The guest is allergic to {self.config.allergen}.')])
        params.request.messages.insert(0, note)

        # 4. Run the model (or the next middleware in use=[...])
        response = await next_fn(params, ctx)

        # 5. After: read or rewrite the response
        print(f'[allergy_guard] {response.text}')
        return response

# 6. Without it vs with it
await ai.generate(prompt='Suggest a dish.')
# => Pad Thai

await ai.generate(prompt='Suggest a dish.', use=[AllergyGuard(allergen='peanuts')])
# => [allergy_guard] Green Curry
# => Green Curry
```

With `use=[A(), B()]`, `A` runs first and wraps `B`: A before, B before, model, B after, A after. `@ai.middleware` registers the class so the Dev UI lists it; plugins ship middleware from `Plugin.list_middleware()`.

`MiddlewareContext` is shared by every middleware in one `generate` call:

- `ctx.ai`: the app's `Genkit` instance. This is the escape hatch; there's no registry handle. It's typed for lookups: `await ctx.ai.lookup_model(name)`, `await ctx.ai.lookup_tool(name)`, `ctx.ai.lookup_value(kind=..., name=...)`.
- `ctx.context`: a copy of the call's `context=` (auth, `secrets`), the same dict flows and tools read as `ctx.context`. Tools called in this turn get it too.
- `ctx.on_chunk`, `ctx.is_streaming`, `ctx.send_chunk(chunk)`: the caller's stream. `ctx.replace_on_chunk(fn)` swaps the callback to rewrite chunks and returns the old one, so restore it in a `finally`.
- `ctx.abort_signal`: an `asyncio.Event` set when the call is cancelled.
- `ctx.telemetry_labels`: labels added to the spans.

---

## 11. Runtime Architecture: Actions, Reflection, Event Loops

### Actions and composites

Every capability is registered as one or more [`Action`](../packages/genkit/src/genkit/_core/_action.py)s: a typed async function with one input, one output, optional chunks, and a registry key. Plugins register theirs at init or resolve them lazily on first use.

The fn you write takes one request and returns a response. A model fn is `ModelRequest[Cfg] -> ModelResponse`, an embedder fn is `EmbedRequest[Cfg] -> EmbedResponse`, and a flow or tool fn takes its own input. Evaluators take `EvaluatorRequest[Cfg]` (one row) or `BatchEvaluatorRequest[Cfg]` (the dataset). Config is always typed as `Request[Cfg]`.

The public classes split by how many actions they are:

- One action each. These subclass `Action`, and calling one is `(await obj.run(x)).response`:
  - `Model(Action[ModelRequest, ModelResponse, ModelResponseChunk])`
  - `Embedder(Action[EmbedRequest, EmbedResponse])`
  - `Evaluator` / `BatchEvaluator(Action[BatchEvaluatorRequest, EvalResponse])`: both actions take the dataset; a per-row fn is called once per row
  - `Flow(Action[In, Out, Chunk])`
  - `Tool(Action[In, Out, Chunk])`: the chunk type comes from `ToolContext[ChunkT]` (see Streaming Tool Chunks). The tool action returns the function's own value, so `await tool(x)` gives a `Quote` or a `str`. Behaves the same as a Flow when executed directly. When the model calls the tool, `generate` turns that value into the tool response the model reads: the value dumped to JSON as `output`, plus `content` parts if the tool returned a `MultipartToolResponse`. The Dev UI runs a companion `tool.v2` action that returns that same envelope.
- Several actions behind one handle, so no single `Action[In, Out]` fits:
  - `BackgroundModel`: `start` (`background-model`), `check` (`check-operation`), and optional `cancel` (`cancel-operation`), each with its own key and types. `.actions` lists them for registration.
  - `Prompt`: a `prompt` action that renders the request and an `executable-prompt` action that runs `generate`. Calling the prompt resolves the model and goes through `generate`.

### Reflection server on a daemon thread

`Genkit()` starts the reflection server when `GENKIT_ENV=dev` (which `genkit start` sets) or `GENKIT_REFLECTION_ENABLED=true`. It runs on its own daemon thread with its own event loop (`asyncio.run` inside the thread), so it works the same under FastAPI, Flask, Django, a script, or no server at all, and never blocks the app's loop. The socket is bound on the calling thread first, so a busy pinned port fails `Genkit()` instead of a thread nobody watches. Production servers without those env vars start no thread.

For scripts, use `ai.run_main(main())`. With reflection on, it keeps the process alive after `main` returns so the Dev UI can keep running flows, until Ctrl+C or SIGTERM. With reflection off, it returns when `main` does, like `asyncio.run`.

### One action, two event loops: `loop_local_client`

This only comes up in dev mode. With the reflection server on, a dev process has two live event loops: your app's, and the reflection daemon thread's. A flow the Dev UI runs executes on the reflection loop; the same flow called from your app runs on yours. Async clients (httpx, database pools, the OpenAI, Anthropic, and google-genai SDKs) bind to the loop they first run on, so one client created at module level and shared by both fails with "attached to a different loop" or hangs. Production servers run one loop and never see it; a test runner that starts a new loop per test can.

If you're writing an app, create short-lived clients inside the flow or tool when you can, so each call uses the loop it runs on.

```python
# 1. Bound to whichever loop first used it; breaks when the Dev UI runs the tool
http = httpx.AsyncClient()

# 2. One client per call; works on either loop
@ai.tool
async def menu_for(restaurant_id: str) -> Menu:
    async with httpx.AsyncClient(timeout=10.0) as http:
        response = await http.get(f'https://api.bistro.dev/menus/{restaurant_id}')
    return Menu.model_validate(response.json())
```

If you're writing a plugin, wrap every long-lived client in `loop_local_client` (`genkit.plugin_api`). It caches one client per running loop, built on first use there and reused after, so model and embedder calls work on both loops. The built-in plugins already do this.

```python
import httpx
from openai import AsyncOpenAI
from genkit.plugin_api import Plugin, loop_local_client

# 1. Module level: a plain transport shared by every plugin instance, one per loop
@loop_local_client
def http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=60.0)

# 2. Per instance: a client built from plugin settings, one per instance per loop
class Bistro(Plugin):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @loop_local_client
    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self._api_key)

# 3. Same loop, same client; the reflection loop gets its own
print(http_client() is http_client())
# => True
```

The cache holds each loop weakly, so a client goes away with its loop. Don't close a shared client. The method form is #6708.
