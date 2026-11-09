# Migrate from Genkit Python 0.12 to 0.13

Genkit Python 0.13 makes the API feel native to Python ahead of 1.0, and it has breaking changes. This guide covers every change you need to make, with old and new code for each. Most apps only touch a few sections. Skim the headings and apply the ones whose old code appears in your project.

## Upgrade the packages

Upgrade `genkit` and every `genkit-*` package to 0.13 together. Keep them all on the same version.

```bash
uv add 'genkit==0.13.*' 'genkit-google-genai==0.13.*'   # plus each genkit-* package you use
```

If you call Claude or Llama through Vertex AI Model Garden, install the extra for it:

```bash
uv add 'genkit-vertexai[anthropic]'   # Claude
uv add 'genkit-vertexai[openai]'      # Llama, Mistral and other OpenAI-compatible models
```

## Using a coding agent

You can point a coding agent at this page:

```text
Migrate this project from Genkit Python 0.12 to 0.13 by following https://github.com/genkit-ai/genkit/blob/main/py/docs/migrating-from-0.12.md.
Apply every section whose old code appears in this codebase, then run the type checker and tests and list anything you couldn't migrate.
```

After migrating, run your type checker (pyright, mypy or ty). Most of the renames in this guide show up there as errors.

## Model and embedder helpers

Model helpers drop the `_model` suffix, and `.embedding()` is now `.embedder()`.

**Old:**

```python
GoogleAI.gemini_model('gemini-flash-latest')
GoogleAI.gemini_image_model('gemini-2.5-flash-image')
GoogleAI.gemini_tts_model('gemini-2.5-flash-preview-tts')
GoogleAI.embedding('gemini-embedding-001')
VertexAI.veo_model('veo-3.1-generate-001')
Anthropic.claude_model('claude-sonnet-4-6')
```

**New:**

```python
GoogleAI.gemini('gemini-flash-latest')
GoogleAI.gemini_image('gemini-2.5-flash-image')
GoogleAI.gemini_tts('gemini-2.5-flash-preview-tts')
GoogleAI.embedder('gemini-embedding-001')
VertexAI.veo('veo-3.1-generate-001')
Anthropic.claude('claude-sonnet-4-6')
```

## Imports

`genkit` now exports only what apps use. Every public name lives in exactly one namespace, and each namespace's `__init__.py` lists its exports in `__all__`. If an import breaks, use the layout and lookup below to find its new home instead of guessing.

**The layout**

```text
genkit              app code: Genkit, Message, Part, Document, FlowContext, ToolContext, GenkitError, Failure, ...
genkit.model        defining or inspecting models: Model, ModelRef, ModelRequest, ModelInfo, ModelSupports, ...
genkit.embedder     defining or inspecting embedders: Embedder, EmbedderRef, EmbedRequest, ...
genkit.evaluator    evaluator types
genkit.middleware   writing middleware
genkit.formats      custom output formats
genkit.plugin_api   writing plugins: Plugin, Action, GenkitConfig, provider_error, ...
genkit.web          serving flows: RequestData, ContextProvider, to_sse
genkit.telemetry    tracing backends
genkit.testing      fake models for tests
genkit.exp.agent    experimental agent APIs
```

The rule: if an application uses it, it's in `genkit`. If you need it to define a model, embedder, evaluator, middleware or format, it's in that resource's namespace. Plugin-only primitives are in `genkit.plugin_api`. Plugin packages export everything from their top-level module, such as `genkit_google_genai`; their submodules are private.

**Find where a name went**

```bash
# list one namespace's exports
python -c "import genkit.model as m; print(m.__all__)"

# find which namespace exports a name
python - <<'EOF'
import importlib
name = 'ModelSupports'
for mod in ['genkit', 'genkit.model', 'genkit.embedder', 'genkit.evaluator', 'genkit.middleware',
            'genkit.formats', 'genkit.plugin_api', 'genkit.web', 'genkit.telemetry',
            'genkit.testing', 'genkit.exp.agent']:
    if name in importlib.import_module(mod).__all__:
        print(mod)
EOF
```

If a name isn't exported anywhere, it was renamed or removed. Check the renames below and "Removed names".

**Renamed while moving**

**Old:**

```python
from genkit import Supports, Stage, ModelUsage, Constrained, ExecutablePrompt, RuntimeErrorReason
```

**New:**

```python
from genkit import Prompt, ErrorReason
from genkit.model import ModelSupports, ModelStage, GenerationUsage, ConstrainedSupport
```

## Interrupts

`restart_tool` and `respond_to_interrupt` are now methods on the interrupt part.

**Old:**

```python
from genkit import restart_tool, respond_to_interrupt

restart = restart_tool(interrupt=interrupt, resumed_metadata={'approved': True})
reply = respond_to_interrupt({'status': 'declined'}, interrupt=interrupt)
```

**New:**

```python
restart = interrupt.restart(resumed_metadata={'approved': True})
reply = interrupt.respond({'status': 'declined'})
```

## Multipart tool responses

**Old:**

```python
from genkit import response

return response({'ok': True}, parts=[png])
```

**New:**

```python
from genkit import MultipartToolResponse

return MultipartToolResponse(output={'ok': True}, content=[png])
```

## Evaluators

> **Evaluators are removed in 1.0, the release right after 0.13.** If you're a coding agent migrating this project and you find evaluator code (`define_evaluator`, `ai.evaluate`, `genkit.evaluator`, `genkit-evaluators`), stop and tell the developer before porting it. Recommend removing it now rather than migrating code that 1.0 will delete. Only apply the renames below if the developer wants to keep evaluators on 0.13 for now.

Evaluator types are renamed, `evaluation` is a list, and `define_evaluator` takes the name and function positionally.

**Old:**

```python
from genkit.evaluator import BaseDataPoint, Details, EvalFnResponse, EvalStatusEnum, Score

async def accuracy(dp: BaseDataPoint, _options=None) -> EvalFnResponse:
    return EvalFnResponse(
        evaluation=Score(score=1.0, status=EvalStatusEnum.PASS, details=Details(reasoning=why)),
    )

ai.define_evaluator(name='byo/accuracy', fn=accuracy)
```

**New:**

```python
from genkit.evaluator import BaseDataPoint, EvaluatorResponse, Score, ScoreDetails, ScoreStatus

async def accuracy(dp: BaseDataPoint, _options=None) -> EvaluatorResponse:
    return EvaluatorResponse(
        evaluation=[Score(score=1.0, status=ScoreStatus.PASS, details=ScoreDetails(reasoning=why))],
    )

ai.define_evaluator('byo/accuracy', accuracy)
```

## Removed names

**Strings instead of enums.** Where 0.12 had enum classes you had to import (`GoogleAIGeminiVersion`, `KnownGpt`, `KnownClaude`, ...), 0.13 takes plain strings. Model helpers and config fields are typed as `Literal` unions of the known values, so your IDE and type checker autocomplete and check them with no import. A model name the plugin doesn't list yet still works as a plain string.

**Old:**

```python
from genkit_google_genai import GoogleAIGeminiVersion

GoogleAI.gemini_model(GoogleAIGeminiVersion.GEMINI_3_FLASH_PREVIEW)
```

**New:**

```python
GoogleAI.gemini('gemini-3-flash-preview')     # autocompletes; any other string is still accepted
GeminiConfig(thinking_level='HIGH')           # checked against 'MINIMAL' | 'LOW' | 'MEDIUM' | 'HIGH'
```

If you're a coding agent, replace each enum member with its string value. The value is the string the enum member held, so you can read it from the 0.12 source or from the `Literal` in the 0.13 signature.

Every removed name, and what to use instead:

- **Model enums** (`KnownGeminiModels`, `KnownGemini*`, `KnownGpt`, `KnownClaude`, `*Version`, `GeminiEmbeddingModels`, `VertexEmbeddingModels`): pass a typed ref such as `GoogleAI.gemini('gemini-flash-latest')` or `OpenAI.gpt('gpt-4o')`, or the model name as a string, such as `'openai/gpt-4o'`.
- **Name helpers** (`bedrock_name`, `anthropic_name`, `ollama_name`, `openai_model`, `genkit_eval_name`, `package_name`): use the full `'plugin/model'` string.
- **`Candidate`**: use the response's `message`.
- **`Document.data`, `Document.data_type`, `Document.from_data()`**: build a `Document` with its constructor.
- **`ai.registry`**: use the lookup methods on `Genkit`.
- **Plugin submodules** such as `genkit_openai.models`: import from the plugin's top-level package.

## Keyword-only arguments

Constructors, plugin constructors, and every option after an action's input are keyword-only. If you see `TypeError: takes N positional arguments`, name the argument.

**Old:**

```python
ai = Genkit([GoogleAI()], GoogleAI.gemini_model('gemini-flash-latest'))
await my_flow.run(data, on_chunk, ctx)
```

**New:**

```python
ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))
await my_flow.run(data, on_chunk=on_chunk, context=ctx)
```

## Plugin constructor arguments

Google Cloud plugins take `project=`, matching the `google-cloud-*` clients. Every plugin that talks to a server takes one `base_url=`.

**Old:**

```python
VertexAI(project_id='my-proj')
Ollama(server_address='http://localhost:11434')
```

**New:**

```python
VertexAI(project='my-proj')
Ollama(base_url='http://localhost:11434')
```

## Config class names

**Old:**

```python
from genkit_google_genai import GeminiConfigSchema, GeminiImageConfigSchema, GeminiTtsConfigSchema, GemmaConfigSchema
```

**New:**

```python
from genkit_google_genai import GeminiConfig, GeminiImageConfig, GeminiTtsConfig, GemmaConfig
```

`EmbeddingTaskType` and `VertexAIEvaluationMetricType` are no longer exported. Pass the string value instead.

## Gemini config

Gemini config is flat. A nested key such as `thinking_config` now raises an error that names the flat field to use.

**Old:**

```python
config={'thinking_config': {'thinking_level': 'HIGH', 'include_thoughts': True}}
config={'google_search_retrieval': True}
config={'speech_config': {'voice_config': {'prebuilt_voice_config': {'voice_name': 'Kore'}}}}
```

**New:**

```python
config=GeminiConfig(thinking_level='HIGH', include_thoughts=True)
config=GeminiConfig(google_search=True)
config=GeminiTtsConfig(voice='Kore')
```

## Anthropic config

`AnthropicConfig` is flat. `ThinkingConfig`, `OutputConfig`, `TaskBudget`, `RequestMetadata` and the `*ToolChoice` types are removed.

**Old:**

```python
config={'thinking': {'enabled': True, 'budgetTokens': 1024}}
```

**New:**

```python
config=AnthropicConfig(thinking='enabled', thinking_budget=1024)

# effort, task_budget and user_id are flat fields too
config=AnthropicConfig(thinking='adaptive', effort='high', task_budget=20000, user_id='diner-42')
```

## Tool choice

Tool choice moves out of provider config and onto `generate`.

**Old:**

```python
await ai.generate(prompt=p, tools=[lookup], config={'tool_choice': {'type': 'any'}})
```

**New:**

```python
await ai.generate(prompt=p, tools=[lookup], tool_choice='required')   # 'auto' | 'required' | 'none'
```

## Provider settings without a field

Settings a config class doesn't declare go in `extra`. This applies to Bedrock and Gemini Interactions models.

**Old:**

```python
config={'some_provider_option': 1}
```

**New:**

```python
config=BedrockConfig(extra={'some_provider_option': 1})
```

`OpenAIConfig` no longer has a `user` field.

## options= on embed and evaluate

**Old:**

```python
await ai.embed(embedder=e, content=doc, options={'task_type': 'RETRIEVAL_DOCUMENT'})
result = await ai.evaluate(evaluator=ev, dataset=rows, options={...})
```

**New:**

```python
await ai.embed(embedder=e, content=doc, config={'task_type': 'RETRIEVAL_DOCUMENT'})
results = await ai.evaluate(evaluator=ev, dataset=rows, config={...})   # now returns a list
```

## Unknown config keys

An unknown config key now raises before the model is called. In 0.12 it was silently dropped, so this can surface typos you didn't know about.

```python
await ai.generate(prompt=p, config={'temprature': 0.2})
# raises GenkitError (INVALID_ARGUMENT) naming 'temprature'
```

Config defaults merge key by key: the model ref's default, then the prompt's config, then the call's. Pass `None` to clear an inherited value.

## API keys

An API key in model `config` now raises in every plugin. In 0.12, Google GenAI already required `context.secrets`, but Ollama silently dropped a key in config and Anthropic could ignore it. If your code passed a key in config to OpenAI, Anthropic or Ollama, move it to the plugin or to `context`.

**Old:**

```python
await ai.generate(model=m, prompt=p, config={'api_key': user_key})
```

**New:**

```python
# one key for the whole app
ai = Genkit(plugins=[OpenAI(api_key=os.environ['OPENAI_API_KEY'])])

# or a different key per request
await ai.generate(model=m, prompt=p, context={'secrets': {'api_key': user_key}})
```

Vertex AI Model Garden authenticates with ADC and rejects a per-request key. `ai.embed()` takes no `context`, so embedders always use the plugin's key.

## Flow contexts

`ActionRunContext` is replaced by a typed context for each kind of action. The chunk type moves from `@ai.flow(chunk_type=...)` into the context's type parameter.

**Old:**

```python
from genkit import ActionRunContext

@ai.flow(chunk_type=Status)
async def run(n: int, ctx: ActionRunContext) -> str:
    ctx.send_chunk(Status(step='search'))
```

**New:**

```python
from genkit import FlowContext

@ai.flow()
async def run(n: int, ctx: FlowContext[Status]) -> str:
    ctx.send_chunk(Status(step='search'))
```

## Tool contexts

Tools can now stream typed chunks too.

**Old:**

```python
from genkit import ToolRunContext

@ai.tool()
async def prepare(dish: str, ctx: ToolRunContext) -> str: ...
```

**New:**

```python
from genkit import ToolContext

@ai.tool()
async def prepare(dish: str, ctx: ToolContext[Progress]) -> str:
    ctx.send_chunk(Progress(stage='searing', percent=40))
```

## Model contexts

If you implement a model yourself:

**Old:**

```python
from genkit import ActionRunContext

async def generate(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse: ...
```

**New:**

```python
from genkit.model import ModelContext

async def generate(request: ModelRequest, ctx: ModelContext) -> ModelResponse: ...
```

Genkit finds the context by its type annotation, and an action takes exactly one input argument. The new contexts subclass `ActionRunContext`, so an old annotation still type-checks; only the import from `genkit` breaks.

## Cancellation

`ctx.abort_signal` is removed. Cancellation is now standard `asyncio.CancelledError`, and it reaches the model call, any retry backoff, and every running tool.

**Old:**

```python
@ai.tool()
async def slow(x: dict, ctx: ToolRunContext) -> dict:
    while not done:
        if ctx.abort_signal.is_set():
            raise GenkitError(status='ABORTED', message='aborted')
        await step()
```

**New:**

```python
@ai.tool()
async def slow(x: dict) -> dict:
    while not done:
        await step()   # cancellation raises asyncio.CancelledError here
```

To cancel a call, cancel its task:

```python
task = asyncio.create_task(ai.generate(prompt=p))
task.cancel()
```

To stop a stream early, use `async with` (or call `aclose()`). A bare `break` without it leaves the flow running.

```python
stream = my_flow.stream(x)
async with stream:
    async for chunk in stream:
        if enough(chunk):
            break   # cancels the flow; stream.response raises CANCELLED
```

Cancellation can now land at any `await` in your tool, so put cleanup for partial work in `finally`. If you catch `asyncio.CancelledError` yourself, re-raise it.

```python
@ai.tool()
async def book_trip(trip: Trip) -> Booking:
    hold = await reserve_seats(trip)
    try:
        return await confirm(hold)
    finally:
        if not hold.confirmed:
            await release(hold)
```

## Returned errors

`response.error` (and `.error` on operations and agent results) is now typed `Failure`, a public type you can import. It has the same fields as before: `status`, `message`, `details` and `reason`. The reason enum is renamed from `RuntimeErrorReason` to `ErrorReason`.

**Old:**

```python
from genkit import RuntimeErrorReason

if res.error and res.error.reason == RuntimeErrorReason.INVALID_OUTPUT: ...
```

**New:**

```python
from genkit import ErrorReason, Failure

if res.error and res.error.reason == ErrorReason.INVALID_OUTPUT: ...

def report(err: Failure) -> None: ...
```

## Raised errors

`GenkitError` keeps fewer attributes, and its constructor is keyword-only.

**Old:**

```python
from genkit import RuntimeErrorReason

except GenkitError as e:
    log(e.original_message, e.http_code, e.trace_id, e.cause)
```

**New:**

```python
from genkit import ErrorReason

except GenkitError as e:
    log(e.message, e.status, e.__cause__)
```

## Plugin exception types

Plugin-specific exceptions such as `OllamaConnectionError` are gone. Check the status instead.

**Old:**

```python
except GenkitError as e:
    if isinstance(e.cause, OllamaConnectionError): ...
```

**New:**

```python
except GenkitError as e:
    if e.status == 'UNAVAILABLE': ...
```

## Interrupts and except Exception

`Interrupt` now subclasses `BaseException`, so `except Exception:` no longer catches it. If your code catches interrupts on purpose, catch `Interrupt` by name.

**Old:**

```python
try:
    require_approval(order)
except Exception as e:   # caught Interrupt too
    ...
```

**New:**

```python
try:
    require_approval(order)
except Interrupt:
    raise                    # only if you caught it on purpose
except Exception as e:
    ...
```

## Context providers

`RequestData` moves to `genkit.web` and exposes `method`, `headers` and `input` directly, the same way on Flask, FastAPI and Django.

**Old:**

```python
from genkit.plugin_api import RequestData

async def auth(req: RequestData[HttpRequest]) -> dict:
    return {'user': req.request.META.get('HTTP_AUTHORIZATION')}
```

**New:**

```python
from genkit.web import RequestData

async def auth(req: RequestData) -> dict:
    return {'user': req.headers.get('authorization')}
```

## Errors from served flows

This only affects flows served over HTTP: `genkit_flask_handler`, the FastAPI and Django handlers, or `genkit.web` in your own route. Calling a flow or model in process (`await my_flow.run(...)`, `await ai.generate(...)`) is unchanged, and you still get the full error.

### One error, two views

- **In process,** you get everything: a `GenkitError` with `status`, `message`, `details` and `__cause__`, or your own exception, unwrapped. Your `try`/`except`, middleware, tests and the Dev UI (with stack trace and trace link) all see it.
- **Over HTTP,** the client gets a body built from that error by these rules:

| What was raised | HTTP status | What the client gets |
| --- | --- | --- |
| `PublicError(status, message, details)` | Matches the status, such as 404 for `NOT_FOUND` | Its `status`, `message` and `details` |
| Invalid input to the served flow | 400 | `Invalid input for flow 'save': qty: Input should be a valid integer`: field names and Pydantic's built-in messages, never the values sent |
| Anything else: other `GenkitError`s, provider errors, your own exceptions | 500 | `{"status": "INTERNAL", "message": "Internal Error"}` |

Streaming responses use the same body inside a `data: {"error": {...}}` SSE frame. Every failure is still logged on the server, tagged with the flow name: 5xx errors with the full traceback, 4xx errors as warnings.

In 0.12, the client got the message and details of any `GenkitError`, and the text and stack trace of any other exception.

### Errors your clients rely on

If a client shows an error's message or branches on its status, raise `PublicError` for it.

**Old:**

```python
@ai.flow()
async def get_order(order_id: str) -> Order:
    order = await db.find(order_id)
    if order is None:
        raise GenkitError(status='NOT_FOUND', message=f'Order {order_id} not found')   # 0.13: client gets a 500
    return order
```

**New:**

```python
from genkit import PublicError

@ai.flow()
async def get_order(order_id: str) -> Order:
    order = await db.find(order_id)
    if order is None:
        raise PublicError('NOT_FOUND', f'Order {order_id} not found')   # client gets 404 and this message
    return order
```

Everything in a `PublicError`'s `message` and `details` goes to the client as written, so keep secrets, IDs from other tenants, and internal details out of it.

### Auth in context providers

A `PublicError` raised in a context provider is handled the same way, which is how you return a 401 or 403.

```python
from genkit import PublicError
from genkit.web import RequestData

async def auth(req: RequestData) -> dict:
    token = req.headers.get('authorization')
    if not token:
        raise PublicError('UNAUTHENTICATED', 'Missing bearer token')   # 401
    return {'user': await verify(token)}
```

### Validation messages

For invalid input, the client sees Pydantic's built-in messages. A custom validator's own message shows as `Invalid value`, because it might contain server data. To show the caller your validator's message, raise `PublicError` from the validator.

### Custom routes

If you serve a flow from your own route, use the same rules with `error_body` and `error_status`:

```python
from genkit.web import error_body, error_status

try:
    result = await my_flow.run(data)
except Exception as e:
    return JSONResponse(error_body(e), status_code=error_status(e))
```

> **If you're a coding agent:** find errors raised in served flows and context providers whose message is meant for end users, such as not-found, permission or validation errors, and convert them to `PublicError`. If you're not sure a message is safe to show a caller, ask the developer before making it public.

## Google Cloud telemetry

**Old:**

```python
add_gcp_telemetry(project_id='my-project', force_dev_export=True)
```

**New:**

```python
enable_google_cloud_telemetry(project='my-project')
```

The `FIREBASE_PROJECT_ID` fallback is gone, so set `project=` or `GOOGLE_CLOUD_PROJECT`.

## Custom trace exporters

**Old:**

```python
from genkit.plugin_api import tracer, add_custom_exporter, AdjustingTraceExporter
```

**New:**

```python
from genkit.telemetry import Instrumentation, configure_instrumentation
```

## Imagen

The Imagen catalog is retired. Use a Gemini image model.

**Old:**

```python
GoogleAI.imagen_model('imagen-4.0-generate-001')
```

**New:**

```python
GoogleAI.gemini_image('gemini-2.5-flash-image')
```

## Vertex AI Model Garden

**Old:**

```python
from genkit_vertexai.model_garden import ModelGardenPlugin

ai = Genkit(plugins=[ModelGardenPlugin()])
```

**New:**

```python
from genkit_vertexai import ModelGarden

ai = Genkit(plugins=[ModelGarden()])
```

Install the `[anthropic]` or `[openai]` extra for the models you call (see the top of this guide).

## OpenAI

**Old:**

```python
from genkit_openai import openai_model

model = openai_model('gpt-4o')
```

**New:**

```python
from genkit_openai import OpenAI

model = OpenAI.gpt('gpt-4o')  # typed ref; autocompletes and binds OpenAIConfig
model = 'openai/gpt-4o'       # plain string also works
```

## Ollama

`Ollama()` reads your pulled models and their capabilities from the server, so you no longer list them.

**Old:**

```python
from genkit_ollama import EmbeddingDefinition, ModelDefinition, Ollama

Ollama(
    models=[ModelDefinition(name='llama3.2')],
    embedders=[EmbeddingDefinition(name='nomic-embed-text')],
    server_address=os.getenv('OLLAMA_HOST'),
)
```

**New:**

```python
from genkit_ollama import Ollama

Ollama(base_url=os.getenv('OLLAMA_HOST'))
```

Ollama no longer reads media from local file paths.

## Amazon Bedrock

**Old:**

```python
from genkit_amazon_bedrock import Bedrock, BedrockImageConfig, ModelDefinition

Bedrock(models=[ModelDefinition(name='us.amazon.nova-lite-v1:0')])
```

**New:**

```python
from genkit_amazon_bedrock import Bedrock, StabilityImageConfig

Bedrock(models=['us.amazon.nova-lite-v1:0'])
```

## Flask

`genkit-flask` no longer installs `genkit-google-genai`. Add it yourself if you use it.

```bash
uv add genkit-google-genai
```

## Media size limit

For every provider, media URLs larger than 20 MB now fail instead of downloading.

## Templates in generate

`ai.generate(prompt=...)` and `system=` are sent exactly as written. Handlebars templates only render in `define_prompt` and `.prompt` files. This change raises no error, so search your code for `{{` inside `generate` calls.

**Old:**

```python
await ai.generate(prompt='Say hi to {{name}}')
```

**New:**

```python
greet = ai.define_prompt(name='greet', prompt='Say hi to {{name}}')
await greet({'name': 'Ana'})
```

## Prompt call options

Calling a prompt takes the same keyword names as `generate`. The output schema is fixed by the prompt and can't be overridden per call. Call config merges over the prompt's config instead of replacing it.

```python
recipe = ai.prompt('recipe')
await recipe({'dish': 'pie'}, config={'temperature': 0.2})
```

Prompt input is now validated, and defaults from `.prompt` files are filled in.

## Truncated structured output

If a JSON reply is cut off, `response.output` is now `None` and `response.error` reports `INVALID_OUTPUT`. Before, you could get a partial object. Check for `None` where you read `output`.

```python
res = await ai.generate(prompt=p, output_schema=City)
if res.output is None:
    ...   # the reply was incomplete; see res.error
```

## Flow input and output validation

Flow input and output are now validated against their type annotations. A flow that returns the wrong shape now raises. For served flows, `{"data": null}` means "use the default".

## Exceptions from actions

An action now raises the exception its body raised, unwrapped. If you caught `GenkitError` around `flow.run()` to handle your own exceptions, catch those types directly. `Retry` and `Fallback` only retry failures a plugin classified, such as `UNAVAILABLE`.

## For plugin and middleware authors

Skip these if you only use Genkit in an app.

### Middleware

**Old:**

```python
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext

@ai.middleware(name='pii_redact')
class PiiRedact(BaseMiddleware[Cfg]):
    async def model(self, params, ctx: GenerateMiddlewareContext, next_fn): ...
```

**New:**

```python
from genkit.middleware import Middleware, MiddlewareContext

@ai.middleware
class PiiRedact(Middleware[Cfg]):
    name = 'pii_redact'
    async def wrap_model(self, params, ctx: MiddlewareContext, next_fn): ...
```

Hooks are now `wrap_generate`, `wrap_model`, and `wrap_tool`. A method still named `model` is never called; the class inherits the pass-through `wrap_model` and the request goes to the provider unchanged, with no error.

`tools()` returns `Tool` handles. `GenerateMiddleware`, `MiddlewareRef`, `MiddlewarePlugin` and `new_middleware` are removed.

### Defining models and embedders

**Old:**

```python
from genkit.plugin_api import model, model_ref, background_model, embedder, embedder_ref, ModelConfig
```

**New:**

```python
from genkit.model import Model, ModelRef, BackgroundModel
from genkit.embedder import Embedder, EmbedderRef
from genkit.plugin_api import GenkitConfig
```

### Resolving models

`resolve()` now receives the model ID without the plugin prefix.

**Old:**

```python
async def resolve(self, kind, name):
    model_id = name.removeprefix('myplugin/')
```

**New:**

```python
async def resolve(self, kind, name):
    model_id = name
```

### Provider errors, clients and logging

**Old:**

```python
from genkit.plugin_api import (
    wrap_http_error, from_http_code, parse_retry_after_ms,
    get_cached_client, GENKIT_CLIENT_HEADER, GENKIT_VERSION,
    tracer, add_custom_exporter,
)
```

**New:**

```python
from genkit import get_logger
from genkit.plugin_api import provider_error, loop_local_client, GENKIT_USER_AGENT
from genkit.telemetry import Instrumentation, configure_instrumentation
```

Also removed: `Plugin.model()`, `Plugin.embedder()` and `Supports.context`. `generate_operation` receives only what a background start uses. Custom formatters import `Formatter`, `FormatterConfig` and `FormatHandler` from `genkit.formats`.

## Experimental APIs

APIs under an `exp` module can change in any release.

**Old:**

```python
from genkit.exp import Agent, Session, InMemorySessionStore, AgentError
from genkit_fastapi import serve_agent
from genkit_middleware import Artifacts
```

**New:**

```python
from genkit.exp.agent import Agent, Session, InMemorySessionStore
from genkit_fastapi.exp import serve_agent
from genkit_middleware.exp import Artifacts
```

`AgentError` is removed. A failed turn returns `res.error`, and a call that produces no turn raises `GenkitError`. Tools in a detached turn are cancelled with `await task.abort()` instead of checking `abort_signal`.

## Evaluators

Evaluators still work in 0.13, with the renames above, but this is their last release. Genkit 1.0, the next release after 0.13, removes the evaluator APIs and the `genkit-evaluators` package while evaluators are redesigned across all the Genkit SDKs. The simplest path is to remove evaluator code as part of this migration. If you need it for now, `genkit-evaluators` 0.13.0 requires `genkit<1`, so keep `genkit` on the 0.13 line until the new design ships.

## Check your upgrade

1. Run your type checker. Most renames in this guide surface as errors.
2. Run your tests.
3. Search for `{{` inside `ai.generate(` calls, and for reads of `response.output` that assume it's never `None`. Neither change raises an error.
