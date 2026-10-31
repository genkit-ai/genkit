---
name: contributing-to-genkit-python
description: How the Genkit Python API is put together, the judgment calls a linter can't make for you, how plugin packages are laid out, and how to write the PR. Read this before you add or change anything a user imports, calls, passes in, or gets back, in core or in a plugin.
---

# Contributing to Genkit Python

Read the rest of this file for any change to the public surface. Read these too when they apply:

- **Changing a plugin package**: [references/plugins.md](references/plugins.md). What a plugin exports and how its config is shaped.
- **Changing a generated wire type** (anything in `genkit/_core/_typing.py`): [references/wire-types.md](references/wire-types.md).
- **Opening a PR**: [references/pr-descriptions.md](references/pr-descriptions.md). How to write the title and description.
- **Running `genkit start` or the Dev UI**: [references/dev-ui-alongside-other-agents.md](references/dev-ui-alongside-other-agents.md). How to test your own app when other agents share the machine.

## What makes something Genkit

The public API surface is the product. Keep it as small and tight as you can. Every name, argument and return type you add is something users learn, depend on, and can't easily lose. When someone reviews a change, they mostly ask four things:

- Is the surface still clean?
- Is it as tight as it can be?
- Does it read like Python?
- Is what it opens up safe? It should be thought through for what Genkit is today and where it's going, not just for the case in front of you.

If you can solve the problem without adding to the surface, do that. If you can't, add the smallest thing that leaves room to grow.

Your code stays plain Python. A flow or a tool is a function you wrote with a decorator on it, and Genkit gets its input and output types from the function's annotations. Users shouldn't have to describe the same thing twice, once in Python and again in a schema or wrapper class that Genkit requires.

One way to call any model. `generate` looks the same whichever provider is behind it, and provider-specific features come in through that model's typed config, not a separate API. Switching models should mean changing the ref, not rewriting the call.

Everything you define, you can see and run. Every action is traced, takes and returns JSON, and shows up in the Dev UI where you can run it by hand. If a new feature can't be run from the Dev UI, or shows up in a trace as an opaque blob, it isn't done.

The core is as unopinionated as possible. It runs what you asked for and nothing more: it doesn't retry a failed call, switch models, or pause for approval on its own. Behavior like that comes from middleware the user adds with `use=` (retry, fallback, tool approval) or from plugins. If a feature takes a stance on how an app should behave, it belongs in one of those, not in the core.

Everything you can run is an action. An action is a function with a name, a kind, input and output schemas, and a trace span around every call, which is what lets the Dev UI list it, run it and show what happened. Every runnable thing in Genkit is an `Action`, a subclass of one (`Evaluator`, `BidiAction`), or a class that wraps or composes actions (`Tool`, `Embedder`, `Prompt`). A new feature should fit one of those shapes. If it truly needs a new kind of runnable thing, that's a big decision, so treat it like one.

Safe in production by default. Secrets are redacted, errors are redacted at the edge, and prompt and reply text stay off spans unless someone turns them on.

The same ideas in every language, written the way that language writes them. The wire format, traces and `.prompt` files are shared across SDKs. The Python surface should read like Python.

## Start with the snippet

Before writing any code, write the thing a user would paste into their app. Use a real-ish domain (orders, a menu, a refund, not `foo`) and number the steps. Use comments to explain what each call does, put `# =>` under anything that comes back, and write the ways it can go wrong as comments too:

```python
# 1. Register a tool the model can call
@ai.tool()
async def check_allergens(order: Order) -> AllergenReport:
    return await kitchen.allergens(order.items)

# 2. Generate with the tool; Genkit runs the tool calls and loops until the model answers
response = await ai.generate(
    model=GoogleAI.gemini('gemini-flash-latest'),
    prompt='Is the pad thai safe for a peanut allergy?',
    tools=[check_allergens],
)
print(response.text)
# => No. The pad thai is garnished with crushed peanuts.

# 3. An unknown config key raises before anything is sent
await ai.generate(prompt='...', config={'temprature': 0.2})
# => GenkitError: INVALID_ARGUMENT, names 'temprature'

# 4. If the tool fails mid-turn, generate doesn't raise; the turn comes back failed
#    with the history so far, so the user can retry without paying for it twice
response = await ai.generate(prompt='Is the curry safe?', tools=[check_allergens])
response.finish_reason == FinishReason.FAILED
# => True
```

If the snippet is awkward to write, or a comment has to explain something surprising, the API is awkward. Change the API. Don't explain it in the docs.

## How the pieces fit

Everything goes through `ai`. The registry is private; if you need to look something up, it's `ai.lookup_model`, `ai.lookup_tool`, `ai.lookup_value`. Experimental features live on `genkit.exp.Genkit`, which subclasses the stable one. The stable class never picks up experimental methods, so changing the import is how you opt in.

Modules are split by who's reading them:

- `genkit` is for people building an app.
- `genkit.model`, `genkit.embedder`, `genkit.evaluator` and `genkit.middleware` are for people writing one of those.
- `genkit.plugin_api` is for plugin packages.
- `genkit.testing` holds fake models (`define_echo_model`, `define_scripted_model`) for testing flows, agents and plugins without a provider.
- `genkit.telemetry` is for people wiring traces into a recording backend, and for resetting instrumentation between tests.
- `genkit.web` is for people writing their own server adapter.

A new namespace needs a distinct reader that none of the existing ones serve, not just a group of related names. Every name has exactly one home. If you're tempted to re-export something "for convenience", it's probably in the wrong module.

Everything is an action: one input, one output, maybe chunks. The function a user writes takes one request. If it needs context, it says so with an annotation (`ToolContext[ChunkT]`, `FlowContext[ChunkT]`, `ModelContext`), and typed config comes in through `Request[Cfg]`. Types flow from those annotations.

If something's wrong before the run starts (bad config, unknown model), raise `GenkitError`. Once the run has started, don't raise. Return the response with `finish_reason`, a `Failure` on `.error`, and `messages` the user can send back to pick up where it stopped. They paid for those tokens, so don't throw them away. Interrupts work the same way: they come back as a value.

Streams always look like `.stream` plus `.response`, whether they come from generate, a flow, a tool or a prompt.

Models are referenced through plugin classmethods: `GoogleAI.gemini('gemini-flash-latest')` gives you a `ModelRef[GeminiConfig]`. There's one flat config class per model family. Passing a dict also works, and it gets validated into that same class.

Auth and secrets ride on `context=`. Nested calls inherit it, traces redact it, and it never gets saved anywhere. Over HTTP, the only error text that leaves the server is a `PublicError` the author raised on purpose.

## Judgment calls

Naming for any public symbols and methods is a very important consideration. Use the same name for the same idea everywhere. If two calls disagree, someone will guess wrong. You want to choose intuitive names so that external contributors can understand the API without having to read the docs.

A decorated function should still work when you call it yourself. `await check_allergens(order)` runs the same action Genkit runs when the model calls the tool: the input is validated the same way and the call gets the same trace span. That's what makes a unit test that calls the tool directly worth having. Don't add a path that only runs inside `generate`, or the test checks something the model never hits. If the function does different things depending on whether it's called by a user or by a model, document that in a comment.

Be very conservative about union types on the veneer layer, and about multiple ways to do the same thing in general. Sometimes a second way is justified, but in most cases one way is more intuitive and easier to document. If you go against this, explain why in the PR description.

Config is the example of a good reason. It accepts a dict or the typed config class, because the Dev UI talks to your app through the reflection server, and everything it sends, config included, arrives as JSON. Genkit has to accept the dict, and it validates it into the same class right away, so code past the boundary only ever sees one shape. Every exception to this rule should have a reason that concrete.

Return types should plug into the next call. If a user gets something back from Genkit and the obvious next step is to pass it to another Genkit call, that call should accept it as-is, with no converting or unwrapping. A tool you get back from `ai.lookup_tool()` goes straight into `generate(tools=[...])`.

One input shape per idea. If the constructor accepts a config entry, the method that adds one later accepts the same entry.

Inputs are `Literal`s. Values you match on are enums. If the user passes the value in, type it as a `Literal`: `thinking_level='high'`, `api_type='chat'`. There's no extra type to import, the IDE still autocompletes it, and a typo fails the type check. Use a `StrEnum` only when callers write code that branches on specific values, which almost always means something Genkit hands back: `response.finish_reason == FinishReason.FAILED`, `err.reason == RuntimeErrorReason.TOOL_FAILED`. `StrEnum` keeps comparison against the plain string working. Don't export an enum just so users can pass one of its values in. If the set keeps growing, like model ids, type it as `KnownGeminiModel | str`: users get autocomplete, and a new id works the day it ships.

Keep state as plain data. History, `Operation`, anything a user might store: it should round-trip through JSON.

The call that causes a problem should be the one that reports it. If a bad argument is accepted at setup and only fails three calls later inside `generate`, the user is debugging the wrong line.

Check at the surface, trust inside. Python doesn't enforce annotations, so a user can pass anything, whatever the signature says. Validate and normalize input where it enters Genkit: `generate`, a plugin constructor, a `define_*` call, an action's input. Turn a dict into the typed config and a model name into a ref, and raise right there if it doesn't fit. Past that point, the data matches its types, and downstream code trusts them instead of checking again. If you find yourself adding `isinstance` checks deep inside, the boundary let something through; fix the boundary.

Raise `GenkitError` when something downstream reads the status. That's anything raised while an action runs: model, embedder, tool, evaluator, or Genkit's own code inside `generate`. Retry and Fallback decide what to do from the status, and Retry retries any exception that isn't a `GenkitError`, so a plain `ValueError` for a bad request gets sent again. Pick the status by asking whether trying again could help:

- **`INVALID_ARGUMENT`**: the caller passed something wrong. Not retried.
- **`NOT_FOUND`**: the model, tool or other thing they named doesn't exist.
- **`FAILED_PRECONDITION`**: the input is fine, but something isn't set up yet, like a missing session store.
- **`INTERNAL`**: a bug in Genkit or the plugin. Retry treats this as transient, so don't use it for the caller's mistakes.
- **Provider failures**: wrap them with `provider_error(...)`. It maps the HTTP status and reads `Retry-After`.

Set `reason=` when callers will branch on the failure. Leave it off otherwise.

Raise a plain Python exception when the user calls Python wrong at setup, before any action runs. Passing the wrong type to a plugin constructor is a `TypeError`, and passing two arguments that conflict is a `ValueError`. These show up on the user's own line, and nothing branches on their status. Validators on a Pydantic config raise `ValueError` too; Pydantic turns that into a `ValidationError`, and Genkit reports it as `INVALID_ARGUMENT`.

`PublicError` is for code that wants the HTTP caller to see the message: app code, and web adapters rejecting a malformed request body. Model, tool and other provider plugins never raise it. Everything else reaches the HTTP caller as a generic message, usually a 500.

An error should name the user's thing and say how to fix it: "server 'linear' isn't a stdio entry", not "2 validation errors" or an interpreter path. It should never echo the value back, because the value might be someone's key.

One rule, one message. If two places check the same thing, have them call the same check, so the user never sees two different errors for one mistake.

For overrides: leaving an argument out keeps the default, `[]` clears it, a list replaces it, and config merges key by key. Before adding a new one, ask who else reads or changes that value after your code runs. Middleware can rewrite the request after your check. Retry and Fallback act on whatever status you pick. Config merges across the ref, the prompt and the call. This is the question most designs miss.

Pick the safe default. Prompt and reply text stay off spans unless someone asks for them, and an explicit setting beats an environment variable. Don't make people pay for things they don't call; prompts load lazily for that reason.

Only export what someone outside the SDK would actually type. You can always add it later; removing it breaks people.

Put behavior on the class it belongs to instead of exporting a loose function. If the user already holds the object, it's a method: `paused.restart(...)`, `stream.to_sse()`. If they're building one, it's a classmethod or the constructor: `GoogleAI.gemini(...)`, `Part.from_text(...)`, `MultipartToolResponse(...)`. That way autocomplete on the thing they have shows them what they can do. A standalone function is fine when there's no object to hang it on, like the `genkit.web` helpers for people writing their own server adapter.

When you explain a choice, in a comment, a docstring or a PR, explain it by what the user experiences. "JS does it this way" is useful research, but it isn't a reason.

## Pydantic, dataclass, or plain class

Ask what the object is for.

If it's ever going to be JSON, it's Pydantic. The base class depends on who owns the keys:

- The app's own data (flow and tool input and output, or config for middleware the app wrote itself) is a plain `BaseModel`. Those are the user's keys, and we don't rename them.
- Settings the user hands to Genkit are a `GenkitConfig` (from `genkit.plugin_api`). That covers model config through `ModelConfig`, plugin and shared middleware config, and whatever's nested inside them. You construct it with `snake_case` arguments that all three type checkers accept. Its JSON is `camelCase`, because the Dev UI form, traces and `.prompt` files all have to agree on one spelling. Unknown keys raise, and so does passing both spellings of one setting. `model_dump()` is plain Pydantic, so provider SDKs get field names; pass `by_alias=True` when you want Genkit JSON. Don't hand-write `alias=` when it would just be `to_camel(name)`.
- `GenkitModel` is internal, for Genkit's own wire types (`Part`, `Message`, `ModelRequest`, spans). It dumps `camelCase` by default, which isn't what provider SDKs or user code expect, so never make it the base of anything a user or plugin author subclasses.

If it's only in-process data, like a handle or a context object passed around (`ModelRef`, `SpanMetadata`, `TurnContext`), use a frozen, keyword-only dataclass. Frozen means nothing downstream can quietly change it.

If it does things or has a lifecycle (`Genkit`, `Tool`, `Prompt`, `StreamResponse`, `Plugin`, `Middleware`), it's a plain class with private state and methods. Don't make it a Pydantic model just to get a free constructor: every internal field becomes public.

## Imports and module structure

Put imports at the top of the file. Import inside a function only when you mean to:

- **Lazy loading**: a heavy module that most callers never need shouldn't slow down `import genkit`.
- **Optional dependencies**: an extra that isn't installed (`uvloop`, a provider SDK) must not crash code that doesn't use it. Import it where it's used and raise a clear error if it's missing.

Leave a comment saying which one it is, so nobody "fixes" it later.

Don't put imports under `if TYPE_CHECKING:`. If you need one to break an import cycle, the modules depend on each other in both directions. Dependencies should run one way: lower-level modules never import higher-level ones, and the folder layout should show that order. Fix the cycle by moving the shared piece down a level, not by hiding the import from the runtime.

There's a runtime cost too. Genkit reads a flow's or tool's annotations when it's defined to build its schemas, so a type that's only imported under `TYPE_CHECKING` fails at definition time.
