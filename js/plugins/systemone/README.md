# System One decision models for Genkit

Adds decision models to Genkit: models that speak System One, the protocol
TypeSafe AI introduced with jev and that Liquid AI's d1, Cloudflare's Clef, and
the gateways serving them also speak. A decision model does not generate text.
It evaluates a state against typed questions and returns one typed answer per
question, with calibrated probabilities. That makes it a fit for the decisions inside an application: routing,
classification, scoring, guardrails, and verification.

> Status: in preview. Its API may change in any minor version release.

One plugin, `systemOne`, serves any server that speaks the protocol.
Constructors set it up for the known ones, and any other server takes a few
options; see [Servers](#servers). The plugin is the TypeScript side of the
System One plugins for
[Genkit Go](https://genkit.dev/docs/go/integrations/system-one/) and
[Genkit Java](https://genkit-ai.github.io/genkit-java/plugins/systemone/), and
takes the same questions and answers.

## Installation

```bash
npm i --save @genkit-ai/systemone
```

It needs a `genkit` release whose schema validation accepts `x-` keywords,
which the question schemas carry. It runs on Node.js 20 or later, through
Node's `fetch`; in a browser, a `fetch` of your own reports redirects and
errors by the browser's rules (an opaque redirect is refused like a 3xx).

## Design principle: the output schema is the question set

A decision model is served as a model that speaks only constrained JSON, which
is the subset of the generate API it fits exactly. The questions are the fields
of the output schema, declared with `choice()`, `score()`, and `noul()`. Each
field's `.describe()` is the question's instructions, and each field's JSON is
the wire answer, so a decision is one typed generate call and the answer lands
in typed fields.

The question helpers belong to the protocol, not to a server, so one decision
schema serves every model that speaks it, and a fallback from one to another
needs no second schema.

```ts
import { choice, noul, score, systemOne } from '@genkit-ai/systemone';
import { genkit, z } from 'genkit';

// The decision: one question per field.
const Triage = z.object({
  // The options of a choice are its keys, with what each one means.
  department: choice({
    billing: 'Payments, invoicing, refunds',
    technical: 'Bugs, outages, integrations',
    other: 'None of the above',
  }).describe('Which team should handle this?'),
  isUrgent: noul().describe(
    'Does the ticket explicitly communicate time pressure?'
  ),
  // The levels of a score, lowest first.
  frustration: score(['Calm', 'Concerned but civil', 'Very angry']).describe(
    'How frustrated is the customer?'
  ),
});

const ai = genkit({ plugins: [systemOne.typeSafe()] }); // TYPESAFE_API_KEY

const { output } = await ai.generate({
  model: 'typesafe/jev-1.13.0',
  prompt: ticket,
  output: { schema: Triage },
});
if ((output!.department.confidence ?? 0) < 0.6) {
  // the caller owns thresholds
  return escalate(ticket);
}
switch (
  output!.department.choice // typed: 'billing' | 'technical' | 'other'
) {
  case 'billing':
  // ...
}
```

That is the whole call: the model, the state, and the schema. The questions
ride on the output schema, which the model reads back, so no output format
needs naming.

Everything else is the generate API as it already is: prompt files and the Dev
UI, model middleware such as fallback across servers, the trace with the
question set on the request and the distributions on the response, and the
token counters.

## Question types

| Field                  | Question                                       | Answer fields                                                                   |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------- |
| `choice(criteria)`     | pick one of the keys of `criteria`             | `choice`, `probabilities` per option, `confidence`                              |
| `score(levels)`        | rate on `levels`, lowest first                 | `score` (expected level, fractional), `probabilities`, `confidence`, `legend`   |
| `noul()`               | is this true?                                  | `noul`, the probability; near 0.5 means "could not tell"                        |
| `noul({ yes, no })`    | is this true, where `yes` and `no` say what each side means | as `noul()`                                                        |

`ranked(answer)` lists a choice's options from most to least likely, and
`margin(answer)` is the probability gap between the top two, which is how
decisive the choice is. `level(answer, levels)` is a score's nearest whole
level, and `label(answer, levels)` that level's text; both fall back to the
answer's legend.

A choice's options go out sorted by name, as Go, Java, and Dart send a typed
choice's options. A server may leave `confidence` out of an answer, as the Go
schema allows, so read it with a default (Go reads a missing one as 0). A
confidence or a probability of a declared option must be a number from 0 to 1,
or the call fails as a malformed answer (INTERNAL). A probability for an option
or level the question does not declare is dropped from the answer without
being checked, where Go keeps it; `responseInfo()` keeps the answers as sent.
An answer outside the question set, such as an option that is not one of the
choices or a noul above 1, fails the generate call's schema validation
(INVALID_ARGUMENT). Criteria and levels are
strings, and the schema and the answers show those strings. The wire format
also takes a structured description, most often an object with labeled parts,
which `guidance` adds: keyed by option for a choice, by level index for a
score, and as `yesGuidance` and `noGuidance` for a noul.

```ts
department: choice(
  {
    billing: 'Payments, invoicing, refunds',
    technical: 'Bugs, outages, integrations',
  },
  {
    guidance: {
      billing: {
        not_for: 'Where an order is, or when it arrives',
        examples: ['I was charged twice for one order.'],
      },
    },
  }
).describe('Which team should handle this?'),
```

An object with no `what` gets the string as its `what`, so guidance adds to the
description. Any other value goes out as it is. A score's legend keeps the
rubric's strings, and the guidance the API echoes back is on
`responseInfo(response).answers`.

The built-in `enum` output format also works, with no decision schema: the
enum values are the options of one choice question, the system message is the
question, and the output is the option.

```ts
const { output: team } = await ai.generate({
  model: 'typesafe/jev-1.13.0',
  system: 'Which team should handle this ticket?',
  prompt: ticket,
  output: { format: 'enum', schema: z.enum(['billing', 'technical', 'sales']) },
});
```

What the enum format does not give is criteria per option or a probability per
option; a `choice()` field gives both.

## Questions built at run time

A question whose options come from data, such as the tools on hand, a tenant's
categories, or the nodes of a taxonomy, is built from values.
`systemOne.schema()` turns
them into the output schema, and the answers come back as a record of
`Answer`:

```ts
import { systemOne, type Answer } from '@genkit-ai/systemone';

const response = await ai.generate({
  model: 'typesafe/jev-1.13.0',
  prompt: request,
  output: {
    jsonSchema: systemOne.schema({
      tool: {
        type: 'choice',
        instructions: 'Which tool serves the request?',
        options: tools.map((t) => ({ name: t.name, criteria: t.description })),
      },
      personal: {
        type: 'noul',
        instructions: "Does the request involve the user's own data?",
        yes: "Names the user's files, mail, or calendar",
        no: 'Asks about the world at large',
      },
    }),
  },
});
const { tool } = response.output as Record<string, Answer>;
if ((tool.confidence ?? 0) >= 0.8) {
  return run(tool.choice!);
}
```

Choice, score, and noul questions take what the field helpers take: options
with criteria and guidance, levels with guidance by index, and yes and no
criteria. A choice's options go out in the order given. Instructions can be a
string or any JSON value, such as an object that describes the field of the
state a question is about; a system message goes beside structured
instructions as the first element of an array.

## State

The state is built from the user and model messages, with no instruction text
mixed in. A system message is never state: it goes in front of every
question's instructions, and for the enum format it is the question. That is
the place for shared context, such as what the state is and what its fields
mean. A description on the output schema itself follows the system message in
every question's instructions. To judge a transcript that has its own system
prompt, leave that prompt out of the messages.

- One message is sent as its value: the string of a text part, or the JSON of a
  data part. `prompt: [{ data: value }]` sends any value as an object state,
  so that a question can name a field.
- Several messages are sent as an array of `{role, content}` records, roles
  included, so a question can refer to what the user said and what the model
  said.
- With documents attached, the state is `{messages, context}`, each document as
  its content, read as a message's is, or as `{content, metadata}` when it has
  metadata.

Output instructions that the generate loop adds to messages (parts with
`metadata.purpose: 'output'`, which it adds for the enum format) are dropped,
as are the ones a history recorded from another model keeps.

A prompt template renders text, so a template that renders JSON needs
`stateJSON: true` in its config to produce an object state. The JSON is sent as
written, so an ID past 2^53 reaches the model unchanged.

Tools are rejected: the model never calls anything.

### Media

Media parts become the request's media, by kind: images go in `images`, audio
in `audio`, and video in `videos`, each in the order its parts appear, messages
first and documents after. The servers place them before the state, and the
text of the same messages stays the state; a message of media alone is a turn
with empty text. A request may carry media and no text, except to Ollama. Media
goes out as a base64 data URL, or as raw base64 to Ollama; a part given as a URL
is refused rather than fetched, so download it first. Any other media type is
refused.

Media is an extension to the protocol, and a server that does not know a field
can drop it and answer as if nothing had been sent. A plugin therefore refuses
media before the request is sent, kind by kind, unless its `images`, `audio`,
or `video` option is `'supported'`. `systemOne.ollama()` and
`systemOne.workersAI()` set `images`, and so does a server of your own that
takes images. Then a model is sent that kind, and the server refuses it for a
model that cannot read it, so a new model needs no setup. A model's entry in
`models` overrides the plugin's setting, kind by kind, either way: Clef-omni on
Workers AI also reads audio and video.

## Prompt files

The decision schema is a registered schema, so a prompt file can name it:

```ts
ai.defineSchema('Triage', Triage);
```

```yaml
---
model: typesafe/jev-1.13.0
config:
  stateJSON: true
input:
  schema:
    ticket: string
    accountTier: string
output:
  schema: Triage
---
{{role "system"}}
The state is a support ticket; accountTier is the customer's plan.

{{role "user"}}
{"ticket": {{json ticket}}, "accountTier": {{json accountTier}} }
```

## Servers

Models are named by the server they are reached through: the plugin's
`provider`, then the ID the server uses, so a model a server adds later works
on the day it ships.

| Constructor                       | Models                                                                                                                                                                 | Key                    | Notes                                                                                                                       |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `systemOne.typeSafe()`            | `typesafe/jev-1.13.0`, `typesafe/jev-latest`                                                                                                                           | `TYPESAFE_API_KEY`     | `TYPESAFE_BASE_URL` is read too; lists models; text only                                                                    |
| `systemOne.openRouter()`          | `openrouter-decisions/liquid/d1`, `openrouter-decisions/typesafe/jev-1.13`, `openrouter-decisions/~typesafe/jev-latest`                                                | `OPENROUTER_API_KEY`   | alpha Decisions API; lists OpenRouter's decision models; cost in `usage.custom.cost`; text only                            |
| `systemOne.ollama()`              | `ollama-decisions/clef`                                                                                                                                                | none needed; an `apiKey`, if given, is sent as a bearer token (for a proxy) | `http://localhost:11434` unless `baseUrl` is set; images as raw base64; a request needs text                               |
| `systemOne.workersAI(accountId)`  | `cloudflare-decisions/@cf/cloudflare/clef`, `cloudflare-decisions/@cf/cloudflare/clef-flash`, `cloudflare-decisions/@cf/cloudflare/clef-omni`, `cloudflare-decisions/typesafe/jev` | `CLOUDFLARE_API_TOKEN` | `CLOUDFLARE_ACCOUNT_ID` when `accountId` is unset; Clef models read images, Clef-omni audio and video too; a long state is truncated; no versions to pin |

A gateway serves several vendors' models under one key, so one plugin reaches
all of them. OpenRouter's is named `openrouter-decisions` so that a plugin for
its chat models can keep `openrouter`; Ollama's is `ollama-decisions` because
the Ollama plugin for chat models has `ollama`.

A constructor takes the same options as `systemOne()`, over the ones it sets
for its server, such as a key from a secret manager rather than the
environment: `systemOne.typeSafe({ apiKey })`. Its `models` are added to the
constructor's, and an entry with one of their IDs replaces that model's whole
spec, as in Go, so `{'@cf/cloudflare/clef-omni': {label: 'Omni'}}` also drops
the audio and video the preset gave it: copy the fields you need. Unlike Go,
where the caller owns the map and can drop a preset model, the constructor's
models are always kept. Naming a constructor's model by its full name is
refused.

The plugin checks its setup when it is created and throws FAILED_PRECONDITION
for one every request would fail on: a provider with a slash; a base URL that
is not http or https with a host, or that carries credentials, a query, or a
fragment (it is never quoted, since it can carry a password); a `timeoutMs`
that is not a whole number from 1 to 2^31 - 1; a header or key HTTP cannot
carry, such as one with a newline, a control byte, or a character outside
Latin-1 (named, never quoted); a connection or framing header (Host,
Connection, Content-Length, Transfer-Encoding, Keep-Alive, Upgrade, Expect); a
missing key or account ID. Go checks only the missing key, account, and base
URL there; a bad base URL or header fails its first request. Keys, base URLs,
and account IDs read from the environment are trimmed, so a key file that ends
in a newline works; Go reads them as they are.

A server that needs a key refuses to start without one, unless `apiKey` is
`false`: then each request takes the key from `config.apiKey` only, the
pattern the anthropic and google-genai plugins share, and a request without
one fails with FAILED_PRECONDITION (google-genai says INVALID_ARGUMENT); the
model listing goes out with no key. A `config.apiKey` replaces the plugin's key
on any plugin, not only with `false`, and one HTTP cannot carry fails the
request with INVALID_ARGUMENT, without a retry. A key passed in `config.apiKey`
is recorded in traces and the Dev UI like any other config field, as is one in
the config of a fallback reference, which the generate span records under
`use`, so where traces are kept, prefer the environment variable or the
plugin's `apiKey`. With per-request keys, give every fallback reference a
config with its own key, or set `isolateConfig` and give the backup plugin its
own key: a reference without a config inherits the request's, so the first
server's key would go to the next server's host.

### OpenRouter

Models go by OpenRouter's IDs, and each response reports what the request cost.
An alias that follows the latest release starts with a tilde, as
`~typesafe/jev-latest` does; `typesafe/jev-latest` is not an ID.

The API is in alpha, and its models are text only here. It takes images inside
the state, and can route them to a provider that drops them and answers anyway,
so the plugin refuses media before the request is sent.

### Cloudflare Workers AI

Workers AI serves Cloudflare's Clef and Clef-flash, which read images,
Clef-omni, which reads images, audio, and video, and partners' models such as
jev, which reads text only. A model is named by its Workers AI ID, `@cf/`
included.

Clef-omni takes up to 4 images, 4 audio clips of up to 5 minutes, and 2 videos
of up to a minute, sampled at 2 frames a second; a video's soundtrack is heard
only when every video in the request has one. Clef-flash takes 24k tokens of
state, media, and questions together, and Clef and Clef-omni 64k.

The base URL has an `{account}` segment, which the account ID fills. A proxy's
`baseUrl` with no such segment needs no account ID:

```ts
systemOne.workersAI(undefined, {
  baseUrl: 'https://ai-proxy.example.com/cloudflare', // takes /ai/run/<model>
});
```

Workers AI names one model per release, with no versions, so a threshold tuned
against a model can move when Cloudflare updates it.

### Several servers

Each server is its own plugin, so one app can use several, and the fallback
middleware moves a request to another server when one is down. Each model's
answers are calibrated on their own, so check which model answered before
applying a threshold tuned for one: `responseInfo(response).model`. With the
`fallback` middleware of `@genkit-ai/middleware`, a reference built with a
config, `plugin.modelRef(id, config)`, sends that config in place of the
request's, and one without inherits the request's unless `isolateConfig` is
set, so give a reference with a config the `stateJSON` it needs, or send the
state as a data part.

### Any other server

Any other server that speaks the protocol takes a few options. `provider` names
the plugin and prefixes its models, and must differ from every other plugin's
name; `baseUrl` is the root that `path` (default `/v1/systemone`, or `/` for
`baseUrl` itself) and `modelsPath` hang under; `apiKey` is optional for a server
that takes none.

```ts
// Liquid AI's d1, from Liquid's own API.
const liquid = systemOne({
  provider: 'liquid',
  baseUrl: 'https://api.liquid.ai/decisions',
  modelsPath: '/v1/models',
  apiKey: process.env.LIQUID_API_KEY,
  images: 'supported', // d1 reads images
});
```

A proxy that forwards the native protocol, such as LiteLLM, is reached the same
way. A server whose wire is not the native one takes two hooks over the native
body, so it needs no plugin release:

```ts
const wrapped = systemOne({
  provider: 'wrapped',
  baseUrl: 'https://decisions.example.com',
  path: '/run',
  // The request: the native body under input, the model in the path.
  route: (model, body) => {
    delete body.model;
    return { path: model, body: { input: body } };
  },
  // The response: the native body out of the server's envelope.
  unwrap: (body: any) => body.result,
});
```

Both hooks run synchronously. A `route` path is a string, joined to `path` with
a slash when it has none. A GenkitError either hook throws keeps its status
(from `unwrap`, under the provider's name); any other error is reported as
UNKNOWN under the provider's name. Go does that for `unwrap` only, and returns
a route's error unchanged. A route that returns no body, a path that is not a
string, or a promise, and an `unwrap` that returns a promise, fail with
INVALID_ARGUMENT.

### Every server

`models` describes the models known ahead, keyed by the server's ID or by the
full model name; they are listed in the Dev UI with whatever the server's
listing adds, which is kept for five minutes. A listing contributes at most
1,000 model IDs, each at most 256 characters; others are left out. `fetch`, `timeoutMs`, and
`headers` are the escape hatches to the transport; `headers` go first, and the
Authorization header the API key makes replaces one given there, in any case. The `extra` config merges fields into
the request body that the plugin does not model, such as a gateway's
`session_id` or `trace`; it cannot replace a field the request builds itself,
and a NaN or infinite number in it is refused rather than sent as null.

Each attempt is bounded at 30 seconds with the global `fetch`, which is looked
up per request, and by your own `fetch` when you pass one, unless
`timeoutMs`, a whole number of milliseconds from 1 to 2^31 - 1, says otherwise.
A request the caller aborts fails with CANCELLED, or DEADLINE_EXCEEDED when its
signal's reason is a timeout; Go returns the context's own cause. Requests that fail to connect, time out, are rate
limited, or hit a server error other than 501 and 505 are retried twice, with
`Retry-After` honored up to 30 seconds. These retries stack with a `retry()`
middleware on top, so a call can make up to three attempts for each of its
retries. A redirect is not followed, so the key's handling never depends on
the transport's redirect rules: the request fails (FAILED_PRECONDITION), and
the base URL should name the final address. A response larger than 8 MiB is
refused (INTERNAL) and not retried; Go reads the first 8 MiB and then fails to
parse them, so both fail, with different messages. A network error names its
cause, such as ECONNREFUSED. A server's error message is quoted in the
error up to 500 characters.

Pin a version in production where the server offers one. Confidence thresholds
tuned against one release do not carry over to the next, nor from one model to
another, and `responseInfo(response).model` is the version that answered, on
every call. A gateway's cost is `response.usage.custom.cost`.

## Limits

- No per-item questions. Score a list of passages with one call per passage.
- No nested decision schemas: a question is a top-level field.
- The instructions of a field in a decision schema are a string, since they
  are its description. Structured instructions need a runtime question.
- Each server sets its own limits on questions and tokens; jev takes up to 64k
  tokens of state and questions together, and up to 32k of state and its
  longest question.
- A server can cut a long state to fit its limit and answer from what is left,
  with no error. Check a long document against the model's limit, or split it.
- No streaming; the answer arrives whole.
- The answer is a text part holding JSON, with no content type of its own, as
  JS text parts have none.
- Token counts the server leaves out are left out of `usage`, rather than
  read as zero.
- A data part whose value is null is dropped from the state, as Go drops it.
- A question ID or an option named `__proto__` does not reach the model: the
  generate loop's copy of the output schema drops that key.

## Tests

`pnpm test` runs against a fake System One server; no key is needed.

Usage information and the API reference are on the
[Genkit documentation](https://genkit.dev/docs/get-started).

License: Apache 2.0
