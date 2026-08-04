# Genkit telemetry redesign

**What this proposes.** Genkit's observability today has two problems. First, Genkit instantiates and owns the OpenTelemetry SDK — the app-level telemetry runtime — inside the framework. In OTel's architecture, that's the application's job: libraries emit spans through the lightweight OTel *API*, and exactly one SDK, owned by the app, collects everything and exports it. Because Genkit takes that role for itself, a developer's own telemetry setup silently conflicts with ours (details below), and Genkit's spans can never join the app's traces. Second, Genkit describes its spans in a private vocabulary (`genkit:input`, `genkit:type`, ...) that no observability backend understands, so even when spans arrive at Langfuse or Datadog, they render as opaque JSON instead of LLM traces with cost, sessions, and chat views.

**The proposal, in four commitments:**
1. **Genkit uses the OTel API only.** Core never instantiates an SDK, never registers globals, never configures exporters. Spans are emitted through `@opentelemetry/api` and land wherever the app's telemetry setup sends everything else.
2. **The app owns the OTel SDK.** The developer's `instrumentation.ts` (or a Genkit-provided opt-in bootstrapper like `enableGoogleCloudTelemetry`) decides what exists and where spans go. If the app sets up nothing, Genkit's spans are near-free no-ops — correct library behavior.
3. **Genkit emits GenAI semantic conventions (`gen_ai.*`) by default.** The OTel-standard vocabulary for LLM telemetry, which Langfuse, Datadog, Grafana and others already translate server-side into their products. Genkit keeps a small `genkit.*` namespace only for concepts the standard lacks.
4. **The Dev UI keeps working with zero config, via a dev-only provider.** In dev, Genkit guarantees an SDK exists (registering a minimal one only if the app didn't, piggybacking on the app's if it did). In prod, this code never runs.

**Why.** These four together produce the target experience: a developer with a standard OTel setup gets one interleaved trace — HTTP span, Genkit flow, model call, tool, DB query — in the backend of their choice, with zero Genkit-specific integration code, and every observability vendor supports Genkit without writing anything, because the translation layer for `gen_ai.*` already exists on their servers. Meanwhile the zero-config Dev UI experience is unchanged, and production apps that configure nothing pay nothing.

The rest of this doc: the mechanical root cause of today's conflict, the code changes per audience (core / providers / developers), the instrumentation spec and attribute mapping, decisions to ratify (D1–D8), cross-language portability, and migration.

## Today's bug

`runInNewSpan` → `ensureBasicTelemetryInstrumentation()` → boots a full NodeSDK, even in prod. If the app has its own OTel SDK, both race for the API's global slot — first writer takes all spans, loser's exporters get nothing, silently:

```js
// @opentelemetry/api global-utils.js — the root cause
const GLOBAL_OPENTELEMETRY_API_KEY = Symbol.for(`opentelemetry.js.api.${major}`);
function registerGlobal(type, instance, diag, allowOverride = false) {
  if (!allowOverride && api[type]) {
    diag.error(`Attempted duplicate registration of API: ${type}`); // diag is OFF by default
    return false;                                                   // second SDK silently discarded
  }
```

This is why flows root their own traces instead of nesting under the app's HTTP span, and why "express + own telemetry setup = a mystery."

## 1. Core framework

Emit via `@opentelemetry/api` (peer dep). No SDK on the prod path. gen_ai.* by default, `genkit.*` only for what the convention lacks.

```ts
// core/src/tracing/instrumentation.ts
export async function runInNewSpan(opts, fn) {
- await ensureBasicTelemetryInstrumentation();          // deleted: no more NodeSDK boot
  const tracer = trace.getTracer('genkit', VERSION);    // reads the global slot, never writes it
  return tracer.startActiveSpan(opts.name, async (span) => { ... });
  // active parent in context → child span, same trace
  // no parent (Dev UI runner, script) → new trace root
  // no provider registered at all → no-op, ~free
}
```

```ts
// core/src/tracing/semconv.ts — every attribute name in ONE file (gen_ai.* is pre-stable)
export const GEN_AI = {
  OPERATION: 'gen_ai.operation.name',        // 'chat' | 'execute_tool' | 'invoke_agent'
  REQ_MODEL: 'gen_ai.request.model',
  IN_TOKENS: 'gen_ai.usage.input_tokens',
  OUT_TOKENS: 'gen_ai.usage.output_tokens',
  CONVERSATION: 'gen_ai.conversation.id',    // replaces genkit:metadata:agent:sessionId
  IN_MSGS: 'gen_ai.input.messages',          // content — gated, see defaults
  OUT_MSGS: 'gen_ai.output.messages',
};
export const GENKIT = {                      // residue the convention lacks; Dev UI reads these
  PATH: 'genkit.path', TYPE: 'genkit.type', FAILURE_SOURCE: 'genkit.isFailureSource',
};
```

```ts
// model span emission — dual vocabulary, one span
span.setAttributes({
  [GEN_AI.OPERATION]: 'chat',                // on EVERY span incl. flows — vendor filters
  [GEN_AI.REQ_MODEL]: req.model,             //   drop spans without gen_ai.* keys
  [GEN_AI.IN_TOKENS]: usage.inputTokens,
  [GENKIT.TYPE]: 'action',
  [GENKIT.PATH]: path,
});
if (captureContent) span.setAttribute(GEN_AI.IN_MSGS, toSemconvMessages(req));
// no more genkit:input/genkit:output JSON blobs — content lives in gen_ai.* only
```

```ts
// core/src/tracing/dev-mode.ts — the ONLY place Genkit touches an SDK
// devUiProcessor() = the EXISTING Dev UI plumbing, reused:
//   RealtimeSpanProcessor(new TraceServerExporter())
//   - RealtimeSpanProcessor: forwards spans as they end (no batching delay,
//     so the Dev UI updates live during a run)
//   - TraceServerExporter: serializes spans and POSTs them to the local
//     telemetry server the `genkit start` CLI spawned (GENKIT_TELEMETRY_SERVER).
//     Also the D7 compat point: its payload down-converts to legacy `genkit:*`
//     names so old Dev UI versions keep working.
export function maybeInitDevTelemetry() {
  if (!isDevEnv()) return;                             // prod: never runs
  if (tryClaimSlot()) return;                          // slot empty → Genkit's minimal dev SDK now owns it
  {                                                    // app registered its own SDK
    tryPiggyback();                                    // piggyback — Dev UI AND their backend
    // NB (D2): dev default = content ON, and the span visits every sink —
    // so the app's own exporters receive full prompts/outputs here too.
    // Genkit does not redact per-sink; suppression/masking is the sink's
    // policy (e.g. GCP forceDevExport, Langfuse mask). Documented behavior.
    return;
  }
}
// tryClaimSlot / tryPiggyback: SDK-agnostic helpers; per-language internals in D1.
```

## Instrumentation spec: where the convention is implemented

Span creation already funnels through `runInNewSpan` at few sites: `core/action.ts:533` (all actions — model/tool/embedder/retriever), `core/flow.ts:216`, `ai/generate/action.ts:142` (generate wrapper), `prompt.ts`, `evaluator.ts`. Attribute writes today: 26 sites across 7 files, in three categories with different fates:
- **~20 sites — already at the choke point** (`metadata.name/input/output`, `subtype`, `SPAN_TYPE_ATTR` labels are set inside the runner callback or passed to `runInNewSpan`): fold mechanically into the enrichers below.
- **4 sites — mid-execution contextual writes** (`agent:sessionId` @agent.ts:1061, `agent:snapshotId` @486, `interrupt` @tool.ts:538, `resumed` @569): information doesn't exist at span start/end, so they stay in place — rewritten from the homegrown `setCustomMetadataAttribute` buffer to the ecosystem-standard `trace.getActiveSpan()?.setAttribute(GEN_AI.CONVERSATION, ...)`.
- **Names centralize 100% regardless**: every site imports constants from `semconv.ts`, so "what can appear on a span" is auditable from two files + a grep for constants — even where write timing is distributed.

Implement type-shape logic as **one enricher module** at the action runner:

```ts
// WHO CALLS IT: the action runner, at the choke point. runInNewSpan wires both hooks:
// core/action.ts (~:533) — every action (model/tool/embedder/retriever) passes here
return runInNewSpan({ name, actionMeta }, async (span) => {
  enrichStart(span, actionMeta);          // ← start-time facts, keyed by action type
  try {
    const result = await fn(input);
    enrichEnd(span, result);              // ← end-time facts (usage, finish, resp model)
    return result;
  } catch (e) { enrichEnd(span, { error: e }); throw e; }
});
// flow.ts:216 and generate/action.ts:142 pass through the same runner path,
// so no other file touches semconv names.

// core/src/tracing/semconv-enrich.ts — the single implementation point
export function enrichStart(span, action) {   // called from the action runner
  switch (action.type) {
    case 'model':  span.updateName(`chat ${action.modelName}`);
                   span.setAttributes({ [GEN_AI.OPERATION]: 'chat',
                     [GEN_AI.REQ_MODEL]: action.modelName,
                     [GEN_AI.PROVIDER]: providerIdFor(action) });  // see below
                   break; // + SpanKind.CLIENT — semconv requires it; today everything is INTERNAL
    case 'tool':   span.updateName(`execute_tool ${action.name}`);
                   span.setAttributes({ [GEN_AI.OPERATION]: 'execute_tool',
                     [GEN_AI.TOOL_NAME]: action.name, [GEN_AI.TOOL_CALL_ID]: callId });
                   break; // SpanKind.INTERNAL
    case 'embedder': /* `embeddings {model}`, operation=embeddings */ break;
    case 'flow':   /* operation value = D4; genkit.type='flow' regardless */ break;
    case 'retriever': case 'reranker': /* no semconv exists → genkit.* only + operation.name for filters */ break;
  }
}
export function enrichEnd(span, result) {     // end-of-span facts (works for streaming: known at close)
  span.setAttributes({ [GEN_AI.RESP_MODEL]: result.modelVersion,
    [GEN_AI.IN_TOKENS]: result.usage?.inputTokens, [GEN_AI.OUT_TOKENS]: result.usage?.outputTokens,
    [GEN_AI.FINISH_REASONS]: result.finishReasons });
  if (result.error) span.setAttribute('error.type', result.error.name);
}
```

Attribute timing: start = operation, span name, request.model, provider, tool identity. End = response.model, usage tokens, finish_reasons, error.type. Content (`gen_ai.input/output.messages`) at whichever point it's known, behind the D2 gate.

`gen_ai.provider.name` requires a **plugin interface addition**: model plugins declare their semconv provider id (`googleai` → `gcp.gen_ai`, `vertexai` → `gcp.vertex_ai`, `openai` → `openai`, ...); fall back to the action-name prefix for unknown plugins.

Resolved dedup: `generate/action.ts:142` wraps the model action — one generate() can contain 1..N inferences plus tool calls, so generate is an orchestration span (see mapping table), and inference-only gen_ai.* attributes live exclusively on the inner model spans. Two spans always; backends never double-count tokens.

## Mapping strategy and output

**Strategy.** Semconv attributes carry requirement levels (Required / Conditionally Required / Recommended / Opt-In) — partial conformance is designed-in, not a compromise. Rules, in order:
1. A semconv attribute exists for the concept → emit it (semconv name, semconv type, semconv value enum).
2. Genkit's value doesn't fit the semconv enum → emit the closest semconv value AND the exact value under `genkit.*` (map lossy, drop nothing).
3. No semconv exists for the concept → `genkit.*` only, plus `gen_ai.operation.name` for vendor-filter survival.
4. Genuine gaps are filed upstream to the semconv WG (spec is pre-stable; field reports are the input it wants) — not solved by inventing parallel keys.
5. gen_ai.* semantics are honored strictly: inference-only attributes (usage, messages, finish_reasons) appear **only on model spans** — never on containers — so backends can sum tokens without double-counting.

Genkit's normalized `GenerateRequest/GenerateResponse` schema means one schema-to-schema mapping, not one per provider.

**Output.** WIP. Will write a separate mapping document using the strategy above.

## 2. Provider (Langfuse / Datadog / any OTel backend)

Required integration code: **none.** Their server-side ingest already translates `gen_ai.*` into their model — a Genkit chat span becomes a Langfuse generation (rendering, cost, sessions via `conversation.id`) through the same path as their own SDK's spans. Langfuse's recommended setup is already this architecture:

```ts
// Langfuse's own docs, today:
new NodeSDK({ spanProcessors: [new LangfuseSpanProcessor()] }).start();
// processor = client-side shipping (filter/mask/batch/POST)
// translator = their server reading gen_ai.* → generations, cost, sessions
```

Vendor-tier extras stay vendor-owned, in app code, optional:

```ts
updateActiveTrace({ userId: req.auth.uid, tags: ['prod'] });  // → langfuse.user.id etc.
```

No `@genkit-ai/<vendor>` packages. Genkit ships a verified-providers docs page + conformance test instead.

## 3. End developer

```ts
// DEV, zero config — unchanged experience
const ai = genkit({ plugins: [googleAI()] });
// `genkit start` → dev provider auto-registers → Dev UI streams traces
```

```ts
// PROD, bring-your-own OTel — THE MAIN CASE. instrumentation.ts, app-owned.
// GCP is not special: it's one processor in the fan, same as any vendor.
import { NodeSDK } from '@opentelemetry/sdk-node';
import { googleCloudSpanProcessor, googleCloudMetricReader } from '@genkit-ai/google-cloud';
import { LangfuseSpanProcessor } from '@langfuse/otel';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';

new NodeSDK({
  resource: myResource,                      // app's service.name etc. (GCP labels merge, never replace)
  sampler: mySampler,                        // app's sampling policy (D6)
  spanProcessors: [
    googleCloudSpanProcessor(),              // → Cloud Trace
    new LangfuseSpanProcessor(),             // → Langfuse — composes; impossible today
  ],
  metricReaders: [googleCloudMetricReader()], // → Cloud Monitoring (current sdk-node: plural)
  instrumentations: [getNodeAutoInstrumentations()],  // express, pg, redis...
}).start();
// one trace: POST /chat ▸ chatFlow ▸ chat gemini-2.5-flash ▸ execute_tool ▸ SELECT
// — in Cloud Trace AND Langfuse. Zero Genkit-specific lines; Genkit emits into this.
```

```ts
// PROD, batteries included — for GCP-only apps that don't want to write the above.
// enableGoogleCloudTelemetry() is EXACTLY this, nothing more:
export function enableGoogleCloudTelemetry(opts?: GcpOpts) {
  assertNoProviderRegistered();          // app already has an SDK? loud error, not a race
  new NodeSDK({
    spanProcessors: [googleCloudSpanProcessor(opts)],
    metricReaders: [googleCloudMetricReader(opts)],
  }).start();                            // + Cloud Logging hookup (outside NodeSDK config)
}
// i.e. the same NodeSDK call as the main case, pre-filled with GCP-only pieces.
// It IS the app instantiating the SDK — one level of sugar, not a parallel system.
// Composing with other sinks? Don't call it; write the main case instead.
enableGoogleCloudTelemetry();
```

```ts
// PROD, no telemetry setup
const ai = genkit({ plugins: [googleAI()] });
// no provider → no-op spans, no SDK loaded, nothing exported. Today: silent NodeSDK.
```

## Decisions

Each: the question, the choice in code, what's still open.

**D1 — When does Genkit ever register an OTel SDK?**

```ts
// HOOKS IN AT: genkit() init, dev branch — same place the reflection server starts.
// Deterministic: the app's instrumentation.ts (by OTel convention) ran before app code,
// so detection sees their provider. Prod: no hook exists at all.
export function genkit(options) {
  const ai = new Genkit(options);
  if (isDevEnv()) { startReflectionServer(ai.registry); maybeInitDevTelemetry(); }
  return ai;
}
// prod:                  never. The app's instrumentation.ts is the only SDK.
// dev, slot empty:       Genkit claims it with a minimal provider → Dev UI works, zero config
// dev, app has an SDK:   Genkit APPENDS the Dev UI processor to the app's provider
if (!isDevEnv()) return;
if (tryClaimSlot()) return;      // atomic try-register (see Open below) — slot was empty
tryPiggyback();                  // app owns the slot → duck-typed addSpanProcessor
```

Open items narrowed by design: don't sniff provider *types* — ask the two questions directly, wrapped in SDK-agnostic helpers (`tryClaimSlot()` / `tryPiggyback()`) whose internals differ per language:

```ts
// Q1 — "is the slot empty?" → don't detect, ATTEMPT. JS registration returns a boolean
// (the same guard we read in registerGlobal's source) — atomic, public API, immune to
// duplicate api copies and minification:
function tryClaimSlot(): boolean {
  return trace.setGlobalTracerProvider(buildDevProvider());  // true → we own dev telemetry
}                                                            // false → app got there first

// Q2 — "can I piggyback?" → duck-type the CAPABILITY, not the class:
function tryPiggyback(): boolean {
  const p = trace.getTracerProvider() as any;
  if (typeof p.addSpanProcessor !== 'function') return false;
  p.addSpanProcessor(devUiProcessor());
  return true;
}

export function maybeInitDevTelemetry() {
  if (!isDevEnv()) return;
  if (tryClaimSlot()) return;                                // zero-config dev: done
  if (!tryPiggyback())                                       // app's SDK: append Dev UI sink
    logger.warn('Dev UI: unrecognized tracer provider; traces will not stream to the Dev UI.');
}
```

Per-language internals (same helper names, same contract):
- **JS**: as above — try-register boolean + `addSpanProcessor` duck-check.
- **Python**: `set_tracer_provider` returns None → `tryClaimSlot` is check-then-set (isinstance of the API's NoOp/Proxy placeholder — public types); `tryPiggyback` = `hasattr(p, 'add_span_processor')`.
- **Go**: two extra wrinkles force the checks into the **opposite order**. (a) `otel.SetTracerProvider` has **no guard — last-writer-wins** — so claiming is the *dangerous* op (it can clobber the app's provider), not the safe probe it is in JS. (b) The unset global isn't the public noop type — it's an **unexported delegating placeholder** (`otel/internal/global`), so there's no clean type assertion for "empty". But the placeholder has no `RegisterSpanProcessor` method and every real SDK provider does — so the *piggyback check doubles as presence detection*, and runs first:

```go
func maybeInitDevTelemetry() {
    if !isDevEnv() { return }
    p := otel.GetTracerProvider()
    // 1. Reliable signal first: a real SDK accepts processors → piggyback, done.
    if sdk, ok := p.(interface{ RegisterSpanProcessor(sdktrace.SpanProcessor) }); ok {
        sdk.RegisterSpanProcessor(devUiProcessor())
        return
    }
    // 2. Only claim if p is provably the unset placeholder (reflection on pkg path —
    //    ugly, pins internal layout; the price of no public "is default" API):
    if isDefaultPlaceholder(p) {                    // reflect.TypeOf(p).PkgPath() ~ "otel/internal/global"
        otel.SetTracerProvider(devProvider())       // last-writer-wins is safe HERE: nothing to clobber
        return
    }
    // 3. Exotic provider (real, but no processor hook): NEVER SetTracerProvider over it.
    slog.Warn("genkit: unrecognized TracerProvider; Dev UI will not receive traces")
}
```

One Go consolation: the placeholder *delegates* — tracers obtained before `SetTracerProvider` route to the eventually-registered provider — so Genkit-Go's emission (`otel.Tracer(...)`) is ordering-proof even though the dev claim isn't. Also offer the explicit escape valve, idiomatic in Go: `genkit.Init(ctx, genkit.WithTracerProvider(tp))` skips detection entirely.

Corroborator in all languages: probe span + `isRecording()` (no-op tracers return false — confirms, never decides, since sampled-out spans also return false). Still open: fallback UX when piggyback fails (Dev UI dark + the logged hint above — acceptable?); app SDK started *after* genkit() in dev → loses silently in JS/Python, gets clobbered in Go; mitigate with "init OTel before genkit()" docs + a visible log line when Genkit claims the slot.

**D2 — When do full prompts/outputs go on spans?**

```ts
// CALLED FROM: the enrichers (see instrumentation spec) — the gate's ONLY two call sites.
// Config resolves once at genkit() init; per-span cost is a boolean read.
//   enrichStart(span, action, input)  → gate → writes inputs  (known at span start)
//   enrichEnd(span, result)           → gate → writes outputs (known at span end — streaming just works)
// Model spans → gen_ai.input/output.messages; flow/tool spans → genkit.input/output.
function shouldCaptureContent(cfg) {
  if (cfg?.captureContent !== undefined) return cfg.captureContent;  // explicit wins
  return isDevEnv();                                                 // dev ON, prod OFF
}
// dev, unconfigured          → ON  (Dev UI shows raw I/O — its core value)
// prod, unconfigured         → OFF (structure/tokens/cost only; nothing to redact)
// prod + captureContent:true → ON  (ships to the app's chosen sinks — informed choice)
```

Two homes, same gate — never both on one span.
Piggyback edge (resolved): in dev with an app SDK, content-carrying spans reach the app's external sinks too. Genkit doesn't redact per-sink; suppression/masking is sink policy (GCP `forceDevExport`, Langfuse `mask`).
Open: knob granularity (global vs per-flow vs inputs/outputs split); truncation limit; redaction hook.

**D3 — Does the vocabulary change between dev and prod?**

```ts
// NO. Same keys everywhere; only listeners and content verbosity differ.
// dev span:   gen_ai.operation.name, gen_ai.usage.*, genkit.path, + messages (D2 on)
// prod span:  gen_ai.operation.name, gen_ai.usage.*, genkit.path              (D2 off)
// NOT this:   if (isProd) omit genkit.*   ← rejected; app-side processors can strip
```

One snapshot test covers every environment.

**D4 — Does every span carry `gen_ai.operation.name`, even flows?**

```ts
// Langfuse's processor default: export spans WITH gen_ai.* keys, drop the rest.
flowSpan: { 'genkit.type': 'flow' }                              // ← DROPPED: tree has holes
flowSpan: { 'genkit.type': 'flow',
            'gen_ai.operation.name': /* D4 open */ }             // ← survives the filter
```

Proposed: yes, on everything. Open: the value for flows — `invoke_agent`? emerging workflow attrs? `genkit.flow`? Track the semconv WG.

**D5 — Who decides trace boundaries?**

```ts
// Nobody, explicitly. Spans inherit ambient context:
app.post('/chat', h)  →  POST /chat ▸ chatFlow ▸ ...     // joins the app's trace
Dev UI "Run" button   →  chatFlow ▸ ...                   // empty context → own trace (unchanged)
// Escape hatch for batch loops:
runFlow(f, item, { newTrace: true })   // empty-context start + span link back
```

Required consequence: `genkit.isRoot` redefined as "root of the *Genkit subtree*" — or Dev UI trace assembly breaks once flows nest under HTTP spans.

**D6 — Whose sampler wins?**

```ts
new NodeSDK({ sampler: new TraceIdRatioBasedSampler(0.1), ... })   // the APP's
// → app drops the HTTP span → the entire Genkit subtree under it is dropped
// → piggybacked Dev UI sees 1 in 10 runs. Document: "why is my flow missing?"
```

Proposed: app's sampler, full stop. Open: a dev-only debug flag for the Dev UI processor to bypass sampling?

**D7 — What happens to the old `genkit:*` keys?**

Each attribute is renamed in two steps: if `gen_ai.*` covers the concept, the data moves there and the old spelling dies (`genkit:metadata:agent:sessionId` → `gen_ai.conversation.id`, no `genkit.sessionId` twin); only concepts with no semconv equivalent survive as `genkit.*` dot-style (`genkit:path` → `genkit.path`).

The problem: existing software parses the old names — the Dev UI (not force-upgradable), the Firebase console, and genkit-tools' eval code. Solution: wire spans carry only new names, and the two exporters that ship in Genkit's own repo translate back to legacy names in their own payloads: (1) core's `TraceServerExporter` (the Dev UI sink), covering old Dev UI versions, and (2) the GCP plugin's `AdjustingTraceExporter` — which also serves `enableFirebaseTelemetry`, a thin wrapper over it — covering the Firebase console and existing Cloud Trace dashboards. Third-party sinks see only the new vocabulary. Shims are deleted at the next major.

```ts
// TraceServerExporter: span has new names; the payload to the (possibly old) Dev UI speaks old ones.
toLegacyPayload(span) {
  const a = span.attributes;
  return { ...span, attributes: { ...a,
    'genkit:path':                     a['genkit.path'],
    'genkit:metadata:agent:sessionId': a['gen_ai.conversation.id'],
    'genkit:input':                    a['gen_ai.input.messages'] ?? a['genkit.input'],
    'genkit:output':                   a['gen_ai.output.messages'] ?? a['genkit.output'],
  }};
}
```

Rejected: dual-emitting both names on the span (doubles the biggest attributes for every external sink) and a clean break (impossible given un-upgradable Dev UIs). Open: full inventory of `genkit:*` consumers, and whether `genkit.state` is needed at all given span status carries the same info.

**D8 — What does the GCP plugin become?**

```ts
// Four layers; each is the previous one wrapped. Developer picks their entry point:
googleCloudTraceExporter()      // Adjusting inside: Cloud Trace truncation + labels (no redaction — D2)
googleCloudSpanProcessor()      // batch-wrapped → one line in ANY NodeSDK (the main case)
googleCloudTelemetry()          // { spanProcessors, metricReaders, resource } — full bundle
enableGoogleCloudTelemetry()    // = assertNoProviderRegistered();
                                //   new NodeSDK(googleCloudTelemetry(opts)).start();
                                //   the batteries one-liner: the APP instantiating the SDK,
                                //   loud error if a provider exists (no more silent race)
```

Metrics compose on current sdk-node (`metricReaders: []`, plural; singular deprecated) — pin the version or document the fallback. Cloud Logging rides the bootstrapper (outside NodeSDK config). Phase 2 direction: core emits standard `gen_ai.client.*` metrics via the metrics API; the plugin's span-derived metrics shrink. Serverless flush moves to the SDK owner. Firebase console is a `genkit:*` consumer → D7 inventory; likely gates the release.

## Cross-language portability (JS, Python; Go, Dart to follow)

Spec each rule as a behavior contract, not a mechanism — mechanisms differ per runtime:

| Contract (spec once) | JS | Python | Go | Dart |
|---|---|---|---|---|
| Never write the global in prod; in dev, write only if verified unset | slot guard is first-writer-wins; detect via ProxyTracerProvider delegate unwrap | first-wins (`Once` guard); isinstance NoOp/Proxy check | **last-writer-wins — no guard**; compare against package default before writing, never clobber | TBD (immature SDK) |
| Spans start from the caller's current context | AsyncLocalStorage (context manager registered by SDK) | `contextvars` (built into API) | explicit `ctx` param — no ambient; Genkit-Go already threads it | Zones |
| Dev piggyback is best-effort: attach Dev UI processor to a *recognized* SDK provider; else fallback (D1 open: dark + logged hint vs. no-piggyback) | `addSpanProcessor` + instanceof (breaks on duplicate api copies) | `add_span_processor` + lazy SDK import | `RegisterSpanProcessor` + type assertion | TBD |
| One semconv registry, generated bindings | codegen → semconv.ts | → semconv.py | → semconv.go | → semconv.dart |
| Depend on API only; SDK is dev-only/optional | peerDependency | `opentelemetry-api` dep; `-sdk` as extra | import `otel`, never `/sdk`, in library paths | TBD |
| Content gate: dev-on / prod-opt-in; `gen_ai.*.messages` serialized to one JSON schema | identical logic + shared schema fixtures across all four |||| 

Conformance suite runs the same assertions per language: the two-SDK race repro (adjusted for Go's last-wins), trace-shape test (one trace_id across app+Genkit spans), attribute snapshot against the shared registry, content-gate defaults per env.

## Migration & risks

- Wire spans switch vocabularies at the redesign major; compatibility lives in Genkit-owned sinks (D7): `TraceServerExporter` and the GCP exporter down-convert to legacy `genkit:*` in their payloads, so old Dev UI versions and the Firebase console work without upgrade. Shims deleted at the following major.
- Mapping table above = the migration guide: `genkit:type`→`genkit.type`, `agent:sessionId`→`gen_ai.conversation.id`, `genkit:input/output`→`gen_ai.input/output.messages` (gated) / `genkit.input/output` on non-model spans, span names→`chat {model}` / `execute_tool {name}`.
- gen_ai.* is pre-stable (Development status, own repo since semconv v1.42) → all names in `semconv.ts`, renames are one-file.
- Riskiest code: `tryClaimSlot`/`tryPiggyback` (slot claim + piggyback), esp. Go's last-writer-wins global. Prototype first.
- Acceptance test: Express app with own NodeSDK → Langfuse; run flow; assert one trace_id across HTTP+Genkit+DB spans, generation+cost render in Langfuse, Dev UI live simultaneously in dev. Mirror in Python.
