# RFC: First-class third-party observability export for Genkit

| | |
|---|---|
| Status | Draft |
| Author | Elliot Hesp |
| Date | 2026-06-08 |
| Affects | JS/TS, Go, Python, Dart |
| Discussion | TBD |

## Summary

Genkit instruments every flow, model, and tool call with OpenTelemetry spans, but
it only ships first-party telemetry export for Google Cloud / Firebase. The raw
spans use a Genkit-proprietary `genkit:*` attribute namespace. When those spans are
exported over OTLP to a third-party LLM observability backend (Langfuse, Arize
Phoenix, Braintrust, Laminar, PostHog, SigNoz, …), they ingest fine but render as
generic spans — not LLM "generations" — so users lose model badges, token/cost
charts, and generation-specific UI.

The gap is semantic, not transport: OTLP is already a universal wire format.
What is missing is a mapping layer that translates Genkit's `genkit:*` attributes
into the emerging industry standard, the
[OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
(`gen_ai.*`).

This RFC proposes a layered approach: ship one first-party `gen_ai.*` mapping
layer (Option B), then *document* how to plug each vendor's OTLP span processor on
top (Option C) — rather than building and maintaining a separate plugin per
vendor. Because all three Genkit runtimes emit the same `genkit:*` attributes, the
mapping is a shared spec; only the per-language exporter wiring differs. This is a
well-trodden path — the Vercel AI SDK converged on the same split (one first-party
`gen_ai.*` bridge plus per-vendor docs), which §4 examines as a useful reference
point.

### The documentation reflects the same Google-only gap

The code gap is mirrored in the docs. Genkit's entire "Observability and
monitoring" section is, in practice, a guide to the Firebase/Google Cloud plugin —
there is no vendor-neutral story for the wider ecosystem:

| Genkit doc page | What it actually covers |
|---|---|
| [Getting started](https://genkit.dev/docs/js/observability/getting-started/) | Add the `@genkit-ai/firebase` plugin, `enableFirebaseTelemetry()`, enable Cloud Logging/Trace/Monitoring APIs |
| [Authentication](https://genkit.dev/docs/js/observability/authentication/) | Google Cloud / Firebase project ID, service-account roles, ADC |
| [Telemetry collection](https://genkit.dev/docs/js/observability/telemetry-collection/) | Which logs/traces/metrics the Firebase plugin sends to Google Cloud |
| [Advanced configuration](https://genkit.dev/docs/js/observability/advanced-configuration/) | `enableFirebaseTelemetry({...})` options (export interval, sampling, etc.) |
| [Troubleshooting](https://genkit.dev/docs/js/observability/troubleshooting/) | Debugging why data isn't showing in Google Cloud Monitoring |
| [Google Cloud plugin](https://genkit.dev/docs/js/integrations/google-cloud/) | `enableGoogleCloudTelemetry()`, Cloud Trace/Monitoring, Model Armor |

Every page assumes the destination is Google Cloud Monitoring. A developer who
exports anywhere else has no first-party path to follow.

This is a framing problem as much as a feature gap. Genkit instruments on open
standards (OpenTelemetry), so it should present observability as ecosystem-wide —
Langfuse, Phoenix, Braintrust, SigNoz, PostHog, and others alongside Google — with
Google Cloud / Firebase positioned as one (very well-supported) provider among
many, not as the definition of "observability." The Vercel AI SDK is the
reference for this posture: a vendor-neutral observability surface plus a page per
backend. The work in this RFC (a shared `gen_ai.*` mapping in §2, the provider
catalog in §3) is what lets the docs be re-framed that way.

---

## 1. Problem

### 1.1 Genkit only ships telemetry for Google Cloud / Firebase

Telemetry export in Genkit is pluggable. The core entry point is
`enableTelemetry(telemetryConfig)`:

```97:106:js/core/src/tracing.ts
export async function enableTelemetry(
  telemetryConfig: TelemetryConfig | Promise<TelemetryConfig>
) {
  if (isOTelInitializationDisabled()) {
    return;
  }
  global[instrumentationKey] =
    telemetryConfig instanceof Promise ? telemetryConfig : Promise.resolve();
  return getTelemetryProvider().enableTelemetry(telemetryConfig);
}
```

`TelemetryConfig` is just the Node OTel SDK configuration:

```24:24:js/core/src/telemetryTypes.ts
export type TelemetryConfig = Partial<NodeSDKConfiguration>;
```

and the Node provider lets a caller supply their own `spanProcessors`:

```66:77:js/core/src/tracing/node-telemetry-provider.ts
  const processors: SpanProcessor[] = [createTelemetryServerProcessor()];
  if (nodeOtelConfig.traceExporter) {
    throw new Error('Please specify spanProcessors instead.');
  }
  if (nodeOtelConfig.spanProcessors) {
    processors.push(...nodeOtelConfig.spanProcessors);
  }
  if (nodeOtelConfig.spanProcessor) {
    processors.push(nodeOtelConfig.spanProcessor);
    delete nodeOtelConfig.spanProcessor;
  }
  nodeOtelConfig.spanProcessors = processors;
```

So the transport plumbing is generic. But the only *first-party* telemetry plugins
in the repo are Google-specific:

- `enableGoogleCloudTelemetry()` →

```38:47:js/plugins/google-cloud/src/index.ts
export function enableGoogleCloudTelemetry(
  options?: GcpTelemetryConfigOptions
) {
  return enableTelemetry(
    configureGcpPlugin(options).then(async (pluginConfig) => {
      logger.init(await new GcpLogger(pluginConfig).getLogger(getCurrentEnv()));
      return new GcpOpenTelemetry(pluginConfig).getConfig();
    })
  );
}
```

- `enableFirebaseTelemetry()` is a thin wrapper over the same call:

```40:45:js/plugins/firebase/src/index.ts
export async function enableFirebaseTelemetry(
  options?: FirebaseTelemetryOptions | GcpTelemetryConfigOptions
) {
  logger.debug('Initializing Firebase Genkit Monitoring.');
  await enableGoogleCloudTelemetry(options);
}
```

The same is true across languages: Go has `go/plugins/googlecloud/` and
`go/plugins/firebase/`; Python has `py/plugins/google-cloud/` and the Firebase
wrapper. There is no generic OTLP / third-party telemetry plugin in any
runtime. (The Python `py/plugins/README.md` advertises a planned `observability`
plugin for Sentry/Honeycomb/Datadog/Grafana/Axiom, but no such package exists in
the repo today — `py/plugins/` contains only `google-cloud` and the Firebase
wrapper for telemetry.)

The practical consequence: if you do not deploy to Google Cloud (self-hosted, AWS,
Fly, Render, Cloud Run paired with a non-Google backend, on-prem, …) Genkit gives
you no supported, generation-aware export path.

### 1.2 `genkit:*` vs the `gen_ai.*` standard

Every Genkit span is created through `runInNewSpan`, and all of its metadata is
serialized into attributes by `metadataToAttributes`, which hard-codes the
`genkit` prefix (`ATTR_PREFIX = 'genkit'`, `SPAN_TYPE_ATTR = 'genkit:type'`):

```221:239:js/core/src/tracing/instrumentation.ts
function metadataToAttributes(metadata: SpanMetadata): Record<string, string> {
  const out = {} as Record<string, string>;
  Object.keys(metadata).forEach((key) => {
    if (
      key === 'metadata' &&
      typeof metadata[key] === 'object' &&
      metadata.metadata
    ) {
      Object.entries(metadata.metadata).forEach(([metaKey, value]) => {
        out[ATTR_PREFIX + ':metadata:' + metaKey] = value;
      });
    } else if (key === 'input' || typeof metadata[key] === 'object') {
      out[ATTR_PREFIX + ':' + key] = JSON.stringify(metadata[key]);
    } else {
      out[ATTR_PREFIX + ':' + key] = metadata[key];
    }
  });
  return out;
}
```

The attributes Genkit emits today are:

| Attribute | Meaning | Set in |
|---|---|---|
| `genkit:type` | `action` / `flow` / `flowStep` / `util` / `userEngagement` | `runInNewSpan` labels |
| `genkit:metadata:subtype` | `model` / `tool` / `flow` | `js/core/src/action.ts:529` |
| `genkit:name` | action/flow/model name (e.g. `googleai/gemini-2.5-flash`) | metadata |
| `genkit:path` | hierarchical path, e.g. `/{myFlow,t:flow}/{generate,t:util}` | `buildPath` |
| `genkit:input` | request payload as a JSON string | metadata |
| `genkit:output` | response payload as a JSON string | metadata |
| `genkit:state` | `success` / `error` | `runInNewSpan` |
| `genkit:isRoot` | whether this is the top-level span | metadata |

The span "subtype" that marks a span as a model generation is attached when the
action runs:

```527:532:js/core/src/action.ts
        labels: {
          [SPAN_TYPE_ATTR]: 'action',
          'genkit:metadata:subtype': config.actionType,
          ...(genkitKey ? { 'genkit:key': genkitKey } : {}),
          ...options?.telemetryLabels,
        },
```

Model generate work is wrapped in a `util` span by the generate helper:

```142:166:js/ai/src/generate/action.ts
  return await runInNewSpan(
    {
      metadata: {
        name: options.rawRequest.stepName || 'generate',
      },
      labels: {
        [SPAN_TYPE_ATTR]: 'util',
      },
    },
    async (metadata) => {
      metadata.name = options.rawRequest.stepName || 'generate';
      metadata.input = options.rawRequest;
      const output = await generateActionImpl(registry, {
        rawRequest: options.rawRequest,
        middleware: options.middleware,
        currentTurn,
        messageIndex,
        abortSignal: options.abortSignal,
        streamingCallback: options.streamingCallback,
        context: options.context,
      });
      metadata.output = JSON.stringify(output);
      return output;
    }
  );
```

and flow steps in `run()` are wrapped as `flowStep` spans:

```196:209:js/core/src/flow.ts
  return runInNewSpan(
    {
      metadata: { name },
      labels: {
        [SPAN_TYPE_ATTR]: 'flowStep',
      },
    },
    async (meta) => {
      meta.input = input;
      const output = hasInput ? await func(input) : await func();
      meta.output = JSON.stringify(output);
      return output;
    }
  );
```

Go emits the identical attribute set:

```344:363:go/core/tracing/tracing.go
		attribute.String("genkit:name", sm.Name),
		attribute.String("genkit:state", string(sm.State)),
		attribute.String("genkit:input", base.JSONString(sm.Input)),
		attribute.String("genkit:path", sm.Path),
		// ...
		kvs = append(kvs, attribute.String("genkit:output", base.JSONString(sm.Output)))
		// ...
		kvs = append(kvs, attribute.String("genkit:type", sm.Type))
		// ...
		kvs = append(kvs, attribute.String("genkit:metadata:subtype", sm.Subtype))
```

(Python's `py/plugins/google-cloud/src/genkit/plugins/google_cloud/telemetry/`
modules consume the same attribute names.)

The industry is converging on the OpenTelemetry GenAI Semantic Conventions,
which use a `gen_ai.*` namespace, e.g. `gen_ai.operation.name`,
`gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
`gen_ai.provider.name` (formerly `gen_ai.system`), `gen_ai.response.finish_reasons`,
`gen_ai.input.messages`, `gen_ai.output.messages`.

### 1.3 What breaks in third-party UIs today

Because Genkit emits `genkit:*` (or `genkit/*` after the GCP plugin's
normalization — see §2.3), an LLM observability backend that keys off `gen_ai.*`
cannot recognize a Genkit model span as a generation. The
[Langfuse OpenTelemetry docs](https://langfuse.com/docs/opentelemetry/get-started)
are explicit about this: spans without `gen_ai.*` still ingest, but they are
shown as generic spans, while `gen_ai.request.model` and `gen_ai.usage.*` are what
map to Langfuse's generation/cost model. The same is true for Arize Phoenix,
Braintrust, Laminar, SigNoz, and PostHog (which converts `gen_ai.*` spans into
`$ai_generation` events).

Net effect for a Genkit user pointing raw OTLP at one of these tools today:

- traces appear (good), but
- model calls are not classified as generations,
- there are no token/cost rollups,
- model/provider badges and prompt/completion panels are empty.

The fix is not a new transport. It is a `genkit:* → gen_ai.*` mapping.

---

## 2. Mapping: `genkit:*` → `gen_ai.*`

### 2.1 Attribute mapping table

For a Genkit model span (`genkit:type == action` and
`genkit:metadata:subtype == model`, or the `util` `generate` span), produce the
following:

| Genkit attribute | Maps to (`gen_ai.*`) | Notes |
|---|---|---|
| `genkit:metadata:subtype == 'model'` | *marks the span as a generation* | gate the entire mapping on this |
| (implied operation) | `gen_ai.operation.name = 'chat'` | `embed` for embedders; `'chat'` is the common default for `generate` |
| `genkit:name` | `gen_ai.request.model` | e.g. `googleai/gemini-2.5-flash` (optionally split the `provider/` prefix into `gen_ai.provider.name`) |
| `genkit:input` (JSON) | `gen_ai.input.messages`, `gen_ai.request.*` | parse, then map messages + request config (temperature, max tokens, etc.) |
| `genkit:output` (JSON) | `gen_ai.output.messages`, `gen_ai.response.finish_reasons` | parse the `GenerateResponseData` |
| `genkit:output.usage` (JSON) | `gen_ai.usage.*` | see §2.2 for field-level mapping |
| `genkit:state` | span status | `success` → OK, `error` → ERROR |
| `genkit:type`, `genkit:path` | keep as custom attributes | preserve Genkit flow hierarchy for drill-down |
| `genkit:isRoot` | keep as custom attribute | useful for "top-level feature" rollups |

Non-model spans (`flow`, `flowStep`, `util`, `tool`) stay as ordinary spans; we
only *add* `gen_ai.*` to spans that represent model calls. Tool spans
(`subtype == 'tool'`) may optionally map to `gen_ai.operation.name = 'execute_tool'`
and `gen_ai.tool.name`.

### 2.2 Usage / token mapping

Genkit's usage shape is `GenerationUsageSchema`:

```260:275:js/ai/src/model-types.ts
export const GenerationUsageSchema = z.object({
  inputTokens: z.number().optional(),
  outputTokens: z.number().optional(),
  totalTokens: z.number().optional(),
  inputCharacters: z.number().optional(),
  outputCharacters: z.number().optional(),
  inputImages: z.number().optional(),
  outputImages: z.number().optional(),
  inputVideos: z.number().optional(),
  outputVideos: z.number().optional(),
  inputAudioFiles: z.number().optional(),
  outputAudioFiles: z.number().optional(),
  custom: z.record(z.number()).optional(),
  thoughtsTokens: z.number().optional(),
  cachedContentTokens: z.number().optional(),
});
```

(This is the same `usage` object the GCP plugin already reads in
`js/plugins/google-cloud/src/telemetry/generate.ts`.) Map it as:

| `GenerationUsage` field | `gen_ai.usage.*` |
|---|---|
| `inputTokens` | `gen_ai.usage.input_tokens` |
| `outputTokens` | `gen_ai.usage.output_tokens` |
| `totalTokens` | `gen_ai.usage.total_tokens` (non-standard but widely accepted) |
| `thoughtsTokens` | `gen_ai.usage.reasoning_tokens` / vendor extension |
| `cachedContentTokens` | `gen_ai.usage.cached_input_tokens` / vendor extension |
| `input/outputCharacters`, `images`, `videos`, `audioFiles`, `custom` | keep as `genkit.usage.*` custom attributes |

### 2.3 Implementation mechanism — Option B (one shared mapping exporter)

The cleanest place to run the mapping is an OTel `SpanExporter` /
`SpanProcessor` wrapper whose `export()` (or `onEnd()`) rewrites attributes — which
is exactly the pattern the GCP plugin already uses. `AdjustingTraceExporter`
wraps a downstream exporter and runs a pipeline of span rewrites on every batch:

```323:335:js/plugins/google-cloud/src/gcpOpenTelemetry.ts
  private adjust(spans: ReadableSpan[]): ReadableSpan[] {
    return spans.map((span) => {
      this.tickTelemetry(span);

      span = this.redactInputOutput(span);
      span = this.markErrorSpanAsError(span);
      span = this.markFailedSpan(span);
      span = this.markGenkitFeature(span);
      span = this.markGenkitModel(span);
      span = this.normalizeLabels(span);
      return span;
    });
  }
```

One of those steps, `normalizeLabels`, already demonstrates wholesale attribute
key rewriting — it converts `genkit:` to `genkit/` by replacing `:` with `/`:

```410:420:js/plugins/google-cloud/src/gcpOpenTelemetry.ts
  private normalizeLabels(span: ReadableSpan): ReadableSpan {
    const normalized = {} as Record<string, any>;
    for (const [key, value] of Object.entries(span.attributes)) {
      normalized[key.replace(/\:/g, '/')] = value;
    }
    return {
      ...span,
      spanContext: span.spanContext,
      attributes: normalized,
    };
  }
```

and `markGenkitModel` already isolates the "this span is a model call" condition we
need to key the mapping on:

```437:445:js/plugins/google-cloud/src/gcpOpenTelemetry.ts
  private markGenkitModel(span: ReadableSpan): ReadableSpan {
    if (
      span.attributes['genkit:metadata:subtype'] === 'model' &&
      !!span.attributes['genkit:name']
    ) {
      span.attributes['genkit:model'] = span.attributes['genkit:name'];
    }
    return span;
  }
```

The proposal is a GCP-independent version of this idea: a small span processor
that adds `gen_ai.*` attributes and can wrap *any* OTLP exporter. Sketch:

```ts
import {
  type ReadableSpan,
  type SpanProcessor,
} from '@opentelemetry/sdk-trace-base';

/ Adds OTel GenAI SemConv (`gen_ai.*`) attributes to Genkit model spans. */
export class GenAiMappingSpanProcessor implements SpanProcessor {
  constructor(private readonly downstream: SpanProcessor) {}

  onStart() {}
  forceFlush() {
    return this.downstream.forceFlush();
  }
  shutdown() {
    return this.downstream.shutdown();
  }

  onEnd(span: ReadableSpan): void {
    const attrs = span.attributes;
    const isModel = attrs['genkit:metadata:subtype'] === 'model';

    if (isModel) {
      // Mutating attributes in onEnd is the same technique the GCP plugin uses.
      const a = attrs as Record<string, unknown>;
      a['gen_ai.operation.name'] = 'chat';
      a['gen_ai.request.model'] = attrs['genkit:name'];

      const name = String(attrs['genkit:name'] ?? '');
      const slash = name.indexOf('/');
      if (slash > 0) a['gen_ai.provider.name'] = name.slice(0, slash);

      const output = safeParse(attrs['genkit:output']);
      const usage = output?.usage;
      if (usage) {
        if (usage.inputTokens != null)
          a['gen_ai.usage.input_tokens'] = usage.inputTokens;
        if (usage.outputTokens != null)
          a['gen_ai.usage.output_tokens'] = usage.outputTokens;
      }
      // ...map gen_ai.input.messages / gen_ai.output.messages /
      //    gen_ai.response.finish_reasons from genkit:input / genkit:output
    }

    this.downstream.onEnd(span);
  }
}

function safeParse(v: unknown): any | undefined {
  if (typeof v !== 'string') return undefined;
  try {
    return JSON.parse(v);
  } catch {
    return undefined;
  }
}
```

Because `enableTelemetry` accepts a `Partial<NodeSDKConfiguration>` with
`spanProcessors`, wiring this on top of any OTLP exporter is a few lines — and it
composes with vendor processors (§3, Option C):

```ts
import { enableTelemetry } from 'genkit/tracing';
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { GenAiMappingSpanProcessor } from '@genkit-ai/otel';

await enableTelemetry({
  spanProcessors: [
    new GenAiMappingSpanProcessor(
      new BatchSpanProcessor(
        new OTLPTraceExporter({
          url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT,
        })
      )
    ),
  ],
});
```

> Note: the Node provider rejects a bare `traceExporter` and requires
> `spanProcessors` (see `node-telemetry-provider.ts:67-68`), so the processor form
> above is the supported shape.

### 2.4 Cross-language parity

Since Go (`go/core/tracing/tracing.go`) and Python
(`py/plugins/google-cloud/.../telemetry/`) emit the same `genkit:*`
attributes, the mapping table in §2.1–§2.2 is a single shared spec. Only the
exporter wiring differs per runtime:

- Go: a `sdktrace.SpanProcessor` wrapper whose `OnEnd(ReadOnlySpan)` adds
  `gen_ai.*` attributes before delegating to an OTLP exporter
  (`go.opentelemetry.io/otel/exporters/otlp/otlptrace`).
- Python: a `opentelemetry.sdk.trace.SpanProcessor` whose `on_end(span)` does
  the same before an `OTLPSpanExporter`.

Publishing the mapping table as a language-neutral spec (e.g. in `engdoc/`) keeps
the three runtimes consistent.

---

## 3. Third-party providers

The relevant question for each backend is how it *attaches*, because that
determines whether Genkit needs code or just docs. The Vercel AI SDK's
[observability provider list](https://ai-sdk.dev/v7/providers/observability)
(mirrored in this repo under `ai-sdk/content/providers/05-observability/`) is a
convenient catalog of the same integrations and how they wire up. They fall into
three mechanisms:

### 3.1 OTLP span-processor / exporter based (the majority)

These consume standard OTLP spans and only need `gen_ai.*` to render generations —
i.e. they "just work" once Option B is in place, with the vendor's span
processor/exporter added downstream (Option C).

| Provider | Wiring | Notes |
|---|---|---|
| Langfuse | `LangfuseSpanProcessor` (`@langfuse/otel`) | also reads `langfuse.*` trace attributes |
| Braintrust | `BraintrustSpanProcessor` / `BraintrustExporter` | supports `filterAISpans`, `parent` |
| PostHog | `PostHogTraceExporter` (`@posthog/ai/otel`) | converts `gen_ai.*` → `$ai_generation`; `posthog_distinct_id` metadata |
| Arize AX / Phoenix | OTLP exporter / OpenInference | Phoenix is the lightweight self-host (see §3.4) |
| SigNoz | OTLP exporter | general-purpose OTel backend |
| Traceloop | OTLP / OpenLLMetry | |
| Axiom | OTLP exporter | |
| LangSmith, LangWatch, MLflow, Maxim, Patronus, Confident AI, Respan, Scorecard, Weave | OTLP / SDK exporter | all OTLP-ingest capable |

### 3.2 Custom tracer routing

| Provider | Wiring | Notes |
|---|---|---|
| Laminar | `Laminar.initialize()`, then pass `getTracer()` into the integration | uses its own tracer rather than a downstream span processor |

For Genkit, this maps to letting the user supply the tracer/provider and disabling
Genkit's default OTel init (`disableGenkitOTelInitialization()` exists for exactly
this).

### 3.3 Proxy / model-wrapper (no OTel at all)

| Provider | Wiring | Notes |
|---|---|---|
| Helicone | `@helicone/ai-sdk-provider` — swap the model | proxy-based; no spans |
| Literal AI | model wrapper | similar |

These are out of scope for the mapping layer — they intercept at the model
provider level, not the telemetry level. Genkit users would integrate them by
wrapping the model, not via telemetry config.

> The AI SDK list also includes HoneyHive and Sentry; both ingest OTLP and
> belong in the §3.1 bucket for Genkit's purposes.

### 3.4 Self-host footprint (a real selection factor)

If a user wants to self-host, infra weight matters and is worth documenting:

- Arize Phoenix — *lightweight*: a single container (SQLite by default), OTLP
  in, UI on `:6006`. The natural "drop-in local backend" recommendation.
- Langfuse (self-host) — *heavier*: ~6 services (web, worker, Postgres,
  ClickHouse, Redis, MinIO); docs recommend ~4 CPU / 16 GB.
- Laminar (default `docker compose`) — *heavier*: Postgres + ClickHouse +
  Quickwit + query-engine + frontend + app-server.

This footprint difference is why the recommendation is to ship a *generic* mapping
layer plus docs, rather than betting the first-party experience on any single
heavyweight backend.

---

## 4. Prior art: the layered approach in practice

The Vercel AI SDK is a useful existence proof for the layered approach, because it
faced the same fan-out problem and resolved it without a plugin-per-vendor. A key
nuance: the AI SDK does NOT build a plugin per provider. It ships a small
telemetry *core* plus one first-party OTel bridge, and leaves everything else
to vendor packages + docs.

### 4.1 The small core

`registerTelemetry(...integrations)` simply collects integrations on a global:

```6:15:ai-sdk/packages/ai/src/telemetry/telemetry-registry.ts
export function registerTelemetry(...integrations: Telemetry[]): void {
  if (!globalThis.AI_SDK_TELEMETRY_INTEGRATIONS) {
    globalThis.AI_SDK_TELEMETRY_INTEGRATIONS = [];
  }
  globalThis.AI_SDK_TELEMETRY_INTEGRATIONS.push(...integrations);
}
```

`create-telemetry-dispatcher.ts` fans the generation lifecycle out to every
registered integration, and the `Telemetry` interface
(`ai-sdk/packages/ai/src/telemetry/telemetry.ts`) is just a set of lifecycle
callbacks — `onStart`, `onStepStart`, `onLanguageModelCallStart/End`,
`onToolExecutionStart/End`, `onStepEnd`, `onEnd`, `onError`, plus
`executeLanguageModelCall` / `executeTool` wrappers for nesting. There is also a
passive Node `diagnostics_channel` (`ai.telemetry`) that vendors can subscribe to
without registering an integration
(`ai-sdk/packages/ai/src/telemetry/diagnostic-channel-publisher.ts`).

### 4.2 The one first-party bridge

The *only* observability integrations vendored in the AI SDK repo are:

- `@ai-sdk/otel` (`ai-sdk/packages/otel/`) — the `OpenTelemetry` class emits
  `gen_ai.*` GenAI-SemConv spans (`gen_ai.operation.name`,
  `gen_ai.provider.name`, `gen_ai.input.messages`, `gen_ai.output.messages`, …;
  see `gen-ai-format-messages.ts`), and `LegacyOpenTelemetry` emits the older
  `ai.*` spans. Its public surface is tiny:

```1:8:ai-sdk/packages/otel/src/index.ts
export { OpenTelemetry } from './open-telemetry';
export type {
  EnrichSpan,
  OpenTelemetryOptions,
  OpenTelemetrySpanType,
} from './open-telemetry';
export { LegacyOpenTelemetry } from './legacy-open-telemetry';
```

- `@ai-sdk/devtools` (`ai-sdk/packages/devtools/`) — a local SQLite-backed
  trace viewer for development (analogous to Genkit's Dev UI).

### 4.3 Everything else is docs + a vendor package

Every other backend — Langfuse, PostHog, Braintrust, Laminar, SigNoz, Arize,
Axiom, etc. — is not vendored. Each is:

1. an MDX doc page under
   `ai-sdk/content/providers/05-observability/*.mdx` (19 pages today), and
2. a vendor-maintained npm package the user installs and wires themselves.

The [PostHog page](https://ai-sdk.dev/v7/providers/observability/posthog) is the
canonical example — it is *docs only*: install `@posthog/ai`, then wire
`PostHogTraceExporter` into your own OTel `NodeSDK` alongside the AI SDK bridge:

```ts
// from ai-sdk/content/providers/05-observability/posthog.mdx
import { registerTelemetry } from 'ai';
import { LegacyOpenTelemetry } from '@ai-sdk/otel';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { PostHogTraceExporter } from '@posthog/ai/otel';

registerTelemetry(new LegacyOpenTelemetry());

const sdk = new NodeSDK({
  traceExporter: new PostHogTraceExporter({
    apiKey: process.env.POSTHOG_API_KEY!,
    host: 'https://us.i.posthog.com',
  }),
});
sdk.start();
```

### 4.4 Contrast with Genkit today

| | AI SDK | Genkit (today) |
|---|---|---|
| Lifecycle/telemetry core | `registerTelemetry` + dispatcher + diagnostics channel | OTel spans via `runInNewSpan` |
| First-party standard bridge | `@ai-sdk/otel` → `gen_ai.*` | none (only `genkit:*`) |
| Local dev viewer | `@ai-sdk/devtools` | Dev UI + telemetry server (`TraceServerExporter`) |
| Vendor backends | docs + vendor npm packages | GCP/Firebase only, vendored |

Genkit already has the lifecycle instrumentation and a dev viewer. The one piece
it lacks is the first-party `gen_ai.*` bridge and the per-vendor docs — the
same two pieces the comparison above shows the AI SDK leaning on.

---

## 5. Proposal / Recommendation

Adopt the layered approach: one first-party mapping layer, then per-vendor docs.

### Option B (build): ship one first-party `gen_ai.*` mapping layer

Add a single, GCP-independent mapping that translates `genkit:*` → `gen_ai.*` on
model spans (and optionally tool spans), packaged either as:

- a new `@genkit-ai/otel` plugin exporting a `GenAiMappingSpanProcessor`
  (and a convenience `enableOtelTelemetry({ exporter })`), or
- a flag on the existing telemetry config (e.g.
  `enableTelemetry({ genAiSemconv: true, spanProcessors: [...] })`).

A processor/exporter wrapper is the right mechanism because Genkit already proves
the pattern with `AdjustingTraceExporter.adjust()` (§2.3) — we are generalizing
"rewrite spans on export" away from Google Cloud.

### Option C (document): per-provider OTLP processors layered on top

Do not maintain a plugin per vendor. Instead, ship docs (one page per backend,
mirroring `ai-sdk/content/providers/05-observability/`) that show how to add each
vendor's OTLP span processor downstream of the mapping layer — e.g.
`LangfuseSpanProcessor`, `PostHogTraceExporter`, `BraintrustSpanProcessor`,
SigNoz/Traceloop/Axiom OTLP exporters, and the Phoenix lightweight self-host quick
start. Vendor-specific extras (`langfuse.*` trace attributes,
`posthog_distinct_id`, Braintrust `parent` / `filterAISpans`) are configured in the
vendor processor, on top of the shared `gen_ai.*` mapping.

### Cross-language note

Define the §2 mapping table once as a language-neutral spec. Implement the wrapper
three times (JS `SpanProcessor`, Go `sdktrace.SpanProcessor`, Python
`SpanProcessor`). The mapping is shared; only wiring differs.

### Caveat: what raw OTLP export loses vs the GCP plugin

The GCP plugin does more than export spans — at export time it also derives, via
`tickTelemetry`, a set of metrics (`genkit/feature/requests`,
`genkit/ai/generate/input/tokens`, `genkit/ai/generate/latency`, …) and structured
logs (`Input[...]`, `Output[...]`, `Error[...]`) from the telemetry handler
modules in `js/plugins/google-cloud/src/telemetry/` (`feature.ts`, `generate.ts`,
`action.ts`, `path.ts`, `engagement.ts`):

```337:376:js/plugins/google-cloud/src/gcpOpenTelemetry.ts
  private tickTelemetry(span: ReadableSpan) {
    const attributes = span.attributes;
    if (!Object.keys(attributes).includes('genkit:type')) {
      return;
    }

    const type = attributes['genkit:type'] as string;
    const subtype = attributes['genkit:metadata:subtype'] as string;
    const isRoot = !!span.attributes['genkit:isRoot'];

    pathsTelemetry.tick(span, this.logInputAndOutput, this.projectId);
    if (isRoot) {
      featuresTelemetry.tick(span, this.logInputAndOutput, this.projectId);
      span.attributes['genkit:rootState'] = span.attributes['genkit:state'];
    } else {
      if (type === 'action' && subtype === 'model') {
        generateTelemetry.tick(span, this.logInputAndOutput, this.projectId);
      }
      // ...
    }
  }
```

Pure OTLP span export does not reproduce these derived metrics/logs. The RFC
should decide one of:

1. Replicate in the mapping layer — emit equivalent OTel *metrics*
   (`gen_ai.client.token.usage`, operation duration histograms) from the same span
   stream, so third-party backends with metrics dashboards get parity. This is more
   work but closes the gap.
2. Leave GCP-only — ship spans (with `gen_ai.*`) for third parties and keep
   derived metrics/logs as a Google Cloud value-add. Simpler; most third-party LLM
   UIs derive their own token/cost rollups from `gen_ai.usage.*` on spans anyway.

Recommendation: start with (2) (spans-first, since that unlocks all of §3 with
the least surface area), and treat (1) as a fast-follow if users ask for
metric-dashboard parity.

### Suggested phasing

1. Phase 1 — JS `@genkit-ai/otel` `GenAiMappingSpanProcessor` (model spans →
   `gen_ai.*` + usage), plus generic OTLP docs.
2. Phase 2 — per-vendor doc pages (Langfuse, PostHog, Braintrust, Phoenix,
   SigNoz, Axiom, …) following the AI SDK structure.
3. Phase 3 — Go / Python / Dart parity processors against the shared spec.
4. Phase 4 (optional) — derived `gen_ai.*` metrics for dashboard parity.

### Open questions

- Package vs. config flag: standalone `@genkit-ai/otel` plugin, or an option on
  `enableTelemetry`?
- Should the mapping keep `genkit:*` alongside `gen_ai.*` (recommended, for
  flow drill-down) or replace it?
- Embedders/rerankers/tools — map now (`embed` / `execute_tool`) or model-only
  first?
- Do we adopt `gen_ai.input.messages` / `gen_ai.output.messages` (current SemConv)
  vs. the older event-based content convention, given backend support varies?
- How to handle the `genkit:input`/`genkit:output` JSON being potentially large or
  sensitive (the GCP plugin redacts via `redactInputOutput`) — opt-in capture?

---

## References

- OpenTelemetry GenAI Semantic Conventions — https://opentelemetry.io/docs/specs/semconv/gen-ai/
- AI SDK observability providers — https://ai-sdk.dev/v7/providers/observability
- AI SDK PostHog (docs-only example) — https://ai-sdk.dev/v7/providers/observability/posthog
- Langfuse OpenTelemetry — https://langfuse.com/docs/opentelemetry/get-started
- Arize Phoenix (self-host) — https://docs.arize.com/phoenix/deployment
- Genkit code: `js/core/src/tracing/instrumentation.ts`, `js/core/src/tracing.ts`,
  `js/core/src/tracing/node-telemetry-provider.ts`,
  `js/plugins/google-cloud/src/gcpOpenTelemetry.ts`,
  `js/ai/src/model-types.ts`, `go/core/tracing/tracing.go`
