# Genkit Telemetry Overview

This document outlines the core OpenTelemetry instrumentation strategy for Genkit, organized by Trace Spans and Trace Events.

---

## Trace Spans

Trace spans are the primary building blocks of Genkit's observability. Each major operation (Flow, Model, Tool, etc.) is represented as a span with a specific set of attributes.

### Span Attributes Overview

Genkit uses a `genkit:` prefix for its custom attributes.

#### Core Attributes
These attributes are fundamental to Genkit's trace structure and are present on most Genkit-instrumented spans.

| Attribute | Sample | Description | Consumer |
| :--- | :--- | :--- | :--- |
| `genkit:key` | `"/flow/myFlow"` | Registry key for the action. | Dev UI, AI Monitoring |
| `genkit:type` | `"action"` | High-level span type (e.g. `action`, `util`, `flowStep`). | Dev UI, AI Monitoring |
| `genkit:name` | `"myFlow"` | Friendly name of the operation. | Dev UI, AI Monitoring |
| `genkit:input` | `"{\"id\": 123}"` | JSON-stringified input. | Dev UI, AI Monitoring |
| `genkit:output` | `"{\"status\": \"ok\"}"`| JSON-stringified result. | Dev UI, AI Monitoring |
| `genkit:state` | `"success"` | Operation outcome (`success` or `error`). | Dev UI, AI Monitoring |
| `genkit:isRoot`| `true` | (Optional) Marks the entry point span of a Genkit execution. | Dev UI, AI Monitoring |
| `genkit:metadata:subtype`| `"flow"` | Specific category for spans of type `action`. | Dev UI, AI Monitoring |

#### Supplemental Attributes
These attributes support advanced monitoring features, specific application patterns, or cloud-specific visualizations.

| Source | Attribute | Sample | Description | Consumer |
| :--- | :--- | :--- | :--- | :--- |
| Genkit | `genkit:path` | `"/flow1/{step1,t:action}"`| Unique execution path in trace hierarchy. | AI Monitoring |
| Genkit | `genkit:isFailureSource`| `true` | Marks the specific span where an error originated. | AI Monitoring |
| Genkit | `genkit:sessionId`| `"sess-abc-123"` | (Optional) Unique identifier for a chat session. | Dev UI, AI Monitoring |
| Genkit | `genkit:threadName`| `"main"` | (Optional) Identifier for a specific chat thread. | Dev UI, AI Monitoring |
| Genkit | `genkit:metadata:context`| `"{\"auth\":...}"`| (Optional) JSON-stringified `ActionContext`. | Dev UI, AI Monitoring |
| Genkit | `genkit:lastKnownParentSpanId`| `"a1b2c3d4..."` | Used to maintain Genkit hierarchy across non-Genkit spans. | Dev UI, AI Monitoring |
| GCP | `genkit:feature`| `"menuSuggestionFlow"` | (Root only) Maps the trace to a logical Genkit feature. | AI Monitoring |
| GCP | `genkit:model` | `"googleai/gemini-1.5-flash"`| (Model only) Used for aggregating metrics by model ID. | AI Monitoring |
| GCP | `genkit:rootState`| `"success"` | (Root only) Captures overall outcome of feature execution. | AI Monitoring |
| GCP | `genkit:failedSpan`| `"getWeather"` | (Failure only) The name of the action that failed. | AI Monitoring |
| GCP | `genkit:failedPath`| `"/flow/{getWeather,t:action}"`| (Failure only) The exact path where failure occurred. | AI Monitoring |
| GCP | `/http/status_code` | `"599"` | Workaround to trigger red exclamation mark in GCP Trace. | AI Monitoring (GCP Trace) |
| GCP | `cloud.*` | `"gcp_cloud_run"` | (Resource) Environmental metadata (Project ID, Zone, etc.). | GCP Console |

### Action Index

| Action Type / Feature   | `genkit:type` (Span Type) | `genkit:metadata:subtype` | Other Notable Attributes                |
| :---------------------- | :------------------------ | :------------------------ | :-------------------------------------- |
| **Flow**                | `action`                  | `flow`                    | `genkit:isRoot`, `...metadata:context`  |
| **Flow Step (`run`)**   | `flowStep`                | -                         | `genkit:metadata:flow:stepType`         |
| **Model**               | `action`                  | `model`                   | `genkit:isRoot` (opt), `...context` (opt)|
| **Tool**                | `action`                  | `tool`                    | `genkit:isRoot` (opt), `...resumed`     |
| **Prompt**              | `action`                  | `prompt`                  | `genkit:isRoot` (opt), `...context` (opt)|
| **Retriever**           | `action`                  | `retriever`               | `genkit:isRoot` (opt), `...context` (opt)|
| **Indexer**             | `action`                  | `indexer`                 | `genkit:isRoot` (opt), `...context` (opt)|
| **Evaluator**           | `action`                  | `evaluator`               | `genkit:isRoot` (opt), `...context` (opt)|
| **Evaluator Batch**     | -                         | -                         | `genkit:metadata:evaluator:evalRunId`   |
| **Evaluator Test Case** | -                         | -                         | `genkit:metadata:evaluator:evalRunId`   |
| **Generate (util)**     | `action`                  | `util`                    | `genkit:isRoot` (opt), `...context` (opt)|
| **Generate Helper**     | `util`                    | -                         |                                         |
| **Dotprompt Render**    | `promptTemplate`          | -                         |                                         |
| **ExecutablePrompt**    | `dotprompt`               | -                         |                                         |
| **Chat Send**           | `helper`                  | -                         | `genkit:sessionId`, `genkit:threadName` |

### Detailed Span Definitions

#### Flow
High-level orchestration actions. Flows **are** actions.

| Source | Attribute | Sample | Description |
| :--- | :--- | :--- | :--- |
| Genkit | `genkit:key` | `"/flow/menuSuggestionFlow"` | Registry key for the flow. |
| Genkit | `genkit:type` | `"action"` | Fixed to `action`. |
| Genkit | `genkit:name` | `"menuSuggestionFlow"` | Flow name. |
| Genkit | `genkit:input` | `"{\"restaurantId\": \"123\"}"` | Flow input. |
| Genkit | `genkit:output` | `"{\"suggestions\": [...]}"` | Flow result. |
| Genkit | `genkit:state` | `"success"` | Operation state. |
| Genkit | `genkit:isRoot` | `true` | Always true for the entry-point flow span. |
| Genkit | `genkit:path` | `"/menuSuggestionFlow"` | Execution path (specific to AI monitoring). |
| Genkit | `genkit:metadata:subtype` | `"flow"` | Fixed to `flow`. |
| Genkit | `genkit:metadata:context` | `"{\"auth\":...}"` | (Optional) JSON-stringified `ActionContext`. |
| GCP | `genkit:feature` | `"menuSuggestionFlow"` | (Root only) Maps trace to logical feature. |
| GCP | `genkit:rootState` | `"success"` | (Root only) Captures final state of the entire feature. |
| GCP | `genkit:failedSpan` | `"getWeather"` | (Failure only) Name of failing span. |
| GCP | `genkit:failedPath` | `"/flow/{getWeather...}"` | (Failure only) Path of failing span. |

#### Flow Step (`run`)
Internal steps within a flow created using `genkit.run()`.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:key` | - | Not present on steps. |
| `genkit:type` | `"flowStep"` | Fixed to `flowStep`. |
| `genkit:name` | `"call-llm"` | Friendly name of the step. |
| `genkit:input` | `"{\"prompt\": \"...\"}"` | Data passed to the step. |
| `genkit:output` | `"{\"text\": \"...\"}"` | Result returned from the step. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:path` | `"/flow/{step,t:flowStep}"` | Execution path (specific to AI monitoring). |
| `genkit:metadata:flow:stepType` | `"run"` | The type of flow interaction. |

#### Model
Actions wrapping LLM providers.

| Source | Attribute | Sample | Description |
| :--- | :--- | :--- | :--- |
| Genkit | `genkit:key` | `"/model/googleai/gemini-1.5-flash"` | Registry key for the model. |
| Genkit | `genkit:type` | `"action"` | Fixed to `action`. |
| Genkit | `genkit:name` | `"googleai/gemini-1.5-flash"` | Model ID. |
| Genkit | `genkit:input` | `"{\"messages\": [...]}"` | Generation request. |
| Genkit | `genkit:output` | `"{\"candidates\": [...]}"` | Generation response. |
| Genkit | `genkit:state` | `"success"` | Operation state. |
| Genkit | `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| Genkit | `genkit:path` | `"/myFlow/{...flash,t:action}"` | Execution path (specific to AI monitoring). |
| Genkit | `genkit:metadata:subtype` | `"model"` | Fixed to `model`. |
| Genkit | `genkit:metadata:context` | `"{\"auth\":...}"` | (Optional) JSON-stringified `ActionContext`. |
| GCP | `genkit:model` | `"googleai/gemini-1.5-flash"`| (Model only) Used for aggregating metrics by model ID. |

#### Tool
Actions used for function calling. Supports standard and multipart (v2) tools.

| Source | Attribute | Sample | Description |
| :--- | :--- | :--- | :--- |
| Genkit | `genkit:key` | `"/tool/getWeather"` | Registry key for the tool. |
| Genkit | `genkit:type` | `"action"` | Fixed to `action`. |
| Genkit | `genkit:name` | `"getWeather"` | Tool name. |
| Genkit | `genkit:input` | `"{\"location\": \"New York\"}"` | Tool arguments. |
| Genkit | `genkit:output` | `"\"Sunny, 75°F\""` | Tool execution result. |
| Genkit | `genkit:state` | `"success"` | Operation state. |
| Genkit | `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| Genkit | `genkit:path` | `"/myFlow/{getWeather,t:action}"` | Execution path (specific to AI monitoring). |
| Genkit | `genkit:metadata:subtype` | `"tool"` or `"tool.v2"` | Category of tool. |
| Genkit | `genkit:metadata:context` | `"{\"auth\":...}"` | (Optional) JSON-stringified `ActionContext`. |
| Genkit | `genkit:metadata:resumed` | `"true"` | (Optional) Metadata if resumed after interrupt. |
| Genkit | `genkit:metadata:interrupt`| `"{\"reason\":\"...\"}"` | (Optional) Metadata if execution interrupted. |

#### Prompt
Prompts typically produce multiple spans during execution.

**Execution:** Top-level action span for the executable prompt.

| Source | Attribute | Sample | Description |
| :--- | :--- | :--- | :--- |
| Genkit | `genkit:key` | `"/executable-prompt/summarizer"` | Registry key for the prompt action. |
| Genkit | `genkit:type` | `"action"` | Fixed to `action`. |
| Genkit | `genkit:name` | `"summarizer"` | Name of the prompt. |
| Genkit | `genkit:input` | `"{\"text\": \"...\"}"` | Input to prompt execution. |
| Genkit | `genkit:output` | `"{\"candidates\": [...]}"` | Result of prompt execution. |
| Genkit | `genkit:state` | `"success"` | Operation state. |
| Genkit | `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| Genkit | `genkit:path` | `"/summarizer"` | Execution path (specific to AI monitoring). |
| Genkit | `genkit:metadata:subtype` | `"executable-prompt"` | Action category. |
| Genkit | `genkit:metadata:context` | `"{\"auth\":...}"` | (Optional) JSON-stringified `ActionContext`. |

**Render:** Span covering the logic of populating the template.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:type` | `"promptTemplate"` | Fixed type for render steps. |
| `genkit:name` | `"render"` | Fixed name for render steps. |
| `genkit:input` | `"{\"text\": \"...\"}"` | Data passed to the template. |
| `genkit:output` | `"{\"model\": \"...\", ...}"` | The resulting `GenerateOptions`. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:path` | `"/summarizer/{render,t:promptTemplate}"`| Execution path (specific to AI monitoring). |

**Generate:** The executable prompt wraps a standard generation call. See the [Generate](#generate) section below for details.

#### Retriever
Actions used to query and fetch relevant documents.

| Source | Attribute | Sample | Description |
| :--- | :--- | :--- | :--- |
| Genkit | `genkit:key` | `"/retriever/myRetriever"` | Registry key for the retriever. |
| Genkit | `genkit:type` | `"action"` | Fixed to `action`. |
| Genkit | `genkit:name` | `"myRetriever"` | Retriever name. |
| Genkit | `genkit:input` | `"{\"query\": {...}, \"options\": {...}}"` | Retrieval query and options. |
| Genkit | `genkit:output` | `"{\"documents\": [...]}"` | Resulting documents. |
| Genkit | `genkit:state` | `"success"` | Operation state. |
| Genkit | `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| Genkit | `genkit:path` | `"/flow/{myRetriever,t:action}"` | Execution path (specific to AI monitoring). |
| Genkit | `genkit:metadata:subtype` | `"retriever"` | Category of action. |
| Genkit | `genkit:metadata:context` | `"{\"auth\":...}"` | (Optional) JSON-stringified `ActionContext`. |

#### Indexer
Actions used to add documents to a searchable index.

| Source | Attribute | Sample | Description |
| :--- | :--- | :--- | :--- |
| Genkit | `genkit:key` | `"/indexer/myIndexer"` | Registry key for the indexer. |
| Genkit | `genkit:type` | `"action"` | Fixed to `action`. |
| Genkit | `genkit:name` | `"myIndexer"` | Indexer name. |
| Genkit | `genkit:input` | `"{\"documents\": [...], ...}"` | Documents to be indexed. |
| Genkit | `genkit:output` | `"null"` | Indexers typically return void. |
| Genkit | `genkit:state` | `"success"` | Operation state. |
| Genkit | `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| Genkit | `genkit:path` | `"/flow/{myIndexer,t:action}"` | Execution path (specific to AI monitoring). |
| Genkit | `genkit:metadata:subtype` | `"indexer"` | Category of action. |
| Genkit | `genkit:metadata:context` | `"{\"auth\":...}"` | (Optional) JSON-stringified `ActionContext`. |

#### Embedder
Actions that convert data into vector embeddings.

| Source | Attribute | Sample | Description |
| :--- | :--- | :--- | :--- |
| Genkit | `genkit:key` | `"/embedder/googleai/text-embedding-004"` | Registry key for the embedder. |
| Genkit | `genkit:type` | `"action"` | Fixed to `action`. |
| Genkit | `genkit:name` | `"googleai/text-embedding-004"` | Embedder ID. |
| Genkit | `genkit:input` | `"{\"input\": [...], ...}"` | Data to be embedded. |
| Genkit | `genkit:output` | `"{\"embeddings\": [...]}"` | Resulting vector embeddings. |
| Genkit | `genkit:state` | `"success"` | Operation state. |
| Genkit | `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| Genkit | `genkit:path` | `"/flow/{...004,t:action}"` | Execution path (specific to AI monitoring). |
| Genkit | `genkit:metadata:subtype` | `"embedder"` | Category of action. |
| Genkit | `genkit:metadata:context` | `"{\"auth\":...}"` | (Optional) JSON-stringified `ActionContext`. |

#### Reranker
Actions that re-order documents based on relevance.

| Source | Attribute | Sample | Description |
| :--- | :--- | :--- | :--- |
| Genkit | `genkit:key` | `"/reranker/myReranker"` | Registry key for the reranker. |
| Genkit | `genkit:type` | `"action"` | Fixed to `action`. |
| Genkit | `genkit:name` | `"myReranker"` | Reranker name. |
| Genkit | `genkit:input` | `"{\"query\": {...}, \"documents\": [...]}"` | Query and documents to rerank. |
| Genkit | `genkit:output` | `"{\"documents\": [...]}"` | Re-ordered documents with scores. |
| Genkit | `genkit:state` | `"success"` | Operation state. |
| Genkit | `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| Genkit | `genkit:path` | `"/flow/{myReranker,t:action}"` | Execution path (specific to AI monitoring). |
| Genkit | `genkit:metadata:subtype` | `"reranker"` | Category of action. |
| Genkit | `genkit:metadata:context` | `"{\"auth\":...}"` | (Optional) JSON-stringified `ActionContext`. |

#### Evaluator
Evaluators assess the quality of AI outputs.

**Execution:** Top-level action span.

| Source | Attribute | Sample | Description |
| :--- | :--- | :--- | :--- |
| Genkit | `genkit:key` | `"/evaluator/myEvaluator"` | Registry key for the evaluator. |
| Genkit | `genkit:type` | `"action"` | Fixed to `action`. |
| Genkit | `genkit:name` | `"myEvaluator"` | Evaluator name. |
| Genkit | `genkit:input` | `"{\"dataset\": [...], \"evalRunId\": \"...\"}"` | Evaluation request. |
| Genkit | `genkit:output` | `"[{\"testCaseId\": \"...\", ...}]"` | Full evaluation results. |
| Genkit | `genkit:state` | `"success"` | Operation outcome. |
| Genkit | `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| Genkit | `genkit:path` | `"/myEvaluator"` | Execution path (specific to AI monitoring). |
| Genkit | `genkit:metadata:subtype` | `"evaluator"` | Category of action. |
| Genkit | `genkit:metadata:context` | `"{\"auth\":...}"` | (Optional) JSON-stringified `ActionContext`. |

**Batch / Test Case:** Sub-spans created during the loop (**Untyped**).

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:name` | `"Batch 0"` or `"Test Case X"` | Step identifier. |
| `genkit:input` | `"{...}"` | Input data for the step. |
| `genkit:output` | `"{...}"` | Result of the evaluation step. |
| `genkit:state` | `"success"` | Operation outcome. |
| `genkit:path` | `"/evaluator/{Batch 0}"` | Simplified path (no type). |
| `genkit:metadata:evaluator:evalRunId` | `"uuid-123"` | Links steps to the evaluation run. |

#### Generate
The core generation utility wrapper.

| Source | Attribute | Sample | Description |
| :--- | :--- | :--- | :--- |
| Genkit | `genkit:key` | `"/util/generate"` | Registry key (only if registered). |
| Genkit | `genkit:type` | `"util"` | Span type (often `util`). |
| Genkit | `genkit:name` | `"generate"` | Utility name. |
| Genkit | `genkit:input` | `"{\"model\": \"...\"}"` | Request options. |
| Genkit | `genkit:output` | `"{\"message\": {...}}"` | Generation result. |
| Genkit | `genkit:state` | `"success"` | Operation state. |
| Genkit | `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| Genkit | `genkit:path` | `"/myFlow/{generate,t:util}"` | Execution path (specific to AI monitoring). |
| Genkit | `genkit:metadata:context` | `"{\"auth\":...}"` | (Optional) JSON-stringified `ActionContext`. |

#### Chat (Helper)
Spans created by the Chat session manager.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:type` | `"helper"` | Fixed type for chat send operations. |
| `genkit:name` | `"send"` | Friendly name. |
| `genkit:sessionId` | `"sess-123"` | Unique session identifier. |
| `genkit:threadName` | `"main"` | Thread identifier. |
| `genkit:input` | `"{\"prompt\": \"...\"}"` | Chat input. |
| `genkit:output` | `"{...}"` | Chat response. |

### Action Registry Subtypes

Valid `ActionType` values appearing as `genkit:metadata:subtype` when span type is `action`:

- `background-model`, `cancel-operation`, `check-operation`, `custom`, `dynamic-action-provider`, `embedder`, `evaluator`, `executable-prompt`, `flow`, `indexer`, `model`, `prompt`, `reranker`, `resource`, `retriever`, `tool.v2`, `tool`, `util`

### Span

#### Attribute Naming Conventions
Genkit uses colons (`:`) as a namespace delimiter for its custom attributes (e.g., `genkit:metadata:subtype`).

- **Deviation from OTel Standard:** This naming style intentionally deviates from the [OpenTelemetry Attribute Naming Conventions](https://opentelemetry.io/docs/specs/semconv/general/attribute-naming/), which mandate lowercase, dot-separated names (e.g., `genkit.metadata.subtype`).
- **GCP Transformation:** When exporting to Google Cloud via the `google-cloud` plugin, these colons are automatically transformed into forward slashes (`/`) to align with legacy Cloud Trace UI conventions (e.g., `genkit/metadata/subtype`).
- **Visual Distinction:** The use of colons provides a clear visual distinction between Genkit-internal metadata and standard OTel auto-instrumentation attributes (which use dots).

#### Untyped Spans
Spans marked as **Untyped** (like Evaluator batches) use `runInNewSpan` but lack a `genkit:type` label. They are treated as generic trace steps, have simplified paths (`/{name}`), and may bypass internal instrumentation like `lastKnownParentSpanId` tracking.

#### Cross-Language Parity: Go

While the core Genkit telemetry strategy is shared across languages, the **Go** implementation introduces several specific attributes and path decoration logic to maintain compatibility and provide additional context.

##### Path Decoration (Subtypes)
Unlike the JS SDK, which primarily uses `t:type` in paths, the Go SDK's `decoratePathWithSubtype` function adds an explicit `s:subtype` annotation to the final path segment:
- **JS Style:** `/{myFlow}/{myTool,t:action}`
- **Go Style:** `/{myFlow,t:action,s:flow}/{myTool,t:action,s:tool}`

##### Automatic Flow Context Injection
In Go, any action executed within a flow context automatically has the following metadata attribute injected:
- `genkit:metadata:flow:name`: The name of the surrounding flow. 

**Why Go injects this:** While the JS SDK relies primarily on regex-based parsing of the `genkit:path` attribute to identify the root feature, the Go SDK explicitly propagates the flow name via context. This provides a more robust mechanism for telemetry plugins (like GCP) to attribute metrics to the correct root feature, especially in complex execution environments where the full path string might be truncated or unavailable.

##### Additional Flow Metadata
Go's flow implementation and conformance tests record more verbose flow state information than the current JS core:
- `genkit:metadata:flow:id`: A unique execution ID for the flow instance.
- `genkit:metadata:flow:dispatchType`: e.g., `"start"`, `"resume"`.
- `genkit:metadata:flow:state`: e.g., `"run"`, `"done"`.
- `genkit:metadata:flow:stepName`: The original name of the step before resolution.
- `genkit:metadata:flow:resolvedStepName`: The final unique name of the step.
- `genkit:metadata:flow:wrapperAction`: Set to `"true"` on the action span that wraps a flow.

##### Error Handling (markedError)
Go uses a specialized `markedError` type to track whether an error has already been recorded by a span. This ensures that `genkit:isFailureSource` is only set once on the original failing span and prevents redundant error recording as the exception bubbles up.

## Trace Events

Trace events capture point-in-time occurrences within a span.

### Error Handling (Exceptions)

When an exception occurs during execution, Genkit records the error as an OpenTelemetry `exception` event.

#### OTel Exception Event
Standard OpenTelemetry mappings are used:
- `exception.type`: The error class name.
- `exception.message`: The error message.
- `exception.stacktrace`: The full stack trace.

#### Associated Span Updates
On error, the following updates are made to the parent span:
- **Span Status:** Set to `SpanStatusCode.ERROR`.
- **Status Message:** Set to the error message (or `String(e)`).
- **genkit:state:** Set to `"error"`.
- **genkit:isFailureSource:** Set to `true` on the **first** span in the call stack where the error originated.

### Span Links
While not events, **Links** allow spans to reference other traces. Genkit uses these to link evaluation test cases back to the original generation traces.
