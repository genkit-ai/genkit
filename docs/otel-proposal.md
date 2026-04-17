# Genkit Telemetry Proposal

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
| GCP | `genkit:feature`| `"menuSuggestionFlow"` | (Root only) Maps the trace to a logical Genkit feature. | AI Monitoring |
| GCP | `genkit:model` | `"googleai/gemini-1.5-flash"`| (Model only) Used for aggregating metrics by model ID. | AI Monitoring |
| GCP | `genkit:rootState`| `"success"` | (Root only) Captures overall outcome of feature execution. | AI Monitoring |
| GCP | `genkit:failedSpan`| `"getWeather"` | (Failure only) The name of the action that failed. | AI Monitoring |
| GCP | `genkit:failedPath`| `"/flow/{getWeather,t:action}"`| (Failure only) The exact path where failure occurred. | AI Monitoring |
| GCP | `/http/status_code` | `"599"` | Workaround to trigger red exclamation mark in GCP Trace. | AI Monitoring (GCP Trace) |

### Action Index (Span Types)

| Action Type / Feature   | `genkit:type` (Span Type) | `genkit:metadata:subtype` | Other Notable Attributes                |
| :---------------------- | :------------------------ | :------------------------ | :-------------------------------------- |
| **Flow**                | `action`                  | `flow`                    | `genkit:isRoot`                         |
| **Flow Step (`run`)**   | `flowStep`                | -                         | `genkit:metadata:flow:stepType`         |
| **Model**               | `action`                  | `model`                   | `genkit:isRoot` (optional)              |
| **Tool**                | `action`                  | `tool`                    | `genkit:isRoot` (optional), `...resumed`|
| **Prompt**              | `action`                  | `prompt`                  | `genkit:isRoot` (optional)              |
| **Retriever**           | `action`                  | `retriever`               | `genkit:isRoot` (optional)              |
| **Indexer**             | `action`                  | `indexer`                 | `genkit:isRoot` (optional)              |
| **Evaluator**           | `action`                  | `evaluator`               | `genkit:isRoot` (optional)              |
| **Evaluator Batch**     | -                         | -                         | `genkit:metadata:evaluator:evalRunId`   |
| **Evaluator Test Case** | -                         | -                         | `genkit:metadata:evaluator:evalRunId`   |
| **Generate (util)**     | `action`                  | `util`                    | `genkit:isRoot` (optional)              |
| **Generate Helper**     | `util`                    | -                         |                                         |
| **Dotprompt Render**    | `promptTemplate`          | -                         |                                         |
| **ExecutablePrompt**    | `dotprompt`               | -                         |                                         |
| **Chat Send**           | `helper`                  | -                         | `genkit:sessionId`, `genkit:threadName` |

### Detailed Span Definitions

#### Flow
High-level orchestration actions. Flows **are** actions.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:key` | `"/flow/menuSuggestionFlow"` | Registry key for the flow. |
| `genkit:type` | `"action"` | Fixed to `action`. |
| `genkit:name` | `"menuSuggestionFlow"` | Flow name. |
| `genkit:input` | `"{\"restaurantId\": \"123\"}"` | Flow input. |
| `genkit:output` | `"{\"suggestions\": [...]}"` | Flow result. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:path` | `"/menuSuggestionFlow"` | Execution path (specific to AI monitoring). |
| `genkit:isRoot` | `true` | Always true for the entry-point flow span. |
| `genkit:metadata:subtype` | `"flow"` | Fixed to `flow`. |

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

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:key` | `"/model/googleai/gemini-1.5-flash"` | Registry key for the model. |
| `genkit:type` | `"action"` | Fixed to `action`. |
| `genkit:name` | `"googleai/gemini-1.5-flash"` | Model ID. |
| `genkit:input` | `"{\"messages\": [...]}"` | Generation request. |
| `genkit:output` | `"{\"candidates\": [...]}"` | Generation response. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| `genkit:path` | `"/myFlow/{...flash,t:action}"` | Execution path (specific to AI monitoring). |
| `genkit:metadata:subtype` | `"model"` | Fixed to `model`. |

#### Tool
Actions used for function calling. Supports standard and multipart (v2) tools.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:key` | `"/tool/getWeather"` | Registry key for the tool. |
| `genkit:type` | `"action"` | Fixed to `action`. |
| `genkit:name` | `"getWeather"` | Tool name. |
| `genkit:input` | `"{\"location\": \"New York\"}"` | Tool arguments. |
| `genkit:output` | `"\"Sunny, 75°F\""` | Tool execution result. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:path` | `"/myFlow/{getWeather,t:action}"` | Execution path (specific to AI monitoring). |
| `genkit:metadata:subtype` | `"tool"` or `"tool.v2"` | Category of tool. |
| `genkit:metadata:resumed` | `"true"` | (Optional) Metadata if resumed after interrupt. |
| `genkit:metadata:interrupt`| `"{\"reason\":\"...\"}"` | (Optional) Metadata if execution interrupted. |

#### Prompt
Prompts typically produce multiple spans during execution.

**Execution:** Top-level action span for the executable prompt.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:key` | `"/executable-prompt/summarizer"` | Registry key for the prompt action. |
| `genkit:type` | `"action"` | Fixed to `action`. |
| `genkit:name` | `"summarizer"` | Name of the prompt. |
| `genkit:input` | `"{\"text\": \"...\"}"` | Input to prompt execution. |
| `genkit:output` | `"{\"candidates\": [...]}"` | Result of prompt execution. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| `genkit:path` | `"/summarizer"` | Execution path (specific to AI monitoring). |
| `genkit:metadata:subtype` | `"executable-prompt"` | Action category. |

**Render:** Span covering the logic of populating the template.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:type` | `"promptTemplate"` | Fixed type for render steps. |
| `genkit:name` | `"render"` | Fixed name for render steps. |
| `genkit:input` | `"{\"text\": \"...\"}"` | Data passed to the template. |
| `genkit:output` | `"{\"model\": \"...\", ...}"` | The resulting `GenerateOptions`. |
| `genkit:state`            | `"success"`                       | Operation state.                            |
| `genkit:path`             | `"/summarizer/{render,t:promptTemplate}"`| Execution path (specific to AI monitoring). |

**Generate:** The executable prompt wraps a standard generation call to perform the actual model interaction. See the [Generate](#generate) section below for detailed span attributes.

#### Retriever

Actions used to query and fetch relevant documents.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:key` | `"/retriever/myRetriever"` | Registry key for the retriever. |
| `genkit:type` | `"action"` | Fixed to `action`. |
| `genkit:name` | `"myRetriever"` | Retriever name. |
| `genkit:input` | `"{\"query\": {...}, \"options\": {...}}"` | Retrieval query and options. |
| `genkit:output` | `"{\"documents\": [...]}"` | Resulting documents. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| `genkit:path` | `"/flow/{myRetriever,t:action}"` | Execution path (specific to AI monitoring). |
| `genkit:metadata:subtype` | `"retriever"` | Category of action. |

#### Indexer
Actions used to add documents to a searchable index.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:key` | `"/indexer/myIndexer"` | Registry key for the indexer. |
| `genkit:type` | `"action"` | Fixed to `action`. |
| `genkit:name` | `"myIndexer"` | Indexer name. |
| `genkit:input` | `"{\"documents\": [...], ...}"` | Documents to be indexed. |
| `genkit:output` | `"null"` | Indexers typically return void. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| `genkit:path` | `"/flow/{myIndexer,t:action}"` | Execution path (specific to AI monitoring). |
| `genkit:metadata:subtype` | `"indexer"` | Category of action. |

#### Embedder
Actions that convert data into vector embeddings.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:key` | `"/embedder/googleai/text-embedding-004"` | Registry key for the embedder. |
| `genkit:type` | `"action"` | Fixed to `action`. |
| `genkit:name` | `"googleai/text-embedding-004"` | Embedder ID. |
| `genkit:input` | `"{\"input\": [...], ...}"` | Data to be embedded. |
| `genkit:output` | `"{\"embeddings\": [...]}"` | Resulting vector embeddings. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| `genkit:path` | `"/flow/{...004,t:action}"` | Execution path (specific to AI monitoring). |
| `genkit:metadata:subtype` | `"embedder"` | Category of action. |

#### Reranker
Actions that re-order documents based on relevance.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:key` | `"/reranker/myReranker"` | Registry key for the reranker. |
| `genkit:type` | `"action"` | Fixed to `action`. |
| `genkit:name` | `"myReranker"` | Reranker name. |
| `genkit:input` | `"{\"query\": {...}, \"documents\": [...]}"` | Query and documents to rerank. |
| `genkit:output` | `"{\"documents\": [...]}"` | Re-ordered documents with scores. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| `genkit:path` | `"/flow/{myReranker,t:action}"` | Execution path (specific to AI monitoring). |
| `genkit:metadata:subtype` | `"reranker"` | Category of action. |

#### Evaluator
Evaluators assess the quality of AI outputs.

**Execution:** Top-level action span.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:key` | `"/evaluator/myEvaluator"` | Registry key for the evaluator. |
| `genkit:type` | `"action"` | Fixed to `action`. |
| `genkit:name` | `"myEvaluator"` | Evaluator name. |
| `genkit:input` | `"{\"dataset\": [...], \"evalRunId\": \"...\"}"` | Evaluation request. |
| `genkit:output` | `"[{\"testCaseId\": \"...\", ...}]"` | Full evaluation results. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| `genkit:path` | `"/myEvaluator"` | Execution path (specific to AI monitoring). |
| `genkit:metadata:subtype` | `"evaluator"` | Category of action. |

**Batch / Test Case:** Sub-spans created during the loop (**Untyped**).

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:name` | `"Batch 0"` or `"Test Case X"` | Step identifier. |
| `genkit:input` | `"{...}"` | Input data for the step. |
| `genkit:output` | `"{...}"` | Result of the evaluation step. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:path` | `"/evaluator/{Batch 0}"` | Simplified path (no type). |
| `genkit:metadata:evaluator:evalRunId` | `"uuid-123"` | Links steps to the evaluation run. |

#### Generate
The core generation utility wrapper.

| Attribute | Sample | Description |
| :--- | :--- | :--- |
| `genkit:key` | `"/util/generate"` | Registry key (only if registered). |
| `genkit:type` | `"util"` | Span type (often `util`). |
| `genkit:name` | `"generate"` | Utility name. |
| `genkit:input` | `"{\"model\": \"...\"}"` | Request options. |
| `genkit:output` | `"{\"message\": {...}}"` | Generation result. |
| `genkit:state` | `"success"` | Operation state. |
| `genkit:isRoot` | `true` | (Optional) Present if called directly. |
| `genkit:path` | `"/myFlow/{generate,t:util}"` | Execution path (specific to AI monitoring). |

### Action Registry Subtypes

Valid `ActionType` values appearing as `genkit:metadata:subtype` when span type is `action`:

- `background-model`, `cancel-operation`, `check-operation`, `custom`, `dynamic-action-provider`, `embedder`, `evaluator`, `executable-prompt`, `flow`, `indexer`, `model`, `prompt`, `reranker`, `resource`, `retriever`, `tool.v2`, `tool`, `util`

### Caveats

#### Untyped Spans
Spans marked as **Untyped** (like Evaluator batches) use `runInNewSpan` but lack a `genkit:type` label. They are treated as generic trace steps, have simplified paths (`/{name}`), and may bypass internal instrumentation like `lastKnownParentSpanId` tracking.

#### Attribute Naming Conventions
Genkit uses colons (`:`) as a namespace delimiter for its custom attributes (e.g., `genkit:metadata:subtype`). 

- **Deviation from OTel Standard:** This naming style intentionally deviates from the [OpenTelemetry Attribute Naming Conventions](https://opentelemetry.io/docs/specs/semconv/general/attribute-naming/), which mandate lowercase, dot-separated names (e.g., `genkit.metadata.subtype`).
- **GCP Transformation:** When exporting to Google Cloud via the `google-cloud` plugin, these colons are automatically transformed into forward slashes (`/`) to align with legacy Cloud Trace UI conventions (e.g., `genkit/metadata/subtype`).
- **Visual Distinction:** The use of colons provides a clear visual distinction between Genkit-internal metadata and standard OTel auto-instrumentation attributes (which use dots).

---

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
