# Genkit OpenTelemetry Specification

This document defines the official OpenTelemetry instrumentation standards for the Genkit framework across all supported languages. It does not define "how" data should be collected; just "what" data should be collected.

## 1. Instrumentation Scope

To ensure global uniqueness and clear attribution across a polyglot ecosystem, every Genkit SDK must use a **Logical Name** for the primary tracer identity, while providing **Physical Metadata** via scope attributes.

- **Name (Logical):** `genkit/<language>` (e.g., `genkit/go`, `genkit/js`, `genkit/py`, `genkit/dart`)
- **Version:** The semantic version of the SDK (e.g., `1.4.0`).
- **Schema URL:** `https://opentelemetry.io/schemas/1.27.0` (Targeting the most stable widely-supported schema).

### Scope Attributes
To maintain traceability to the source code and specific package releases, the following attributes **must** be included in the instrumentation scope (where supported by the language's OTel SDK):

| Attribute | Description | Example |
| :--- | :--- | :--- |
| `genkit.package` | The fully qualified package name or URL. | `github.com/firebase/genkit/go` |
| `genkit.version` | The specific semantic version of the package. | `1.4.0` |

**Rationale for Logical Names:**
Using `genkit/<language>` ensures a unified experience in distributed traces. It allows developers and tooling (like the Genkit Dev UI) to easily identify Genkit-produced spans regardless of the underlying language, while scope attributes preserve the technical precision required for deep debugging.

*Note: If a specific language SDK does not yet support scope attributes, the logical name must still be used, and attributes should be adopted as soon as API support becomes available.*

## 2. Genkit Semantic Conventions

Genkit follows standard OpenTelemetry naming conventions, using a `genkit.` prefix with dot-separated namespaces. 

### Naming Convention and Transition
*   **New Standard:** All Genkit-specific attributes must use the dot-separated format (e.g., `genkit.name`).
*   **Legacy Compatibility:** During the modernization phase, SDK exporters or OTel Collectors may perform "on-the-fly" conversion from the new `genkit.` format back to the legacy `genkit:` format (colon-separated) to maintain compatibility with the Genkit Dev UI and existing AI Monitoring tools until they are updated.

### Span Attributes
These attributes are fundamental to Genkit's trace structure and should be present on Genkit-instrumented spans as defined by their requirement level.

| Convention | Attribute | Requirement Level | Type | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Genkit | `genkit.name` | Required | string | Friendly name of the operation. | `"myFlow"` |
| Genkit | `genkit.type` | Required | string | High-level span type (`action`, `flowStep`, `util`). | `"action"` |
| Genkit | `genkit.key` | Recommended | string | Registry key for the action. | `"/flow/myFlow"` |
| Genkit | `genkit.input` | Opt-In | string | JSON-stringified input. (Enabled by default for local development). | `"{\"id\": 123}"` |
| Genkit | `genkit.output` | Opt-In | string | JSON-stringified result. (Enabled by default for local development). | `"{\"status\": \"ok\"}"` |
| Genkit | `genkit.state` | Required | string | Operation outcome (`success`, `error`). | `"success"` |
| Genkit | `genkit.isRoot` | Conditionally Required | boolean | Marks the entry point span of a Genkit execution. | `true` |
| Genkit | `genkit.lastKnownParentSpanId` | Recommended | string | Used to maintain Genkit hierarchy across non-Genkit spans. | `"a1b2c3d4..."` |
| Genkit | `genkit.path` | Recommended | string | Unique execution path in trace hierarchy. | `"/{flow1}/{step1,t:action}"` |
| Genkit | `genkit.isFailureSource` | Recommended | boolean | Marks the specific span where an error originated. | `true` |
| Genkit | `genkit.metadata.context` | Opt-In | string | JSON-stringified execution context (e.g., auth data). (Enabled by default for local development). | `"{\"auth\":...}"` |
| Genkit | `genkit.metadata.subtype` | Conditionally Required | string | Specific category for spans of type `action`. | `"flow"` |
| Genkit | `genkit.metadata.resumed` | Recommended | boolean | Marks if the operation was resumed after an interrupt. | `true` |

### Specialized Spans

#### Flows
- `genkit.metadata.subtype`: `flow`
- `genkit.isRoot`: `true` (if entry point)

#### Agents (New)
When a turn is executed within the context of a long-running session or conversation, the following attributes should be propagated to all spans within that run to enable trace grouping.

| Convention | Attribute | Requirement Level | Type | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Session | `session.id` | Required | string | Unique identifier for the user session. | `"sess-abc-123"` |
| Gen AI | `gen_ai.conversation.id` | Recommended | string | Identifier for the specific chat thread. | `"main"` |

WIP: Genkit should align with the OTel GenAI Semantic Conventions for [agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/#spans) for agents.

#### Models

Genkit aligns with the OTel GenAI Semantic Conventions for [model spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#spans). These attributes should be present on spans where `genkit.metadata.subtype` is `model`.

> [!NOTE]
> Genkit by convention does not break out input/output params into separate attributes. e.g. `gen_ai.request.top_k` would actually be included as part of `genkit.input`.
> 
> An audit is necessary to determine which attributes below should *always* be present, and which should be added via a developer choice (e.g. plugin). 
> 
> For example:
>
> - Attributes `gen_ai.operation.name`, `gen_ai.provider.name`, `server.address`, `server.port`, and others, that are not specifically covered by the Genkit spec, are candidates to include _always_.
> - We should (by default) forgo `gen_ai.input.messages` / `gen_ai.output.messages` in favor of `genkit.input` / `genkit.output`. These are very large, and duplication is costly. Input/output is _standard_ for all genkit action types, and are inherent to Genkit observability.
> - It would be prudent to include *some* overlapping data, such as `gen_ai.usage.input_tokens`, because they are not provided in a standard location by each model provider, and will frequently be observed, or aggregated.

| Convention | Attribute | Requirement Level | Type | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gen AI | `gen_ai.operation.name` | Required | string | The operation type. | `"chat"` |
| Gen AI | `gen_ai.provider.name` | Required | string | The provider/system name. | `"gcp.vertex_ai"` |
| Gen AI | `gen_ai.request.model` | Conditionally Required | string | The model ID requested. | `"gemini-1.5-flash"` |
| Gen AI | `gen_ai.request.stream` | Conditionally Required | boolean | Whether streaming was used. | `true` |
| Gen AI | `gen_ai.request.top_k` | Conditionally Required | double | Top-k sampling threshold. | `1.0` |
| Gen AI | `gen_ai.request.seed` | Conditionally Required | int | Seed for deterministic sampling. | `42` |
| Gen AI | `gen_ai.request.choice.count` | Conditionally Required | int | Number of completions to return. | `1` |
| Gen AI | `gen_ai.output.type` | Conditionally Required | string | Requested content type. | `"text"` |
| Gen AI | `gen_ai.conversation.id` | Conditionally Required | string | Identifier for the chat thread. | `"main"` |
| Server | `server.address` | Recommended | string | Hostname or IP of the remote GenAI API/provider. | `"example.com"` |
| Server | `server.port` | Conditionally Required | int | Port of the remote GenAI API/provider. | `443` |
| Error | `error.type` | Conditionally Required | string | Error class if operation failed. | `"timeout"` |
| Gen AI | `gen_ai.response.model` | Recommended | string | The actual model used. | `"gemini-1.5-flash-001"` |
| Gen AI | `gen_ai.request.temperature` | Recommended | double | Sampling temperature. | `0.7` |
| Gen AI | `gen_ai.request.top_p` | Recommended | double | Nucleus sampling probability. | `0.95` |
| Gen AI | `gen_ai.request.presence_penalty` | Recommended | double | Presence penalty. | `0.5` |
| Gen AI | `gen_ai.request.frequency_penalty` | Recommended | double | Frequency penalty. | `0.5` |
| Gen AI | `gen_ai.request.max_tokens` | Recommended | int | Maximum tokens to generate. | `2048` |
| Gen AI | `gen_ai.request.stop_sequences` | Recommended | string[] | Sequences that stop generation. | `["STOP"]` |
| Gen AI | `gen_ai.response.finish_reasons` | Recommended | string[] | Reasons the model stopped. | `["stop"]` |
| Gen AI | `gen_ai.response.id` | Recommended | string | Provider-generated response ID. | `"chatcmpl-123"` |
| Gen AI | `gen_ai.response.time_to_first_chunk` | Recommended | double | Latency to first chunk (seconds). | `0.5` |
| Gen AI | `gen_ai.usage.input_tokens` | Recommended | int | Total input tokens (incl. cached). | `150` |
| Gen AI | `gen_ai.usage.output_tokens` | Recommended | int | Total output tokens. | `45` |
| Gen AI | `gen_ai.usage.reasoning.output_tokens` | Recommended | int | Tokens used for reasoning. | `50` |
| Gen AI | `gen_ai.usage.cache_read.input_tokens` | Recommended | int | Input tokens served from cache. | `100` |
| Gen AI | `gen_ai.usage.cache_creation.input_tokens` | Recommended | int | Input tokens written to cache. | `150` |
| Gen AI | `gen_ai.input.messages` | Opt-In | any[] | Structured conversation history. | `[...]` |
| Gen AI | `gen_ai.output.messages` | Opt-In | any[] | Structured completion messages. | `[...]` |
| Gen AI | `gen_ai.system_instructions` | Opt-In | any[] | System-level instructions. | `[...]` |
| Gen AI | `gen_ai.tool.definitions` | Opt-In | any[] | List of tool definitions available to the model. | `[...]` |

#### Tools

Genkit aligns with the OTel GenAI Semantic Convention for [tool execution spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span). These attributes should be present on spans where `genkit.metadata.subtype` is `tool`.

> [!NOTE]
> An audit is necessary to determine which attributes below should *always* be present, and which should be added via a developer choice (e.g. plugin). 
>
> For example:
>
> - We might forgo `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` in favor of `genkit.input` / `genkit.output`. These are very large, and duplication is costly. Input/output is _standard_ for all genkit action types, and are inherent to Genkit observability.

| Convention | Attribute | Requirement Level | Type | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gen AI | `gen_ai.operation.name` | Required | string | The operation type. | `"execute_tool"` |
| Gen AI | `gen_ai.tool.name` | Required | string | Name of the tool being executed. | `"get_weather"` |
| Server | `server.address` | Recommended | string | Hostname or IP of the remote tool provider (if applicable). | `"api.weather.com"` |
| Server | `server.port` | Conditionally Required | int | Port of the remote tool provider. | `443` |
| Error | `error.type` | Conditionally Required | string | Error class if operation failed. | `"timeout"` |
| Gen AI | `gen_ai.tool.type` | Recommended | string | Type of the tool. | `"function"` |
| Gen AI | `gen_ai.tool.description` | Recommended | string | Description of the tool. | `"Get current weather"` |
| Gen AI | `gen_ai.tool.call.id` | Recommended | string | The tool call identifier. | `"call_abc_123"` |
| Gen AI | `gen_ai.tool.call.arguments` | Opt-In | any | Parameters passed to the tool. | `{"city": "Paris"}` |
| Gen AI | `gen_ai.tool.call.result` | Opt-In | any | Result returned by the tool. | `{"temp": 72}` |
| Genkit | `genkit.metadata.resumed` | Recommended | boolean | True if resumed after interrupt. | `true` |

## 3. Schema Management & Validation

Genkit maintains a central Telemetry Schema file: `genkit-telemetry.yaml`.

- **Tooling:** [OpenTelemetry Weaver](https://github.com/open-telemetry/weaver) is used to validate this schema against the official [OpenTelemetry Semantic Conventions](https://github.com/open-telemetry/semantic-conventions).
- **Code Generation:** SDKs should use Weaver or similar tools to generate attribute constants from the schema to ensure cross-language consistency.
