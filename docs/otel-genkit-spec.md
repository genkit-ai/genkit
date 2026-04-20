# Genkit OpenTelemetry Specification

This document defines the official OpenTelemetry instrumentation standards for the Genkit framework across all supported languages.

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

### Specialized Spans

#### Flows
- `genkit.metadata.subtype`: `flow`
- `genkit.isRoot`: `true` (if entry point)

#### Session Flows
When a flow is executed within the context of a long-running session or conversation, the following attributes should be propagated to all spans within that flow to enable trace grouping.

| Convention | Attribute | Requirement Level | Type | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Session | `session.id` | Required | string | Unique identifier for the user session. | `"sess-abc-123"` |
| Gen AI | `gen_ai.conversation.id` | Recommended | string | Identifier for the specific chat thread. | `"main"` |

#### Models (GenAI)
Genkit should align with the [OTel GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) where possible.
- `gen_ai.system`: The provider name (e.g., `googleai`, `vertexai`).
- `gen_ai.request.model`: The model ID.
- `gen_ai.response.model`: The actual model used (if different).
- `genkit.metadata.subtype`: `model`

#### Tools
- `genkit.metadata.subtype`: `tool`
- `genkit.metadata.resumed`: `true` (if resumed after interrupt)

## 3. Schema Management & Validation

Genkit maintains a central Telemetry Schema file: `genkit-telemetry.yaml`.

- **Tooling:** [OpenTelemetry Weaver](https://github.com/open-telemetry/weaver) is used to validate this schema against the official [OpenTelemetry Semantic Conventions](https://github.com/open-telemetry/semantic-conventions).
- **Code Generation:** SDKs should use Weaver or similar tools to generate attribute constants from the schema to ensure cross-language consistency.
