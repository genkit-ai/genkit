# Genkit Model Action Specification

This document specifies the contract for Genkit Model Actions. Genkit models are implemented as actions with specific input, output, and streaming schemas. They encapsulate the logic for communicating with AI models, handling multimodal input, tool calling, and structured output.

## Model Action Definition

A Genkit Model is an Action with the following characteristics:

- **Action Type**: `model`
- **Input Schema**: `GenerateRequest`
- **Output Schema**: `GenerateResponse`
- **Streaming Schema**: `GenerateResponseChunk`

### Metadata

Model actions should define the following metadata:

- `model`: Object containing model capability information.
  - `label`: Human-readable name (e.g., "Google AI - Gemini Pro").
  - `versions`: Array of supported version strings.
  - `supports`: Object defining supported capabilities:
    - `multiturn`: Boolean (history support).
    - `media`: Boolean (multimodal input support).
    - `tools`: Boolean (tool calling support).
    - `systemRole`: Boolean (system message support).
    - `output`: Array of supported output formats (e.g., `['json', 'text']`).
    - `contentType`: Array of supported output content types.
    - `context`: Boolean (document context support).
    - `constrained`: Enum (`'none'`, `'all'`, `'no-tools'`) - native constrained generation support.
    - `toolChoice`: Boolean (forcing tool selection).
    - `longRunning`: Boolean (long running operation support).
  - `stage`: Development stage (`'featured'`, `'stable'`, `'unstable'`, `'legacy'`, `'deprecated'`).
  - `customOptions`: JSON Schema for model-specific configuration (exposed as `config` in request).

## Data Structures

### GenerateRequest

The input to a model action.

| Field | Type | Description |
|---|---|---|
| `messages` | `Message[]` | **(Required)** List of messages in the conversation history. |
| `config` | `any` | Model-specific configuration options (e.g., temperature, topK). Validated against the model's config schema. |
| `tools` | `ToolDefinition[]` | List of tools available for the model to call. |
| `toolChoice` | `enum` | Tool selection strategy: `'auto'`, `'required'`, or `'none'`. |
| `output` | `OutputConfig` | Configuration for the desired output format/schema. |
| `docs` | `DocumentData[]` | Retrieved documents to be used as context. |

#### OutputConfig

| Field | Type | Description |
|---|---|---|
| `format` | `string` | Desired format (e.g., `'json'`, `'text'`). |
| `schema` | `Record<string, any>` | JSON schema defining the expected output structure. |
| `constrained` | `boolean` | Whether to enforce the schema constraints natively. |
| `contentType` | `string` | Specific content type for the output. |

### GenerateResponse

The output from a model action.

| Field | Type | Description |
|---|---|---|
| `message` | `Message` | The generated message. |
| `finishReason` | `enum` | Reason for generation completion: `'stop'`, `'length'`, `'blocked'`, `'interrupted'`, `'other'`, `'unknown'`. |
| `finishMessage` | `string` | Additional information about the finish reason. |
| `usage` | `GenerationUsage` | Token and character usage statistics. |
| `latencyMs` | `number` | Time taken for generation in milliseconds. |
| `custom` | `any` | Model-specific extra information. |
| `request` | `GenerateRequest` | The request that triggered this response. |

### GenerateResponseChunk

The chunk format for streaming responses.

| Field | Type | Description |
|---|---|---|
| `role` | `Role` | Role of the message being generated (usually `'model'`). |
| `index` | `number` | Index of the message in the response (typically 0). |
| `content` | `Part[]` | **(Required)** Content parts in this chunk. |
| `aggregated` | `boolean` | If true, this chunk contains the full accumulated content so far. |
| `custom` | `any` | Model-specific extra information. |

### Message

| Field | Type | Description |
|---|---|---|
| `role` | `enum` | **(Required)** The role of the message sender: `'system'`, `'user'`, `'model'`, `'tool'`. |
| `content` | `Part[]` | **(Required)** The content of the message, composed of one or more parts. |
| `metadata` | `Record<string, any>` | Arbitrary metadata associated with the message. |

### Parts

Genkit uses a unified `Part` structure to represent different types of content. A `Part` is a union of specific part types.

#### Text Part

Represents plain text content.

```json
{
  "text": "Hello, world!"
}
```

#### Media Part

Represents multimodal content. Inline data should be encoded as `data:` URIs (base64).

**Image:**

```json
{
  "media": {
    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "contentType": "image/jpeg"
  }
}
```

**Audio:**

```json
{
  "media": {
    "url": "data:audio/L16;codec=pcm;rate=24000;base64,AAAAAA...",
    "contentType": "audio/L16;codec=pcm;rate=24000"
  }
}
```

**Video:**

```json
{
  "media": {
    "url": "https://example.com/video.mp4",
    "contentType": "video/mp4"
  }
}
```

**Metadata:**
All parts can include a `metadata` field for provider-specific information that doesn't fit into the main schema. Common uses include `mediaResolution` for images/video, `videoMetadata` (e.g. duration, offset), or internal signatures like `thoughtSignature`.

```json
{
  "media": { "url": "..." },
  "metadata": {
    "mediaResolution": { "level": "MEDIA_RESOLUTION_HIGH" },
    "videoMetadata": { "startOffset": { "seconds": 10 } }
  }
}
```

#### Tool Request Part

Represents a request from the model to execute a tool.

```json
{
  "toolRequest": {
    "name": "weatherTool",
    "ref": "call_123", // Optional correlation ID
    "input": { "city": "New York" }
  }
}
```

#### Tool Response Part

Represents the result of a tool execution, sent back to the model.

```json
{
  "toolResponse": {
    "name": "weatherTool",
    "ref": "call_123", // Must match the request ref
    "output": { "temperature": 72 }, // Structured output
    "content": [ ... ] // Optional content parts (e.g. if tool returns artifacts)
  }
}
```

#### Custom Part

Represents provider-specific content not covered by other types. A common use case is returning the result of server-side tools like Code Execution.

```json
{
  "custom": {
    "executableCode": {
      "code": "print('Hello World')",
      "language": "PYTHON"
    },
    "codeExecutionResult": {
      "outcome": "OUTCOME_OK",
      "output": "Hello World\n"
    }
  }
}
```

## Provider-Specific Features

Many models offer server-side features that go beyond standard text generation or client-side tool calling. These are typically handled via the `config` object or specific metadata.

### Server-Side Tools

Features like **Web Search** (Grounding), **Code Execution**, or **URL Context** are often implemented as "server-side tools". Since the client does not execute them, they are configured in the `config` object rather than the `tools` list.

**Example (Web Search Configuration):**

```json
{
  "config": {
    "googleSearch": {}, // Provider-specific key
    "tools": [{ "googleSearch": {} }] // Some providers might use a tools config key
  }
}
```

**Example (URL Context):**

```json
{
  "config": {
    "urlContext": { "urls": ["https://example.com/article"] }
  }
}
```

### Encoding Guidelines

- **Requests**: Use `config` for enabling/configuring server-side features. Do not use `ToolRequestPart` unless the client is expected to execute the tool.
- **Responses**:
  - If a server-side tool produces content (e.g., code execution output), it may appear as a `TextPart` (if integrated into the answer) or a `CustomPart`.
  - Metadata about the execution (e.g., search sources, grounding metadata) should be placed in the `custom` field of the `GenerateResponse` or `Message` metadata.

#### Reasoning Part

Represents chain-of-thought or reasoning text provided by the model.

```json
{
  "reasoning": "First, I will calculate..."
}
```


#### Data Part

This part is reserved for future use and is not currently supported by any known plugins. Represents generic structured data. 

```json
{
  "data": { "key": "value" }
}
```

## Behavior

### Request Processing

1. **Validation**: The model action validates the `GenerateRequest`.
2. **Context**: If `docs` are provided, the model action should incorporate them into the context, typically by augmenting the message history.
3. **Tools**: If `tools` are provided, they are converted to the format expected by the underlying model API.
4. **Configuration**: `config` options are applied.

### System Message Handling

Genkit standardizes system instructions as messages with `role: 'system'` within the `messages` array. However, many model providers (e.g., Google GenAI) require system instructions to be passed as a separate configuration field rather than part of the conversation history.

**Implementation Requirement:**
- The model action MUST accept `role: 'system'` messages in the input `messages` array.
- If the underlying provider requires separate system instructions:
  1. Extract the system message(s) from the `messages` array.
  2. Convert/format them as required by the provider (e.g., `systemInstruction` field).
  3. Ensure they are NOT passed in the regular conversation history if the provider doesn't support `system` role there.

### Configuration Handling

Model plugins should follow the "passthrough" pattern for configuration options. This ensures that new features added to the underlying model API can be used immediately by users without requiring plugin updates.

1.  **Extract Known Options**: Explicitly destructure known configuration keys (e.g., `temperature`, `topK`, `topP`) to handle them according to Genkit's common schema or specific logic.
2.  **Pass Through the Rest**: Pass all remaining unknown keys directly to the underlying model API's configuration object.

**Example (TypeScript):**

```typescript
const {
  temperature,
  topK,
  ...restOfConfig
} = request.config || {};

const apiRequest = {
  model: modelName,
  temperature: temperature, // Handle known keys
  top_k: topK,
  ...restOfConfig // Pass through unknown keys
};
```

3. **Merge Tools**: If the provider supports passing tools via configuration (e.g., `config.tools`) in addition to the standard `request.tools`, these should be merged. This allows users to pass provider-specific tool definitions (like server-side tools) alongside standard Genkit tools.

```typescript
const tools = request.tools?.map(toProviderTool) || [];
if (config.tools) {
  tools.push(...config.tools);
}
```

### Response Generation

1. **Content**: The model output is parsed into `Part` objects. Text is mapped to `TextPart`, function calls to `ToolRequestPart`.
2. **Streaming**: When streaming, the model emits `GenerateResponseChunk`s.
   - Chunks should ideally contain incremental updates.
   - If the underlying model only supports full responses during streaming, `aggregated: true` should be set.
3. **Finish Reason**: The model must map provider-specific finish reasons to the standard Genkit enum.

### Tool Handling

Tools are a central capability of Genkit models. Implementation involves converting definitions, handling requests (including streaming), and processing responses.

#### Tool Definition Conversion
The model action must convert Genkit's `ToolDefinition` into the format expected by the provider.

- **Name**: Sanitize tool names if the provider has strict rules (e.g., replace `/` with `__` for Gemini).
- **Input Schema**: Convert the JSON Schema in `inputSchema` to the provider's schema format.
- **Description**: Pass the tool description.

#### Tool Requests
When the model decides to call a tool, it emits a `ToolRequestPart`.

- **Ref**: Assign a stable `ref` (call ID) if the provider supports it, to correlate with the response.
- **Input**: The arguments for the tool.

**Partial Tool Requests (Streaming)**
Some models (like Gemini 3.0) support streaming tool calls. In this case, the model emits `ToolRequestPart`s with `partial: true`.

- The `input` field in a partial request should contain the **accumulated** arguments so far (if supported by the plugin logic) or the current delta, depending on how the plugin manages state.
- The final chunk for the tool call should have `partial: false` (or omitted).

#### Tool Responses
The result of a tool execution is passed back to the model as a `ToolResponsePart` in a message with `role: 'tool'`.

- **Ref**: Must match the `ref` of the corresponding `ToolRequestPart`.
- **Output**: The result of the tool execution (usually a JSON object).
- **Content**: Optional list of Parts (e.g., if the tool returns an image or other rich content).

#### Multi-turn Flow
Models supporting tools must handle the conversation loop:
1. `User Message`
2. `Model Message` (containing `ToolRequestPart`s)
3. `Tool Message` (containing `ToolResponsePart`s)
4. `Model Message` (Final Answer)

### Structured Output

- If `output.schema` is provided, the model should attempt to generate content matching that schema.
- If `output.constrained` is true and the model supports it, the schema is enforced by the model generation process.
- Otherwise, the schema may be included in the prompt instructions.
- The resulting structured data should be typically serialized in a `TextPart`.
