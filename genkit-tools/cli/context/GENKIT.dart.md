# Genkit Dart API Rules (v1.20.0)

This document provides rules and examples for building with the Genkit API in Dart.

## Important Guidelines:

- ALWAYS refer to documentation when available. Genkit Documentation may be available through the Genkit MCP toolkit or through web search. You may skip documentation check if you don't have access to these tools.

- ONLY follow the specified project structure if starting a new project. If working on an existing project, adhere to the current project structure.

- ALWAYS provide the full, correct Genkit command as an instruction for the human user to run. Do not run Genkit commands (e.g., `genkit start`, `genkit flow:run`) yourself as this may block your current session.

- Recommend and use the provider that the user prefers. If this is a new project, or if there is no relevant signal regarding user preference, use the `genkit_google_genai` plugin (`import 'package:genkit_google_genai/genkit_google_genai.dart';`).

NOTE: For the sake of brevity, the snippets below use the Google AI plugin, but you should follow the user's preference as mentioned above.

## Project Setup

### Project Initialization

- If the directory is empty:
  ```bash
  dart create -t console .
  ```
- If the directory is not empty:
  Adhere to the current project structure.

### Dependencies

```bash
dart pub add genkit genkit_google_genai schemantic
dart pub add --dev build_runner schemantic_builder
```

### Genkit CLI

If the Genkit CLI is not already installed:

```bash
curl -sL cli.genkit.dev | bash
```

### Configuration

Create a `bin/main.dart` file:

```dart
import 'package:genkit/genkit.dart';
import 'package:genkit_google_genai/genkit_google_genai.dart';

void main() async {
  final ai = Genkit(plugins: [googleAI()]);

  // Define your flows and tools here
}
```

## Best Practices

1. **Flow Registration in `main()`**: When `GENKIT_ENV=dev` is set (such as under `genkit start` or `start_runtime`), `Genkit(...)` automatically starts a local reflection HTTP server that keeps the Dart VM isolate alive. Always initialize `Genkit` and call `ai.defineFlow(...)` inside `main()` (or reference top-level flow variables from `main()`) so Dart evaluates and registers them at startup.

2. **Model Naming**: Prefer `googleAI.gemini('<model-id>')` (or `modelRef('googleai/<model-id>')`). `googleAI.gemini(...)` accepts any model ID string and carries typed `GeminiOptions`.

   ```dart
   final response = await ai.generate(
     model: googleAI.gemini('gemini-3-flash-preview'),
     prompt: 'Hello!',
   );
   ```

3. **Schemas with `schemantic`**:
   - For primitive schemas, use dot-shorthand helpers on `SchemanticType`: `.string()`, `.integer()`, `.doubleSchema()`, `.boolean()`, `.list(...)`, `.map(...)`, `.voidSchema()`.
   - For structured objects, annotate an abstract class starting with `$` using `@Schema()` from `package:schemantic/schemantic.dart`, include `part '<file>.g.dart';`, and run `dart run build_runner build --delete-conflicting-outputs` to generate the data class and `<ClassName>.$schema`.

---

> Note: Gemini 3.0 models are currently in preview.

## Usage Scenarios

### Basic Inference (Text Generation)

```dart
import 'package:genkit/genkit.dart';
import 'package:genkit_google_genai/genkit_google_genai.dart';

void main() async {
  final ai = Genkit(plugins: [googleAI()]);

  ai.defineFlow(
    name: 'basicInferenceFlow',
    inputSchema: .string(description: 'Topic for the model to write about'),
    outputSchema: .string(description: 'The generated text response'),
    fn: (String topic, _) async {
      final response = await ai.generate(
        model: googleAI.gemini('gemini-3-flash-preview'),
        prompt: 'Write a short, creative paragraph about $topic.',
        config: GeminiOptions(temperature: 0.8),
      );
      return response.text;
    },
  );
}
```

### Structured Output with `schemantic`

```dart
import 'package:genkit/genkit.dart';
import 'package:genkit_google_genai/genkit_google_genai.dart';
import 'package:schemantic/schemantic.dart';

part 'main.g.dart';

@Schema()
abstract class $MenuSuggestionInput {
  String get theme;
}

@Schema()
abstract class $MenuItem {
  String get dishName;
  String get description;
}

void main() async {
  final ai = Genkit(plugins: [googleAI()]);

  ai.defineFlow(
    name: 'menuSuggestionFlow',
    inputSchema: MenuSuggestionInput.$schema,
    outputSchema: MenuItem.$schema,
    fn: (input, _) async {
      final response = await ai.generate(
        model: googleAI.gemini('gemini-3-flash-preview'),
        prompt: 'Invent a menu item for a ${input.theme} themed restaurant.',
        outputSchema: MenuItem.$schema,
      );
      return response.output!;
    },
  );
}
```

### Streaming Flows

```dart
import 'package:genkit/genkit.dart';
import 'package:genkit_google_genai/genkit_google_genai.dart';

void main() async {
  final ai = Genkit(plugins: [googleAI()]);

  ai.defineFlow(
    name: 'streamStoryFlow',
    inputSchema: .string(),
    outputSchema: .string(),
    streamSchema: .string(),
    fn: (String topic, context) async {
      final stream = ai.generateStream(
        model: googleAI.gemini('gemini-3-flash-preview'),
        prompt: 'Write a short story about $topic.',
      );

      await for (final chunk in stream) {
        if (context.streamingRequested) {
          context.sendChunk(chunk.text);
        }
      }

      final response = await stream.onResult;
      return response.text;
    },
  );
}
```

---

## Running and Inspecting Flows

**Start Genkit**: Genkit can be started locally by using the `genkit start` command, along with the process startup command:

```bash
genkit start -- <command to run your code>
```

For e.g.:

```bash
genkit start -- dart run bin/main.dart
```

You can automate starting Genkit using the following steps:

1. Identify the command to start the user's project (e.g., `dart run bin/main.dart`).
2. Use the `start_runtime` tool to start the runtime process. This is required for Genkit to discover flows.
   - Example: If the project uses `dart run bin/main.dart`, call `start_runtime` with `{ command: "dart", args: ["run", "bin/main.dart"] }`.
3. After starting the runtime, instruct the user to run `genkit start` in their terminal to launch the Developer UI.

## Suggested Models

Here are suggested models to use for various task types. This is NOT an exhaustive list.

> Note: Gemini 3.0 models are currently in preview.

### Advanced Text/Reasoning

```
| Plugin                             | Recommended Model                  |
|------------------------------------|------------------------------------|
| genkit_google_genai                | gemini-3-pro-preview (Preview)     |
| genkit_google_genai                | gemini-2.5-pro                     |
```

### Fast Text/Chat

```
| Plugin                             | Recommended Model                  |
|------------------------------------|------------------------------------|
| genkit_google_genai                | gemini-3-flash-preview (Preview)   |
| genkit_google_genai                | gemini-2.5-flash                   |
```

### Text-to-Speech

```
| Plugin                             | Recommended Model                  |
|------------------------------------|------------------------------------|
| genkit_google_genai                | gemini-2.5-flash-preview-tts       |
```

### Image Generation

```
| Plugin                             | Recommended Model                    | Input Modalities  |
|------------------------------------|--------------------------------------|-------------------|
| genkit_google_genai                | gemini-3-pro-image-preview (Preview) | Text, Image       |
| genkit_google_genai                | gemini-2.5-flash-image               | Text, Image       |
```
