# Prompt Management Demo

A comprehensive demo of Genkit's [Dotprompt](https://genkit.dev/docs/dotprompt)
system — executable prompt templates stored as `.prompt` files with YAML
frontmatter for model config, schemas, tools, and more.

## Features Demonstrated

| Feature | Prompt File(s) | Flow | Description |
|---------|---------------|------|-------------|
| Structured Output (Picoschema) | `extract_info.prompt` | `extract_info_flow` | JSON output with optional fields |
| Model Config | `menu_item.prompt` | `menu_item_flow` | temperature, topK, topP, maxOutputTokens |
| Input Defaults | `menu_item.prompt` | `menu_item_flow` | Default values for input fields |
| Multi-message (roles) | `multi_message.prompt` | `multi_message_flow` | `{{role "system"}}` + `{{role "user"}}` |
| Multimodal (images) | `describe_image.prompt` | `describe_image_flow` | `{{media url=photoUrl}}` helper |
| Tool Calling | `trip_planner.prompt` | `trip_planner_flow` | `tools:` in frontmatter |
| Partials (reusable) | `_style.prompt`, `_destination.prompt` | — | `{{>style}}`, `{{>destination this}}` |
| `{{#each}}` iteration | `choose_destination.prompt` | `choose_destination_flow` | Loop over array with partial |
| Prompt Variants | `recipe.creative.prompt` | `creative_chef_flow` | `[name].[variant].prompt` pattern |
| Registered Schema | `recipe.prompt` | `chef_flow` | `output: schema: Recipe` (code-defined) |
| Custom Helpers | `greeting.prompt` | `greeting_flow` | `{{shout name}}`, `{{list items}}` |
| Streaming + Partials | `story.prompt` | `tell_story` | `{{>style}}` partial + streaming |
| Complex Picoschema | `article.prompt` | `article_flow` | Nested objects, arrays, enums, descriptions |
| Conditionals | `greeting.prompt` | `greeting_flow` | `{{#if}}` / `{{else}}` |

## Prompt File Inventory

```
prompts/
├── _style.prompt              # Partial: system role with personality
├── _destination.prompt        # Partial: renders one destination item
├── article.prompt             # Complex Picoschema (nested objects, enums)
├── choose_destination.prompt  # {{#each}} + partial iteration
├── describe_image.prompt      # Multimodal: {{media url=...}}
├── extract_info.prompt        # Picoschema input/output
├── greeting.prompt            # Custom helpers: {{shout}}, {{list}}
├── menu_item.prompt           # Model config overrides + defaults
├── multi_message.prompt       # Multi-message: system + user roles
├── recipe.prompt              # Registered schema output
├── recipe.creative.prompt     # Variant of recipe (high temperature)
├── story.prompt               # Streaming + partial inclusion
└── trip_planner.prompt        # Tool calling from frontmatter
```

## ELI5: Key Concepts

| Concept | ELI5 |
|---------|------|
| **Dotprompt** | Store prompts in files instead of code — like templates for AI instructions |
| **Picoschema** | A compact way to define schemas in YAML — `name: string`, `age?: integer` |
| **Partials** | Reusable prompt fragments (files starting with `_`) — like shared paragraphs |
| **Helpers** | Custom functions in templates — `{{shout name}}` calls your Python function |
| **Variants** | A/B test prompts — `recipe.creative.prompt` is a variant of `recipe.prompt` |
| **`{{role}}`** | Switch between system/user/model messages inside one `.prompt` file |
| **`{{media}}`** | Embed images or audio in prompts — `{{media url=photoUrl}}` |
| **`{{#each}}`** | Loop over arrays — render each item with a partial |

## Example Inputs

Copy-paste these JSON inputs into the DevUI to test each flow.

| Flow | Input JSON |
|------|-----------|
| `chef_flow` | `{"food": "banana bread"}` |
| `creative_chef_flow` | `{"food": "chocolate lava cake"}` |
| `tell_story` | `{"subject": "a brave little toaster", "personality": "courageous"}` |
| `extract_info_flow` | `{"text": "John Doe is a 35-year-old software engineer living in San Francisco."}` |
| `menu_item_flow` | `{"theme": "medieval"}` |
| `multi_message_flow` | `{"userQuestion": "Why is the sky blue?"}` |
| `describe_image_flow` | `{"photoUrl": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png", "style": "poet"}` |
| `trip_planner_flow` | `{"destination": "Paris", "days": 3}` |
| `choose_destination_flow` | `{"budget": "$2000"}` |
| `article_flow` | `{"topic": "quantum computing", "audience": "beginners"}` |
| `greeting_flow` | `{"name": "Developer", "items": ["docs", "samples", "tests"]}` |

## Setup environment

### How to Get Your Gemini API Key

To use the Google GenAI plugin, you need a Gemini API key.

1.  **Visit AI Studio**: Go to [Google AI Studio](https://aistudio.google.com/).
2.  **Create API Key**: Click on "Get API key" and create a key in a new or existing Google Cloud project.

For more details, check out the [official documentation](https://ai.google.dev/gemini-api/docs/api-key).

Export the API key as env variable `GEMINI_API_KEY` in your shell configuration.

```bash
export GEMINI_API_KEY='<Your api key>'
```

### Monitoring and Running

For an enhanced development experience, use the provided `run.sh` script to start the sample with automatic reloading:

```bash
./run.sh
```

This script uses `watchmedo` to monitor changes in:
- `src/` (Python logic)
- `../../packages` (Genkit core)
- `../../plugins` (Genkit plugins)
- File patterns: `*.py`, `*.prompt`, `*.json`

Changes will automatically trigger a restart of the sample. You can also pass command-line arguments directly to the script, e.g., `./run.sh --some-flag`.

## Run the sample

```bash
genkit start -- uv run src/main.py
```

## Testing This Demo

1. **Prerequisites**:

   ```bash
   export GEMINI_API_KEY=your_api_key
   ```

   Or the demo will prompt for the key interactively.

2. **Run the demo**:

   ```bash
   cd py/samples/framework-prompt-demo
   ./run.sh
   ```

3. **Open DevUI** at [http://localhost:4000](http://localhost:4000)

4. **Test the following flows**:
   - [ ] `chef_flow` — Structured output with registered `Recipe` schema
   - [ ] `tell_story` — Streaming with `_style.prompt` partial
   - [ ] `extract_info_flow` — Picoschema JSON extraction
   - [ ] `menu_item_flow` — Model config overrides + input defaults
   - [ ] `multi_message_flow` — System + user role messages
   - [ ] `describe_image_flow` — Multimodal `{{media}}` helper
   - [ ] `trip_planner_flow` — Tool calling from frontmatter
   - [ ] `choose_destination_flow` — `{{#each}}` + partial iteration
   - [ ] `creative_chef_flow` — Prompt variant (`recipe.creative`)
   - [ ] `article_flow` — Complex Picoschema (nested objects, enums)
   - [ ] `greeting_flow` — Custom helpers (`{{shout}}`, `{{list}}`)

5. **Test with file changes**:
   - Edit a `.prompt` file and verify hot reload works
   - Try different input parameters in DevUI

6. **Expected behavior**:
   - All 13 prompt files load from the `prompts/` directory
   - JSON outputs match the Picoschema defined in `.prompt` files
   - Streaming shows incremental text generation
   - Tools are called by the model when declared in frontmatter
   - Variants produce different styles for the same input
