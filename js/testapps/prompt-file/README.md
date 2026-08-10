# Dotprompt Files

Demonstrates Genkit's [Dotprompt](https://genkit.dev/docs/dotprompt) system —
executable prompt templates stored as `.prompt` files with YAML frontmatter
for model config, schemas, and more.

## Features Demonstrated

| Feature | Flow | Prompt File | Description |
|---------|------|-------------|-------------|
| Structured Output | `chefFlow` | `recipe.prompt` | Recipe generation with `Recipe` schema |
| Prompt Variants | `robotChefFlow` | `recipe.robot.prompt` | Robot-themed variant of the recipe prompt |
| Streaming + Partials | `tellStory` | `story.prompt` | Story generation with `_style.prompt` partial |
| Custom Helpers | — | — | `{{list items}}` helper for array rendering |
| Registered Schema | `Recipe` | — | Code-defined Zod schema referenced by prompts |

## Prompt File Inventory

```
prompts/
├── _style.prompt          # Partial: personality/style for the storyteller
├── recipe.prompt          # Recipe generation with structured output
├── recipe.robot.prompt    # Variant: robot personality for recipes
└── story.prompt           # Streaming story with partial inclusion
```

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager

### API Keys

```bash
export GEMINI_API_KEY='<your-api-key>'
```

### Build and Install

From the repo root:

```bash
pnpm install
pnpm run setup
```

## Run the Sample

```bash
pnpm build
```

Then use the Genkit CLI:

```bash
genkit flow:run chefFlow '{"food": "banana bread"}'
```

## Testing This Demo

1. **Test recipe generation**:
   ```bash
   genkit flow:run chefFlow '{"food": "banana bread"}'
   ```

2. **Test robot variant**:
   ```bash
   genkit flow:run robotChefFlow '{"food": "chocolate cake"}'
   ```

3. **Test streaming story**:
   ```bash
   genkit flow:run tellStory '{"subject": "a brave little toaster"}' -s
   ```

4. **Expected behavior**:
   - `chefFlow` returns structured JSON matching the `Recipe` schema
   - `robotChefFlow` uses the robot personality variant
   - `tellStory` streams text chunks incrementally
   - `_style.prompt` partial is included in the story prompt
