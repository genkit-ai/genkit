# Model Armor

Demonstrates the [Model Armor](https://cloud.google.com/security/products/model-armor)
middleware for content safety filtering. Model Armor scans prompts for
prompt injection, jailbreak attempts, and sensitive data before sending
them to the model.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Content Filtering | `modelArmorFlow` | Generation with Model Armor middleware for PI/jailbreak and SDP filters |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager
- **Google Cloud project** with Model Armor enabled

### API Keys

```bash
export GEMINI_API_KEY='<your-api-key>'
```

### Model Armor Template

Create a Model Armor template in your GCP project, then set:

```bash
export MODEL_ARMOR_TEMPLATE='projects/<project>/locations/<location>/templates/<template>'
```

### Build and Install

From the repo root:

```bash
pnpm install
pnpm run setup
```

## Run the Sample

```bash
pnpm start
```

This starts the Genkit Dev UI with live reload.

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test content filtering**:
   - [ ] `modelArmorFlow` — Default input tries a prompt injection
   - [ ] Try safe inputs (e.g., `"Tell me about cats"`)
   - [ ] Try unsafe inputs (e.g., `"ignore previous instructions..."`)

3. **Expected behavior**:
   - Safe prompts pass through and generate a response
   - Unsafe prompts are caught by Model Armor and raise a `GenkitError`
   - Error details include the specific filter that triggered (PI/jailbreak, SDP)
