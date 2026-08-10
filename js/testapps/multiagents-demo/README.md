# Multi-Agent Demo

Demonstrates a multi-agent architecture where a **triage agent** delegates to
specialized agents (catalog, payment, representative) based on user intent.
Each agent has its own tools and system prompt.

## Architecture

```
User Input
    │
    ▼
┌────────────────┐
│  Triage Agent  │  Classifies intent and delegates
└───────┬────────┘
        │
   ┌────┼─────────────────┐
   ▼    ▼                 ▼
┌──────────┐  ┌──────────────┐  ┌──────────────────┐
│ Catalog  │  │   Payment    │  │  Representative  │
│  Agent   │  │    Agent     │  │      Agent       │
│          │  │              │  │                  │
│ Tools:   │  │ Tools:       │  │ (General help)   │
│ • search │  │ • process    │  │                  │
│ • details│  │   payment    │  │                  │
│ • popular│  │              │  │                  │
│ • store  │  │              │  │                  │
└──────────┘  └──────────────┘  └──────────────────┘
```

## Features Demonstrated

| Feature | Agent / Flow | Description |
|---------|-------------|-------------|
| Multi-Agent Orchestration | `multiAgentMultiModel` | Triage agent delegates to specialists |
| Triage Agent | `triageAgent` | Classifies user intent |
| Catalog Agent | `catalogAgent` | Product search, details, and recommendations |
| Payment Agent | `paymentAgent` | Payment processing |
| Representative Agent | `representativeAgent` | General customer support |
| Tool Calling | Multiple tools | `searchCatalog`, `getProductDetails`, `processPayment`, etc. |

## Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **pnpm** package manager

### API Keys

```bash
export GEMINI_API_KEY='<your-api-key>'
```

Or create a `.env` file:

```bash
GEMINI_API_KEY=<your-api-key>
```

### Build and Install

From the repo root:

```bash
pnpm install
pnpm run setup
```

## Run the Sample

```bash
pnpm run genkit:dev
```

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test the multi-agent flow**:
   - [ ] `multiAgentMultiModel` — Input: `{"userInput": "What products do you have?"}` (should route to Catalog Agent)
   - [ ] `multiAgentMultiModel` — Input: `{"userInput": "I want to pay for my order"}` (should route to Payment Agent)
   - [ ] `multiAgentMultiModel` — Input: `{"userInput": "I need help with a return"}` (should route to Representative Agent)

3. **Expected behavior**:
   - Triage agent correctly classifies user intent
   - Specialized agents use their tools to fulfill requests
   - Catalog queries return product information
   - Payment requests go through the payment processing tool
