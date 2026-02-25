# Restaurant Menu Demo

A multi-file Genkit application demonstrating how to organize prompts, flows,
and tools across separate modules — simulating a restaurant menu analysis system.

## Project Structure

```
src/
├── main.py              # Entry point — imports all case modules
├── menu_ai.py           # Shared Genkit instance
├── case_01/
│   └── prompts.py       # Basic menu prompts
├── case_02/
│   ├── prompts.py       # Menu analysis prompts
│   ├── flows.py         # Menu analysis flows
│   └── tools.py         # Restaurant data tools
├── case_03/
│   ├── prompts.py       # Recommendation prompts
│   └── flows.py         # Recommendation flows
├── case_04/
│   ├── prompts.py       # Dietary restriction prompts
│   └── flows.py         # Dietary restriction flows
└── case_05/
    ├── prompts.py       # Multi-language prompts
    └── flows.py         # Multi-language flows
```

## Features Demonstrated

| Feature | Case | Description |
|---------|------|-------------|
| Multi-file Organization | All | Prompts, flows, tools in separate modules |
| Flow Registration via Import | `main.py` | Importing a module registers its flows |
| Basic Prompts | Case 01 | Simple menu-related prompts |
| Flows + Tools | Case 02 | Menu analysis with tool integration |
| Recommendation Flows | Case 03 | AI-powered menu recommendations |
| Dietary Restrictions | Case 04 | Handling dietary requirements |
| Multi-language Support | Case 05 | Menu translation and localization |

## ELI5: Key Concepts

| Concept | ELI5 |
|---------|------|
| **Multi-file App** | Code split across files — each file handles one part (prompts, flows, tools) |
| **Module Organization** | Separate files for different concerns — `case_01/prompts.py`, `case_02/flows.py` |
| **Flow Registration** | Importing a module registers its flows — just import it and Genkit knows about it |
| **Shared `ai` Instance** | One `Genkit` instance in `menu_ai.py`, imported by all modules |

## Quick Start

```bash
export GEMINI_API_KEY=your-api-key
./run.sh
```

Then open the Dev UI at http://localhost:4000.

## Setup

### Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API key" and create a key

```bash
export GEMINI_API_KEY='your-api-key'
```

### Run the Sample

```bash
./run.sh
```

Or manually:

```bash
genkit start -- uv run python -m src.main
```

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test the different cases**:
   - [ ] Case 01: Basic menu prompts
   - [ ] Case 02: Menu analysis with tools
   - [ ] Case 03: Menu recommendations flow
   - [ ] Case 04: Dietary restrictions handling
   - [ ] Case 05: Multi-language menu support

3. **Expected behavior**:
   - All prompts/flows appear in DevUI sidebar
   - Menu items are analyzed correctly
   - Tools provide realistic restaurant data
   - Each case module's flows are independently testable

## Development

The `run.sh` script uses `watchmedo` for hot reloading on file changes.
