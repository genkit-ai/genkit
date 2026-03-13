# Tool Interrupts Demo

Human-in-the-loop AI interactions using Genkit's tool interruption mechanism.
The AI pauses execution, asks the human a question, and resumes with the answer.

## Features Demonstrated

| Feature | Flow / API | Description |
|---------|-----------|-------------|
| Tool Interruption | `ctx.interrupt(payload)` | Pause AI execution and return data to the caller |
| Interrupt Detection | `response.interrupts` | Check if the AI is waiting for human input |
| Resume with Response | `tool_response(request, answer)` | Continue AI execution with the human's answer |
| Interactive CLI Loop | `while True: input()` | Multi-turn game loop with interrupt handling |
| Pydantic Tool Schema | `TriviaQuestions` | Structured tool input with validation |

## How Tool Interrupts Work

```
┌─────────────────────────────────────────────────────────────────┐
│              TOOL INTERRUPT FLOW (Trivia Game)                   │
│                                                                  │
│  1. User starts game         2. AI calls tool                   │
│  ┌──────────┐                ┌──────────────────┐               │
│  │ "Science"│───► AI ──────►│ present_questions │               │
│  └──────────┘    (host)      └────────┬─────────┘               │
│                                       │                          │
│                              3. Tool calls ctx.interrupt()       │
│                                       │                          │
│                                       ▼                          │
│                              ┌──────────────────┐               │
│                              │  EXECUTION PAUSES │               │
│                              │  response.interrupts              │
│                              │  has the question  │               │
│                              └────────┬─────────┘               │
│                                       │                          │
│                              4. Human answers                    │
│                                       │                          │
│                                       ▼                          │
│                              ┌──────────────────┐               │
│                              │ tool_response()   │               │
│                              │ resumes AI        │               │
│                              └────────┬─────────┘               │
│                                       │                          │
│                              5. AI continues                     │
│                                       ▼                          │
│                              "That's correct!"                   │
└─────────────────────────────────────────────────────────────────┘
```

## ELI5: Key Concepts

| Concept | ELI5 |
|---------|------|
| **Tool Interrupt** | AI pauses and asks you a question — like a waiter asking "How do you want that cooked?" |
| **Human-in-the-Loop** | A person reviews/approves AI actions before they happen |
| **`ctx.interrupt()`** | The function that pauses execution and sends data back to the caller |
| **`tool_response()`** | Resume execution by passing the human's answer back to the AI |
| **`response.interrupts`** | Check if AI is waiting for input — non-empty means there's a question |

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

**Dev UI mode** (recommended for exploring the `play_trivia` flow):

```bash
./run.sh
```

**CLI mode** (interactive trivia game in your terminal):

```bash
uv run python src/main.py
```

## Testing This Demo

### Via Dev UI (http://localhost:4000)

- [ ] `play_trivia` — Enter a theme (e.g., "Science"), see the AI greet you and return an interrupted question

### Via CLI (`uv run python src/main.py`)

- [ ] AI greets you and asks for a trivia theme
- [ ] Type a theme (e.g., "science", "movies")
- [ ] Questions appear as tool interrupts — answer by number
- [ ] AI reacts dramatically to your answers
- [ ] Game loop continues until Ctrl+C

### Expected Behavior

- AI acts as an enthusiastic trivia game host
- Questions pause execution (interrupt) and wait for human input
- Answers are evaluated with dramatic responses
- In Dev UI: flow returns `INTERRUPTED: <question>` with answer choices

## Development

The `run.sh` script uses `watchmedo` for hot reloading on file changes.
