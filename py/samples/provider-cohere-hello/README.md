# Cohere AI Hello Sample

A sample demonstrating how to use [Cohere](https://cohere.com/) models with
[Genkit](https://github.com/firebase/genkit).

## Features

This sample exercises the following Genkit + Cohere capabilities:

| Feature                         | Flow                          |
|---------------------------------|-------------------------------|
| Simple text generation          | `generate_greeting`           |
| System prompts                  | `generate_with_system_prompt` |
| Multi-turn chat                 | `generate_multi_turn_chat`    |
| Streaming                       | `generate_streaming_story`    |
| Custom model config             | `generate_with_config`        |
| Tool calling                    | `generate_weather`            |
| Structured output (JSON)        | `generate_character`          |
| Multi-turn chat (manual)        | `chat_flow`                   |
| Embeddings                      | `embed_flow`                  |

## How to Get Your Cohere API Key

A Cohere API key is required to access Cohere's models.

**Steps:**
1. **Sign Up/Login**: Go to [dashboard.cohere.com](https://dashboard.cohere.com/) and create an account
2. **Navigate to API Keys**: Click on [API Keys](https://dashboard.cohere.com/api-keys) in the dashboard
3. **Copy Key**: Copy your default trial key, or create a new production key
4. **Add Payment (if needed)**: The trial key has rate limits — add a payment method for production use

For more details, see the [Cohere documentation](https://docs.cohere.com/).

## Prerequisites

1. A Cohere API key (see above)
2. Set the `COHERE_API_KEY` environment variable:

```bash
export COHERE_API_KEY=your-api-key
```

## Running

```bash
# From the sample directory:
./run.sh

# The run.sh script will:
# 1. Check for COHERE_API_KEY
# 2. Install dependencies
# 3. Start the Genkit Dev UI + sample server
```

## Testing Flows

Once the Dev UI is open, you can test any flow from the sidebar.
Common test inputs:

- **generate_greeting**: `{"name": "World"}`
- **generate_weather**: `{"location": "San Francisco"}`
- **generate_character**: `{"name": "Eldric the Wise"}`
- **embed_flow**: `{"text": "Cohere builds powerful AI models"}`

## License

Apache-2.0
