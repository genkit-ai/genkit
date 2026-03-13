# BugBot: AI Code Reviewer

AI-powered code review using Genkit and FastAPI.

## Features

- **Security Analysis**: Detects SQL injection, XSS, and other vulnerabilities
- **Bug Detection**: Finds potential bugs and logic errors
- **Style Review**: Checks code style and best practices
- **Streaming Support**: Real-time analysis via Server-Sent Events
- **Dev UI Integration**: Test flows directly in Genkit Developer UI

## Quick Start

```bash
export GEMINI_API_KEY="your-api-key"
./run.sh
```

The API will start on http://localhost:8080 with:
- Swagger UI at http://localhost:8080/docs
- Genkit Dev UI at http://localhost:4000

## API Endpoints

### Native FastAPI Endpoints

```bash
# Review code
curl -X POST http://localhost:8080/review \
  -H "Content-Type: application/json" \
  -d '{"code": "query = f\"SELECT * FROM users WHERE id={user_input}\"", "language": "python"}'

# Security analysis only
curl -X POST http://localhost:8080/review/security \
  -H "Content-Type: application/json" \
  -d '{"code": "eval(user_input)", "language": "python"}'
```

### Genkit Flow Endpoints (with streaming)

```bash
# Review with streaming (SSE)
curl -X POST http://localhost:8080/flow/review?stream=true \
  -H "Content-Type: application/json" \
  -d '{"data": {"code": "x = eval(input())", "language": "python"}}'
```

## Architecture

This sample demonstrates the Genkit FastAPI plugin:

- **`genkit_lifespan(ai)`**: Connects FastAPI to Genkit Dev UI
- **`@genkit_fastapi_handler(ai)`**: Exposes flows as HTTP endpoints with streaming
- **Parallel Flow Execution**: Runs multiple analyzers concurrently
- **Typed Prompts**: Strongly-typed inputs and outputs using Pydantic
