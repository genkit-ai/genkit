# GitHub PR Explainer (Genkit + MCP)

This sample summarizes a GitHub pull request by calling GitHub tools via an MCP server and asking a model to produce a concise explanation (TL;DR, what changed, why).

## Prerequisites

- Install the GitHub MCP server and ensure the binary is on your PATH, or set an explicit command via `GITHUB_MCP_CMD`.
  - Official repo: https://github.com/github/github-mcp-server
  - Build from source (no Docker): https://github.com/github/github-mcp-server?tab=readme-ov-file#build-from-source

- Point to the built MCP server:

```bash
export GITHUB_MCP_CMD=/absolute/path/to/github-mcp-server
```

- Export a GitHub Personal Access Token with repo read access:

```bash
export GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here
```

- Export a Google AI API key (required by the model plugin):

```bash
export GOOGLE_API_KEY=your_gemini_api_key
```

## Run

From this directory:

```bash
go run . -repo owner/name -pr 1234
```