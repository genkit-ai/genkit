# Genkit and Ollama Test App

This example demonstrates a Retrieval-Augmented Generation (RAG) flow using Genkit and Ollama. It uses a pretrained model to embed descriptions of different Pokemon, allowing you to find the most relevant Pokemon based on user queries.

## Prerequisites

- **Node.js** (version 18 or higher recommended)
- **Ollama** - a local AI model server, which is used to handle embeddings and generate responses.

## Setup

### Step 1: Install Ollama

1. Go to the [Ollama website](https://ollama.com/download) to download and install Ollama for your operating system.
2. Once installed, start the Ollama server by running:

```bash
ollama serve
```

The server will run at http://localhost:11434 by default.

### Step 2: Pull the Required Models

In this example, we use two models with Ollama:

An embedding model (nomic-embed-text)
A generation model (phi3.5:latest)
Run the following commands in your terminal to pull these models:

```bash
ollama pull nomic-embed-text
ollama pull phi3.5:latest
```

Now you may start the testapp and the Developer UI with `pnpm run genkit:dev`
