# Genkit Valkey Plugin

Valkey vector store plugin for [Genkit](https://github.com/firebase/genkit).

Provides indexing and retrieval backed by Valkey with the valkey-search module
using HNSW vector similarity search.

## Installation

```bash
pip install genkit-plugin-valkey
```

## Requirements

- Valkey 8+ with the valkey-search module
- Python 3.11+

## Usage

```python
from genkit import Genkit, Document
from genkit.plugins.valkey import Valkey, ValkeyConfig

cfg = ValkeyConfig(
    index_name="my-index",
    embedder="ollama/nomic-embed-text",
    dimension=768,
)

ai = Genkit(plugins=[Valkey(configs=[cfg])])

# Index documents
await ai.index(indexer="valkey/my-index", documents=[...])

# Retrieve
resp = await ai.retrieve(retriever="valkey/my-index", query="search text", options={"k": 5})
```
