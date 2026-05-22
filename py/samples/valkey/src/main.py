# Genkit (Python) + Valkey RAG sample
# Indexes 5 documents into Valkey, retrieves the top-3 matches for a query,
# then uses an Ollama LLM to answer a question grounded in those results.
#
# Prerequisites:
#   Valkey:  docker run -d --name valkey-search -p 6379:6379 valkey/valkey-search:latest
#   Ollama:  ollama pull nomic-embed-text && ollama pull gemma4:e2b
#   Install: cd py && uv sync
#
# How to run:
#   uv run src/main.py

from __future__ import annotations

import asyncio

from genkit import Document, Genkit
from genkit.plugins.ollama import Ollama
from genkit.plugins.ollama.embedders import EmbeddingDefinition
from genkit.plugins.ollama.models import ModelDefinition
from genkit.plugins.valkey import Valkey, ValkeyConfig

EMBEDDER = 'ollama/nomic-embed-text'
MODEL = 'ollama/gemma4:e2b'
INDEX_NAME = 'genkit-sample-py'
DIMENSION = 768  # nomic-embed-text output dimension


async def main() -> None:
    # --- Bootstrap ---
    cfg = ValkeyConfig(
        index_name=INDEX_NAME,
        embedder=EMBEDDER,
        dimension=DIMENSION,
        host='localhost',
        port=6379,
    )

    ai = Genkit(
        plugins=[
            Ollama(
                models=[ModelDefinition(name='gemma4:e2b')],
                embedders=[EmbeddingDefinition(name='nomic-embed-text', dimensions=DIMENSION)],
            ),
            Valkey(configs=[cfg]),
        ],
        model=MODEL,
    )
    await ai.registry.initialize_all_plugins()

    # --- Index documents ---
    docs = [
        Document.from_text(
            'Valkey is an open-source, high-performance in-memory data store '
            'forked from Redis after the license change in 2024.',
            metadata={'source': 'valkey-overview'},
        ),
        Document.from_text(
            'The valkey-search module adds vector similarity search, full-text '
            'search, and secondary indexing capabilities to Valkey.',
            metadata={'source': 'valkey-search-docs'},
        ),
        Document.from_text(
            'Genkit is a framework for building AI-powered applications that '
            'supports flows, tools, retrievers, and indexers.',
            metadata={'source': 'genkit-overview'},
        ),
        Document.from_text(
            'HNSW (Hierarchical Navigable Small World) is an approximate nearest '
            'neighbour algorithm used for fast vector similarity search.',
            metadata={'source': 'hnsw-paper'},
        ),
        Document.from_text(
            'Ollama lets you run large language models like Llama 3 and Mistral '
            'locally on your own hardware without cloud dependencies.',
            metadata={'source': 'ollama-docs'},
        ),
    ]

    print('Indexing documents...')
    await ai.index(indexer=f'valkey/{INDEX_NAME}', documents=docs)
    print(f'Indexed {len(docs)} documents.\n')

    # --- Retrieve ---
    query = 'How does Valkey support vector search?'
    print(f'Query: {query}')
    response = await ai.retrieve(
        retriever=f'valkey/{INDEX_NAME}',
        query=query,
        options={'k': 3},
    )
    print(f'Retrieved {len(response.documents)} documents:')
    for i, doc in enumerate(response.documents, 1):
        src = (doc.metadata or {}).get('source', 'unknown')
        print(f'  [{i}] ({src}) {doc.text[:80]}...')
    print()

    # --- Generate answer grounded in retrieved context ---
    answer = await ai.generate(
        prompt=f'Answer the question concisely: {query}',
        docs=response.documents,
    )
    print('Answer:', answer.text)


if __name__ == '__main__':
    asyncio.run(main())
