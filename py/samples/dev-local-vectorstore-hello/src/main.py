# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


"""Dev local vector store sample - Local RAG without external services.

This sample demonstrates Genkit's local vector store for development,
which allows testing RAG (Retrieval Augmented Generation) without
setting up external vector databases.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ RAG                 │ Retrieval-Augmented Generation. AI looks up        │
    │                     │ your docs before answering. Fewer hallucinations!  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Vector Store        │ A database that finds "similar" items by meaning.  │
    │                     │ "Happy" finds docs about "joyful" too.             │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Local Store         │ Runs on your computer, no cloud needed. Perfect    │
    │                     │ for testing before deploying to production.        │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Indexing            │ Adding documents to the store. Like a librarian    │
    │                     │ cataloging new books.                              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Retrieval           │ Finding documents that match a query. "Show me     │
    │                     │ docs about sci-fi films" returns matching results. │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow (RAG Pipeline)::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │               HOW RAG FINDS ANSWERS FROM YOUR DOCUMENTS                 │
    │                                                                         │
    │    STEP 1: INDEX (one-time setup)                                       │
    │    ──────────────────────────────                                       │
    │    Your Documents: ["The Godfather...", "The Matrix...", ...]           │
    │         │                                                               │
    │         │  (1) Convert each doc to numbers (embeddings)                 │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Embedder       │   "sci-fi film" → [0.2, -0.5, 0.8, ...]          │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) Store in local vector store                          │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Local Store    │   All docs + embeddings saved locally            │
    │    └─────────────────┘                                                  │
    │                                                                         │
    │    STEP 2: RETRIEVE (at query time)                                     │
    │    ────────────────────────────────                                     │
    │    Query: "What's a good sci-fi movie?"                                 │
    │         │                                                               │
    │         │  (3) Convert query to embedding                               │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Embedder       │   Query → [0.21, -0.48, 0.79, ...] (similar!)    │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (4) Find nearest matches                                 │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Local Store    │   "The Matrix" (0.95 match)                      │
    │    │                 │   "Inception" (0.89 match)                       │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| Local Vector Store Definition           | `define_dev_local_vector_store`     |
| Document Indexing                       | `ai.index()`                        |
| Document Retrieval                      | `ai.retrieve()`                     |
| Document Structure                      | `Document.from_text()`              |

See README.md for testing instructions.
"""

import asyncio
import os

from genkit.ai import Genkit
from genkit.plugins.dev_local_vectorstore import define_dev_local_vector_store
from genkit.plugins.google_genai import VertexAI
from genkit.types import Document, RetrieverResponse
from samples.shared.logging import setup_sample

setup_sample()

if 'GCLOUD_PROJECT' not in os.environ:
    os.environ['GCLOUD_PROJECT'] = input('Please enter your GCLOUD_PROJECT: ')

ai = Genkit(
    plugins=[VertexAI()],
    model='vertexai/gemini-3-flash-preview',
)

# Define dev local vector store
define_dev_local_vector_store(
    ai,
    name='films',
    embedder='vertexai/gemini-embedding-001',
)

films = [
    'The Godfather is a 1972 crime film directed by Francis Ford Coppola.',
    'The Dark Knight is a 2008 superhero film directed by Christopher Nolan.',
    'Pulp Fiction is a 1994 crime film directed by Quentin Tarantino.',
    "Schindler's List is a 1993 historical drama directed by Steven Spielberg.",
    'Inception is a 2010 sci-fi film directed by Christopher Nolan.',
    'The Matrix is a 1999 sci-fi film directed by the Wachowskis.',
    'Fight Club is a 1999 film directed by David Fincher.',
    'Forrest Gump is a 1994 drama directed by Robert Zemeckis.',
    'Star Wars is a 1977 sci-fi film directed by George Lucas.',
    'The Shawshank Redemption is a 1994 drama directed by Frank Darabont.',
]


@ai.flow()
async def index_documents() -> None:
    """Indexes the film documents in the local vector store."""
    genkit_documents = [Document.from_text(text=film) for film in films]
    await ai.index(
        indexer='films',
        documents=genkit_documents,
    )


@ai.flow()
async def retrieve_documents() -> RetrieverResponse:
    """Retrieve documents from the vector store."""
    return await ai.retrieve(
        query=Document.from_text('sci-fi film'),
        retriever='films',
        options={'limit': 3},
    )


async def main() -> None:
    """Main entry point for the sample - keep alive for Dev UI."""
    # Keep the process alive for Dev UI
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
