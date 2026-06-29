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

"""Vertex AI Plugin for Genkit.

This plugin provides integration with Google Cloud's Vertex AI platform,
including Model Garden for accessing third-party models and Vector Search
for RAG applications.

Example:
    ```python
    from genkit import Genkit
    from genkit_googleai import VertexAI
    from genkit_vertexai import (
        ModelGardenPlugin,
        define_vertex_vector_search_firestore,
    )

    # 1. Initialize Genkit with VertexAI and Model Garden plugins
    ai = Genkit(
        plugins=[
            VertexAI(project='my-project', location='us-central1'),
            ModelGardenPlugin(project='my-project', location='us-central1'),
        ],
    )

    # 2. Define Vector Search store backed by Firestore
    store = define_vertex_vector_search_firestore(
        ai,
        name='my_store',
        collection='documents',
        embedder='vertexai/text-embedding-005',
    )

    # 3. Retrieve relevant documents for RAG
    docs = await store.retrieve(query='How to reset my password', limit=2)
    print(docs[0].content)
    # => "To reset your password, navigate to Account Settings..."
    ```

Requirements:
    - Requires Google Cloud Application Default Credentials (ADC) or explicit credentials.

See Also:
    - Vertex AI Model Garden: https://cloud.google.com/vertex-ai/docs/model-garden
    - Vertex AI Vector Search: https://cloud.google.com/vertex-ai/docs/vector-search
"""

from genkit_vertexai.model_garden.modelgarden_plugin import (
    ModelGardenPlugin,
)


def package_name() -> str:
    """Get the package name for the Vertex AI plugin.

    Returns:
        The fully qualified package name as a string.
    """
    return 'genkit_vertexai'


__all__ = [
    'ModelGardenPlugin',
    'package_name',
]
