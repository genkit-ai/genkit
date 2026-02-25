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

"""Vertex AI Rerankers and Evaluators Demo.

This sample demonstrates:
- Semantic document reranking for RAG quality improvement
- Model output evaluation using Vertex AI metrics (BLEU, ROUGE, fluency, safety, etc.)

Prerequisites:
- GOOGLE_CLOUD_PROJECT environment variable set
- gcloud auth application-default login
- Discovery Engine API enabled (for rerankers)
- Vertex AI API enabled (for evaluators)
"""

from typing import Any, cast

import structlog
from pydantic import BaseModel

from genkit.ai import Genkit
from genkit.blocks.document import Document
from genkit.core.typing import BaseDataPoint, DocumentData, Score
from genkit.plugins.google_genai import VertexAI
from samples.shared.logging import setup_sample

setup_sample()

logger = structlog.get_logger(__name__)


ai = Genkit(
    plugins=[
        VertexAI(location='us-central1'),
    ],
    model='vertexai/gemini-2.5-flash',
)


class RerankResult(BaseModel):
    """Result of a rerank operation."""

    query: str
    ranked_documents: list[dict[str, Any]]


@ai.flow()
async def rerank_documents(query: str = 'How do neural networks learn?') -> RerankResult:
    """Rerank documents based on relevance to query.

    This demonstrates using Vertex AI's semantic reranker to re-order
    documents by their semantic relevance to a query. Useful for improving
    RAG (Retrieval-Augmented Generation) quality.
    """
    # Sample documents to rerank (in a real app, these would come from a retriever)
    documents: list[Document] = [
        Document.from_text('Neural networks learn through backpropagation, adjusting weights based on errors.'),
        Document.from_text('Python is a popular programming language for machine learning.'),
        Document.from_text('The gradient descent algorithm minimizes the loss function during training.'),
        Document.from_text('Cats are popular pets known for their independence.'),
        Document.from_text('Deep learning models use multiple layers to extract hierarchical features.'),
        Document.from_text('The weather today is sunny with a high of 75 degrees.'),
        Document.from_text('Transformers use attention mechanisms to process sequential data efficiently.'),
    ]

    # Rerank documents using Vertex AI semantic reranker
    # Document extends DocumentData, so we can cast and pass documents directly
    ranked_docs = await ai.rerank(
        reranker='vertexai/semantic-ranker-default@latest',
        query=query,
        documents=cast(list[DocumentData], documents),
        options={'top_n': 5},
    )

    # Format results
    results: list[dict[str, Any]] = []
    for doc in ranked_docs:
        results.append({
            'text': doc.text(),
            'score': doc.score,
        })

    return RerankResult(query=query, ranked_documents=results)


@ai.flow()
async def rag_with_reranking(question: str = 'What is machine learning?') -> str:
    """Full RAG pipeline with reranking.

    Demonstrates a two-stage retrieval pattern:
    1. Initial retrieval (simulated with sample docs)
    2. Reranking for quality
    3. Generation using top-k results
    """
    # Simulated retrieval results (in production, use a real retriever)
    retrieved_docs: list[Document] = [
        Document.from_text('Machine learning is a subset of artificial intelligence.'),
        Document.from_text('Supervised learning uses labeled data to train models.'),
        Document.from_text('The stock market closed higher today.'),
        Document.from_text('ML algorithms can identify patterns in large datasets.'),
        Document.from_text('Unsupervised learning finds hidden patterns without labels.'),
        Document.from_text('Pizza is a popular Italian dish.'),
        Document.from_text('Deep learning uses neural networks with many layers.'),
        Document.from_text('Reinforcement learning learns from rewards and penalties.'),
    ]

    # Stage 2: Rerank for quality
    # Document extends DocumentData, so we can cast and pass documents directly
    ranked_docs = await ai.rerank(
        reranker='vertexai/semantic-ranker-default@latest',
        query=question,
        documents=cast(list[DocumentData], retrieved_docs),
        options={'top_n': 3},
    )

    # Build context from top-ranked documents
    context = '\n'.join([f'- {doc.text()}' for doc in ranked_docs])

    # Stage 3: Generate answer using reranked context
    response = await ai.generate(
        model='vertexai/gemini-2.5-flash',
        prompt=f"""Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Answer:""",
    )

    return response.text


class EvalResult(BaseModel):
    """Result of an evaluation."""

    metric: str
    scores: list[dict[str, Any]]


def _extract_score(evaluation: Score | list[Score]) -> float | str | bool | None:
    """Extract score from evaluation result."""
    if isinstance(evaluation, list):
        return evaluation[0].score if evaluation else None
    return evaluation.score


def _extract_reasoning(evaluation: Score | list[Score]) -> str | None:
    """Extract reasoning from evaluation result."""
    if isinstance(evaluation, list):
        if evaluation and evaluation[0].details:
            return evaluation[0].details.reasoning
        return None
    if evaluation.details:
        return evaluation.details.reasoning
    return None


@ai.flow()
async def evaluate_fluency() -> EvalResult:
    """Evaluate text fluency using Vertex AI.

    Fluency measures how natural and readable the text is.
    Scores range from 1 (poor) to 5 (excellent).
    """
    dataset: list[BaseDataPoint] = [
        BaseDataPoint(
            input='Write a sentence about AI.',
            output=(
                'Artificial intelligence is transforming industries by enabling '
                'machines to learn from data and make intelligent decisions.'
            ),
        ),
        BaseDataPoint(
            input='Describe machine learning.',
            # Intentionally poor grammar for demonstration
            output='Machine learning algorithms data patterns the find automatically.',
        ),
        BaseDataPoint(
            input='Explain neural networks.',
            output=(
                'Neural networks are computational models inspired by the human brain, '
                'consisting of interconnected nodes that process information in layers.'
            ),
        ),
    ]

    results = await ai.evaluate(
        evaluator='vertexai/fluency',
        dataset=dataset,
    )

    scores: list[dict[str, Any]] = []
    for i, result in enumerate(results.root):
        scores.append({
            'sample_index': i,
            'test_case_id': result.test_case_id,
            'score': _extract_score(result.evaluation),
            'reasoning': _extract_reasoning(result.evaluation),
        })

    return EvalResult(metric='fluency', scores=scores)


@ai.flow()
async def evaluate_safety() -> EvalResult:
    """Evaluate text safety using Vertex AI.

    Safety checks for harmful, biased, or inappropriate content.
    Higher scores indicate safer content.
    """
    dataset: list[BaseDataPoint] = [
        BaseDataPoint(
            input='Write about technology.',
            output=(
                'Technology has revolutionized communication, enabling people to connect instantly across the globe.'
            ),
        ),
        BaseDataPoint(
            input='Describe a helpful assistant.',
            output=(
                'A helpful AI assistant provides accurate information, '
                'respects user privacy, and declines harmful requests.'
            ),
        ),
    ]

    results = await ai.evaluate(
        evaluator='vertexai/safety',
        dataset=dataset,
    )

    scores: list[dict[str, Any]] = []
    for i, result in enumerate(results.root):
        scores.append({
            'sample_index': i,
            'test_case_id': result.test_case_id,
            'score': _extract_score(result.evaluation),
        })

    return EvalResult(metric='safety', scores=scores)


@ai.flow()
async def evaluate_groundedness() -> EvalResult:
    """Evaluate groundedness using Vertex AI.

    Groundedness checks if the output is factually grounded in the provided context.
    This helps detect hallucinations in RAG applications.
    """
    dataset: list[BaseDataPoint] = [
        BaseDataPoint(
            input='What is the capital of France?',
            output='The capital of France is Paris.',
            context=[
                'France is a country in Western Europe. Its capital city is Paris, which is known for the Eiffel Tower.'
            ],
        ),
        BaseDataPoint(
            input='What is the population of Paris?',
            # Hallucinated - context doesn't mention population
            output='Paris has a population of about 12 million people.',
            context=['Paris is the capital of France. It is known for art, fashion, and culture.'],
        ),
        BaseDataPoint(
            input='What is France known for?',
            output='France is known for wine, cheese, and the Eiffel Tower.',
            context=[
                'France is famous for its cuisine, especially wine and cheese. '
                'The Eiffel Tower in Paris is a major landmark.'
            ],
        ),
    ]

    results = await ai.evaluate(
        evaluator='vertexai/groundedness',
        dataset=dataset,
    )

    scores: list[dict[str, Any]] = []
    for i, result in enumerate(results.root):
        scores.append({
            'sample_index': i,
            'test_case_id': result.test_case_id,
            'score': _extract_score(result.evaluation),
            'reasoning': _extract_reasoning(result.evaluation),
        })

    return EvalResult(metric='groundedness', scores=scores)


@ai.flow()
async def evaluate_bleu() -> EvalResult:
    """Evaluate using BLEU score.

    BLEU (Bilingual Evaluation Understudy) compares output to a reference.
    Commonly used for translation and text generation quality.
    Scores range from 0 to 1, with higher being better.
    """
    dataset: list[BaseDataPoint] = [
        BaseDataPoint(
            input='Translate to French: Hello, how are you?',
            output='Bonjour, comment allez-vous?',
            reference='Bonjour, comment allez-vous?',  # Perfect match
        ),
        BaseDataPoint(
            input='Translate to French: Good morning',
            output='Bon matin',
            reference='Bonjour',  # Different but valid translation
        ),
    ]

    results = await ai.evaluate(
        evaluator='vertexai/bleu',
        dataset=dataset,
    )

    scores: list[dict[str, Any]] = []
    for i, result in enumerate(results.root):
        scores.append({
            'sample_index': i,
            'test_case_id': result.test_case_id,
            'score': _extract_score(result.evaluation),
        })

    return EvalResult(metric='bleu', scores=scores)


@ai.flow()
async def evaluate_summarization() -> EvalResult:
    """Evaluate summarization quality using Vertex AI.

    Summarization quality assesses how well a summary captures the key points
    of the original text.
    """
    dataset: list[BaseDataPoint] = [
        BaseDataPoint(
            input='Summarize this article about climate change.',
            output='Climate change is causing rising temperatures and extreme weather events globally.',
            context=[
                'Climate change refers to long-term shifts in temperatures and weather patterns. '
                'Human activities have been the main driver since the 1800s, primarily due to '
                'burning fossil fuels. This has led to rising global temperatures, melting ice '
                'caps, rising sea levels, and more frequent extreme weather events like '
                'hurricanes, droughts, and floods.'
            ],
        ),
    ]

    results = await ai.evaluate(
        evaluator='vertexai/summarization_quality',
        dataset=dataset,
    )

    scores: list[dict[str, Any]] = []
    for i, result in enumerate(results.root):
        scores.append({
            'sample_index': i,
            'test_case_id': result.test_case_id,
            'score': _extract_score(result.evaluation),
            'reasoning': _extract_reasoning(result.evaluation),
        })

    return EvalResult(metric='summarization_quality', scores=scores)


async def main() -> None:
    """Main function."""
    # Example run logic can go here or be empty for pure flow server
    pass


if __name__ == '__main__':
    ai.run_main(main())
