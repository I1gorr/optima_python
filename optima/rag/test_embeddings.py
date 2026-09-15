"""
Test embeddings module for when sentence-transformers/torch are not available.
Provides a simple mock embedding for testing the RAG pipeline structure.
"""

import logging
import hashlib
from typing import List
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class MockEmbeddings(Embeddings):
    """Mock embeddings for testing when real embeddings are not available."""

    def __init__(self, dimension: int = 384):
        """
        Initialize mock embeddings.

        Args:
            dimension: Embedding dimension size
        """
        self.dimension = dimension
        logger.info(f"Initialized MockEmbeddings with dimension {dimension}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Create mock embeddings for documents.

        Args:
            texts: List of text strings

        Returns:
            List of mock embedding vectors
        """
        logger.info(f"Creating mock embeddings for {len(texts)} documents")
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()

            # Convert to list of floats between -1 and 1
            embedding = []
            for i in range(self.dimension):
                # Use cyclic bytes from hash
                byte_val = hash_bytes[i % len(hash_bytes)]
                # Normalize to [-1, 1]
                normalized = (byte_val / 255.0) * 2 - 1
                embedding.append(normalized)
            embeddings.append(embedding)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Create mock embedding for a query.

        Args:
            text: Query string

        Returns:
            Mock embedding vector
        """
        return self.embed_documents([text])[0]


# Test the mock embeddings
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    embeddings = MockEmbeddings(dimension=10)
    test_texts = [
        "This is a test function.",
        "Another test function with different content.",
        "Yet another function for testing purposes."
    ]

    # Test document embeddings
    doc_embeddings = embeddings.embed_documents(test_texts)
    print(f"Document embeddings shape: {len(doc_embeddings)} x {len(doc_embeddings[0])}")

    # Test query embedding
    query_embedding = embeddings.embed_query("test query")
    print(f"Query embedding shape: {len(query_embedding)}")

    # Check that same text produces same embedding
    emb1 = embeddings.embed_documents(["same text"])[0]
    emb2 = embeddings.embed_documents(["same text"])[0]
    assert emb1 == emb2, "Same text should produce same embedding"

    print("✓ Mock embeddings test passed")