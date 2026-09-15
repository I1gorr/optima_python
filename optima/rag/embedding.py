"""
Embedding module for Optima RAG system.
Handles creation of embeddings and vector indexes using LangChain.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class OptimaEmbedder:
    """Handles embedding generation and vector index creation."""

    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5",
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the embedder.

        Args:
            embedding_model_name: Name of the HuggingFace embedding model
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize the HuggingFace embeddings model."""
        try:
            logger.info(f"Initializing embedding model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of LangChain Documents

        Returns:
            List of embedding vectors
        """
        if not self.embeddings:
            raise RuntimeError("Embeddings model not initialized")

        logger.info(f"Generating embeddings for {len(documents)} documents")
        start_time = time.time()

        # Extract text content
        texts = [doc.page_content for doc in documents]

        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)

        elapsed_time = time.time() - start_time
        logger.info(f"Generated embeddings in {elapsed_time:.2f} seconds")
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        if not self.embeddings:
            raise RuntimeError("Embeddings model not initialized")

        return self.embeddings.embed_query(query)

    def create_vector_index(self, documents: List[Document],
                          index_path: Path) -> FAISS:
        """
        Create a FAISS vector index from documents.

        Args:
            documents: List of LangChain Documents
            index_path: Path to save the index

        Returns:
            FAISS vector store
        """
        if not documents:
            raise ValueError("Cannot create index from empty document list")

        logger.info(f"Creating vector index for {len(documents)} documents")
        start_time = time.time()

        # Split documents into chunks
        logger.info("Splitting documents into chunks")
        chunked_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunked_docs)} chunks")

        # Create vector store
        logger.info("Creating FAISS vector store")
        vector_store = FAISS.from_documents(chunked_docs, self.embeddings)

        # Save index
        index_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(index_path))
        logger.info(f"Vector index saved to {index_path}")

        elapsed_time = time.time() - start_time
        logger.info(f"Vector index created in {elapsed_time:.2f} seconds")

        return vector_store

    def load_vector_index(self, index_path: Path) -> Optional[FAISS]:
        """
        Load a FAISS vector index from disk.

        Args:
            index_path: Path to the index

        Returns:
            FAISS vector store or None if not found
        """
        if not index_path.exists():
            logger.warning(f"Index path does not exist: {index_path}")
            return None

        try:
            logger.info(f"Loading vector index from {index_path}")
            vector_store = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector index loaded successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            return None

    def get_embedding_metrics(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get metrics about the embedding process.

        Args:
            documents: List of documents

        Returns:
            Dictionary of metrics
        """
        if not documents:
            return {}

        # Calculate basic stats
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chars = total_chars / len(documents) if documents else 0

        # Estimate tokens (rough approximation: 4 chars per token)
        estimated_tokens = total_chars // 4

        metrics = {
            "num_documents": len(documents),
            "total_characters": total_chars,
            "avg_characters_per_document": avg_chars,
            "estimated_tokens": estimated_tokens,
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

        return metrics


def embed_corpus(json_path: Path, enrichment_model: Optional[str],
                embedding_model_name: str, output_base_dir: Path,
                representation_mode: str = "hybrid") -> Dict[str, Any]:
    """
    Embed a single JSON corpus and create vector index.

    Args:
        json_path: Path to JSON file
        enrichment_model: Name of enrichment model (None for raw)
        embedding_model_name: Name of embedding model to use
        output_base_dir: Base directory for output
        representation_mode: Document representation mode

    Returns:
        Dictionary with results and metrics
    """
    logger.info(f"Starting embedding process for {json_path}")
    start_time = time.time()

    # Construct documents
    from .document_constructor import OptimaDocumentConstructor
    constructor = OptimaDocumentConstructor(representation_mode=representation_mode)
    documents = constructor.construct_documents_from_json(
        json_path, enrichment_model=enrichment_model
    )

    if not documents:
        raise ValueError(f"No documents constructed from {json_path}")

    # Get document construction metrics
    doc_constructor_metrics = {
        "num_documents": len(documents),
        "representation_mode": representation_mode
    }

    # Initialize embedder
    embedder = OptimaEmbedder(embedding_model_name=embedding_model_name)

    # Get embedding metrics
    embedding_metrics = embedder.get_embedding_metrics(documents)

    # Create vector index
    if enrichment_model:
        corpus_name = enrichment_model
        corpus_type = "enriched"
    else:
        corpus_name = "raw"
        corpus_type = "raw"

    index_dir = output_base_dir / "indexes" / embedding_model_name / corpus_name
    vector_store = embedder.create_vector_index(documents, index_dir)

    # Get index metrics
    index_size_mb = 0
    try:
        index_file = index_dir / "index.faiss"
        if index_file.exists():
            index_size_mb = index_file.stat().st_size / (1024 * 1024)
    except Exception:
        pass

    index_metrics = {
        "index_path": str(index_dir),
        "index_size_mb": round(index_size_mb, 2),
        "num_chunks_in_index": vector_store.index.ntotal if hasattr(vector_store.index, 'ntotal') else 0
    }

    elapsed_time = time.time() - start_time

    result = {
        "success": True,
        "corpus_name": corpus_name,
        "corpus_type": corpus_type,
        "json_path": str(json_path),
        "num_documents": len(documents),
        "elapsed_time_seconds": round(elapsed_time, 2),
        "document_constructor_metrics": doc_constructor_metrics,
        "embedding_metrics": embedding_metrics,
        "index_metrics": index_metrics
    }

    logger.info(f"Completed embedding process for {json_path} in {elapsed_time:.2f} seconds")
    return result


def embed_all_corpora(output_dir: Path, embedding_model_name: str,
                     representation_mode: str = "hybrid") -> Dict[str, Any]:
    """
    Embed all discovered JSON corpora.

    Args:
        output_dir: Directory containing JSON files
        embedding_model_name: Name of embedding model to use
        representation_mode: Document representation mode

    Returns:
        Dictionary with results for all corpora
    """
    from .document_constructor import discover_json_files

    logger.info("Starting embedding process for all corpora")
    start_time = time.time()

    # Discover JSON files
    json_files = discover_json_files(output_dir)

    if not json_files:
        raise ValueError(f"No JSON files found in {output_dir}")

    # Prepare output directory
    rag_report_dir = output_dir.parent / "rag_report"
    rag_report_dir.mkdir(exist_ok=True)

    results = {
        "embedding_model": embedding_model_name,
        "representation_mode": representation_mode,
        "corpora": {},
        "summary": {}
    }

    # Embed each corpus
    for corpus_name, json_path in json_files.items():
        try:
            enrichment_model = None if corpus_name == "base" else corpus_name
            logger.info(f"Processing corpus: {corpus_name}")

            corpus_result = embed_corpus(
                json_path=json_path,
                enrichment_model=enrichment_model,
                embedding_model_name=embedding_model_name,
                output_base_dir=rag_report_dir,
                representation_mode=representation_mode
            )

            results["corpora"][corpus_name] = corpus_result

        except Exception as e:
            logger.error(f"Failed to process corpus {corpus_name}: {e}")
            results["corpora"][corpus_name] = {
                "success": False,
                "error": str(e),
                "corpus_name": corpus_name,
                "json_path": str(json_path)
            }

    # Calculate summary
    successful = sum(1 for c in results["corpora"].values() if c.get("success", False))
    total_time = time.time() - start_time

    results["summary"] = {
        "total_corpora": len(json_files),
        "successful_corpora": successful,
        "failed_corpora": len(json_files) - successful,
        "total_elapsed_time_seconds": round(total_time, 2)
    }

    logger.info(f"Completed embedding all corpora: {successful}/{len(json_files)} successful")
    return results


if __name__ == "__main__":
    # Test the embedding module
    logging.basicConfig(level=logging.INFO)

    output_dir = Path("/home/igorr/Projects/op-python/optima/output")
    embedding_model = "BAAI/bge-small-en-v1.5"

    # Test with just base.json first
    base_path = output_dir / "base.json"
    if base_path.exists():
        print(f"Testing embedding with base.json using {embedding_model}")
        result = embed_corpus(
            json_path=base_path,
            enrichment_model=None,
            embedding_model_name=embedding_model,
            output_base_dir=output_dir.parent / "rag_report",
            representation_mode="hybrid"
        )
        print(f"Embedding result: {result['success']}")
        print(f"Documents: {result['num_documents']}")
        print(f"Time: {result['elapsed_time_seconds']}s")