"""
RAG pipeline module for Optima RAG system.
Handles the full Retrieval-Augmented Generation process.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)


class OptimaRAGPipeline:
    """RAG pipeline for Optima system."""

    def __init__(self, vector_store: FAISS, llm: BaseLLM,
                 k: int = 5, prompt_template: Optional[str] = None):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: FAISS vector store for retrieval
            llm: Language model for generation
            k: Number of documents to retrieve
            prompt_template: Custom prompt template (optional)
        """
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
        self.prompt_template = prompt_template or self._default_prompt_template()
        self.retriever = None
        self.rag_chain = None
        self._setup_chain()

    def _default_prompt_template(self) -> str:
        """Default prompt template for code-related queries."""
        return """You are an expert software engineer analyzing code functions.
Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know,
don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """

    def _setup_chain(self):
        """Set up the RAG chain."""
        try:
            # Set up retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.k}
            )

            # Set up prompt
            prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["context", "question"]
            )

            # Set up RAG chain
            self.rag_chain = (
                {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            logger.info("RAG chain initialized successfully")
        except Exception as e:
            logger.error(f"Failed to setup RAG chain: {e}")
            raise

    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for inclusion in prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline.

        Args:
            question: Question to ask

        Returns:
            Dictionary with answer and source documents
        """
        if not self.rag_chain:
            raise RuntimeError("RAG chain not initialized")

        logger.info(f"Processing query: {question}")
        start_time = time.time()

        try:
            # Get answer
            answer = self.rag_chain.invoke(question)

            # Get source documents
            source_docs = self.retriever.get_relevant_documents(question)

            elapsed_time = time.time() - start_time

            result = {
                "question": question,
                "answer": answer,
                "source_documents": source_docs,
                "num_source_documents": len(source_docs),
                "response_time_seconds": round(elapsed_time, 3)
            }

            logger.info(f"Query processed in {elapsed_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def query_with_metadata(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline and return detailed metadata.

        Args:
            question: Question to ask

        Returns:
            Dictionary with answer, sources, and detailed metadata
        """
        result = self.query(question)

        # Add metadata about source documents
        source_metadata = []
        for doc in result["source_documents"]:
            metadata = {
                "function_id": doc.metadata.get("function_id", ""),
                "function_name": doc.metadata.get("function_name", ""),
                "qualified_name": doc.metadata.get("qualified_name", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "corpus_type": doc.metadata.get("corpus_type", ""),
                "enrichment_model": doc.metadata.get("enrichment_model", ""),
                "representation_mode": doc.metadata.get("representation_mode", "")
            }
            source_metadata.append(metadata)

        result["source_metadata"] = source_metadata
        return result


def create_rag_pipeline_from_corpus(corpus_name: str, embedding_model_name: str,
                                  llm: BaseLLM, k: int = 5,
                                  output_base_dir: Path = None) -> OptimaRAGPipeline:
    """
    Create a RAG pipeline from a specific corpus.

    Args:
        corpus_name: Name of the corpus (e.g., "raw", "llama-3.2-3b-instruct")
        embedding_model_name: Name of embedding model used
        llm: Language model for generation
        k: Number of documents to retrieve
        output_base_dir: Base directory for output (defaults to project rag_report)

    Returns:
        Configured OptimaRAGPipeline
    """
    if output_base_dir is None:
        # Default to project rag_report directory
        output_base_dir = Path("/home/igorr/Projects/op-python/optima/rag_report")

    from optima.rag.embedding_simple import OptimaEmbedder, index_directory

    # Load vector store
    embedder = OptimaEmbedder(embedding_model_name=embedding_model_name)
    index_path = index_directory(output_base_dir, corpus_name, embedding_model_name)
    vector_store = embedder.load_vector_index(index_path)
    if vector_store is None:
        # Preserve the historical layout for callers with existing indexes.
        from .evaluation import load_corpus_indices
        legacy = load_corpus_indices(output_base_dir, embedding_model_name)
        vector_store = legacy.get(corpus_name)

    if not vector_store:
        raise ValueError(f"Could not load vector index for corpus {corpus_name}")

    # Create and return pipeline
    return OptimaRAGPipeline(vector_store=vector_store, llm=llm, k=k)


def rag_query(corpus_name: str, embedding_model_name: str, llm: BaseLLM,
              question: str, k: int = 5,
              output_base_dir: Path = None) -> Dict[str, Any]:
    """
    Convenience function for a single RAG query.

    Args:
        corpus_name: Name of the corpus to query
        embedding_model_name: Name of embedding model used
        llm: Language model for generation
        question: Question to ask
        k: Number of documents to retrieve
        output_base_dir: Base directory for output

    Returns:
        Query result dictionary
    """
    pipeline = create_rag_pipeline_from_corpus(
        corpus_name=corpus_name,
        embedding_model_name=embedding_model_name,
        llm=llm,
        k=k,
        output_base_dir=output_base_dir
    )
    return pipeline.query(question)


if __name__ == "__main__":
    # Test the RAG pipeline module
    logging.basicConfig(level=logging.INFO)
    print("RAG pipeline module loaded successfully")