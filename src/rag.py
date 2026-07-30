import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class VectorStoreNotFoundError(Exception):
    """Raised when the vector store directory does not exist."""
    pass

class LowConfidenceRetrievalError(Exception):
    """Raised when retrieved chunks fail to meet the similarity threshold."""
    pass

def retrieve_chunks(query: str, vector_store, config) -> List[Dict[str, Any]]:
    """Retrieve chunks with fallback logic for low similarity scores."""
    if not query.strip():
        logger.warning("Empty query provided to retriever.")
        return []

    # Run similarity search with relevance scores
    results_with_scores = vector_store.similarity_search_with_score(
        query, k=config.retrieval.top_k
    )

    if not results_with_scores:
        logger.info("No matching chunks found in vector store.")
        return []

    valid_chunks = []
    for doc, score in results_with_scores:
        # Distance-to-similarity normalization depends on vector store distance metric
        if score >= config.retrieval.similarity_threshold:
            valid_chunks.append(doc)

    if not valid_chunks:
        logger.warning(f"All retrieved chunks were below threshold {config.retrieval.similarity_threshold}")
        return []

    return valid_chunks

def generate_rag_response(query: str, vector_store, config) -> Dict[str, Any]:
    """Generates response with graceful error fallback."""
    try:
        chunks = retrieve_chunks(query, vector_store, config)
        
        if not chunks:
            return {
                "answer": "I do not have enough relevant information in the customer complaint records to answer your question accurately.",
                "sources": []
            }

        # Build prompt & query LLM...
        # response = llm_chain.run(...)
        return {"answer": "Generated answer here...", "sources": chunks}

    except VectorStoreNotFoundError:
        logger.critical("Vector store files are missing.")
        return {"answer": "System Error: Index not loaded. Please contact support.", "sources": []}
    except Exception as e:
        logger.error(f"Unexpected error during generation: {str(e)}", exc_info=True)
        return {"answer": "An internal error occurred while processing your query.", "sources": []}