import time
from typing import List, Dict, Set

def evaluate_retrieval(
    test_cases: List[Dict[str, any]], retriever, k: int = 5
) -> Dict[str, float]:
    """
    test_cases schema:
    [
      {
        "query": "Why was my card fee charged?",
        "expected_ids": {"complaint_101", "complaint_205"}
      }
    ]
    """
    total_precision = 0.0
    total_recall = 0.0
    total_latency = 0.0
    hits = 0

    for test in test_cases:
        query = test["query"]
        expected_ids: Set[str] = set(test["expected_ids"])

        start_time = time.perf_counter()
        retrieved_docs = retriever.get_relevant_documents(query, k=k)
        latency = (time.perf_counter() - start_time) * 1000  # in ms

        retrieved_ids = {doc.metadata["complaint_id"] for doc in retrieved_docs}
        
        relevant_retrieved = len(retrieved_ids.intersection(expected_ids))
        
        precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0.0
        recall = relevant_retrieved / len(expected_ids) if expected_ids else 0.0

        total_precision += precision
        total_recall += recall
        total_latency += latency
        if relevant_retrieved > 0:
            hits += 1

    num_samples = len(test_cases)
    return {
        f"precision@{k}": total_precision / num_samples,
        f"recall@{k}": total_recall / num_samples,
        f"hit_rate@{k}": hits / num_samples,
        "avg_latency_ms": total_latency / num_samples
    }