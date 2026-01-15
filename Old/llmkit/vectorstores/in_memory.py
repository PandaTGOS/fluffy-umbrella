import math
from typing import List, Dict, Any
from ..interfaces import VectorStore

class InMemoryVectorStore(VectorStore):
    def __init__(self):
        # List of {"vector": [...], "metadata": {...}}
        self.store: List[Dict[str, Any]] = []

    def _dot_product(self, v1: List[float], v2: List[float]) -> float:
        return sum(a * b for a, b in zip(v1, v2))

    def _magnitude(self, v: List[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot = self._dot_product(v1, v2)
        mag1 = self._magnitude(v1)
        mag2 = self._magnitude(v2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
            
        return dot / (mag1 * mag2)

    def add(self, vectors: List[List[float]], metadatas: List[dict]) -> None:
        if len(vectors) != len(metadatas):
            raise ValueError("Number of vectors must match number of metadatas")
            
        for vec, meta in zip(vectors, metadatas):
            self.store.append({
                "vector": vec,
                "metadata": meta
            })

    def search(self, query_vector: List[float], k: int) -> List[dict]:
        if not self.store:
            return []

        scored_results = []
        for item in self.store:
            score = self._cosine_similarity(query_vector, item["vector"])
            scored_results.append({
                "score": score,
                "metadata": item["metadata"]
            })

        # Sort by score descending
        scored_results.sort(key=lambda x: x["score"], reverse=True)

        return scored_results[:k]