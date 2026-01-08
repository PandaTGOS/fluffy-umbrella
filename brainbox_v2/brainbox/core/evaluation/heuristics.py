from typing import List, Dict, Any
from brainbox.core.evaluation.signals import ConfidenceSignals

def compute_retrieval_support(documents: List[Any]) -> float:
    if not documents:
        return 0.0
    
    # Handle both Dict and Document object
    scores = []
    for doc in documents:
        if isinstance(doc, dict):
            scores.append(doc.get("score", 0.0))
        else:
            scores.append(getattr(doc, "score", 0.0))
    
    if not scores:
        return 0.0
        
    max_score = max(scores)
    
    # Normalize to [0, 1] by clamping
    return min(max(max_score, 0.0), 1.0)


def compute_answer_coverage(answer: str, documents: List[Any]) -> float:
    if not answer:
        return 0.0

    # Pre-processing: Remove <think> blocks if present
    import re
    clean_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
    
    # 1. Tokenize answer (lowercase, whitespace split, filter short tokens)
    answer_tokens = [t for t in clean_answer.lower().split() if len(t) >= 3]
    
    if not answer_tokens:
        return 0.0
        
    # 2. Build context string from all docs
    context_parts = []
    for d in documents:
        if isinstance(d, dict):
            context_parts.append(d.get("content", ""))
        else:
            context_parts.append(getattr(d, "content", ""))
            
    context_text = " ".join(context_parts).lower()
    
    # 3. Count matches
    # Note: simple substring check for each token in the massive context string
    matches = 0
    for token in answer_tokens:
        if token in context_text:
            matches += 1
            
    # 4. Compute coverage
    coverage = matches / len(answer_tokens)
    
    # 5. Clamp
    return min(max(coverage, 0.0), 1.0)


# Runs both of the above heuristics and returns a ConfidenceSignals object
def evaluate_confidence(answer: str, documents: List[Dict[str, Any]]) -> ConfidenceSignals:
    retrieval_support = compute_retrieval_support(documents)
    answer_coverage = compute_answer_coverage(answer, documents)
    
    return ConfidenceSignals(
        retrieval_support=retrieval_support,
        answer_coverage=answer_coverage,
        metadata={
            "num_documents": len(documents),
            "answer_length": len(answer),
        }
    )


