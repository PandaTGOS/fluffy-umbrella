from typing import List, Dict, Any, Optional

def has_answer_evidence(
    question: str,
    documents: List[Dict[str, Any]],
    required_terms: Optional[List[str]] = None,
) -> bool:

    if not documents:
        return False

    text_blob = " ".join(doc["content"].lower() for doc in documents)

    # Default: If no specific terms required, assume retriever did its job
    if required_terms is None:
        return True

    return any(term in text_blob for term in required_terms)