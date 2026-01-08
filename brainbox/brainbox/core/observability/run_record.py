from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from brainbox.core.evaluation.signals import ConfidenceSignals

@dataclass
class RunRecord:
    question: str
    answer: str
    retriever_type: str
    num_documents: int
    confidence: Optional[ConfidenceSignals]
    attempts: List[Dict[str, Any]]
    final_decision: str  # e.g. "ACCEPT" | "RETRY_ACCEPT" | "REFUSE"
    token_usage: Dict[str, Any]
