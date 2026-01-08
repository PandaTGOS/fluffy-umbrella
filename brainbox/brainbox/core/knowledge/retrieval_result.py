from dataclasses import dataclass
from typing import List, Dict, Any
from brainbox.core.knowledge.documents import Document

@dataclass
class RetrievalResult:
    documents: List[Document]
    signals: Dict[str, Any]   # scores, diagnostics, retriever stats
