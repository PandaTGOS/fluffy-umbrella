from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ConfidenceSignals:
    retrieval_support: float
    answer_coverage: float
    metadata: Dict[str, Any] = field(default_factory=dict)
