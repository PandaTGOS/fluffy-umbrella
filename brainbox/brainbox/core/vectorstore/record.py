from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class VectorRecord:
    vector: List[float]
    metadata: Dict[str, Any]
