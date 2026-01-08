from dataclasses import dataclass
from typing import Dict

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, str]
    score: float = 0.0
