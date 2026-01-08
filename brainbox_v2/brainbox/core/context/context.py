from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Context:
    documents: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
