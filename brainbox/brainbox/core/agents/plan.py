from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class PlanStep:
    action: str            # "tool" | "answer" | "thought"
    thought: str
    name: Optional[str] = None       # tool name
    input: Dict[str, Any] = field(default_factory=dict)  # tool input or empty

@dataclass
class Plan:
    steps: List[Dict[str, Any]] # Storing as dicts for easy JSON serialization/loading
