from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class PromptSpec:
    system_instruction: str
    user_input: str
    context: Optional[List[Dict[str, Any]]] = None
    output_schema: Optional[dict] = None    

class PromptBuilder:
    def build(self) -> PromptSpec:
        raise NotImplementedError
