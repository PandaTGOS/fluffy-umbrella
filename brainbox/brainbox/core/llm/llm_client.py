from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LLMResponse:
    text: str
    token_usage: Dict[str, int]
    model_name: str
    raw_output: Any


class LLMClient:
    def generate(
            self,
            system_instruction: str,
            user_input: str,
            context: Optional[Dict[str, Any]] = None,
            output_schema: Optional[Any] = None,
            runtime_options: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        raise NotImplementedError
