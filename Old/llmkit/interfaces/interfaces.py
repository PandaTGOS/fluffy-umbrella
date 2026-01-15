from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

@dataclass
class LLMResponse:
    text: Union[str, BaseModel]
    token_usage: Dict[str, int]
    model_name: str
    raw_output: Any

class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        system_instruction: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        output_schema: Optional[type[BaseModel]] = None,
        runtime_options: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        pass

@dataclass
class PromptSpec:
    system_instruction: str
    user_input: str
    context: Optional[List[Dict[str, Any]]] = None
    output_schema: Optional[dict] = None    

class PromptBuilder:
    def build(self) -> PromptSpec:
        raise NotImplementedError
    
class Pipeline:
    def run(self, input_data: Any) -> Dict[str, Any]:
        raise NotImplementedError