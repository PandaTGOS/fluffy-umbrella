from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from brainbox.core.evaluation.signals import ConfidenceSignals
from brainbox.core.llm import LLMResponse

@dataclass
class RAGState:
    # Input
    question: str

    # Retrieval
    documents: List[Dict[str, Any]] = field(default_factory=list)
    retriever_type: Optional[str] = None

    # Prompting
    prompt_spec: Optional[Any] = None

    # LLM
    response: Optional[LLMResponse] = None
    tier: Optional[str] = None

    # Evaluation
    confidence: Optional[ConfidenceSignals] = None

    # Control
    final_decision: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)

    # Tools (Legacy - to be migrated or synced)
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    
    # Agent Loop (ReAct)
    tool_request: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    step_count: int = 0
    agent_history: List[Dict[str, Any]] = field(default_factory=list)
    
    plan: Optional[Dict[str, Any]] = None
    
    # Observability
    retrieval_signals: Dict[str, Any] = field(default_factory=dict)
    
    # Tool Chaining (Memory)
    tool_memory: Dict[str, Any] = field(default_factory=dict)
    
    # Tool Routing
    next_step: Optional[str] = None
