from dataclasses import dataclass
from typing import Any
from brainbox.pipelines.agent_graph import build_agent_graph
from brainbox.core.state.rag_state import RAGState

@dataclass
class AgentPipelineResult:
    answer: str
    final_decision: str
    raw_result: dict

class AgentPipeline:
    def __init__(self, retriever, client):
        self.graph = build_agent_graph(retriever, client)

    def run(self, question: str) -> AgentPipelineResult:
        state = RAGState(question=question)
        result = self.graph.invoke(state)
        
        # Extract fields
        answer_text = "No answer generated"
        if result.get("response"):
            answer_text = result["response"].text
            
        return AgentPipelineResult(
            answer=answer_text,
            final_decision=result.get("final_decision", "UNKNOWN"),
            raw_result=result
        )
