from dataclasses import dataclass
from typing import Any
from brainbox.pipelines.rag_graph import build_rag_graph
from brainbox.core.state.rag_state import RAGState

@dataclass
class PipelineResult:
    answer: str
    final_decision: str
    confidence: Any
    raw_result: dict

class RAGPipeline:
    def __init__(self, retriever, client):
        self.graph = build_rag_graph(retriever, client)

    def run(self, question: str) -> PipelineResult:
        state = RAGState(question=question)
        result = self.graph.invoke(state)
        
        # Extract fields
        answer_text = "No answer generated"
        if result.get("response"):
            answer_text = result["response"].text
            
        return PipelineResult(
            answer=answer_text,
            final_decision=result.get("final_decision", "UNKNOWN"),
            confidence=result.get("confidence", 0.0),
            raw_result=result
        )
