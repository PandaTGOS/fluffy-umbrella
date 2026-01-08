from typing import Optional
from brainbox.core.state.rag_state import RAGState

class ToolRouter:
    def route(self, state: RAGState) -> Optional[str]:
        """
        Decide the next node WITHOUT calling the LLM.
        Return:
          - tool name (e.g. "calculator", "python")
          - "LLM" to invoke agent/tool_decision
          - None to end
        """
        raise NotImplementedError
