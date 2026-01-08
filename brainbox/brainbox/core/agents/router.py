from brainbox.core.state.rag_state import RAGState

class AgentRouter:
    def route(self, state: RAGState) -> str:
        """
        Decides which AGENT (Brain) to invoke.
        This is called only when the System Router decides we need an LLM.
        """
        # Heuristic: If we have documents retrieved, use Retrieval Agent?
        # Or if the Question implies retrieval?
        # For now, if documents are present (pre-retrieved), use Retrieval Agent?
        # BUT ToolAgent uses RAG Context too.
        # User Logic: "if state.documents: return 'retrieval_agent'"
        
        if state.documents:
            return "retrieval_agent"
            
        # Simple heuristic for tools
        tool_keywords = ["calc", "python", "code", "weather", "search", "multiply", "divide"]
        q = state.question.lower()
        if any(k in q for k in tool_keywords):
            return "tool_agent"
            
        return "answer_agent"
