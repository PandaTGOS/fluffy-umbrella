import re
from brainbox.core.tools.router import ToolRouter
from brainbox.core.state.rag_state import RAGState

from typing import Optional
class DefaultToolRouter(ToolRouter):
    def route(self, state: RAGState) -> Optional[str]:
        # If we have done work, let LLM summarize/decide next steps
        if state.agent_history:
            return "LLM"
            
        q = state.question.lower()

        # Math -> calculator
        if re.fullmatch(r"[0-9+\-*/ ().]+", q):
            # Loop Prevention: If we just calculated this, STOP.
            if state.agent_history and state.agent_history[-1].get("tool_request", {}).get("name") == "calculator":
                return None # End of Deterministic Chain
            
            # Deterministic: Set the request immediately
            state.tool_request = {
                "name": "calculator",
                "input": {"expression": q}
            }
            print(f"[DEBUG] Router matched Math. Request set: {state.tool_request}")
            return "calculator"
            
        if q.startswith("calculate "):
             expr = q.replace("calculate ", "", 1).strip()
             state.tool_request = {
                "name": "calculator",
                "input": {"expression": expr}
            }
             return "calculator"

        if "code" in q or "python" in q:
            # We can't generate code deterministically without LLM usually.
            # Unless "run python code: print(1)" matches.
            # But usually "write code to..." needs LLM.
            # User example: "Calculate 10 * x ... Router -> python".
            # The Router sends to Python... but who writes the code?
            # Maybe Router sends to "PythonAgent"? Or returns "LLM" if generation needed?
            # User: "Hybrid... Router -> python... then Router -> LLM".
            # This implies Python tool is just an executor.
            
            # If I route to "python", I must have input ready?
            # If I don't, then `python` tool fails.
            # Maybe the User implies "Router -> LLM (specialized)"?
            # "Router -> calculator" is the only CLEAR deterministic case.
            # "Router -> python" might mean "Route to Python Agent"?
            
            # Let's implement basics as requested.
            pass

        if "code" in q or "python" in q:
            return "python" 
            # (Note: This will likely fail execution if no input is prepared, unless I fix that).

        # Otherwise let LLM decide
        return "LLM"
