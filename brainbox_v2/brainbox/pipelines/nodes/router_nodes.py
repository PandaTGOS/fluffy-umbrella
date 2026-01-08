from typing import Dict, Any
from brainbox.core.state.rag_state import RAGState
from brainbox.core.tools.router import ToolRouter

def router_node(state: RAGState, router: ToolRouter) -> Dict[str, Any]:
    next_step = router.route(state)
    
    if next_step is None:
        # Default behavior if router returns None? End or Error?
        # Specification says "None to end".
        return {"final_decision": "NO_ROUTE"}

    updates = {"next_step": next_step}
    
    # If the Router populated tool_request (deterministic route), persist it!
    # Note: DefaultToolRouter modifies state in-place. We must return it to update Graph State.
    if state.tool_request:
        updates["tool_request"] = state.tool_request
        
    return updates
