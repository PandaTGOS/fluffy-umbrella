from typing import Dict, Any
from brainbox.core.state.rag_state import RAGState
from brainbox.core.agents.registry import AgentRegistry
from brainbox.core.agents.router import AgentRouter

def agent_node(state: RAGState, agent_registry: AgentRegistry, agent_router: AgentRouter) -> Dict[str, Any]:
    """
    Generic Agent Node.
    Orchestrates: Router -> Selection -> Execution.
    """
    # 1. Decide which Agent to use
    agent_name = agent_router.route(state)
    print(f"[AGENT_NODE] Selected Agent: {agent_name}")
    
    # 2. Retrieve Agent
    try:
        agent = agent_registry.get(agent_name)
    except ValueError:
         return {"final_decision": f"ERROR: Agent '{agent_name}' not found."}

    # 3. Execute
    try:
        result = agent.run(state)
    except Exception as e:
        return {"final_decision": f"ABORT_AGENT_EXEC_ERROR: {str(e)}"}
        
    return result
