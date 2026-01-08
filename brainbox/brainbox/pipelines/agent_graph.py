from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from brainbox.core.state.rag_state import RAGState
from brainbox.pipelines.nodes.agent_nodes import tool_execution_node, observation_node
from brainbox.pipelines.nodes.router_nodes import router_node
from brainbox.pipelines.nodes.agent_node import agent_node
from brainbox.core.tools.default_router import DefaultToolRouter
from brainbox.pipelines.nodes.tool_execute import TOOLS 
from brainbox.core.tools.registry import ToolRegistry
from brainbox.core.agents.registry import AgentRegistry
from brainbox.core.agents.router import AgentRouter
from brainbox.core.agents.tool_agent import ToolAgent
from brainbox.core.agents.retrieval_agent import RetrievalAgent
from brainbox.core.agents.critic_agent import CriticAgent
from brainbox.core.agents.answer_agent import AnswerAgent

def retrieve_node(state: RAGState, retriever) -> Dict[str, Any]:
    """Retrieves documents based on the question."""
    # Returns RetrievalResult
    result = retriever.retrieve(state.question)
    return {
        "documents": result.documents,
        "retrieval_signals": result.signals
    }

# Router Conditional Edge (System Router)
def router_conditional(state: RAGState):
    next_step = state.next_step
    
    if next_step == "LLM":
        return "agent_node" # Route to Agent Selector
    elif next_step:
        return "tool_execute" # Deterministic Tool
    else:
        return "agent_node" # Default Fallback

def agent_conditional(state: RAGState):
    """Decides where to go after Agent Execution."""
    if state.final_decision == "ANSWER_READY":
        return "critic_node" # Verify before ending
    
    if state.step_count >= 5:
        return END 
        
    if state.tool_request:
        return "tool_execute"
    
    return END

def build_agent_graph(retriever, client):
    """Builds and compiles the Multi-Agent LangGraph."""
    graph = StateGraph(RAGState)
    
    # 1. Tool Logic
    tool_registry = ToolRegistry(TOOLS)
    tool_router = DefaultToolRouter()
    
    # 2. Agent Logic
    agent_registry = AgentRegistry()
    agent_registry.register(ToolAgent(client, tool_registry))
    agent_registry.register(RetrievalAgent(client))
    agent_registry.register(CriticAgent())
    agent_registry.register(AnswerAgent(client))
    
    agent_router = AgentRouter()

    # Add Nodes
    graph.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    graph.add_node("router", lambda s: router_node(s, tool_router))
    graph.add_node("agent_node", lambda s: agent_node(s, agent_registry, agent_router))
    graph.add_node("tool_execute", lambda s: tool_execution_node(s, tool_registry))
    graph.add_node("observation", observation_node)
    
    critic_agent = agent_registry.get("critic_agent")
    graph.add_node("critic_node", lambda s: critic_agent.run(s))
    
    # Edges
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "router")
    
    graph.add_conditional_edges("router", router_conditional)
    graph.add_conditional_edges("agent_node", agent_conditional)
    graph.add_edge("tool_execute", "observation")
    graph.add_edge("observation", "router")
    graph.add_edge("critic_node", END)
    
    return graph.compile()
