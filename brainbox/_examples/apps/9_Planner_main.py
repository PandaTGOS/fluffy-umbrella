from langgraph.graph import StateGraph, END
from brainbox.core.state.rag_state import RAGState
from brainbox.core.llm import OllamaClient
from brainbox.core.tools.registry import ToolRegistry
from brainbox.pipelines.nodes.planner_node import planner_node
from brainbox.pipelines.nodes.executor_node import executor_node
from brainbox.core.tools.calculator import CalculatorTool
from brainbox.core.tools.python_exec import PythonExecutionTool

def executor_conditional(state: RAGState):
    """
    If plan still has steps, continue executing.
    Else, if we have an answer, end.
    """
    if state.final_decision == "ANSWER_READY":
        return END
        
    if state.plan and state.plan.get("steps"):
        return "executor"
        
    return END

def build_planner_graph(client):
    print("Building Planner/Executor Graph...")
    graph = StateGraph(RAGState)
    
    # Tools
    tools = {
        "calculator": CalculatorTool(),
        "python": PythonExecutionTool()
    }
    tool_registry = ToolRegistry(tools)

    # Nodes
    graph.add_node("planner", lambda s: planner_node(s, client, tool_registry))
    graph.add_node("executor", lambda s: executor_node(s, tool_registry))

    # Edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    
    # Conditional Loop for Executor
    graph.add_conditional_edges("executor", executor_conditional)
    
    return graph.compile()


def verify_planner():
    print("Verifying Planner/Executor Architecture (Phase B)...")
    client = OllamaClient()
    pipeline = build_planner_graph(client)
    
    # Scenario: Multi-step Math
    # "Calculate 10 + 5 then multiply by 2"
    print("\n[Scenario 1] Multi-step Plan")
    state = RAGState(question="Calculate 10 + 5 then multiply by 2")
    
    result = pipeline.invoke(state)
    
    print(f"Final Plan State: Steps remaining: {len(result.get('plan', {}).get('steps', []))}")
    print(f"History Length: {len(result.get('agent_history', []))}")
    
    # Check if answer produced
    response = result.get("response")
    if response:
        print(f"Final Response: {response.text}")
    else:
        # Check history for results
        history = result.get('agent_history', [])
        for entry in history:
            print(f"Step Result: {entry.get('tool')} -> {entry.get('result')}")

if __name__ == "__main__":
    verify_planner()
