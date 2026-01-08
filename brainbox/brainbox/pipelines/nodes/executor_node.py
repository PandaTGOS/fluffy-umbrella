from typing import Dict, Any
from brainbox.core.state.rag_state import RAGState
from brainbox.core.llm import LLMResponse

def executor_node(state: RAGState, tool_registry) -> Dict[str, Any]:
    plan = state.plan
    if not plan or not plan.get("steps"):
        return {"final_decision": "DONE"}

    # Pop the next step
    step = plan["steps"].pop(0)

    print(f"[EXECUTOR] Processing Step: {step.get('action')} - {step.get('name', '')}")
    
    if step["action"] == "answer":
        # Final Answer
        answer_text = step.get("input", {}).get("answer", "No answer provided.")
        return {
            "response": LLMResponse(text=answer_text, model_name="planner", token_usage={}, raw_output=None),
            "final_decision": "ANSWER_READY",
            "plan": plan # Update plan (consumed step)
        }

    if step["action"] == "tool":
        tool_name = step.get("name")
        try:
            tool = tool_registry.get(tool_name)
            tool_input = step.get("input", {})
            
            print(f"[EXECUTOR] Running Tool: {tool_name} with {tool_input}")
            result = tool.run(tool_input)
            
            # Log to history
            entry = {
                "thought": step.get("thought", ""),
                "tool": tool_name,
                "result": result
            }
            # Append to history (must return list update if state defines history as list)
            # RAGState defines agent_history as List. We append to a new list or return updated list?
            # LangGraph usually merges.
            new_history = state.agent_history + [entry]
            
            return {
                "plan": plan,
                "agent_history": new_history
            }
            
        except Exception as e:
            print(f"[EXECUTOR] Tool Error: {e}")
            return {"plan": plan} # Continue? Or Fail? For now continue.

    return {"plan": plan}
