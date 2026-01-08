import json
import re
from typing import Dict, Any, TYPE_CHECKING
from brainbox.core.state.rag_state import RAGState
from brainbox.core.tools.registry import ToolRegistry
from brainbox.core.llm import LLMResponse
from brainbox.core.prompts.react import ReActPrompt
from brainbox.core.context import Context

if TYPE_CHECKING:
    from brainbox.core.llm.llm_client import LLMClient

# --- 1. Tool Decision Node ---
def tool_decision_node(
    state: RAGState, 
    client: "LLMClient", 
    tool_registry: ToolRegistry
) -> Dict[str, Any]:
    
    # Build Prompt
    prompt = ReActPrompt(
        question=state.question,
        tool_specs=tool_registry.specs(),
        history=state.agent_history,
        context=Context(documents=state.documents) # Include retrieved docs if any
    )
    
    # Generate
    response = client.generate(
        system_instruction=prompt.system_instruction,
        user_input=prompt.user_input,
        context=None,
        output_schema=None, 
        # schema=None because we manually instruct JSON in prompt for broader compatibility
        # though supported models could use output_schema used in previous phases
        runtime_options={"temperature": 0.0}
    )
    
    text = response.text.strip()
    
    # Extract JSON
    match = re.search(r"(\{.*\})", text, re.DOTALL) # Relaxed match
    if match:
        json_str = match.group(1)
    else:
        # Fallback: Check for "Final Answer:" pattern if JSON is missing
        final_answer_match = re.search(r"Final Answer:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
        if final_answer_match:
            print(f"[WARN] Agent returned text but 'Final Answer' detected. Using fallback.")
            json_str = json.dumps({
                "final_answer": final_answer_match.group(1).strip(),
                "tool": None,
                "thought": "Fallback extraction from text."
            })
        else:
            # Fail safe - PRINT FULL OUTPUT for debugging
            print(f"[ERROR] Agent returned no JSON. Full Output:\n{text}")
            return {"final_decision": "ABORT_AGENT_ERROR"}

    try:
        decision = json.loads(json_str)
    except Exception:
        # Double check fallback if JSON parse failed (unlikely if we just reconstructed it, but good practice)
        print(f"[ERROR] Agent JSON parse error. Extracted string:\n{json_str}\nFull Output:\n{text}")
        return {"final_decision": "ABORT_AGENT_ERROR"}

    # Process Decision
    updates = {}
    
    # Add valid thought to history (partial step)
    # We await execution to commit full step, OR we accept thought now?
    # Better to just return updates. The history is updated in OBSERVATION node usually?
    # Or we append this "turn" to history?
    # Let's start a new history item if tool request.
    
    if decision.get("tool") and decision["tool"].get("name"):
        tool_req = decision["tool"]
        # Embed thought so it persists to observation step
        tool_req["thought"] = decision.get("thought", "")
        
        updates["tool_request"] = tool_req
        updates["step_count"] = state.step_count + 1
        
    elif decision.get("final_answer"):
        final_ans = decision["final_answer"]
        updates["tool_request"] = None
        updates["final_decision"] = "ANSWER_READY"
        
        updates["response"] = LLMResponse(
            text=final_ans,
            model_name=response.model_name,
            token_usage={},
            raw_output=None
        ) 
        
    return updates


# --- 2. Tool Execution Node ---
def resolve_tool_input(input_data: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
    def resolve(value):
        if isinstance(value, dict) and "from_memory" in value:
            path = value["from_memory"].split(".")
            cur = memory
            for p in path:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return f"<Error: Path {p} not found in memory>" # Or handle gracefully
            return cur
        return value

    return {k: resolve(v) for k, v in input_data.items()}

def tool_execution_node(state: RAGState, tool_registry: ToolRegistry) -> Dict[str, Any]:
    # Support Router-based execution (next_step) OR LLM-based execution (tool_request)
    tool_name = state.next_step
    
    # Fallback to tool_request if next_step unset
    if not tool_name and state.tool_request:
        tool_name = state.tool_request.get("name")
    
    if not tool_name:
        return {} 
        
    req = state.tool_request or {} # Input comes from tool_request (set by Router or LLM)
    raw_input = req.get("input", {})
    
    tool = tool_registry.get(tool_name)
    if not tool:
        return {"tool_result": f"Error: Tool '{tool_name}' not found."}
        
    # Resolve Inputs from Memory
    resolved_input = resolve_tool_input(raw_input, state.tool_memory)

    try:
        print(f"[AGENT] Executing {tool_name} with {resolved_input}")
        result = tool.run(resolved_input)
    except Exception as e:
        result = f"Error executing tool: {str(e)}"
        
    # Update Memory
    new_memory = state.tool_memory.copy() # Shallow copy
    new_memory[tool_name] = result # Store by tool name (simple)
    
    return {
        "tool_result": result,
        "tool_memory": new_memory
    }


# --- 3. Observation Node ---
def observation_node(state: RAGState) -> Dict[str, Any]:
    # Commit the step to history
    # tool_request might be None if Router short-circuited and state usage failed?
    # Or if tool_execution failed?
    
    req = state.tool_request or {}
    thought = req.get("thought", "Router Shortcut")
    tool_name = req.get("name") or state.next_step or "Unknown"
    tool_input = req.get("input", {})
    
    new_step = {
        "thought": thought,
        "tool_request": {
            "name": tool_name, 
            "input": tool_input
        },
        "tool_result": state.tool_result
    }
    
    updated_history = state.agent_history + [new_step]
    
    return {
        "agent_history": updated_history,
        "tool_request": None, # Reset
        "tool_result": None,
        "next_step": None # Reset router step
    }
