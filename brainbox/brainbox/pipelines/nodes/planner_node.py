import json
import re
from typing import Dict, Any
from brainbox.core.prompts.planner import PlannerPrompt
from brainbox.core.state.rag_state import RAGState

def planner_node(state: RAGState, client, tool_registry) -> Dict[str, Any]:
    print("[PLANNER] Generating Plan...")
    prompt = PlannerPrompt(
        question=state.question,
        tools=list(tool_registry.tools.keys())
    )

    response = client.generate(
        system_instruction=prompt.system_instruction,
        user_input=prompt.user_input,
        runtime_options={"temperature": 0.0}
    )
    
    text = response.text.strip()
    print(f"[DEBUG_PLANNER_RAW] {text}")
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    
    plan = {"steps": []}
    if match:
        try:
            plan = json.loads(match.group(1))
        except:
            pass
            
    # Normalize keys if needed or validate
    if "steps" not in plan:
        plan = {"steps": []}
        
    print(f"[PLANNER] Plan Generated: {len(plan['steps'])} steps.")
    return {"plan": plan}
