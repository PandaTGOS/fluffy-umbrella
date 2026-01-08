from typing import Dict, Any, TYPE_CHECKING
import json
import re
from brainbox.core.agents.base import Agent
from brainbox.core.prompts.react import ReActPrompt
from brainbox.core.context import Context
from brainbox.core.llm import LLMResponse
from brainbox.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from brainbox.core.llm.llm_client import LLMClient

class ToolAgent(Agent):
    name = "tool_agent"

    def __init__(self, client: "LLMClient", tool_registry: ToolRegistry):
        self.client = client
        self.tool_registry = tool_registry

    def run(self, state) -> Dict[str, Any]:
        prompt = ReActPrompt(
            question=state.question,
            tool_specs=self.tool_registry.specs(),
            history=state.agent_history,
            context=Context(documents=state.documents) 
        )

        response = self.client.generate(
            system_instruction=prompt.system_instruction,
            user_input=prompt.user_input,
            context=None,
            output_schema=None,
            runtime_options={"temperature": 0.0}
        )
        
        # --- Parsing Logic (Ported from tool_decision_node) ---
        text = response.text.strip()
        
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        json_str = None
        
        if match:
            json_str = match.group(1)
        else:
            # Fallback for "Final Answer:" text
            final_answer_match = re.search(r"Final Answer:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
            if final_answer_match:
                json_str = json.dumps({
                    "final_answer": final_answer_match.group(1).strip(),
                    "tool": None,
                    "thought": "Fallback extraction from text."
                })
        
        if not json_str:
             return {"final_decision": "ABORT_AGENT_ERROR"}

        try:
            decision = json.loads(json_str)
        except Exception:
            return {"final_decision": "ABORT_AGENT_ERROR"}
            
        updates = {}
        
        if decision.get("tool") and decision["tool"].get("name"):
            tool_req = decision["tool"]
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
