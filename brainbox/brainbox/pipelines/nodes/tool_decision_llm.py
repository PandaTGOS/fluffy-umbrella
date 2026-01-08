import json
from typing import Dict, Any, TYPE_CHECKING
from brainbox.core.state.rag_state import RAGState
from brainbox.core.tools.registry import ToolRegistry

# Avoid circular import if needed (but here imports are clean)
if TYPE_CHECKING:
    from brainbox.core.llm.llm_client import LLMClient

TOOL_DECISION_SYSTEM = """
You are a tool-using agent.

Available tools:
{tools}

Decide whether a tool is required to answer the user's question.
- If the question involves math, ALWAYS use the 'calculator' tool.
- If the question involves code, ALWAYS use the 'python' tool.
- Do NOT answer the question yourself if a tool can do it.

Respond ONLY with valid JSON in one of the following forms:

1. If a tool is needed:
{{
  "tool": "<tool_name>",
  "input": {{ ... }}
}}

2. If no tool is needed:
{{
  "tool": null
}}
"""

import re

def tool_decision_llm_node(
    state: RAGState,
    client: "LLMClient",
    tool_registry: ToolRegistry,
) -> Dict[str, Any]:

    tool_specs = tool_registry.specs()
    # Format specs nicely for prompt
    specs_str = json.dumps(tool_specs, indent=2)

    response = client.generate(
        system_instruction=TOOL_DECISION_SYSTEM.format(tools=specs_str),
        user_input=state.question,
        context=None,
        output_schema=None,
        runtime_options={"temperature": 0.0},
    )

    text = response.text.strip()
    
    # Robust extraction: look for JSON block
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Try finding first { to last }
        match_loose = re.search(r"(\{.*\})", text, re.DOTALL)
        if match_loose:
             json_str = match_loose.group(1)
        else:
             json_str = text # Hope for the best

    try:
        decision = json.loads(json_str)
    except Exception as e:
        print(f"[ERROR] LLM returned invalid JSON: {text[:100]}...")
        return {}  # fail closed
    except Exception as e:
        print(f"[ERROR] LLM returned invalid JSON: {text}")
        return {}  # fail closed

    if decision.get("tool"):
        # Validate tool exists
        if tool_registry.get(decision["tool"]):
            print(f"[DECISION] LLM selected tool: {decision['tool']}")
            return {
                "tool_name": decision["tool"],
                "tool_input": decision.get("input", {}),
            }
        else:
             print(f"[WARNING] LLM selected unknown tool: {decision['tool']}")

    return {}
