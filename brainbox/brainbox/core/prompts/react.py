from dataclasses import dataclass
from typing import Optional, List, Any, Dict
import json
from brainbox.core.context import Context

@dataclass
class ReActPrompt:
    question: str
    tool_specs: List[Dict[str, Any]]
    history: List[Dict[str, Any]] # List of {thought, tool_request, tool_result}
    context: Optional[Context] = None

    @property
    def system_instruction(self) -> str:
        specs_str = json.dumps(self.tool_specs, indent=2)
        base = f"""You are a reasoning agent capable of using tools to answer questions.

Available Tools:
{specs_str}

**Instructions**:
1. Reasoning: Always think about what to do next. checking previous observations.
2. Tool Use: If you need more information or need to perform an action, output a tool request. You MUST provide 'input' arguments.
3. Final Answer: If you have enough information, output the final answer.

**Output Format**:
Respond ONLY with valid JSON.

Example 1 (Math):
{{
  "thought": "I need to calculate 15 * 6 first.",
  "tool": {{
    "name": "calculator",
    "input": {{ "expression": "15 * 6" }}
  }},
  "final_answer": null
}}

Example 2 (Final Answer):
{{
  "thought": "I have calculated the result to be 80.",
  "tool": null,
  "final_answer": "80"
}}

Your Output:
"""

        return base

    @property
    def user_input(self) -> str:
        # Build prompt with history
        prompt = f"Question: {self.question}\n\n"
        
        if self.context and self.context.documents:
            prompt += "Retrieved Context (if helpful):\n"
            for i, doc in enumerate(self.context.documents):
                prompt += f"[{i+1}] {doc.content}\n"
            prompt += "\n"

        prompt += "History:\n"
        for step in self.history:
            prompt += f"Thought: {step.get('thought')}\n"
            if step.get('tool_request'):
                req = step['tool_request']
                prompt += f"Action: Call {req['name']} with {json.dumps(req['input'])}\n"
            if step.get('tool_result') is not None:
                # Truncate result if too long?
                res = str(step['tool_result'])
                if len(res) > 500: res = res[:500] + "...(truncated)"
                prompt += f"Observation: {res}\n"
            prompt += "\n"
        
        prompt += "Next Step:"
        return prompt

    def build(self):
        # Returns self or dict representation for LLMClient
        # LLMClient expects schema-agnostic usually, providing system/user separately
        return self
