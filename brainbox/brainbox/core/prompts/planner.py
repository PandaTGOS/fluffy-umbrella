class PlannerPrompt:
    def __init__(self, question: str, tools: list[str]):
        self.system_instruction = (
            "You are a planner. "
            "Break the task into minimal steps. "
            "Do NOT execute anything. "
            "Focus on sequential logic."
        )

        self.user_input = f"""
Question:
{question}

Available tools:
{tools}

RULES:
1. Output valid JSON list of steps.
2. For Calculator, use input format: {{ "expression": "10 + 5" }}
3. For Python, use input format: {{ "code": "print('hello')" }}
4. Ensure commas between steps.

Example Output:
{{
  "steps": [
    {{
      "action": "tool",
      "name": "calculator",
      "input": {{ "expression": "10 + 5" }},
      "thought": "First calculate input."
    }},
    {{
      "action": "answer",
      "name": null,
      "input": {{ "answer": "15" }},
      "thought": "Final answer."
    }}
  ]
}}

Output JSON ONLY:
"""
