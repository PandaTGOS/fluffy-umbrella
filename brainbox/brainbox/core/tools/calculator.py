import math
from typing import Dict, Any
from brainbox.core.tools.base import Tool

class CalculatorTool(Tool):
    name = "calculator"
    description = "Performs mathematical calculations safely."
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Mathematical expression to evaluate (e.g., '2 + 2')"}
        },
        "required": ["expression"]
    }

    def run(self, input: Dict[str, Any]) -> Any:
        expression = input.get("expression")
        if not expression:
            raise ValueError("Missing expression")

        # VERY IMPORTANT: restricted eval
        allowed = {
            "__builtins__": {},
            "math": math,
        }

        # We'll allow basic float/int types in the return
        return eval(expression, allowed, {})
