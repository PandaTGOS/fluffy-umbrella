from typing import Dict, Any
from brainbox.core.tools.base import Tool

class PythonExecutionTool(Tool):
    name = "python"
    description = "Executes Python code and returns stdout or result."
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"}
        },
        "required": ["code"]
    }

    def run(self, input: Dict[str, Any]) -> Any:
        code = input.get("code")
        if not code:
            raise ValueError("Missing code")

        # Normalize code
        import textwrap
        code = textwrap.dedent(code).strip()
        print(f"[DEBUG] Executing Code:\n{code!r}")

        local_env = {}
        stdout = []

        def safe_print(*args):
            stdout.append(" ".join(map(str, args)))

        local_env["print"] = safe_print

        try:
            exec(code, {}, local_env)
        except Exception as e:
            return {"error": str(e)}

        # Filter out internal/callable items from locals to keep output clean
        clean_locals = {k: v for k, v in local_env.items() if k != "print" and not callable(v)}

        return {
            "stdout": stdout,
            "locals": clean_locals,
        }
