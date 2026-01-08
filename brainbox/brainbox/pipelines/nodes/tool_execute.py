from typing import Dict, Any
from brainbox.core.state.rag_state import RAGState
from brainbox.core.tools.calculator import CalculatorTool
from brainbox.core.tools.python_exec import PythonExecutionTool

TOOLS = {
    "calculator": CalculatorTool(),
    "python": PythonExecutionTool(),
}

def tool_execute_node(state: RAGState) -> Dict[str, Any]:
    tool = TOOLS.get(state.tool_name)
    if not tool:
        return {}

    try:
        output = tool.run(state.tool_input)
        return {"tool_output": output}
    except Exception as e:
        return {"tool_output": f"Error: {str(e)}"}
