from brainbox.core.state.rag_state import RAGState
from brainbox.core.tools.registry import ToolRegistry
from brainbox.core.tools.calculator import CalculatorTool
from brainbox.pipelines.nodes.agent_nodes import tool_execution_node

def verify_chaining_mechanism():
    print("Verifying Tool Chaining Logic...")
    
    # 1. Setup Registry
    registry = ToolRegistry({"calculator": CalculatorTool()})
    
    # 2. Simulate State with partial memory
    # Pretend previous step calculated 50
    state = RAGState(
        question="dummy",
        tool_memory={"calculator": 50}, # Stored result
        tool_request={
            "name": "calculator",
            "input": {
                "expression": {"from_memory": "calculator"} # Reference to memory
                # expression expects string, but calculator receives resolved value.
                # If memory has 50 (int), resolved input is 50.
                # Calculator tool needs to handle int or string?
                # Calculator `eval(expression)` might fail on int?
                # Let's check CalculatorTool.
                # It does `eval(expression)`. `eval(50)` -> TypeError? No, eval arg must be string.
                # `eval("50")` is fine.
                # So if memory has 50 (int), we might need to cast to string or update CalculatorTool?
                # Or we assume memory stores "50" string.
                # Let's assume memory stores 50 (int).
                # We should update CalculatorTool to str(expression) if needed, or pass string " + 10".
                # Wait, if input is `{"expression": {"from_memory": ...}}`, resolved input is `{"expression": 50}`.
                # CalculatorTool: result = eval(input["expression"]).
                # `eval(50)` raises TypeError.
                # So CalculatorTool is brittle for chaining non-strings unless modified.
                # BUT the user Example B shows: `{"from_memory": "python.locals.x"}`.
                # Ideally Tools should handle typed inputs.
                # I will define `expression` as "50 + 50" (String construction) which needs F-string-like behavior?
                # The provided resolution logic only replaces the *entire value*. 
                # It does NOT do string interpolation `"{from_memory} + 10"`.
                # So `calculator` can only consume the *entire output* of another tool?
                # If Calculator output is 50. Next step `calculator` needs `50 + 10`.
                # How to construct `50 + 10` using `from_memory`?
                # I cannot: `{"expression": {"from_memory": "calculator"}}` -> expression=50. `eval(50)` fails.
                # So Calculator is a bad example for Direct Chaining without interpolation.
                # UNLESS the *previous tool* outputted "50 + 10".
                # OR I use Python tool which is flexible.
            }
        }
    )
    
    # Let's test Python Tool instead, it's generic.
    from brainbox.core.tools.python_exec import PythonExecutionTool
    registry = ToolRegistry({
        "python": PythonExecutionTool()
    })
    
    # Scenario:
    # Memory has `{"python": 42}`.
    # Next request: `python` code: `print(x + 10)`? NO.
    # Python tool input is `code`.
    # How to pass `42` into `code`?
    # `{"code": {"from_memory": ...}}` -> code=42. Executing `42` does nothing.
    # We need `code="x = 42; print(x+10)"`.
    # The resolution logic provided by user ("Resolve Tool Inputs BEFORE Execution") 
    # ONLY resolves exact dict values `{"key": {"from_memory": ...}}`.
    # It does NOT inject variables into code strings.
    # This mechanism is for passing *Data Objects* to tools that accept *Data Arguments*.
    # e.g. `summarize(text=previous_output)`.
    # Calculator accepts `expression`.
    # If I make a `IdentityTool` or `AggregatorTool`, it works.
    
    # Let's Modify Calculator Tool to accept `val1, val2, op`? No, it takes expression.
    
    # Wait, `eval(str(50))` works.
    # So if I wrap the resolution to stringify?
    # Or just store string in memory?
    
    # Let's manually store string "50" in memory for the test.
    state = RAGState(
        question="dummy",
        tool_memory={"step1": "50+50"}, # Previous tool (hypothetically) returned "50+50" string
        tool_request={
            "name": "calculator",
            "input": {"expression": {"from_memory": "step1"}}
        }
    )
    registry = ToolRegistry({"calculator": CalculatorTool()})
    
    print(f"Memory: {state.tool_memory}")
    print(f"Request: {state.tool_request}")
    
    result = tool_execution_node(state, registry)
    print(f"Result: {result}")
    
    assert result["tool_result"] == 100
    print("SUCCESS: Input resolved from memory.")

if __name__ == "__main__":
    verify_chaining_mechanism()
