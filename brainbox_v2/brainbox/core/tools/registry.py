from typing import Dict, List, Optional
from brainbox.core.tools.base import Tool

class ToolRegistry:
    _global_tools: Dict[str, Tool] = {}

    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools.copy()
        # Merge global tools
        self.tools.update(self._global_tools)

    @classmethod
    def register(cls, tool: Tool):
        cls._global_tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def specs(self) -> List[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self.tools.values()
        ]
